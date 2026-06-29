"""
Training Module
================
Full training pipeline for the Siamese clone detection model.

Features:
  - Mixed precision training (FP16)
  - Gradient accumulation
  - Learning rate scheduling (cosine, linear, warmup)
  - Early stopping
  - Checkpoint management
  - TensorBoard logging
  - Threshold optimization on validation set
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class SiameseTrainer:
    """
    Trainer for the Siamese clone detection model.

    Handles the full training loop including:
    - Optimizer and scheduler setup
    - Mixed precision training
    - Validation and metric computation
    - Early stopping
    - Checkpoint saving/loading
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        output_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        train_config = config.get("training", {})
        self.epochs = train_config.get("epochs", 10)
        self.lr = train_config.get("learning_rate", 2e-5)
        self.weight_decay = train_config.get("weight_decay", 0.01)
        self.warmup_ratio = train_config.get("warmup_ratio", 0.1)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)
        self.fp16 = train_config.get("fp16", True)
        self.log_every = train_config.get("log_every_n_steps", 100)
        self.eval_every = train_config.get("eval_every_n_steps", 500)
        self.patience = train_config.get("early_stopping_patience", 3)
        self.save_best_only = train_config.get("save_best_only", True)

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Total steps
        self.total_steps = self.epochs * len(self.train_loader)
        warmup_steps = int(self.total_steps * self.warmup_ratio)
        self.scheduler = self._build_scheduler(warmup_steps)

        # FP16 scaler
        self.scaler = GradScaler() if (self.fp16 and self.device.type == "cuda") else None

        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.output_dir / "logs"))
        except ImportError:
            self.writer = None
            logger.warning("TensorBoard not available")

        # Training state
        self.global_step = 0
        self.best_val_metric = -float("inf")
        self.patience_counter = 0
        self.training_history = []

        logger.info(
            f"Trainer initialized: device={self.device}, "
            f"epochs={self.epochs}, steps={self.total_steps}, "
            f"fp16={self.fp16 and self.scaler is not None}"
        )

    def _build_optimizer(self):
        """Build AdamW optimizer with layer-wise learning rate decay."""
        # Separate parameters for LR decay: encoder layers get lower LR
        no_decay = {"bias", "LayerNorm.weight", "layer_norm.weight"}

        optimizer_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]

        return torch.optim.AdamW(optimizer_groups, lr=self.lr)

    def _build_scheduler(self, warmup_steps: int):
        """Build cosine learning rate scheduler with warmup."""
        from transformers import get_cosine_schedule_with_warmup

        return get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

    def train(self) -> Dict:
        """
        Full training loop.

        Returns:
            Training history dict with metrics per epoch.
        """
        logger.info(f"Starting training for {self.epochs} epochs")
        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{self.epochs}")
            logger.info(f"{'='*60}")

            # Train one epoch
            train_metrics = self._train_epoch(epoch)

            # Validate
            val_metrics = self._validate()

            # Log epoch results
            epoch_metrics = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_roc_auc": val_metrics["roc_auc"],
                "lr": self.scheduler.get_last_lr()[0],
            }
            self.training_history.append(epoch_metrics)

            self._log_metrics(epoch_metrics, step=epoch, prefix="epoch")

            logger.info(
                f"Epoch {epoch}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_f1={val_metrics['f1']:.4f}, "
                f"val_auc={val_metrics['roc_auc']:.4f}"
            )

            # Save checkpoint
            is_best = val_metrics["f1"] > self.best_val_metric
            if is_best:
                self.best_val_metric = val_metrics["f1"]
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_metrics, is_best=True)
                logger.info(f"  ✓ New best model (val_f1={self.best_val_metric:.4f})")
            else:
                self.patience_counter += 1
                if not self.save_best_only:
                    self._save_checkpoint(epoch, val_metrics, is_best=False)

            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(
                    f"Early stopping triggered after {epoch} epochs "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        elapsed = time.time() - start_time
        logger.info(f"\nTraining complete in {elapsed/60:.1f} minutes")
        logger.info(f"Best val F1: {self.best_val_metric:.4f}")

        # Save training history
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        if self.writer:
            self.writer.close()

        return {"history": self.training_history, "best_val_f1": self.best_val_metric}

    def _train_epoch(self, epoch: int) -> Dict:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(pbar):
            loss = self._train_step(batch)

            total_loss += loss
            n_batches += 1

            # Logging
            if self.global_step % self.log_every == 0:
                avg_loss = total_loss / n_batches
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                if self.writer:
                    self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                    self.writer.add_scalar("train/lr", lr, self.global_step)

        avg_loss = total_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        logger.debug(f"Epoch {epoch} train: loss={avg_loss:.4f}, time={epoch_time:.1f}s")

        return {"loss": avg_loss}

    def _train_step(self, batch: Dict) -> float:
        """Single training step."""
        # Move batch to device
        input_ids_1 = batch["input_ids_1"].to(self.device)
        attn_mask_1 = batch["attention_mask_1"].to(self.device)
        input_ids_2 = batch["input_ids_2"].to(self.device)
        attn_mask_2 = batch["attention_mask_2"].to(self.device)
        labels = batch["label"].float().to(self.device)

        self.optimizer.zero_grad()

        if self.scaler:
            with autocast():
                output = self.model(
                    input_ids_1, attn_mask_1,
                    input_ids_2, attn_mask_2,
                    labels=labels,
                )
                loss = output["loss"]
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            output = self.model(
                input_ids_1, attn_mask_1,
                input_ids_2, attn_mask_2,
                labels=labels,
            )
            loss = output["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.scheduler.step()
        self.global_step += 1

        return loss.item()

    @torch.no_grad()
    def _validate(self) -> Dict:
        """Run validation and compute all metrics."""
        self.model.eval()

        all_sims = []
        all_labels = []
        total_loss = 0.0
        n_batches = 0

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            input_ids_1 = batch["input_ids_1"].to(self.device)
            attn_mask_1 = batch["attention_mask_1"].to(self.device)
            input_ids_2 = batch["input_ids_2"].to(self.device)
            attn_mask_2 = batch["attention_mask_2"].to(self.device)
            labels = batch["label"].float().to(self.device)

            output = self.model(
                input_ids_1, attn_mask_1,
                input_ids_2, attn_mask_2,
                labels=labels,
            )

            all_sims.extend(output["similarity"].cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if "loss" in output:
                total_loss += output["loss"].item()
                n_batches += 1

        # Compute metrics
        from src.evaluation import compute_metrics, find_optimal_threshold

        sims = np.array(all_sims)
        labels_arr = np.array(all_labels)

        # Find optimal threshold on validation set
        threshold, _ = find_optimal_threshold(sims, labels_arr)
        metrics = compute_metrics(sims, labels_arr, threshold=threshold)
        metrics["loss"] = total_loss / max(n_batches, 1)
        metrics["optimal_threshold"] = threshold

        return metrics

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
        }

        if is_best:
            path = self.output_dir / "best_model.pt"
        else:
            path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def _log_metrics(self, metrics: Dict, step: int, prefix: str = ""):
        """Log metrics to TensorBoard."""
        if not self.writer:
            return
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                tag = f"{prefix}/{k}" if prefix else k
                self.writer.add_scalar(tag, v, step)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
    ) -> "SiameseTrainer":
        """Resume training from a checkpoint."""
        trainer = cls(model, train_loader, val_loader, config)
        checkpoint = torch.load(checkpoint_path, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        trainer.global_step = checkpoint["global_step"]
        logger.info(f"Resumed from checkpoint: {checkpoint_path} (epoch {checkpoint['epoch']})")
        return trainer

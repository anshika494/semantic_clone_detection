#!/usr/bin/env python3
"""
Training Script
================
Main entry point for training the Siamese clone detection model.

Usage:
  python scripts/train.py --config configs/config.yaml
  python scripts/train.py --config configs/config.yaml --backbone microsoft/graphcodebert-base
  python scripts/train.py --config configs/config.yaml --epochs 20 --lr 1e-5
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from src.utils import load_config, setup_logging, set_seed, get_environment_info, model_summary
from src.dataset import BigCloneBenchLoader, BigCloneBenchDataset, SyntheticCloneDataset
from src.preprocessing import CodePreprocessor, BatchPreprocessor
from src.ast_processing import ASTProcessor
from src.embeddings import create_embedding_model
from src.models import SiameseCloneDetector
from src.models.trainer import SiameseTrainer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Siamese Clone Detection Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument("--backbone", type=str, default=None,
                        help="Override backbone model (e.g., microsoft/graphcodebert-base)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["contrastive", "cosine", "triplet", "ntxent"],
                        help="Override loss function")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for development without BCB)")
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Limit number of pairs (for quick experiments)")
    parser.add_argument("--use-ast", action="store_true",
                        help="Use AST-linearized tokens as input instead of raw code")
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    return parser.parse_args()


def build_dataloaders(
    train_pairs, val_pairs, test_pairs,
    tokenizer, config, use_ast: bool = False,
):
    """Build train/val/test DataLoaders."""
    max_length = config.get("ast", {}).get("max_token_length", 512)
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = config.get("performance", {}).get("num_workers", 4)
    pin_memory = config.get("performance", {}).get("pin_memory", True)

    # Use AST-linearized or preprocessed code
    if use_ast:
        code1_key = "code1_ast"
        code2_key = "code2_ast"
    else:
        code1_key = "code1_processed"
        code2_key = "code2_processed"

    def remap_pairs(pairs, c1k, c2k):
        """Remap code keys if AST keys don't exist, fall back to processed."""
        remapped = []
        for p in pairs:
            rp = dict(p)
            rp["code1"] = p.get(c1k) or p.get("code1_processed") or p["code1"]
            rp["code2"] = p.get(c2k) or p.get("code2_processed") or p["code2"]
            remapped.append(rp)
        return remapped

    train_data = remap_pairs(train_pairs, code1_key, code2_key)
    val_data = remap_pairs(val_pairs, code1_key, code2_key)
    test_data = remap_pairs(test_pairs, code1_key, code2_key)

    train_dataset = BigCloneBenchDataset(
        train_data, tokenizer=tokenizer, max_length=max_length, split="train"
    )
    val_dataset = BigCloneBenchDataset(
        val_data, tokenizer=tokenizer, max_length=max_length, split="val"
    )
    test_dataset = BigCloneBenchDataset(
        test_data, tokenizer=tokenizer, max_length=max_length, split="test"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def main():
    args = parse_args()

    # Setup logging
    setup_logging(level=args.log_level, log_file="results/train.log")
    logger.info("=" * 70)
    logger.info("Semantic Code Clone Detection — Training")
    logger.info("=" * 70)

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.loss:
        config["model"]["siamese"]["loss"] = args.loss
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.no_fp16:
        config["training"]["fp16"] = False

    # Reproducibility
    set_seed(args.seed)

    # Log environment
    env_info = get_environment_info()
    logger.info(f"Environment: {env_info}")

    # -----------------------------------------------------------------------
    # 1. Load Dataset
    # -----------------------------------------------------------------------
    logger.info("\n[1/5] Loading dataset...")

    if args.synthetic:
        logger.info("Using synthetic dataset for development")
        from sklearn.model_selection import train_test_split
        all_pairs = SyntheticCloneDataset.generate_pairs(n_pairs=2000)
        labels = [p["label"] for p in all_pairs]
        train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2,
                                                    stratify=labels, random_state=42)
        t_labels = [p["label"] for p in train_pairs]
        train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1,
                                                   stratify=t_labels, random_state=42)
    else:
        loader = BigCloneBenchLoader(config)
        train_pairs, val_pairs, test_pairs = loader.load_pairs()

    # Limit pairs if requested
    if args.max_pairs:
        train_pairs = train_pairs[:args.max_pairs]
        val_pairs = val_pairs[:args.max_pairs // 5]

    logger.info(
        f"Dataset: {len(train_pairs)} train, "
        f"{len(val_pairs)} val, {len(test_pairs)} test"
    )

    # -----------------------------------------------------------------------
    # 2. Preprocessing
    # -----------------------------------------------------------------------
    logger.info("\n[2/5] Preprocessing code...")

    batch_preprocessor = BatchPreprocessor(config)

    train_pairs = batch_preprocessor.process_dataset(train_pairs)
    val_pairs = batch_preprocessor.process_dataset(val_pairs, show_progress=False)
    test_pairs = batch_preprocessor.process_dataset(test_pairs, show_progress=False)

    # Optional: AST linearization
    if args.use_ast:
        logger.info("Generating AST token sequences...")
        ast_processor = ASTProcessor(config)

        def add_ast_tokens(pairs):
            for pair in pairs:
                pair["code1_ast"] = ast_processor.get_token_string(pair["code1"])
                pair["code2_ast"] = ast_processor.get_token_string(pair["code2"])
            return pairs

        train_pairs = add_ast_tokens(train_pairs)
        val_pairs = add_ast_tokens(val_pairs)
        test_pairs = add_ast_tokens(test_pairs)

    # -----------------------------------------------------------------------
    # 3. Model & Tokenizer
    # -----------------------------------------------------------------------
    logger.info("\n[3/5] Initializing model...")

    model_instance, tokenizer = create_embedding_model(config)
    model = SiameseCloneDetector(config)

    # Print model summary
    summary = model_summary(model)
    logger.info(summary)

    # -----------------------------------------------------------------------
    # 4. DataLoaders
    # -----------------------------------------------------------------------
    logger.info("\n[4/5] Building DataLoaders...")

    train_loader, val_loader, test_loader = build_dataloaders(
        train_pairs, val_pairs, test_pairs,
        tokenizer, config, use_ast=args.use_ast
    )

    logger.info(
        f"Batches: {len(train_loader)} train, "
        f"{len(val_loader)} val, {len(test_loader)} test"
    )

    # -----------------------------------------------------------------------
    # 5. Training
    # -----------------------------------------------------------------------
    logger.info("\n[5/5] Starting training...")

    output_dir = config.get("training", {}).get("output_dir", "checkpoints")

    if args.resume:
        trainer = SiameseTrainer.load_from_checkpoint(
            args.resume, model, train_loader, val_loader, config
        )
    else:
        trainer = SiameseTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            output_dir=output_dir,
        )

    results = trainer.train()

    logger.info(f"\nTraining complete! Best val F1: {results['best_val_f1']:.4f}")
    logger.info(f"Best model saved to: {output_dir}/best_model.pt")

    # -----------------------------------------------------------------------
    # Final Test Evaluation
    # -----------------------------------------------------------------------
    logger.info("\nRunning final test evaluation...")

    from src.evaluation import CloneDetectionEvaluator
    import numpy as np

    # Load best model
    best_ckpt = Path(output_dir) / "best_model.pt"
    if best_ckpt.exists():
        checkpoint = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    device = next(model.parameters()).device
    model.eval()

    all_sims, all_labels, all_types = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids_1 = batch["input_ids_1"].to(device)
            attn_mask_1 = batch["attention_mask_1"].to(device)
            input_ids_2 = batch["input_ids_2"].to(device)
            attn_mask_2 = batch["attention_mask_2"].to(device)

            output = model(input_ids_1, attn_mask_1, input_ids_2, attn_mask_2)
            all_sims.extend(output["similarity"].cpu().numpy())
            all_labels.extend(batch["label"].numpy())
            all_types.extend(batch["clone_type"].numpy())

    evaluator = CloneDetectionEvaluator(config)
    eval_results = evaluator.evaluate(
        similarities=np.array(all_sims),
        labels=np.array(all_labels),
        clone_types=np.array(all_types),
    )
    evaluator.plot_results(np.array(all_sims), np.array(all_labels))

    logger.info("\nFinal Test Results:")
    logger.info(f"  F1:      {eval_results['overall']['f1']:.4f}")
    logger.info(f"  Recall:  {eval_results['overall']['recall']:.4f}")
    logger.info(f"  ROC-AUC: {eval_results['overall']['roc_auc']:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

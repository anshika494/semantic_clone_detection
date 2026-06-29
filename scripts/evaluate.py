#!/usr/bin/env python3
"""
Evaluation Script
==================
Run comprehensive evaluation of a trained clone detection model.

Usage:
  python scripts/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt
  python scripts/evaluate.py --config configs/config.yaml --model checkpoints/best_model.pt --bce
"""

import sys
import json
import logging
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils import load_config, setup_logging, set_seed
from src.dataset import BigCloneBenchLoader, BigCloneBenchDataset, SyntheticCloneDataset
from src.preprocessing import BatchPreprocessor
from src.embeddings import create_embedding_model
from src.models import SiameseCloneDetector
from src.evaluation import (
    CloneDetectionEvaluator,
    compute_metrics,
    find_optimal_threshold,
    evaluate_per_clone_type,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Clone Detection Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pt")
    parser.add_argument("--split", choices=["val", "test", "all"], default="test")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override similarity threshold")
    parser.add_argument("--optimize-threshold", action="store_true",
                        help="Search for optimal threshold on val set")
    parser.add_argument("--bce", action="store_true",
                        help="Run BigCloneEval evaluation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic dataset")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--plots", action="store_true", help="Generate plots")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def run_inference(model, dataloader, device) -> tuple:
    """Run model inference on a dataloader, return similarities, labels, types."""
    model.eval()
    all_sims, all_labels, all_types = [], [], []

    for batch in tqdm(dataloader, desc="Running inference"):
        input_ids_1 = batch["input_ids_1"].to(device)
        attn_mask_1 = batch["attention_mask_1"].to(device)
        input_ids_2 = batch["input_ids_2"].to(device)
        attn_mask_2 = batch["attention_mask_2"].to(device)

        output = model(input_ids_1, attn_mask_1, input_ids_2, attn_mask_2)

        all_sims.extend(output["similarity"].cpu().numpy().tolist())
        all_labels.extend(batch["label"].numpy().tolist())
        all_types.extend(batch["clone_type"].numpy().tolist())

    return np.array(all_sims), np.array(all_labels), np.array(all_types)


def print_results_table(results: dict):
    """Pretty-print evaluation results."""
    overall = results.get("overall", {})

    print("\n" + "=" * 65)
    print("  EVALUATION RESULTS")
    print("=" * 65)
    print(f"  {'Metric':<25} {'Value':>10}")
    print("-" * 65)

    metric_order = [
        ("accuracy",   "Accuracy"),
        ("precision",  "Precision"),
        ("recall",     "Recall (Detection Rate)"),
        ("f1",         "F1 Score"),
        ("roc_auc",    "ROC-AUC"),
        ("pr_auc",     "PR-AUC"),
        ("threshold",  "Decision Threshold"),
    ]

    for key, label in metric_order:
        if key in overall:
            val = overall[key]
            print(f"  {label:<25} {val:>10.4f}")

    if "true_positives" in overall:
        print("-" * 65)
        print(f"  {'True Positives':<25} {overall['true_positives']:>10}")
        print(f"  {'True Negatives':<25} {overall['true_negatives']:>10}")
        print(f"  {'False Positives':<25} {overall['false_positives']:>10}")
        print(f"  {'False Negatives':<25} {overall['false_negatives']:>10}")

    print("=" * 65)

    if "per_clone_type" in results:
        print("\n  PER CLONE-TYPE BREAKDOWN")
        print("-" * 65)
        for type_name, type_metrics in results["per_clone_type"].items():
            print(f"\n  {type_name}:")
            for k, v in type_metrics.items():
                if isinstance(v, float):
                    print(f"    {k:<20} {v:.4f}")
                else:
                    print(f"    {k:<20} {v}")

    if "bigcloneeval" in results:
        print("\n  BIGCLONEEVAL RESULTS")
        print("-" * 65)
        for k, v in results["bigcloneeval"].items():
            if isinstance(v, float):
                print(f"  {k:<25} {v:.4f}")
            else:
                print(f"  {k:<25} {v}")

    print("=" * 65 + "\n")


def main():
    args = parse_args()
    setup_logging(log_file=f"{args.output_dir}/evaluate.log")

    logger.info("=" * 65)
    logger.info("Semantic Code Clone Detection — Evaluation")
    logger.info("=" * 65)

    set_seed(args.seed)
    config = load_config(args.config)
    config["evaluation"]["results_dir"] = args.output_dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    logger.info(f"Loading model from: {args.model}")
    model = SiameseCloneDetector(config)
    _, tokenizer = create_embedding_model(config)

    if not Path(args.model).exists():
        logger.error(f"Checkpoint not found: {args.model}")
        return 1

    checkpoint = torch.load(args.model, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)

    # Threshold from checkpoint
    ckpt_threshold = checkpoint.get("metrics", {}).get("optimal_threshold", 0.5)
    logger.info(f"Checkpoint threshold: {ckpt_threshold:.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    logger.info("Loading dataset...")

    if args.synthetic:
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

    # Preprocess
    preprocessor = BatchPreprocessor(config)
    val_pairs = preprocessor.process_dataset(val_pairs, show_progress=False)
    test_pairs = preprocessor.process_dataset(test_pairs, show_progress=False)

    batch_size = config.get("training", {}).get("batch_size", 32) * 2
    max_length = config.get("ast", {}).get("max_token_length", 512)

    def make_loader(pairs):
        dataset = BigCloneBenchDataset(pairs, tokenizer=tokenizer,
                                       max_length=max_length, split="eval")
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # -----------------------------------------------------------------------
    # Threshold optimization (optional)
    # -----------------------------------------------------------------------
    threshold = args.threshold or ckpt_threshold

    if args.optimize_threshold:
        logger.info("Optimizing threshold on validation set...")
        val_loader = make_loader(val_pairs)
        val_sims, val_labels, _ = run_inference(model, val_loader, device)
        threshold, best_f1 = find_optimal_threshold(val_sims, val_labels)
        logger.info(f"Optimal threshold: {threshold:.4f} (val F1={best_f1:.4f})")

    logger.info(f"Using threshold: {threshold:.4f}")

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    evaluator = CloneDetectionEvaluator(config)

    if args.split in ("val", "all"):
        logger.info("\nEvaluating on validation set...")
        val_loader = make_loader(val_pairs)
        val_sims, val_labels, val_types = run_inference(model, val_loader, device)
        val_results = evaluator.evaluate(
            similarities=val_sims,
            labels=val_labels,
            clone_types=val_types,
            threshold=threshold,
        )
        logger.info("Validation Results:")
        print_results_table(val_results)

        # Save
        with open(f"{args.output_dir}/val_results.json", "w") as f:
            json.dump(val_results, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

    if args.split in ("test", "all"):
        logger.info("\nEvaluating on test set...")
        test_loader = make_loader(test_pairs)
        test_sims, test_labels, test_types = run_inference(model, test_loader, device)
        test_results = evaluator.evaluate(
            similarities=test_sims,
            labels=test_labels,
            clone_types=test_types,
            pairs=test_pairs if args.bce else None,
            threshold=threshold,
            run_bce=args.bce,
        )
        print_results_table(test_results)

        # Save
        with open(f"{args.output_dir}/test_results.json", "w") as f:
            json.dump(test_results, f, indent=2,
                      default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else str(x))

        # Plots
        if args.plots:
            evaluator.plot_results(test_sims, test_labels, save_dir=args.output_dir)
            logger.info(f"Plots saved to {args.output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())

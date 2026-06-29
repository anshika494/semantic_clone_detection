#!/usr/bin/env python3
"""
Preprocessing Script
=====================
Preprocess the BigCloneBench dataset and generate AST tokens.
Run this once before training to cache all processed data.

Usage:
  python scripts/preprocess.py --config configs/config.yaml
  python scripts/preprocess.py --config configs/config.yaml --use-ast --workers 8
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.utils import load_config, setup_logging, timer
from src.dataset import BigCloneBenchLoader, SyntheticCloneDataset
from src.preprocessing import CodePreprocessor
from src.ast_processing import ASTProcessor

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess BigCloneBench Dataset")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--use-ast", action="store_true",
                        help="Also generate AST token sequences")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel workers for AST generation")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for testing)")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild even if cache exists")
    return parser.parse_args()


def process_pair_ast(args_tuple):
    """Worker function for parallel AST processing."""
    pair, ast_config = args_tuple
    try:
        processor = ASTProcessor({"ast": ast_config})
        pair = dict(pair)
        pair["code1_ast"] = processor.get_token_string(pair["code1"])
        pair["code2_ast"] = processor.get_token_string(pair["code2"])
        return pair
    except Exception as e:
        pair["code1_ast"] = pair.get("code1_processed", pair["code1"])
        pair["code2_ast"] = pair.get("code2_processed", pair["code2"])
        return pair


def preprocess_split(
    pairs: list,
    preprocessor: CodePreprocessor,
    split_name: str,
) -> list:
    """Preprocess a dataset split."""
    processed = []
    for pair in tqdm(pairs, desc=f"Preprocessing {split_name}"):
        pair = dict(pair)
        pair["code1_processed"] = preprocessor.preprocess_for_model(pair["code1"])
        pair["code2_processed"] = preprocessor.preprocess_for_model(pair["code2"])

        # Validate
        t1 = pair["code1_processed"].split()
        t2 = pair["code2_processed"].split()
        if len(t1) >= preprocessor.min_tokens and len(t2) >= preprocessor.min_tokens:
            processed.append(pair)

    logger.info(
        f"{split_name}: {len(pairs)} → {len(processed)} pairs "
        f"({len(pairs) - len(processed)} filtered)"
    )
    return processed


def add_ast_tokens(
    pairs: list,
    ast_config: dict,
    n_workers: int,
    split_name: str,
) -> list:
    """Add AST token sequences using parallel processing."""
    logger.info(f"Generating AST tokens for {split_name} ({len(pairs)} pairs)...")

    args_list = [(pair, ast_config) for pair in pairs]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(process_pair_ast, args) for args in args_list]
        results = []
        for future in tqdm(
            as_completed(futures), total=len(futures),
            desc=f"AST {split_name}"
        ):
            results.append(future.result())

    return results


def compute_statistics(pairs: list) -> dict:
    """Compute dataset statistics."""
    import numpy as np
    labels = [p["label"] for p in pairs]
    clone_types = [p.get("clone_type", 0) for p in pairs]
    token_counts1 = [len(p.get("code1_processed", "").split()) for p in pairs]
    token_counts2 = [len(p.get("code2_processed", "").split()) for p in pairs]

    stats = {
        "total_pairs": len(pairs),
        "clone_pairs": sum(labels),
        "non_clone_pairs": len(pairs) - sum(labels),
        "clone_ratio": sum(labels) / max(len(pairs), 1),
        "type_distribution": {
            str(t): clone_types.count(t)
            for t in set(clone_types)
        },
        "token_stats": {
            "code1_mean": float(np.mean(token_counts1)),
            "code1_median": float(np.median(token_counts1)),
            "code1_max": int(np.max(token_counts1)) if token_counts1 else 0,
            "code2_mean": float(np.mean(token_counts2)),
            "code2_median": float(np.median(token_counts2)),
        },
    }
    return stats


def main():
    args = parse_args()
    setup_logging(log_file="results/preprocess.log")

    logger.info("=" * 65)
    logger.info("Semantic Code Clone Detection — Preprocessing")
    logger.info("=" * 65)

    config = load_config(args.config)
    processed_dir = Path(config["dataset"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # Load raw pairs
    # -----------------------------------------------------------------------
    with timer("Dataset loading"):
        if args.synthetic:
            logger.info("Using synthetic dataset")
            from sklearn.model_selection import train_test_split
            all_pairs = SyntheticCloneDataset.generate_pairs(n_pairs=5000)
            labels = [p["label"] for p in all_pairs]
            train_pairs, test_pairs = train_test_split(all_pairs, test_size=0.2,
                                                        stratify=labels, random_state=42)
            t_labels = [p["label"] for p in train_pairs]
            train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.1,
                                                       stratify=t_labels, random_state=42)
        else:
            loader = BigCloneBenchLoader(config)
            stats = loader.get_statistics()
            logger.info(f"BigCloneBench stats: {json.dumps(stats, indent=2)}")
            train_pairs, val_pairs, test_pairs = loader.load_pairs(force_rebuild=args.force)

    logger.info(
        f"Raw pairs: {len(train_pairs)} train, "
        f"{len(val_pairs)} val, {len(test_pairs)} test"
    )

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------
    preprocessor = CodePreprocessor(config)

    with timer("Preprocessing"):
        train_pairs = preprocess_split(train_pairs, preprocessor, "train")
        val_pairs = preprocess_split(val_pairs, preprocessor, "val")
        test_pairs = preprocess_split(test_pairs, preprocessor, "test")

    # -----------------------------------------------------------------------
    # AST generation (optional)
    # -----------------------------------------------------------------------
    if args.use_ast:
        ast_config = config.get("ast", {})

        with timer("AST generation"):
            train_pairs = add_ast_tokens(train_pairs, ast_config, args.workers, "train")
            val_pairs = add_ast_tokens(val_pairs, ast_config, args.workers, "val")
            test_pairs = add_ast_tokens(test_pairs, ast_config, args.workers, "test")

    # -----------------------------------------------------------------------
    # Save processed data
    # -----------------------------------------------------------------------
    logger.info("Saving processed data...")

    suffix = "_ast" if args.use_ast else ""

    for split_name, split_data in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        out_path = processed_dir / f"bcb_{split_name}_processed{suffix}.json"
        with open(out_path, "w") as f:
            json.dump(split_data, f)
        logger.info(f"Saved {len(split_data)} {split_name} pairs → {out_path}")

    # -----------------------------------------------------------------------
    # Statistics
    # -----------------------------------------------------------------------
    logger.info("\nDataset Statistics:")
    for split_name, split_data in [
        ("train", train_pairs), ("val", val_pairs), ("test", test_pairs)
    ]:
        stats = compute_statistics(split_data)
        logger.info(f"\n  [{split_name.upper()}]")
        logger.info(f"    Total pairs:   {stats['total_pairs']:,}")
        logger.info(f"    Clone pairs:   {stats['clone_pairs']:,}")
        logger.info(f"    Non-clones:    {stats['non_clone_pairs']:,}")
        logger.info(f"    Clone ratio:   {stats['clone_ratio']:.2%}")
        logger.info(f"    Avg tokens:    {stats['token_stats']['code1_mean']:.0f}")
        logger.info(f"    Type dist:     {stats['type_distribution']}")

    # Save stats
    stats_path = processed_dir / "dataset_statistics.json"
    all_stats = {
        "train": compute_statistics(train_pairs),
        "val": compute_statistics(val_pairs),
        "test": compute_statistics(test_pairs),
    }
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\nPreprocessing complete! Stats saved to {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

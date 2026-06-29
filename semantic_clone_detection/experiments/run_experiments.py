#!/usr/bin/env python3
"""
Experiment Runner
==================
Systematic experiment management for ablation studies,
hyperparameter sweeps, and model comparisons.

Usage:
  python experiments/run_experiments.py --suite ablation
  python experiments/run_experiments.py --suite hyperparam --backbone microsoft/codebert-base
  python experiments/run_experiments.py --exp single --name codebert_contrastive
"""

import sys
import os
import json
import time
import copy
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, save_config, merge_configs, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Experiment Definitions
# ---------------------------------------------------------------------------

# Base config overrides for all experiments
BASE_OVERRIDE = {
    "training": {
        "epochs": 10,
        "batch_size": 32,
        "fp16": True,
        "early_stopping_patience": 3,
    },
    "dataset": {
        "max_pairs_per_type": 25000,
    },
}

# Ablation study: vary one factor at a time
ABLATION_EXPERIMENTS = {
    # Loss function ablation
    "loss_contrastive": {
        "model": {"siamese": {"loss": "contrastive", "margin": 1.0}},
    },
    "loss_cosine": {
        "model": {"siamese": {"loss": "cosine", "margin": 0.5}},
    },
    "loss_triplet": {
        "model": {"siamese": {"loss": "triplet", "margin": 0.5}},
    },
    "loss_ntxent": {
        "model": {"siamese": {"loss": "ntxent", "temperature": 0.07}},
    },

    # Backbone ablation
    "backbone_codebert": {
        "model": {"backbone": "microsoft/codebert-base"},
    },
    "backbone_graphcodebert": {
        "model": {"backbone": "microsoft/graphcodebert-base"},
    },

    # Preprocessing ablation
    "no_normalize_ids": {
        "preprocessing": {"normalize_identifiers": False},
    },
    "no_normalize_lits": {
        "preprocessing": {"normalize_literals": False},
    },
    "no_remove_comments": {
        "preprocessing": {"remove_comments": False},
    },

    # Projection dimension ablation
    "proj_64": {
        "model": {"projection_dim": 64},
    },
    "proj_128": {
        "model": {"projection_dim": 128},
    },
    "proj_256": {
        "model": {"projection_dim": 256},
    },
    "proj_512": {
        "model": {"projection_dim": 512},
    },

    # Pooling strategy ablation
    "pool_cls": {
        # Requires modifying model pooling (set in embedding model)
        "model": {"pooling": "cls"},
    },
    "pool_mean": {
        "model": {"pooling": "mean"},
    },
    "pool_max": {
        "model": {"pooling": "max"},
    },
}

# Hyperparameter search experiments
HYPERPARAM_EXPERIMENTS = {
    "lr_1e5": {
        "training": {"learning_rate": 1e-5},
    },
    "lr_2e5": {
        "training": {"learning_rate": 2e-5},
    },
    "lr_5e5": {
        "training": {"learning_rate": 5e-5},
    },
    "lr_1e4": {
        "training": {"learning_rate": 1e-4},
    },
    "batch_16": {
        "training": {"batch_size": 16},
    },
    "batch_32": {
        "training": {"batch_size": 32},
    },
    "batch_64": {
        "training": {"batch_size": 64},
    },
    "warmup_0": {
        "training": {"warmup_ratio": 0.0},
    },
    "warmup_01": {
        "training": {"warmup_ratio": 0.1},
    },
    "warmup_02": {
        "training": {"warmup_ratio": 0.2},
    },
}

# Quick smoke-test experiments (small scale)
QUICK_EXPERIMENTS = {
    "quick_codebert": {
        "model": {"backbone": "microsoft/codebert-base"},
        "training": {"epochs": 3, "batch_size": 16},
        "dataset": {"max_pairs_per_type": 1000},
    },
}


EXPERIMENT_SUITES = {
    "ablation": ABLATION_EXPERIMENTS,
    "hyperparam": HYPERPARAM_EXPERIMENTS,
    "quick": QUICK_EXPERIMENTS,
}


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------

class ExperimentRunner:
    """
    Manages running multiple experiments and collecting results.
    """

    def __init__(
        self,
        base_config_path: str,
        experiments_dir: str = "experiments/runs",
    ):
        self.base_config = load_config(base_config_path)
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}

    def run_experiment(
        self,
        name: str,
        override: Dict,
        use_synthetic: bool = False,
        dry_run: bool = False,
    ) -> Dict:
        """Run a single experiment."""
        logger.info(f"\n{'='*65}")
        logger.info(f"Running experiment: {name}")
        logger.info(f"Overrides: {json.dumps(override, indent=2)}")
        logger.info(f"{'='*65}")

        # Build config
        config = merge_configs(self.base_config, BASE_OVERRIDE)
        config = merge_configs(config, override)

        # Experiment output dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = self.experiments_dir / f"{name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        config["training"]["output_dir"] = str(exp_dir / "checkpoints")
        config["evaluation"]["results_dir"] = str(exp_dir / "results")

        # Save experiment config
        config_path = exp_dir / "config.yaml"
        save_config(config, str(config_path))

        if dry_run:
            logger.info(f"[DRY RUN] Would run: {name}")
            return {"name": name, "status": "dry_run"}

        # Run training
        start_time = time.time()
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", str(config_path),
        ]
        if use_synthetic:
            cmd.append("--synthetic")
            cmd.extend(["--max-pairs", "2000"])

        result = {"name": name, "config": override, "exp_dir": str(exp_dir)}

        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=7200
            )

            result["status"] = "success" if proc.returncode == 0 else "failed"
            result["returncode"] = proc.returncode
            result["elapsed_seconds"] = time.time() - start_time

            if proc.returncode != 0:
                logger.error(f"Experiment {name} failed:\n{proc.stderr[-2000:]}")
            else:
                # Parse final metrics from results
                results_file = exp_dir / "results" / "evaluation_results.json"
                if results_file.exists():
                    with open(results_file) as f:
                        eval_results = json.load(f)
                    result["metrics"] = eval_results.get("overall", {})

            # Save logs
            (exp_dir / "stdout.txt").write_text(proc.stdout)
            (exp_dir / "stderr.txt").write_text(proc.stderr)

        except subprocess.TimeoutExpired:
            result["status"] = "timeout"
            result["elapsed_seconds"] = time.time() - start_time
            logger.error(f"Experiment {name} timed out after 2 hours")

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            logger.exception(f"Error running {name}")

        # Save individual result
        with open(exp_dir / "result.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        return result

    def run_suite(
        self,
        suite_name: str,
        experiments: Optional[List[str]] = None,
        use_synthetic: bool = False,
        dry_run: bool = False,
    ) -> List[Dict]:
        """Run a suite of experiments."""
        suite = EXPERIMENT_SUITES.get(suite_name, {})

        if not suite:
            logger.error(f"Unknown suite: {suite_name}. Available: {list(EXPERIMENT_SUITES.keys())}")
            return []

        # Filter experiments if specified
        if experiments:
            suite = {k: v for k, v in suite.items() if k in experiments}

        logger.info(f"Running suite '{suite_name}' with {len(suite)} experiments")
        all_results = []

        for name, override in suite.items():
            result = self.run_experiment(
                name=f"{suite_name}/{name}",
                override=override,
                use_synthetic=use_synthetic,
                dry_run=dry_run,
            )
            all_results.append(result)
            self.results[name] = result

        # Save aggregated results
        summary_path = self.experiments_dir / f"{suite_name}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        self._print_summary(all_results)
        return all_results

    def _print_summary(self, results: List[Dict]):
        """Print a summary table of all experiment results."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"{'Name':<35} {'Status':<10} {'F1':>8} {'ROC-AUC':>10} {'Time':>8}")
        print("-" * 80)

        for r in sorted(results, key=lambda x: x.get("metrics", {}).get("f1", 0), reverse=True):
            name = r["name"].split("/")[-1][:34]
            status = r.get("status", "?")
            metrics = r.get("metrics", {})
            f1 = metrics.get("f1", 0)
            auc = metrics.get("roc_auc", 0)
            elapsed = r.get("elapsed_seconds", 0)
            time_str = f"{elapsed/60:.0f}m" if elapsed else "?"

            print(f"{name:<35} {status:<10} {f1:>8.4f} {auc:>10.4f} {time_str:>8}")

        print("=" * 80)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run Experiments")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--suite", type=str, choices=list(EXPERIMENT_SUITES.keys()),
        help="Experiment suite to run"
    )
    parser.add_argument(
        "--exp", type=str, default="single",
        help="Individual experiment name (from suite)"
    )
    parser.add_argument(
        "--experiments", nargs="+", default=None,
        help="Specific experiments to run from suite"
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic data (for testing)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be run without executing")
    parser.add_argument("--output-dir", type=str, default="experiments/runs")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(log_file="experiments/experiment_runner.log")

    runner = ExperimentRunner(
        base_config_path=args.config,
        experiments_dir=args.output_dir,
    )

    if args.suite:
        results = runner.run_suite(
            suite_name=args.suite,
            experiments=args.experiments,
            use_synthetic=args.synthetic,
            dry_run=args.dry_run,
        )
        logger.info(f"Completed {len(results)} experiments")
    else:
        logger.error("Please specify --suite")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

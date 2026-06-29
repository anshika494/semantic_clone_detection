"""
Evaluation Module
==================
Comprehensive evaluation of clone detection performance:
  - Standard ML metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC
  - Per-clone-type breakdown
  - Threshold optimization
  - BigCloneEval integration
  - Confusion matrix and visualization
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    similarities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all classification metrics given similarity scores and true labels.

    Args:
        similarities: Cosine similarity scores in [-1, 1]
        labels: Binary labels (1 = clone, 0 = non-clone)
        threshold: Decision threshold

    Returns:
        Dict of metric_name → float
    """
    predictions = (similarities >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "threshold": threshold,
    }

    # ROC-AUC (requires both classes)
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = roc_auc_score(labels, similarities)

        # PR-AUC
        precision_curve, recall_curve, _ = precision_recall_curve(labels, similarities)
        metrics["pr_auc"] = auc(recall_curve, precision_curve)
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0

    # Confusion matrix
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["true_positives"] = int(tp)
        metrics["true_negatives"] = int(tn)
        metrics["false_positives"] = int(fp)
        metrics["false_negatives"] = int(fn)

        # Additional metrics
        metrics["specificity"] = tn / max(tn + fp, 1)
        metrics["npv"] = tn / max(tn + fn, 1)  # Negative predictive value

    return metrics


def find_optimal_threshold(
    similarities: np.ndarray,
    labels: np.ndarray,
    metric: str = "f1",
    n_steps: int = 100,
    threshold_range: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[float, float]:
    """
    Find the threshold that maximizes a given metric.

    Args:
        similarities: Similarity scores
        labels: True binary labels
        metric: Metric to optimize ('f1', 'accuracy', 'balanced_accuracy')
        n_steps: Number of thresholds to evaluate
        threshold_range: (min, max) threshold values to search

    Returns:
        (optimal_threshold, best_metric_value)
    """
    thresholds = np.linspace(threshold_range[0], threshold_range[1], n_steps)
    best_threshold = 0.5
    best_metric_val = -1.0

    for thresh in thresholds:
        preds = (similarities >= thresh).astype(int)

        if metric == "f1":
            val = f1_score(labels, preds, zero_division=0)
        elif metric == "accuracy":
            val = accuracy_score(labels, preds)
        elif metric == "balanced_accuracy":
            recall = recall_score(labels, preds, zero_division=0)
            specificity = recall_score(1 - labels, 1 - preds, zero_division=0)
            val = (recall + specificity) / 2
        else:
            val = f1_score(labels, preds, zero_division=0)

        if val > best_metric_val:
            best_metric_val = val
            best_threshold = thresh

    return best_threshold, best_metric_val


def evaluate_per_clone_type(
    similarities: np.ndarray,
    labels: np.ndarray,
    clone_types: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics broken down by clone type.

    Args:
        similarities: Similarity scores
        labels: True binary labels
        clone_types: Clone type per pair (0=non-clone, 1=T1, 2=T2, 3=T3, 4=T4)
        threshold: Decision threshold

    Returns:
        Dict of type_name → metrics dict
    """
    type_names = {
        0: "Non-Clone",
        1: "T1 (Exact)",
        2: "T2 (Renamed)",
        3: "T3 (Near-Miss)",
        4: "T4 (Semantic)",
    }

    results = {}

    for type_id, type_name in type_names.items():
        if type_id == 0:
            # Non-clone pairs
            mask = labels == 0
        else:
            # Clone pairs of this specific type
            mask = (labels == 1) & (clone_types == type_id)

        if mask.sum() == 0:
            continue

        type_sims = similarities[mask]
        type_labels = labels[mask]

        if type_id == 0:
            # For non-clones, check if correctly classified as non-clone
            preds = (type_sims >= threshold).astype(int)
            results[type_name] = {
                "count": int(mask.sum()),
                "true_negative_rate": accuracy_score(type_labels, preds),
                "false_positive_rate": float(preds.mean()),
            }
        else:
            preds = (type_sims >= threshold).astype(int)
            results[type_name] = {
                "count": int(mask.sum()),
                "recall": recall_score(type_labels, preds, zero_division=0),
                "precision": precision_score(type_labels, preds, zero_division=0),
                "f1": f1_score(type_labels, preds, zero_division=0),
                "avg_similarity": float(type_sims.mean()),
            }

    return results


# ---------------------------------------------------------------------------
# BigCloneEval Integration
# ---------------------------------------------------------------------------

class BigCloneEvalIntegration:
    """
    Integration with the BigCloneEval evaluation framework.

    BigCloneEval (https://github.com/jeffsvajlenko/BigCloneEval) provides
    standardized evaluation on the BigCloneBench benchmark.

    Usage:
      1. Generate clone detection results in BCE format
      2. Run BCE evaluation tool
      3. Parse and return results
    """

    # BCE expects: id1 \t id2 \t similarity_score
    BCE_FORMAT = "{id1}\t{id2}\t{score:.6f}\n"

    def __init__(self, bce_dir: str, results_dir: str = "results"):
        self.bce_dir = Path(bce_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def write_results_file(
        self,
        pairs: List[Dict],
        similarities: np.ndarray,
        output_path: str = "results/bce_results.txt",
    ) -> str:
        """
        Write detection results in BigCloneEval format.

        Args:
            pairs: List of pair dicts with 'id1' and 'id2' fields
            similarities: Similarity scores for each pair
            output_path: Output file path

        Returns:
            Path to written file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for pair, sim in zip(pairs, similarities):
                f.write(self.BCE_FORMAT.format(
                    id1=pair["id1"],
                    id2=pair["id2"],
                    score=float(sim),
                ))

        logger.info(f"Wrote {len(pairs)} results to {output_path}")
        return str(output_path)

    def run_evaluation(
        self,
        results_file: str,
        threshold: float = 0.5,
    ) -> Optional[Dict]:
        """
        Run BigCloneEval's evaluation script.

        Requires BigCloneEval to be installed at self.bce_dir.

        Args:
            results_file: Path to results file in BCE format
            threshold: Similarity threshold for clone classification

        Returns:
            Dict of evaluation results, or None if BCE not available
        """
        eval_script = self.bce_dir / "evaluateTool.sh"

        if not eval_script.exists():
            logger.warning(
                f"BigCloneEval not found at {self.bce_dir}. "
                "Download from: https://github.com/jeffsvajlenko/BigCloneEval"
            )
            return None

        cmd = [
            "bash", str(eval_script),
            "--results", results_file,
            "--threshold", str(threshold),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode != 0:
                logger.error(f"BigCloneEval failed: {result.stderr}")
                return None

            return self._parse_bce_output(result.stdout)

        except subprocess.TimeoutExpired:
            logger.error("BigCloneEval timed out after 5 minutes")
            return None
        except FileNotFoundError:
            logger.error("bash not found; cannot run BigCloneEval")
            return None

    @staticmethod
    def _parse_bce_output(output: str) -> Dict:
        """Parse BigCloneEval stdout into a metrics dict."""
        metrics = {}
        lines = output.strip().split("\n")

        for line in lines:
            line = line.strip()
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip().lower().replace(" ", "_")
                    try:
                        value = float(parts[1].strip().rstrip("%")) / 100
                    except ValueError:
                        value = parts[1].strip()
                    metrics[key] = value

        return metrics


# ---------------------------------------------------------------------------
# Full Evaluator
# ---------------------------------------------------------------------------

class CloneDetectionEvaluator:
    """
    Main evaluator for the clone detection system.

    Computes:
    1. Standard ML metrics
    2. Per-clone-type breakdown
    3. Threshold analysis
    4. Visualization
    5. BigCloneEval integration (optional)
    """

    def __init__(self, config: Dict):
        self.config = config
        self.results_dir = Path(config.get("evaluation", {}).get("results_dir", "results"))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        bce_dir = config.get("evaluation", {}).get("bigcloneeval_dir", "")
        self.bce = BigCloneEvalIntegration(bce_dir, str(self.results_dir)) if bce_dir else None

    def evaluate(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        clone_types: Optional[np.ndarray] = None,
        pairs: Optional[List[Dict]] = None,
        threshold: Optional[float] = None,
        run_bce: bool = False,
    ) -> Dict:
        """
        Full evaluation pipeline.

        Args:
            similarities: Model similarity scores [N]
            labels: True binary labels [N]
            clone_types: Clone type per pair [N] (optional)
            pairs: Original pair dicts (for BCE) (optional)
            threshold: Decision threshold (None = auto-optimize on data)
            run_bce: Whether to run BigCloneEval

        Returns:
            Comprehensive evaluation results dict
        """
        results = {}

        # 1. Find optimal threshold if not given
        if threshold is None:
            threshold, _ = find_optimal_threshold(similarities, labels)
            logger.info(f"Optimal threshold: {threshold:.4f}")

        # 2. Standard metrics
        metrics = compute_metrics(similarities, labels, threshold=threshold)
        results["overall"] = metrics

        logger.info(
            f"\nOverall Results (threshold={threshold:.4f}):\n"
            f"  Accuracy:  {metrics['accuracy']:.4f}\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall:    {metrics['recall']:.4f}\n"
            f"  F1:        {metrics['f1']:.4f}\n"
            f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n"
            f"  PR-AUC:    {metrics['pr_auc']:.4f}"
        )

        # 3. Per-type breakdown
        if clone_types is not None:
            type_results = evaluate_per_clone_type(
                similarities, labels, clone_types, threshold
            )
            results["per_clone_type"] = type_results
            logger.info("\nPer Clone-Type Results:")
            for type_name, type_metrics in type_results.items():
                logger.info(f"  {type_name}: {type_metrics}")

        # 4. Threshold analysis
        results["threshold_analysis"] = self._threshold_analysis(similarities, labels)

        # 5. BigCloneEval
        if run_bce and self.bce and pairs:
            bce_results_path = str(self.results_dir / "bce_results.txt")
            self.bce.write_results_file(pairs, similarities, bce_results_path)
            bce_metrics = self.bce.run_evaluation(bce_results_path, threshold)
            if bce_metrics:
                results["bigcloneeval"] = bce_metrics

        # 6. Save results
        results_path = self.results_dir / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(
                {k: v for k, v in results.items() if isinstance(v, dict)},
                f,
                indent=2,
                default=lambda x: float(x) if isinstance(x, np.floating) else int(x),
            )
        logger.info(f"Results saved to {results_path}")

        return results

    def _threshold_analysis(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        n_points: int = 50,
    ) -> List[Dict]:
        """Analyze metrics across a range of thresholds."""
        thresholds = np.linspace(0.1, 0.9, n_points)
        analysis = []

        for thresh in thresholds:
            preds = (similarities >= thresh).astype(int)
            analysis.append({
                "threshold": float(thresh),
                "precision": float(precision_score(labels, preds, zero_division=0)),
                "recall": float(recall_score(labels, preds, zero_division=0)),
                "f1": float(f1_score(labels, preds, zero_division=0)),
                "accuracy": float(accuracy_score(labels, preds)),
            })

        return analysis

    def plot_results(
        self,
        similarities: np.ndarray,
        labels: np.ndarray,
        save_dir: Optional[str] = None,
    ):
        """Generate evaluation plots."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use("Agg")  # Non-interactive backend
        except ImportError:
            logger.warning("matplotlib not available; skipping plots")
            return

        save_dir = Path(save_dir or self.results_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1. ROC Curve
        if len(np.unique(labels)) > 1:
            fpr, tpr, _ = roc_curve(labels, similarities)
            roc_auc = roc_auc_score(labels, similarities)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC={roc_auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve - Clone Detection")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "roc_curve.png", dpi=150)
            plt.close()

        # 2. Similarity Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        clone_sims = similarities[labels == 1]
        non_clone_sims = similarities[labels == 0]

        ax.hist(clone_sims, bins=50, alpha=0.7, color="green", label="Clones", density=True)
        ax.hist(non_clone_sims, bins=50, alpha=0.7, color="red", label="Non-Clones", density=True)
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Density")
        ax.set_title("Similarity Distribution by Class")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "similarity_distribution.png", dpi=150)
        plt.close()

        # 3. Precision-Recall Curve
        if len(np.unique(labels)) > 1:
            precision_vals, recall_vals, _ = precision_recall_curve(labels, similarities)
            pr_auc = auc(recall_vals, precision_vals)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall_vals, precision_vals, "g-", lw=2,
                    label=f"PR Curve (AUC={pr_auc:.4f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall Curve")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_dir / "pr_curve.png", dpi=150)
            plt.close()

        logger.info(f"Plots saved to {save_dir}")

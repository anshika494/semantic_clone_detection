"""
Utilities Module
=================
Common utilities: config loading, logging setup, reproducibility,
progress tracking, and general helpers.
"""

import os
import json
import random
import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Deep merge two config dicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = merge_configs(result[key], val)
        else:
            result[key] = val
    return result


def save_config(config: Dict, path: str):
    """Save config to YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Deterministic mode (may slow training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Random seed set to {seed}")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_str: Optional[str] = None,
) -> logging.Logger:
    """
    Configure logging with console and optional file handler.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_str: Log format string

    Returns:
        Root logger
    """
    if format_str is None:
        format_str = (
            "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
        )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_str))
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        file_handler.setFormatter(logging.Formatter(format_str))
        root_logger.addHandler(file_handler)

    # Silence noisy third-party loggers
    for noisy in ["transformers", "tokenizers", "urllib3", "filelock"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


# ---------------------------------------------------------------------------
# Device utilities
# ---------------------------------------------------------------------------

def get_device(prefer_gpu: bool = True) -> torch.device:
    """Get the best available device."""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def get_gpu_memory_info() -> Dict:
    """Get current GPU memory usage."""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": total - reserved,
    }


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def model_summary(model: torch.nn.Module) -> str:
    """Generate a model summary string."""
    params = count_parameters(model)
    lines = [
        f"Model Summary:",
        f"  Total parameters:     {params['total']:,}",
        f"  Trainable parameters: {params['trainable']:,}",
        f"  Frozen parameters:    {params['frozen']:,}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def batch_iterable(iterable, batch_size: int):
    """Split an iterable into batches."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def flatten_dict(d: Dict, sep: str = "/", prefix: str = "") -> Dict:
    """Flatten a nested dict."""
    result = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, sep=sep, prefix=key))
        else:
            result[key] = v
    return result


def save_jsonl(data: list, path: str):
    """Save list of dicts as JSONL."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


def load_jsonl(path: str) -> list:
    """Load JSONL file as list of dicts."""
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------

import time
from contextlib import contextmanager


@contextmanager
def timer(name: str = ""):
    """Simple timer context manager."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    label = f"[{name}] " if name else ""
    logger.info(f"{label}Elapsed: {elapsed:.3f}s")


# ---------------------------------------------------------------------------
# Progress tracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """Tracks training progress and detects improvement."""

    def __init__(self, metric: str = "val_f1", mode: str = "max", patience: int = 3):
        assert mode in {"max", "min"}
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.patience_counter = 0
        self.history = []

    def update(self, value: float) -> bool:
        """
        Update with new metric value.

        Returns:
            True if this is a new best value.
        """
        self.history.append(value)
        improved = (
            (self.mode == "max" and value > self.best_value) or
            (self.mode == "min" and value < self.best_value)
        )

        if improved:
            self.best_value = value
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return improved

    @property
    def should_stop(self) -> bool:
        return self.patience_counter >= self.patience


# ---------------------------------------------------------------------------
# Environment info
# ---------------------------------------------------------------------------

def get_environment_info() -> Dict:
    """Collect environment information for reproducibility."""
    import platform
    import sys

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        pass

    return info

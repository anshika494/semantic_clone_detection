#!/usr/bin/env python3
"""
Inference Script
=================
Run semantic clone detection on new code pairs or a codebase.

Usage:
  # Check if two code snippets are clones
  python scripts/inference.py --mode pair --file1 method1.java --file2 method2.java

  # Scan a directory for clone pairs
  python scripts/inference.py --mode scan --source-dir /path/to/java/project

  # Run on a CSV of code pairs
  python scripts/inference.py --mode batch --input pairs.csv --output results.csv
"""

import sys
import csv
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from src.utils import load_config, setup_logging, set_seed
from src.preprocessing import CodePreprocessor
from src.embeddings import load_model_checkpoint, EmbeddingGenerator
from src.evaluation import find_optimal_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class CloneDetector:
    """
    Inference engine for semantic clone detection.

    Usage:
        detector = CloneDetector.from_checkpoint("checkpoints/best_model.pt", config)
        sim, is_clone = detector.detect(code1, code2)
    """

    def __init__(
        self,
        model,
        tokenizer,
        preprocessor: CodePreprocessor,
        device: torch.device,
        threshold: float = 0.5,
        max_length: int = 512,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.device = device
        self.threshold = threshold
        self.max_length = max_length

        self.model = self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: Dict,
        threshold: Optional[float] = None,
    ) -> "CloneDetector":
        """Load detector from a saved checkpoint."""
        from src.embeddings import create_embedding_model
        from src.models import SiameseCloneDetector

        model = SiameseCloneDetector(config)
        _, tokenizer = create_embedding_model(config)

        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state)
        logger.info(f"Loaded model from {checkpoint_path}")

        # Get threshold from checkpoint if not provided
        if threshold is None:
            threshold = ckpt.get("metrics", {}).get("optimal_threshold", 0.5)
            logger.info(f"Using threshold from checkpoint: {threshold:.4f}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        preprocessor = CodePreprocessor(config)

        return cls(
            model=model,
            tokenizer=tokenizer,
            preprocessor=preprocessor,
            device=device,
            threshold=threshold,
            max_length=config.get("ast", {}).get("max_token_length", 512),
        )

    @torch.no_grad()
    def detect(self, code1: str, code2: str) -> Tuple[float, bool]:
        """
        Detect if two code snippets are semantic clones.

        Args:
            code1, code2: Java source code strings

        Returns:
            (similarity_score, is_clone)
        """
        # Preprocess
        code1_proc = self.preprocessor.preprocess_for_model(code1)
        code2_proc = self.preprocessor.preprocess_for_model(code2)

        # Tokenize
        enc1 = self.tokenizer(
            code1_proc,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc2 = self.tokenizer(
            code2_proc,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids_1 = enc1["input_ids"].to(self.device)
        attn_mask_1 = enc1["attention_mask"].to(self.device)
        input_ids_2 = enc2["input_ids"].to(self.device)
        attn_mask_2 = enc2["attention_mask"].to(self.device)

        # Forward pass
        output = self.model(input_ids_1, attn_mask_1, input_ids_2, attn_mask_2)
        similarity = float(output["similarity"].cpu().item())

        is_clone = similarity >= self.threshold
        return similarity, is_clone

    @torch.no_grad()
    def detect_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32,
    ) -> List[Dict]:
        """
        Detect clones for a batch of code pairs.

        Args:
            pairs: List of (code1, code2) tuples
            batch_size: Processing batch size

        Returns:
            List of dicts with 'similarity' and 'is_clone'
        """
        results = []
        codes1 = [self.preprocessor.preprocess_for_model(p[0]) for p in pairs]
        codes2 = [self.preprocessor.preprocess_for_model(p[1]) for p in pairs]

        for start in range(0, len(pairs), batch_size):
            batch_c1 = codes1[start:start + batch_size]
            batch_c2 = codes2[start:start + batch_size]

            enc1 = self.tokenizer(
                batch_c1, max_length=self.max_length, truncation=True,
                padding=True, return_tensors="pt"
            )
            enc2 = self.tokenizer(
                batch_c2, max_length=self.max_length, truncation=True,
                padding=True, return_tensors="pt"
            )

            output = self.model(
                enc1["input_ids"].to(self.device),
                enc1["attention_mask"].to(self.device),
                enc2["input_ids"].to(self.device),
                enc2["attention_mask"].to(self.device),
            )

            sims = output["similarity"].cpu().numpy()
            for sim in sims:
                results.append({
                    "similarity": float(sim),
                    "is_clone": bool(sim >= self.threshold),
                    "confidence": float(abs(sim - self.threshold)),
                })

        return results

    def scan_codebase(
        self,
        java_files: List[Path],
        min_method_lines: int = 5,
    ) -> List[Dict]:
        """
        Scan a list of Java files for clone pairs.

        Extracts all methods and computes all-pairs similarity
        (efficient for small-medium codebases).

        Returns:
            List of detected clone pairs sorted by similarity
        """
        from src.ast_processing import ASTProcessor

        logger.info(f"Scanning {len(java_files)} Java files...")

        # Extract methods
        methods = []
        for jfile in java_files:
            try:
                source = jfile.read_text(encoding="utf-8", errors="replace")
                # Simple method extraction: find public/private/protected ... { ... }
                extracted = self._extract_methods_simple(source, str(jfile))
                methods.extend(extracted)
            except Exception as e:
                logger.warning(f"Error processing {jfile}: {e}")

        logger.info(f"Extracted {len(methods)} methods")

        if len(methods) < 2:
            logger.warning("Fewer than 2 methods found; no pairs to compare")
            return []

        # Generate embeddings for all methods
        codes = [m["code"] for m in methods]
        codes_proc = [self.preprocessor.preprocess_for_model(c) for c in codes]

        embeddings = self._encode_all(codes_proc)

        # Compute all-pairs similarities (upper triangle)
        logger.info("Computing pairwise similarities...")
        n = len(methods)
        clone_pairs = []

        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(embeddings[i], embeddings[j]))
                if sim >= self.threshold:
                    clone_pairs.append({
                        "method1": methods[i],
                        "method2": methods[j],
                        "similarity": sim,
                        "is_clone": True,
                    })

        # Sort by similarity
        clone_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        logger.info(f"Found {len(clone_pairs)} clone pairs")
        return clone_pairs

    @torch.no_grad()
    def _encode_all(self, codes: List[str]) -> np.ndarray:
        """Encode all codes and return embeddings matrix."""
        all_embs = []
        batch_size = 32

        for start in range(0, len(codes), batch_size):
            batch = codes[start:start + batch_size]
            enc = self.tokenizer(
                batch, max_length=self.max_length, truncation=True,
                padding=True, return_tensors="pt"
            )
            embs = self.model.encode(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
            )
            all_embs.append(embs.cpu().numpy())

        return np.vstack(all_embs)

    @staticmethod
    def _extract_methods_simple(source: str, file_path: str) -> List[Dict]:
        """Simple regex-based Java method extraction."""
        import re

        METHOD_PATTERN = re.compile(
            r"((?:public|private|protected|static|final|synchronized|abstract)\s+)*"
            r"[\w<>\[\]]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{",
            re.MULTILINE
        )

        methods = []
        for match in METHOD_PATTERN.finditer(source):
            start = match.start()
            # Find matching closing brace
            brace_count = 0
            pos = match.end() - 1  # start at the opening brace
            end = pos

            for i in range(pos, len(source)):
                if source[i] == "{":
                    brace_count += 1
                elif source[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break

            method_code = source[start:end]
            lines = method_code.count("\n")

            if lines >= 5:  # Filter short methods
                methods.append({
                    "code": method_code,
                    "file": file_path,
                    "start_char": start,
                    "end_char": end,
                    "lines": lines,
                })

        return methods


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Clone Detection Inference")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model", type=str, default="checkpoints/best_model.pt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--mode", choices=["pair", "batch", "scan"], default="pair",
        help="Inference mode"
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Similarity threshold (default: from checkpoint)")

    # Pair mode
    parser.add_argument("--file1", type=str, help="First Java file (pair mode)")
    parser.add_argument("--file2", type=str, help="Second Java file (pair mode)")
    parser.add_argument("--code1", type=str, help="First code snippet (pair mode)")
    parser.add_argument("--code2", type=str, help="Second code snippet (pair mode)")

    # Batch mode
    parser.add_argument("--input", type=str, help="Input CSV with code1,code2 columns")
    parser.add_argument("--output", type=str, help="Output CSV path")

    # Scan mode
    parser.add_argument("--source-dir", type=str, help="Directory to scan for Java files")
    parser.add_argument("--scan-output", type=str, default="results/clone_pairs.json")

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    config = load_config(args.config)

    if not Path(args.model).exists():
        logger.error(f"Model checkpoint not found: {args.model}")
        logger.info("Train a model first with: python scripts/train.py")
        return 1

    # Load detector
    detector = CloneDetector.from_checkpoint(
        args.model, config, threshold=args.threshold
    )

    if args.mode == "pair":
        # Single pair comparison
        if args.file1 and args.file2:
            code1 = Path(args.file1).read_text()
            code2 = Path(args.file2).read_text()
        elif args.code1 and args.code2:
            code1, code2 = args.code1, args.code2
        else:
            logger.error("Pair mode requires --file1/--file2 or --code1/--code2")
            return 1

        sim, is_clone = detector.detect(code1, code2)
        print(f"\n{'='*50}")
        print(f"Similarity Score: {sim:.4f}")
        print(f"Threshold:        {detector.threshold:.4f}")
        print(f"Classification:   {'CLONE ✓' if is_clone else 'NOT A CLONE ✗'}")
        print(f"{'='*50}")

    elif args.mode == "batch":
        if not args.input:
            logger.error("Batch mode requires --input")
            return 1

        # Read input CSV
        with open(args.input, "r") as f:
            reader = csv.DictReader(f)
            pairs_data = list(reader)

        pairs = [(p["code1"], p["code2"]) for p in pairs_data]
        results = detector.detect_batch(pairs)

        # Write output
        output_path = args.output or "results/batch_results.csv"
        with open(output_path, "w", newline="") as f:
            fieldnames = list(pairs_data[0].keys()) + ["similarity", "is_clone", "confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for pair_data, result in zip(pairs_data, results):
                writer.writerow({**pair_data, **result})

        n_clones = sum(1 for r in results if r["is_clone"])
        print(f"\nProcessed {len(pairs)} pairs")
        print(f"Detected {n_clones} clone pairs ({100*n_clones/len(pairs):.1f}%)")
        print(f"Results saved to: {output_path}")

    elif args.mode == "scan":
        if not args.source_dir:
            logger.error("Scan mode requires --source-dir")
            return 1

        source_dir = Path(args.source_dir)
        java_files = list(source_dir.rglob("*.java"))
        logger.info(f"Found {len(java_files)} Java files")

        clone_pairs = detector.scan_codebase(java_files)

        # Save results
        output = args.scan_output
        with open(output, "w") as f:
            json.dump([
                {
                    "file1": p["method1"]["file"],
                    "file2": p["method2"]["file"],
                    "similarity": p["similarity"],
                    "lines1": p["method1"]["lines"],
                    "lines2": p["method2"]["lines"],
                }
                for p in clone_pairs
            ], f, indent=2)

        print(f"\nFound {len(clone_pairs)} clone pairs")
        print(f"Results saved to: {output}")

        if clone_pairs:
            print(f"\nTop 5 clones:")
            for pair in clone_pairs[:5]:
                print(
                    f"  sim={pair['similarity']:.4f} | "
                    f"{Path(pair['method1']['file']).name} ↔ "
                    f"{Path(pair['method2']['file']).name}"
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())

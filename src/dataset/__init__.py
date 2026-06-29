"""
BigCloneBench Dataset Loader
==============================
Handles loading and processing of the BigCloneBench dataset,
including pair generation for Type III and Type IV clones.

BigCloneBench schema:
  - functions table: (id, name, file, startline, endline, ...)
  - clones table: (function_id_one, function_id_two, syntactic_type, ...)
  Syntactic types: 1=T1, 2=T2, 3=T3, 4=T4
"""

import os
import sqlite3
import logging
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Iterator
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class CodeMethod:
    """Represents a single Java method from IJaDataset."""
    id: int
    name: str
    file_path: str
    start_line: int
    end_line: int
    source_code: str = ""
    tokens: List[str] = field(default_factory=list)
    ast_tokens: List[str] = field(default_factory=list)

    def __hash__(self):
        return hash(self.id)


@dataclass
class ClonePair:
    """Represents a clone pair with its type label."""
    method1: CodeMethod
    method2: CodeMethod
    clone_type: int   # 1, 2, 3, 4 (0 = non-clone)
    is_clone: bool

    @property
    def type_label(self) -> str:
        if not self.is_clone:
            return "Non-Clone"
        return {1: "T1", 2: "T2", 3: "T3", 4: "T4"}.get(self.clone_type, "Unknown")


# ---------------------------------------------------------------------------
# BigCloneBench Database Interface
# ---------------------------------------------------------------------------

class BigCloneBenchDB:
    """
    Interface to the BigCloneBench SQLite database.

    Database tables:
      functions (id INTEGER, name TEXT, file TEXT, startline INTEGER, endline INTEGER)
      clones    (function_id_one INTEGER, function_id_two INTEGER, syntactic_type INTEGER,
                 min_lines INTEGER, max_lines INTEGER, min_tokens INTEGER, max_tokens INTEGER,
                 min_mt INTEGER, max_mt INTEGER, min_mb INTEGER, max_mb INTEGER)
    """

    CLONE_TYPES = {1: "T1", 2: "T2", 3: "T3", 4: "T4"}

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self):
        if not self._conn:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def get_function_count(self) -> int:
        conn = self.connect()
        cursor = conn.execute("SELECT COUNT(*) FROM functions")
        return cursor.fetchone()[0]

    def get_clone_count(self, clone_types: Optional[List[int]] = None) -> int:
        conn = self.connect()
        if clone_types:
            placeholders = ",".join("?" * len(clone_types))
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM clones WHERE syntactic_type IN ({placeholders})",
                clone_types
            )
        else:
            cursor = conn.execute("SELECT COUNT(*) FROM clones")
        return cursor.fetchone()[0]

    def get_functions_batch(
        self,
        offset: int = 0,
        limit: int = 10000
    ) -> List[Dict]:
        conn = self.connect()
        cursor = conn.execute(
            "SELECT id, name, file, startline, endline FROM functions LIMIT ? OFFSET ?",
            (limit, offset)
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_clone_pairs(
        self,
        clone_types: Optional[List[int]] = None,
        min_lines: int = 5,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Retrieve clone pairs filtered by type.

        Args:
            clone_types: List of syntactic types to include (1-4). None = all.
            min_lines: Minimum number of lines for a method.
            limit: Max pairs to return. None = all.
        """
        conn = self.connect()

        type_filter = ""
        params: List = []

        if clone_types:
            placeholders = ",".join("?" * len(clone_types))
            type_filter = f"WHERE c.syntactic_type IN ({placeholders})"
            params.extend(clone_types)

        limit_clause = f"LIMIT {limit}" if limit else ""

        query = f"""
            SELECT
                c.function_id_one,
                c.function_id_two,
                c.syntactic_type,
                c.min_lines,
                c.max_lines,
                c.min_tokens,
                c.max_tokens,
                f1.name AS name1,
                f1.file AS file1,
                f1.startline AS start1,
                f1.endline AS end1,
                f2.name AS name2,
                f2.file AS file2,
                f2.startline AS start2,
                f2.endline AS end2
            FROM clones c
            JOIN functions f1 ON c.function_id_one = f1.id
            JOIN functions f2 ON c.function_id_two = f2.id
            {type_filter}
            {limit_clause}
        """
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

    def get_all_function_ids(self, clone_types: Optional[List[int]] = None) -> List[int]:
        """Get all function IDs that appear in clone pairs of given types."""
        conn = self.connect()
        if clone_types:
            placeholders = ",".join("?" * len(clone_types))
            query = f"""
                SELECT DISTINCT function_id_one AS id FROM clones
                WHERE syntactic_type IN ({placeholders})
                UNION
                SELECT DISTINCT function_id_two AS id FROM clones
                WHERE syntactic_type IN ({placeholders})
            """
            params = clone_types * 2
        else:
            query = """
                SELECT DISTINCT function_id_one AS id FROM clones
                UNION
                SELECT DISTINCT function_id_two AS id FROM clones
            """
            params = []
        cursor = conn.execute(query, params)
        return [row[0] for row in cursor.fetchall()]


# ---------------------------------------------------------------------------
# Source Code Extractor
# ---------------------------------------------------------------------------

class SourceExtractor:
    """Extracts source code for methods from IJaDataset files."""

    def __init__(self, ijadataset_dir: str):
        self.ijadataset_dir = Path(ijadataset_dir)
        self._cache: Dict[str, List[str]] = {}

    def get_file_lines(self, relative_path: str) -> List[str]:
        """Read file lines with caching."""
        if relative_path not in self._cache:
            full_path = self.ijadataset_dir / relative_path
            try:
                with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                    self._cache[relative_path] = f.readlines()
            except FileNotFoundError:
                logger.warning(f"File not found: {full_path}")
                self._cache[relative_path] = []
        return self._cache[relative_path]

    def extract_method(
        self,
        file_path: str,
        start_line: int,
        end_line: int
    ) -> str:
        """Extract method source code from file (1-indexed lines)."""
        lines = self.get_file_lines(file_path)
        if not lines:
            return ""
        # Convert to 0-indexed
        start = max(0, start_line - 1)
        end = min(len(lines), end_line)
        return "".join(lines[start:end])

    def clear_cache(self):
        self._cache.clear()


# ---------------------------------------------------------------------------
# BigCloneBench Dataset
# ---------------------------------------------------------------------------

class BigCloneBenchDataset(Dataset):
    """
    PyTorch Dataset for BigCloneBench clone pairs.

    Generates positive (clone) and negative (non-clone) pairs for
    training the Siamese network.
    """

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer=None,
        max_length: int = 512,
        split: str = "train",
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        pair = self.pairs[idx]

        code1 = pair["code1"]
        code2 = pair["code2"]
        label = float(pair["label"])  # 1.0 = clone, 0.0 = non-clone
        clone_type = pair.get("clone_type", 0)

        if self.tokenizer is not None:
            enc1 = self.tokenizer(
                code1,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            enc2 = self.tokenizer(
                code2,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids_1": enc1["input_ids"].squeeze(0),
                "attention_mask_1": enc1["attention_mask"].squeeze(0),
                "input_ids_2": enc2["input_ids"].squeeze(0),
                "attention_mask_2": enc2["attention_mask"].squeeze(0),
                "label": label,
                "clone_type": clone_type,
            }

        return {
            "code1": code1,
            "code2": code2,
            "label": label,
            "clone_type": clone_type,
        }


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------

class BigCloneBenchLoader:
    """
    Main dataset loader for BigCloneBench.

    Responsibilities:
    1. Connect to BCB database
    2. Extract source code from IJaDataset
    3. Generate balanced clone/non-clone pairs
    4. Split into train/val/test
    5. Cache processed data
    """

    def __init__(self, config: Dict):
        self.config = config
        self.db_path = config["dataset"]["bcb_db_path"]
        self.ijadataset_dir = config["dataset"]["ijadataset_dir"]
        self.processed_dir = Path(config["dataset"]["processed_dir"])
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self.target_types = [3, 4]  # T3 and T4 by default
        self.max_pairs = config["dataset"].get("max_pairs_per_type", 50000)
        self.negative_ratio = config["dataset"].get("negative_ratio", 1.0)
        self.val_split = config["dataset"].get("val_split", 0.1)
        self.test_split = config["dataset"].get("test_split", 0.1)

        self.db = BigCloneBenchDB(self.db_path)
        self.extractor = SourceExtractor(self.ijadataset_dir)

    def _cache_path(self, split: str) -> Path:
        return self.processed_dir / f"bcb_{split}_pairs.json"

    def _is_cached(self) -> bool:
        return all(
            self._cache_path(s).exists()
            for s in ["train", "val", "test"]
        )

    def load_pairs(self, force_rebuild: bool = False) -> Tuple[List, List, List]:
        """
        Load or build train/val/test pairs.

        Returns:
            (train_pairs, val_pairs, test_pairs)
        """
        if not force_rebuild and self._is_cached():
            logger.info("Loading cached pairs...")
            return self._load_from_cache()

        logger.info("Building dataset from BigCloneBench database...")
        all_pairs = self._build_pairs()

        train, val, test = self._split_pairs(all_pairs)
        self._save_to_cache(train, val, test)

        logger.info(
            f"Dataset built: {len(train)} train, {len(val)} val, {len(test)} test pairs"
        )
        return train, val, test

    def _build_pairs(self) -> List[Dict]:
        """Build all clone and non-clone pairs."""
        positive_pairs = self._load_positive_pairs()
        negative_pairs = self._generate_negative_pairs(positive_pairs)

        all_pairs = positive_pairs + negative_pairs
        np.random.seed(42)
        np.random.shuffle(all_pairs)

        logger.info(
            f"Total pairs: {len(all_pairs)} "
            f"({len(positive_pairs)} positive, {len(negative_pairs)} negative)"
        )
        return all_pairs

    def _load_positive_pairs(self) -> List[Dict]:
        """Load clone pairs from database and extract source code."""
        positive_pairs = []

        with self.db:
            raw_pairs = self.db.get_clone_pairs(
                clone_types=self.target_types,
                limit=self.max_pairs * len(self.target_types)
            )

        logger.info(f"Found {len(raw_pairs)} clone pairs in database")

        for pair in tqdm(raw_pairs, desc="Extracting clone sources"):
            code1 = self.extractor.extract_method(
                pair["file1"], pair["start1"], pair["end1"]
            )
            code2 = self.extractor.extract_method(
                pair["file2"], pair["start2"], pair["end2"]
            )

            if not code1.strip() or not code2.strip():
                continue

            positive_pairs.append({
                "code1": code1,
                "code2": code2,
                "label": 1,
                "clone_type": pair["syntactic_type"],
                "id1": pair["function_id_one"],
                "id2": pair["function_id_two"],
            })

        self.extractor.clear_cache()
        return positive_pairs

    def _generate_negative_pairs(self, positive_pairs: List[Dict]) -> List[Dict]:
        """
        Generate non-clone pairs by random pairing of methods
        that are NOT in any clone relationship together.
        """
        n_negatives = int(len(positive_pairs) * self.negative_ratio)

        # Collect all IDs that appear as positives
        clone_id_pairs = {
            (p["id1"], p["id2"]) for p in positive_pairs
        } | {
            (p["id2"], p["id1"]) for p in positive_pairs
        }

        # Get pool of unique method codes
        method_pool = {}
        for pair in positive_pairs:
            method_pool[pair["id1"]] = pair["code1"]
            method_pool[pair["id2"]] = pair["code2"]

        method_ids = list(method_pool.keys())
        np.random.seed(42)

        negative_pairs = []
        attempts = 0
        max_attempts = n_negatives * 10

        while len(negative_pairs) < n_negatives and attempts < max_attempts:
            i, j = np.random.choice(len(method_ids), 2, replace=False)
            id1, id2 = method_ids[i], method_ids[j]

            if (id1, id2) not in clone_id_pairs and id1 != id2:
                negative_pairs.append({
                    "code1": method_pool[id1],
                    "code2": method_pool[id2],
                    "label": 0,
                    "clone_type": 0,
                    "id1": id1,
                    "id2": id2,
                })
            attempts += 1

        logger.info(f"Generated {len(negative_pairs)} negative pairs")
        return negative_pairs

    def _split_pairs(
        self,
        pairs: List[Dict]
    ) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split pairs into train/val/test with stratification."""
        from sklearn.model_selection import train_test_split

        labels = [p["label"] for p in pairs]

        # First split off test
        train_val, test = train_test_split(
            pairs,
            test_size=self.test_split,
            stratify=labels,
            random_state=42
        )

        # Then split val from train
        train_val_labels = [p["label"] for p in train_val]
        val_ratio = self.val_split / (1 - self.test_split)
        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            stratify=train_val_labels,
            random_state=42
        )

        return train, val, test

    def _save_to_cache(
        self,
        train: List[Dict],
        val: List[Dict],
        test: List[Dict]
    ):
        """Save processed pairs to JSON cache."""
        for split, data in [("train", train), ("val", val), ("test", test)]:
            path = self._cache_path(split)
            with open(path, "w") as f:
                json.dump(data, f)
            logger.info(f"Saved {len(data)} {split} pairs to {path}")

    def _load_from_cache(self) -> Tuple[List, List, List]:
        """Load pairs from JSON cache."""
        result = []
        for split in ["train", "val", "test"]:
            path = self._cache_path(split)
            with open(path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} {split} pairs from cache")
            result.append(data)
        return tuple(result)

    def get_statistics(self) -> Dict:
        """Return dataset statistics."""
        with self.db:
            stats = {
                "total_functions": self.db.get_function_count(),
                "clone_pairs_T1": self.db.get_clone_count([1]),
                "clone_pairs_T2": self.db.get_clone_count([2]),
                "clone_pairs_T3": self.db.get_clone_count([3]),
                "clone_pairs_T4": self.db.get_clone_count([4]),
                "clone_pairs_total": self.db.get_clone_count(),
            }
        return stats


# ---------------------------------------------------------------------------
# Synthetic Dataset (for development/testing without BCB)
# ---------------------------------------------------------------------------

class SyntheticCloneDataset:
    """
    Generates synthetic clone pairs for development and unit testing
    when the actual BigCloneBench dataset is unavailable.
    """

    JAVA_TEMPLATES = [
        # Template 1: Sorting
        """public void bubbleSort(int[] arr) {{
    int n = arr.length;
    for (int i = 0; i < n-1; i++)
        for (int j = 0; j < n-i-1; j++)
            if (arr[j] > arr[j+1]) {{
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }}
}}""",
        # Template 2: Search
        """public int linearSearch(int[] arr, int target) {{
    for (int i = 0; i < arr.length; i++) {{
        if (arr[i] == target) return i;
    }}
    return -1;
}}""",
        # Template 3: Sum
        """public int sumArray(int[] nums) {{
    int total = 0;
    for (int num : nums) total += num;
    return total;
}}""",
    ]

    @classmethod
    def generate_pairs(cls, n_pairs: int = 1000) -> List[Dict]:
        """Generate synthetic clone/non-clone pairs."""
        pairs = []
        np.random.seed(42)

        for i in range(n_pairs // 2):
            # Clone pair (same template, renamed variables)
            template_idx = i % len(cls.JAVA_TEMPLATES)
            code1 = cls.JAVA_TEMPLATES[template_idx]
            code2 = cls._rename_vars(code1)
            pairs.append({
                "code1": code1,
                "code2": code2,
                "label": 1,
                "clone_type": 3,
            })

        for i in range(n_pairs // 2):
            # Non-clone pair (different templates)
            t1 = i % len(cls.JAVA_TEMPLATES)
            t2 = (i + 1) % len(cls.JAVA_TEMPLATES)
            pairs.append({
                "code1": cls.JAVA_TEMPLATES[t1],
                "code2": cls.JAVA_TEMPLATES[t2],
                "label": 0,
                "clone_type": 0,
            })

        np.random.shuffle(pairs)
        return pairs

    @staticmethod
    def _rename_vars(code: str) -> str:
        """Rename variables to create a T2/T3 clone."""
        replacements = {
            "arr": "array",
            "temp": "tmp",
            "n": "size",
            "i": "idx",
            "j": "jdx",
        }
        for old, new in replacements.items():
            code = code.replace(f" {old} ", f" {new} ")
            code = code.replace(f"({old})", f"({new})")
            code = code.replace(f"[{old}]", f"[{new}]")
        return code

"""
Unit Tests for Semantic Clone Detection System
================================================
Tests for all major modules: preprocessing, AST, models, evaluation.
"""

import sys
import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------

TEST_CONFIG = {
    "dataset": {
        "bcb_db_path": "data/raw/bigclonebench.db",
        "ijadataset_dir": "data/raw/bigclonebench/IJaDataset",
        "processed_dir": "data/processed",
        "embeddings_dir": "data/embeddings",
        "max_pairs_per_type": 1000,
        "negative_ratio": 1.0,
        "val_split": 0.1,
        "test_split": 0.1,
    },
    "ast": {
        "parser": "javalang",
        "language": "java",
        "traversal": "preorder",
        "include_node_types": True,
        "include_values": True,
        "max_token_length": 256,
    },
    "preprocessing": {
        "remove_comments": True,
        "normalize_identifiers": True,
        "normalize_literals": True,
        "lowercase": False,
        "min_tokens": 5,
        "max_tokens": 256,
    },
    "model": {
        "backbone": "microsoft/codebert-base",
        "embedding_dim": 768,
        "projection_dim": 64,
        "dropout": 0.1,
        "siamese": {
            "loss": "contrastive",
            "margin": 1.0,
            "temperature": 0.07,
        },
    },
    "training": {
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "fp16": False,
        "early_stopping_patience": 2,
    },
    "inference": {
        "threshold": 0.5,
    },
    "evaluation": {
        "results_dir": "results/test",
    },
    "cache": {
        "use_cache": False,
    },
    "performance": {
        "num_workers": 0,
        "pin_memory": False,
    },
}

# Sample Java code snippets
JAVA_BUBBLE_SORT = """
public void bubbleSort(int[] arr) {
    int n = arr.length;
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}
"""

JAVA_BUBBLE_SORT_RENAMED = """
// Sorting algorithm
public void sortArray(int[] array) {
    /* bubble sort implementation */
    int size = array.length;
    for (int idx = 0; idx < size - 1; idx++) {
        for (int jdx = 0; jdx < size - idx - 1; jdx++) {
            if (array[jdx] > array[jdx + 1]) {
                int tmp = array[jdx];
                array[jdx] = array[jdx + 1];
                array[jdx + 1] = tmp;
            }
        }
    }
}
"""

JAVA_LINEAR_SEARCH = """
public int linearSearch(int[] arr, int target) {
    for (int i = 0; i < arr.length; i++) {
        if (arr[i] == target) {
            return i;
        }
    }
    return -1;
}
"""

JAVA_FACTORIAL = """
public long factorial(int n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}
"""


# ---------------------------------------------------------------------------
# Test: Comment Remover
# ---------------------------------------------------------------------------

class TestCommentRemover(unittest.TestCase):

    def setUp(self):
        from src.preprocessing import CommentRemover
        self.remover = CommentRemover()

    def test_remove_single_line_comments(self):
        code = "int x = 1; // this is a comment\nint y = 2;"
        result = self.remover.remove(code)
        self.assertNotIn("//", result)
        self.assertIn("int x = 1;", result)
        self.assertIn("int y = 2;", result)

    def test_remove_block_comments(self):
        code = "/* block comment */\nint x = 1;"
        result = self.remover.remove(code)
        self.assertNotIn("/*", result)
        self.assertIn("int x = 1;", result)

    def test_remove_javadoc(self):
        code = "/** @param n the number */\npublic int f(int n) { return n; }"
        result = self.remover.remove(code)
        self.assertNotIn("@param", result)

    def test_preserve_strings_with_comment_chars(self):
        code = 'String s = "http://example.com"; // real comment'
        result = self.remover.remove(code)
        self.assertIn("http://example.com", result)
        self.assertNotIn("real comment", result)

    def test_nested_comment_chars_in_string(self):
        code = 'String s = "/* not a comment */";'
        result = self.remover.remove(code)
        self.assertIn("/* not a comment */", result)

    def test_empty_code(self):
        result = self.remover.remove("")
        self.assertEqual(result, "")

    def test_code_without_comments(self):
        code = "int x = 1;\nint y = 2;"
        result = self.remover.remove(code)
        self.assertIn("int x", result)


# ---------------------------------------------------------------------------
# Test: Identifier Normalizer
# ---------------------------------------------------------------------------

class TestIdentifierNormalizer(unittest.TestCase):

    def setUp(self):
        from src.preprocessing import IdentifierNormalizer
        self.normalizer = IdentifierNormalizer(
            normalize_identifiers=True,
            normalize_literals=True,
        )

    def test_normalize_keyword_preserved(self):
        result = self.normalizer.normalize_token("if")
        self.assertEqual(result, "if")

    def test_normalize_integer_literal(self):
        result = self.normalizer.normalize_token("42")
        self.assertEqual(result, "<NUM>")

    def test_normalize_float_literal(self):
        result = self.normalizer.normalize_token("3.14")
        self.assertEqual(result, "<NUM>")

    def test_normalize_string_literal(self):
        result = self.normalizer.normalize_token('"hello world"')
        self.assertEqual(result, "<STR>")

    def test_normalize_variable_identifier(self):
        result = self.normalizer.normalize_token("myVariable")
        self.assertEqual(result, "<ID>")

    def test_normalize_type_identifier(self):
        result = self.normalizer.normalize_token("MyClass")
        self.assertEqual(result, "<TYPE>")

    def test_operator_preserved(self):
        result = self.normalizer.normalize_token("+")
        self.assertEqual(result, "+")

    def test_normalize_sequence(self):
        tokens = ["int", "x", "=", "42", ";"]
        result = self.normalizer.normalize_sequence(tokens)
        self.assertIn("<NUM>", result)
        self.assertIn("int", result)


# ---------------------------------------------------------------------------
# Test: Code Preprocessor
# ---------------------------------------------------------------------------

class TestCodePreprocessor(unittest.TestCase):

    def setUp(self):
        from src.preprocessing import CodePreprocessor
        self.preprocessor = CodePreprocessor(TEST_CONFIG)

    def test_preprocess_removes_comments(self):
        result = self.preprocessor.preprocess(JAVA_BUBBLE_SORT_RENAMED)
        self.assertNotIn("//", result)
        self.assertNotIn("/*", result)

    def test_tokenize_produces_tokens(self):
        tokens = self.preprocessor.tokenize(JAVA_BUBBLE_SORT)
        self.assertGreater(len(tokens), 0)
        self.assertIsInstance(tokens, list)

    def test_preprocess_for_model_returns_string(self):
        result = self.preprocessor.preprocess_for_model(JAVA_BUBBLE_SORT)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result.split()), 0)

    def test_normalization_reduces_vocabulary(self):
        raw_tokens = self.preprocessor.tokenize(JAVA_BUBBLE_SORT)
        norm_tokens = self.preprocessor.normalize_tokens(raw_tokens)

        raw_vocab = set(raw_tokens)
        norm_vocab = set(norm_tokens)
        # Normalization should reduce vocabulary size
        self.assertLessEqual(len(norm_vocab), len(raw_vocab))

    def test_clone_pair_has_similar_structure(self):
        """Clones should produce similar normalized token sequences."""
        proc1 = self.preprocessor.preprocess_for_model(JAVA_BUBBLE_SORT)
        proc2 = self.preprocessor.preprocess_for_model(JAVA_BUBBLE_SORT_RENAMED)

        tokens1 = set(proc1.split())
        tokens2 = set(proc2.split())

        # Jaccard similarity should be reasonably high for clones
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        jaccard = len(intersection) / max(len(union), 1)
        self.assertGreater(jaccard, 0.3)

    def test_max_tokens_respected(self):
        """Output should not exceed max_tokens."""
        long_code = JAVA_BUBBLE_SORT * 20
        result = self.preprocessor.preprocess_for_model(long_code)
        tokens = result.split()
        self.assertLessEqual(len(tokens), self.preprocessor.max_tokens)

    def test_is_valid(self):
        """Valid method should pass minimum token check."""
        tokens = self.preprocessor.tokenize(JAVA_BUBBLE_SORT)
        self.assertTrue(self.preprocessor.is_valid(tokens))

    def test_is_valid_short_code(self):
        """Very short code should fail minimum token check."""
        tokens = ["int", "x"]
        self.assertFalse(self.preprocessor.is_valid(tokens))

    def test_batch_preprocessing(self):
        codes = [JAVA_BUBBLE_SORT, JAVA_LINEAR_SEARCH, JAVA_FACTORIAL]
        results = self.preprocessor.preprocess_batch(codes)
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIsInstance(r, str)

    def test_get_stats(self):
        stats = self.preprocessor.get_stats(JAVA_BUBBLE_SORT_RENAMED)
        self.assertIn("original_chars", stats)
        self.assertIn("processed_token_count", stats)
        self.assertGreater(stats["original_chars"], 0)


# ---------------------------------------------------------------------------
# Test: AST Linearizer
# ---------------------------------------------------------------------------

class TestASTLinearizer(unittest.TestCase):

    def setUp(self):
        from src.ast_processing import ASTNode, ASTLinearizer
        self.ASTNode = ASTNode
        self.linearizer = ASTLinearizer(
            traversal="preorder",
            include_node_types=True,
            include_values=True,
            max_tokens=100,
        )

    def _make_tree(self):
        """Build a small test AST."""
        root = self.ASTNode("METHOD")
        block = self.ASTNode("BLOCK")
        stmt = self.ASTNode("RETURN", value="x")
        block.children = [stmt]
        root.children = [block]
        return root

    def test_preorder_traversal(self):
        root = self._make_tree()
        tokens = self.linearizer.linearize(root)
        self.assertIn("[METHOD]", tokens)
        self.assertIn("[BLOCK]", tokens)
        self.assertIn("[RETURN]", tokens)

    def test_preorder_order(self):
        """METHOD should come before BLOCK in preorder."""
        root = self._make_tree()
        tokens = self.linearizer.linearize(root)
        method_idx = next(i for i, t in enumerate(tokens) if t == "[METHOD]")
        block_idx = next(i for i, t in enumerate(tokens) if t == "[BLOCK]")
        self.assertLess(method_idx, block_idx)

    def test_values_included(self):
        root = self._make_tree()
        tokens = self.linearizer.linearize(root)
        self.assertIn("x", tokens)

    def test_max_tokens_respected(self):
        from src.ast_processing import ASTLinearizer
        linearizer = ASTLinearizer(max_tokens=5)
        root = self._make_tree()
        tokens = linearizer.linearize(root)
        self.assertLessEqual(len(tokens), 5)

    def test_linearize_to_string(self):
        root = self._make_tree()
        result = self.linearizer.linearize_to_string(root)
        self.assertIsInstance(result, str)
        self.assertIn("[METHOD]", result)

    def test_bfs_traversal(self):
        from src.ast_processing import ASTLinearizer
        linearizer = ASTLinearizer(traversal="bfs")
        root = self._make_tree()
        tokens = linearizer.linearize(root)
        self.assertIn("[METHOD]", tokens)

    def test_postorder_traversal(self):
        from src.ast_processing import ASTLinearizer
        linearizer = ASTLinearizer(traversal="postorder")
        root = self._make_tree()
        tokens = linearizer.linearize(root)
        # In postorder, RETURN (leaf) comes before METHOD (root)
        return_idx = next(i for i, t in enumerate(tokens) if t == "[RETURN]")
        method_idx = next(i for i, t in enumerate(tokens) if t == "[METHOD]")
        self.assertLess(return_idx, method_idx)


# ---------------------------------------------------------------------------
# Test: Loss Functions
# ---------------------------------------------------------------------------

class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        import torch
        self.torch = torch

    def test_contrastive_loss_clones(self):
        """Clone pairs should have low contrastive loss when close."""
        from src.models import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=1.0)

        emb1 = self.torch.tensor([[1.0, 0.0, 0.0]])
        emb2 = self.torch.tensor([[0.9, 0.1, 0.0]])
        labels = self.torch.tensor([1.0])  # clone

        loss = loss_fn(emb1, emb2, labels)
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)

    def test_contrastive_loss_non_clones(self):
        """Non-clone pairs far apart should have low loss."""
        from src.models import ContrastiveLoss
        loss_fn = ContrastiveLoss(margin=1.0)

        emb1 = self.torch.tensor([[1.0, 0.0, 0.0]])
        emb2 = self.torch.tensor([[-1.0, 0.0, 0.0]])
        labels = self.torch.tensor([0.0])  # non-clone

        loss = loss_fn(emb1, emb2, labels)
        self.assertAlmostEqual(loss.item(), 0.0, places=3)

    def test_cosine_loss(self):
        from src.models import CosineSimilarityLoss
        loss_fn = CosineSimilarityLoss(margin=0.5)

        emb1 = self.torch.randn(4, 64)
        emb2 = self.torch.randn(4, 64)
        labels = self.torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(emb1, emb2, labels)
        self.assertGreaterEqual(loss.item(), 0)

    def test_triplet_loss(self):
        from src.models import TripletLoss
        loss_fn = TripletLoss(margin=0.5)

        anchor = self.torch.tensor([[1.0, 0.0]])
        positive = self.torch.tensor([[0.9, 0.1]])   # close to anchor
        negative = self.torch.tensor([[-1.0, 0.0]])  # far from anchor

        loss = loss_fn(anchor, positive, negative)
        self.assertAlmostEqual(loss.item(), 0.0, places=2)  # margin satisfied

    def test_ntxent_loss(self):
        from src.models import NTXentLoss
        loss_fn = NTXentLoss(temperature=0.07)

        import torch
        emb1 = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)
        emb2 = torch.nn.functional.normalize(torch.randn(8, 64), dim=-1)
        labels = torch.tensor([1, 0, 1, 0, 1, 0, 1, 0], dtype=torch.float)

        loss = loss_fn(emb1, emb2, labels)
        self.assertGreater(loss.item(), 0)

    def test_loss_batch_size_1(self):
        from src.models import ContrastiveLoss
        loss_fn = ContrastiveLoss()
        import torch
        emb1 = torch.randn(1, 32)
        emb2 = torch.randn(1, 32)
        labels = torch.tensor([1.0])
        loss = loss_fn(emb1, emb2, labels)
        self.assertTrue(torch.isfinite(loss))


# ---------------------------------------------------------------------------
# Test: Evaluation Metrics
# ---------------------------------------------------------------------------

class TestEvaluationMetrics(unittest.TestCase):

    def test_compute_metrics_perfect(self):
        from src.evaluation import compute_metrics
        sims = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])

        metrics = compute_metrics(sims, labels, threshold=0.5)
        self.assertAlmostEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["f1"], 1.0)
        self.assertAlmostEqual(metrics["precision"], 1.0)
        self.assertAlmostEqual(metrics["recall"], 1.0)

    def test_compute_metrics_all_wrong(self):
        from src.evaluation import compute_metrics
        sims = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])

        metrics = compute_metrics(sims, labels, threshold=0.5)
        self.assertAlmostEqual(metrics["accuracy"], 0.0)

    def test_roc_auc_perfect(self):
        from src.evaluation import compute_metrics
        sims = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([1, 1, 0, 0])
        metrics = compute_metrics(sims, labels)
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)

    def test_roc_auc_random(self):
        from src.evaluation import compute_metrics
        np.random.seed(42)
        sims = np.random.random(100)
        labels = np.random.randint(0, 2, 100)
        metrics = compute_metrics(sims, labels)
        # Random should be near 0.5
        self.assertGreater(metrics["roc_auc"], 0.2)
        self.assertLess(metrics["roc_auc"], 0.8)

    def test_find_optimal_threshold(self):
        from src.evaluation import find_optimal_threshold
        # Perfect separator at 0.5
        sims = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0, 0])

        threshold, f1 = find_optimal_threshold(sims, labels)
        self.assertAlmostEqual(f1, 1.0, places=2)
        self.assertBetween(threshold, 0.3, 0.7)

    def assertBetween(self, val, lo, hi):
        self.assertGreaterEqual(val, lo)
        self.assertLessEqual(val, hi)

    def test_per_clone_type(self):
        from src.evaluation import evaluate_per_clone_type
        sims = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = np.array([1, 1, 1, 0, 0])
        types = np.array([3, 4, 3, 0, 0])

        results = evaluate_per_clone_type(sims, labels, types, threshold=0.5)
        self.assertIn("T3 (Near-Miss)", results)
        self.assertIn("T4 (Semantic)", results)
        self.assertIn("Non-Clone", results)

    def test_metrics_keys(self):
        from src.evaluation import compute_metrics
        sims = np.random.random(50)
        labels = np.random.randint(0, 2, 50)
        metrics = compute_metrics(sims, labels, threshold=0.5)

        required_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        for key in required_keys:
            self.assertIn(key, metrics, f"Missing metric: {key}")


# ---------------------------------------------------------------------------
# Test: Dataset Generation
# ---------------------------------------------------------------------------

class TestSyntheticDataset(unittest.TestCase):

    def test_generate_pairs(self):
        from src.dataset import SyntheticCloneDataset
        pairs = SyntheticCloneDataset.generate_pairs(n_pairs=100)
        self.assertEqual(len(pairs), 100)

    def test_pair_structure(self):
        from src.dataset import SyntheticCloneDataset
        pairs = SyntheticCloneDataset.generate_pairs(n_pairs=10)
        for pair in pairs:
            self.assertIn("code1", pair)
            self.assertIn("code2", pair)
            self.assertIn("label", pair)
            self.assertIn("clone_type", pair)
            self.assertIn(pair["label"], [0, 1])

    def test_balanced_labels(self):
        from src.dataset import SyntheticCloneDataset
        pairs = SyntheticCloneDataset.generate_pairs(n_pairs=1000)
        labels = [p["label"] for p in pairs]
        clone_ratio = sum(labels) / len(labels)
        # Should be roughly balanced
        self.assertGreater(clone_ratio, 0.3)
        self.assertLess(clone_ratio, 0.7)

    def test_rename_vars(self):
        from src.dataset import SyntheticCloneDataset
        code = "int arr[] = {1, 2, 3};"
        renamed = SyntheticCloneDataset._rename_vars(code)
        # Should be different but same structure
        self.assertNotEqual(code, renamed)


# ---------------------------------------------------------------------------
# Test: BigCloneBenchDataset (PyTorch)
# ---------------------------------------------------------------------------

class TestBigCloneBenchDataset(unittest.TestCase):

    def setUp(self):
        self.pairs = [
            {"code1": JAVA_BUBBLE_SORT, "code2": JAVA_BUBBLE_SORT_RENAMED,
             "label": 1, "clone_type": 3},
            {"code1": JAVA_BUBBLE_SORT, "code2": JAVA_LINEAR_SEARCH,
             "label": 0, "clone_type": 0},
        ]

    def test_dataset_length(self):
        from src.dataset import BigCloneBenchDataset
        dataset = BigCloneBenchDataset(self.pairs)
        self.assertEqual(len(dataset), 2)

    def test_dataset_item_without_tokenizer(self):
        from src.dataset import BigCloneBenchDataset
        dataset = BigCloneBenchDataset(self.pairs)
        item = dataset[0]
        self.assertIn("code1", item)
        self.assertIn("code2", item)
        self.assertIn("label", item)
        self.assertEqual(item["label"], 1.0)

    def test_dataset_item_structure(self):
        from src.dataset import BigCloneBenchDataset
        dataset = BigCloneBenchDataset(self.pairs)
        item = dataset[1]
        self.assertEqual(item["label"], 0.0)
        self.assertEqual(item["clone_type"], 0)


# ---------------------------------------------------------------------------
# Test: Hard Negative Mining
# ---------------------------------------------------------------------------

class TestHardNegativeMiner(unittest.TestCase):

    def test_mine_triplets_basic(self):
        import torch
        from src.models import HardNegativeMiner
        miner = HardNegativeMiner(strategy="hard")

        embeddings = torch.randn(8, 16)
        labels = torch.tensor([1, 1, 0, 0, 1, 0, 1, 0])

        anchors, positives, negatives = miner.mine_triplets(embeddings, labels)

        if anchors is not None:
            self.assertEqual(anchors.shape, positives.shape)
            self.assertEqual(anchors.shape, negatives.shape)

    def test_mine_triplets_semi_hard(self):
        import torch
        from src.models import HardNegativeMiner
        miner = HardNegativeMiner(strategy="semi-hard")

        torch.manual_seed(42)
        embeddings = torch.randn(16, 32)
        labels = torch.tensor([1, 1, 0, 0] * 4)

        anchors, positives, negatives = miner.mine_triplets(embeddings, labels)
        # Semi-hard may return fewer triplets; that's fine
        if anchors is not None:
            self.assertGreater(len(anchors), 0)


# ---------------------------------------------------------------------------
# Test: Utilities
# ---------------------------------------------------------------------------

class TestUtils(unittest.TestCase):

    def test_set_seed(self):
        import torch
        from src.utils import set_seed
        set_seed(42)
        val1 = torch.randn(5).numpy()
        set_seed(42)
        val2 = torch.randn(5).numpy()
        np.testing.assert_array_almost_equal(val1, val2)

    def test_merge_configs(self):
        from src.utils import merge_configs
        base = {"a": 1, "b": {"c": 2, "d": 3}}
        override = {"b": {"c": 99}, "e": 5}
        result = merge_configs(base, override)
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"]["c"], 99)
        self.assertEqual(result["b"]["d"], 3)
        self.assertEqual(result["e"], 5)

    def test_flatten_dict(self):
        from src.utils import flatten_dict
        d = {"a": {"b": 1, "c": {"d": 2}}}
        flat = flatten_dict(d)
        self.assertIn("a/b", flat)
        self.assertIn("a/c/d", flat)

    def test_batch_iterable(self):
        from src.utils import batch_iterable
        items = list(range(10))
        batches = list(batch_iterable(items, batch_size=3))
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[-1], [9])

    def test_progress_tracker_best(self):
        from src.utils import ProgressTracker
        tracker = ProgressTracker(patience=3)
        self.assertTrue(tracker.update(0.5))  # First update is always best
        self.assertFalse(tracker.update(0.4))  # Worse
        self.assertTrue(tracker.update(0.6))  # New best

    def test_progress_tracker_early_stop(self):
        from src.utils import ProgressTracker
        tracker = ProgressTracker(patience=2)
        tracker.update(0.8)
        tracker.update(0.7)
        self.assertFalse(tracker.should_stop)
        tracker.update(0.6)
        self.assertTrue(tracker.should_stop)

    def test_timer_context_manager(self):
        import time
        from src.utils import timer
        with timer("test"):
            time.sleep(0.01)  # Should not raise


# ---------------------------------------------------------------------------
# Integration Test: Full Pipeline (lightweight)
# ---------------------------------------------------------------------------

class TestPipelineIntegration(unittest.TestCase):
    """
    Lightweight integration test that runs the full pipeline
    on a tiny synthetic dataset without downloading any models.
    """

    def test_preprocessing_pipeline(self):
        """Test that full preprocessing pipeline runs end-to-end."""
        from src.preprocessing import CodePreprocessor, BatchPreprocessor
        from src.dataset import SyntheticCloneDataset

        pairs = SyntheticCloneDataset.generate_pairs(n_pairs=10)
        preprocessor = BatchPreprocessor(TEST_CONFIG)
        processed = preprocessor.process_dataset(pairs, show_progress=False)

        self.assertGreater(len(processed), 0)
        for pair in processed:
            self.assertIn("code1_processed", pair)
            self.assertIn("code2_processed", pair)

    def test_evaluation_pipeline(self):
        """Test evaluation metrics computation pipeline."""
        from src.evaluation import CloneDetectionEvaluator

        np.random.seed(42)
        n = 200
        sims = np.random.beta(2, 2, n)
        labels = (sims > 0.5).astype(int)
        # Add noise
        sims += np.random.normal(0, 0.1, n)
        sims = np.clip(sims, 0, 1)

        config = {**TEST_CONFIG}
        evaluator = CloneDetectionEvaluator(config)
        results = evaluator.evaluate(sims, labels)

        self.assertIn("overall", results)
        self.assertIn("f1", results["overall"])
        self.assertIn("roc_auc", results["overall"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)

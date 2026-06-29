"""
Embedding Generation Module
=============================
Generates semantic embeddings for code snippets using pretrained
transformer models (CodeBERT, GraphCodeBERT, CodeT5).

Features:
  - Pooling strategies: [CLS], mean, max, attention-weighted
  - Batch processing with GPU acceleration
  - Disk caching of embeddings
  - Mixed precision (FP16) support
"""

import os
import logging
import hashlib
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pooling strategies
# ---------------------------------------------------------------------------

class PoolingStrategy(nn.Module):
    """Multiple pooling strategies for transformer output."""

    def __init__(self, strategy: str = "mean"):
        super().__init__()
        assert strategy in {"cls", "mean", "max", "attention"}
        self.strategy = strategy

    def forward(
        self,
        token_embeddings: torch.Tensor,       # [B, L, D]
        attention_mask: torch.Tensor,          # [B, L]
    ) -> torch.Tensor:                         # [B, D]

        if self.strategy == "cls":
            return token_embeddings[:, 0, :]

        elif self.strategy == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (token_embeddings * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return summed / count

        elif self.strategy == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            # Replace padding with large negative
            tokens = token_embeddings.masked_fill(~mask, -1e9)
            return tokens.max(dim=1).values

        else:  # attention
            # Weighted mean with learned attention over tokens
            scores = token_embeddings.mean(dim=-1)  # [B, L]
            scores = scores.masked_fill(attention_mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [B, L, 1]
            return (token_embeddings * weights).sum(dim=1)


# ---------------------------------------------------------------------------
# Code Embedding Model (wrapper around pretrained)
# ---------------------------------------------------------------------------

class CodeEmbeddingModel(nn.Module):
    """
    Wraps a pretrained transformer to produce fixed-size code embeddings.

    Architecture:
      Input → Transformer (frozen or fine-tuned) → Pooling → [Optional Projection] → L2 Normalized
    """

    def __init__(
        self,
        model_name: str = "microsoft/codebert-base",
        embedding_dim: int = 768,
        projection_dim: int = 256,
        pooling: str = "mean",
        dropout: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        from transformers import AutoModel, AutoConfig

        logger.info(f"Loading pretrained model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=config)

        if freeze_backbone:
            for param in self.transformer.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")

        self.pooling = PoolingStrategy(pooling)
        self.dropout = nn.Dropout(dropout)

        # Optional projection head (for contrastive learning)
        hidden_size = config.hidden_size
        if projection_dim and projection_dim != hidden_size:
            self.projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, projection_dim),
            )
            self.output_dim = projection_dim
        else:
            self.projection = nn.Identity()
            self.output_dim = hidden_size

        self.embedding_dim = hidden_size

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a batch of code sequences into embeddings."""
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Get token embeddings
        token_embs = outputs.last_hidden_state  # [B, L, D]

        # Pool
        pooled = self.pooling(token_embs, attention_mask)  # [B, D]
        pooled = self.dropout(pooled)

        # Project
        projected = self.projection(pooled)  # [B, projection_dim]

        # L2 normalize
        normalized = F.normalize(projected, p=2, dim=-1)

        return normalized

    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Siamese forward pass: encode both inputs.

        Returns:
            (emb1, emb2) - L2-normalized embeddings for both inputs
        """
        emb1 = self.encode(input_ids_1, attention_mask_1)
        emb2 = self.encode(input_ids_2, attention_mask_2)
        return emb1, emb2

    def get_similarity(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine similarity between two embedding batches."""
        # Since embeddings are L2-normalized, dot product = cosine similarity
        return (emb1 * emb2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Embedding Generator (inference)
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """
    Generates and caches embeddings for code snippets.

    Handles:
    - Batched encoding
    - GPU/CPU device management
    - Disk caching with content hashing
    - Mixed precision inference
    """

    def __init__(
        self,
        model: CodeEmbeddingModel,
        tokenizer,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 512,
        use_fp16: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"EmbeddingGenerator: device={self.device}, "
            f"fp16={self.use_fp16}, batch_size={batch_size}"
        )

    def _cache_key(self, code: str) -> str:
        """Generate cache key from code content hash."""
        return hashlib.md5(code.encode()).hexdigest()

    def _load_from_cache(self, key: str) -> Optional[np.ndarray]:
        if not self.cache_dir:
            return None
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        return None

    def _save_to_cache(self, key: str, embedding: np.ndarray):
        if self.cache_dir:
            path = self.cache_dir / f"{key}.npy"
            np.save(path, embedding)

    @torch.no_grad()
    def encode(
        self,
        codes: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of code strings into embeddings.

        Args:
            codes: List of code strings to encode
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape [N, embedding_dim]
        """
        all_embeddings = []
        cached_indices = {}

        # Check cache
        codes_to_encode = []
        code_indices = []

        for i, code in enumerate(codes):
            key = self._cache_key(code)
            cached = self._load_from_cache(key)
            if cached is not None:
                cached_indices[i] = cached
            else:
                codes_to_encode.append((i, code, key))

        # Encode uncached codes in batches
        if codes_to_encode:
            for batch_start in tqdm(
                range(0, len(codes_to_encode), self.batch_size),
                desc="Generating embeddings",
                disable=not show_progress,
            ):
                batch = codes_to_encode[batch_start : batch_start + self.batch_size]
                indices = [item[0] for item in batch]
                batch_codes = [item[1] for item in batch]
                batch_keys = [item[2] for item in batch]

                # Tokenize
                encoded = self.tokenizer(
                    batch_codes,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )

                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)

                # Forward pass with optional FP16
                with torch.cuda.amp.autocast(enabled=self.use_fp16):
                    embeddings = self.model.encode(input_ids, attention_mask)

                embeddings_np = embeddings.cpu().float().numpy()

                # Store results and cache
                for j, (orig_idx, key) in enumerate(zip(indices, batch_keys)):
                    emb = embeddings_np[j]
                    cached_indices[orig_idx] = emb
                    self._save_to_cache(key, emb)

        # Reassemble in original order
        all_embeddings = [cached_indices[i] for i in range(len(codes))]
        return np.vstack(all_embeddings)

    def compute_similarity(
        self,
        codes1: List[str],
        codes2: List[str],
    ) -> np.ndarray:
        """
        Compute pairwise cosine similarities between two lists of codes.

        Returns:
            numpy array of shape [N] with similarity scores
        """
        embs1 = self.encode(codes1)
        embs2 = self.encode(codes2)

        # Cosine similarity (embeddings are L2-normalized)
        similarities = np.sum(embs1 * embs2, axis=1)
        return similarities

    def encode_dataset(
        self,
        pairs: List[Dict],
        output_path: Optional[str] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Encode all code pairs in a dataset.

        Returns:
            Dict with 'emb1', 'emb2', 'labels', 'similarities' arrays
        """
        codes1 = [p["code1_processed"] if "code1_processed" in p else p["code1"]
                  for p in pairs]
        codes2 = [p["code2_processed"] if "code2_processed" in p else p["code2"]
                  for p in pairs]
        labels = np.array([p["label"] for p in pairs])

        logger.info(f"Encoding {len(pairs)} pairs...")
        embs1 = self.encode(codes1)
        embs2 = self.encode(codes2)
        similarities = np.sum(embs1 * embs2, axis=1)

        result = {
            "emb1": embs1,
            "emb2": embs2,
            "labels": labels,
            "similarities": similarities,
        }

        if output_path:
            np.savez_compressed(output_path, **result)
            logger.info(f"Saved embeddings to {output_path}")

        return result


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_embedding_model(config: Dict) -> Tuple[CodeEmbeddingModel, any]:
    """
    Factory function to create model + tokenizer from config.

    Returns:
        (model, tokenizer)
    """
    from transformers import AutoTokenizer

    model_config = config.get("model", {})
    backbone = model_config.get("backbone", "microsoft/codebert-base")
    embedding_dim = model_config.get("embedding_dim", 768)
    projection_dim = model_config.get("projection_dim", 256)
    dropout = model_config.get("dropout", 0.1)

    logger.info(f"Loading tokenizer: {backbone}")
    tokenizer = AutoTokenizer.from_pretrained(backbone)

    model = CodeEmbeddingModel(
        model_name=backbone,
        embedding_dim=embedding_dim,
        projection_dim=projection_dim,
        dropout=dropout,
    )

    return model, tokenizer


def load_model_checkpoint(
    checkpoint_path: str,
    config: Dict,
) -> Tuple[CodeEmbeddingModel, any]:
    """Load model from a saved checkpoint."""
    model, tokenizer = create_embedding_model(config)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    logger.info(f"Loaded model from {checkpoint_path}")
    return model, tokenizer

"""
Siamese Network Models & Loss Functions
=========================================
Implements Siamese neural networks for learning code similarity,
with contrastive loss and triplet loss options.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.embeddings import CodeEmbeddingModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss Functions
# ---------------------------------------------------------------------------

class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese networks.

    L = (1-y) * (1/2) * D^2 + y * (1/2) * max(0, margin - D)^2

    Where:
      - y = 0 for similar pairs (clones), 1 for dissimilar
      - D = Euclidean distance between embeddings
      - margin = minimum distance for dissimilar pairs

    Note: We use convention y=1 for clones, y=0 for non-clones,
    and adapt the loss accordingly.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,      # [B, D]
        emb2: torch.Tensor,      # [B, D]
        labels: torch.Tensor,    # [B] — 1 for clone, 0 for non-clone
    ) -> torch.Tensor:
        # Euclidean distance
        dist = F.pairwise_distance(emb1, emb2, p=2)

        # Clone pairs (label=1): minimize distance
        # Non-clone pairs (label=0): push apart by margin
        clone_loss = labels * dist.pow(2)
        non_clone_loss = (1 - labels) * F.relu(self.margin - dist).pow(2)

        loss = 0.5 * (clone_loss + non_clone_loss)
        return loss.mean()


class CosineSimilarityLoss(nn.Module):
    """
    Loss based on cosine similarity with margin.

    For clones: maximize cosine similarity (→ 1)
    For non-clones: minimize cosine similarity below margin (→ -1 or < threshold)
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = F.cosine_similarity(emb1, emb2, dim=-1)  # [B]

        # Clone: maximize similarity (loss = 1 - sim)
        clone_loss = labels * (1 - cos_sim)

        # Non-clone: push similarity below margin
        non_clone_loss = (1 - labels) * F.relu(cos_sim - self.margin)

        return (clone_loss + non_clone_loss).mean()


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.

    L = max(0, d(anchor, positive) - d(anchor, negative) + margin)

    Requires triplets: (anchor, positive_clone, negative_non_clone)
    """

    def __init__(self, margin: float = 0.5, distance: str = "cosine"):
        super().__init__()
        self.margin = margin
        self.distance = distance

    def _compute_distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance == "cosine":
            return 1 - F.cosine_similarity(a, b, dim=-1)
        else:  # euclidean
            return F.pairwise_distance(a, b, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        dist_pos = self._compute_distance(anchor, positive)
        dist_neg = self._compute_distance(anchor, negative)
        loss = F.relu(dist_pos - dist_neg + self.margin)
        return loss.mean()


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    Used in SimCLR / contrastive learning frameworks.

    Treats each pair (i, j) where label[i]==label[j]==1 as positives,
    all others as negatives within the batch.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        emb1: torch.Tensor,    # [B, D]
        emb2: torch.Tensor,    # [B, D]
        labels: torch.Tensor,  # [B] — 1 for clone pairs
    ) -> torch.Tensor:
        batch_size = emb1.shape[0]

        # Concatenate all embeddings: [2B, D]
        embeddings = torch.cat([emb1, emb2], dim=0)

        # Compute similarity matrix: [2B, 2B]
        sim_matrix = torch.mm(embeddings, embeddings.T) / self.temperature

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=emb1.device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # For each clone pair (i, i+B) and (i+B, i), treat as positive
        labels_expanded = torch.cat([labels, labels], dim=0)

        # Positive pairs: corresponding cross pairs for clones
        pos_indices = torch.arange(batch_size, device=emb1.device)
        pos_sim_1 = sim_matrix[:batch_size][
            torch.arange(batch_size), pos_indices + batch_size
        ]  # emb1[i] vs emb2[i]
        pos_sim_2 = sim_matrix[batch_size:][
            torch.arange(batch_size), pos_indices
        ]  # emb2[i] vs emb1[i]

        # Loss = -log(exp(pos_sim) / sum(exp(sim_matrix)))
        log_denom = torch.logsumexp(sim_matrix[:batch_size], dim=-1)
        loss_1 = -(pos_sim_1 - log_denom) * labels.float()

        log_denom_2 = torch.logsumexp(sim_matrix[batch_size:], dim=-1)
        loss_2 = -(pos_sim_2 - log_denom_2) * labels.float()

        n_clones = labels.float().sum().clamp(min=1)
        return (loss_1 + loss_2).sum() / (2 * n_clones)


# ---------------------------------------------------------------------------
# Siamese Network
# ---------------------------------------------------------------------------

class SiameseCloneDetector(nn.Module):
    """
    Siamese neural network for code clone detection.

    Architecture:
      - Shared encoder (CodeBERT/GraphCodeBERT)
      - Loss: Contrastive, Cosine, or Triplet
      - Similarity scoring via cosine similarity
    """

    def __init__(self, config: Dict):
        super().__init__()

        model_config = config.get("model", {})
        siamese_config = model_config.get("siamese", {})

        backbone = model_config.get("backbone", "microsoft/codebert-base")
        projection_dim = model_config.get("projection_dim", 256)
        dropout = model_config.get("dropout", 0.1)

        self.loss_type = siamese_config.get("loss", "contrastive")
        margin = siamese_config.get("margin", 1.0)
        temperature = siamese_config.get("temperature", 0.07)

        # Shared encoder
        self.encoder = CodeEmbeddingModel(
            model_name=backbone,
            projection_dim=projection_dim,
            dropout=dropout,
        )

        # Loss function
        if self.loss_type == "contrastive":
            self.criterion = ContrastiveLoss(margin=margin)
        elif self.loss_type == "cosine":
            self.criterion = CosineSimilarityLoss(margin=margin)
        elif self.loss_type == "triplet":
            self.criterion = TripletLoss(margin=margin)
        elif self.loss_type == "ntxent":
            self.criterion = NTXentLoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        logger.info(
            f"SiameseCloneDetector: backbone={backbone}, "
            f"loss={self.loss_type}, margin={margin}"
        )

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a single batch of code into embeddings."""
        return self.encoder.encode(input_ids, attention_mask)

    def forward(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full Siamese forward pass.

        Returns dict with:
          - 'emb1', 'emb2': embeddings
          - 'similarity': cosine similarity scores
          - 'loss': training loss (if labels provided)
        """
        emb1 = self.encode(input_ids_1, attention_mask_1)
        emb2 = self.encode(input_ids_2, attention_mask_2)

        # Cosine similarity (normalized embeddings → dot product)
        similarity = (emb1 * emb2).sum(dim=-1)  # [B]

        output = {
            "emb1": emb1,
            "emb2": emb2,
            "similarity": similarity,
        }

        if labels is not None:
            if self.loss_type == "triplet":
                # For triplet loss, we split batch into anchor/pos/neg
                # Assumes batch is organized as pairs; use positive pairs as anchor-pos
                # and negative samples as negatives
                # This is a simplified approach; full triplet mining is in the trainer
                loss = torch.tensor(0.0, device=emb1.device)
            else:
                loss = self.criterion(emb1, emb2, labels)
            output["loss"] = loss

        return output

    def get_similarity(
        self,
        input_ids_1: torch.Tensor,
        attention_mask_1: torch.Tensor,
        input_ids_2: torch.Tensor,
        attention_mask_2: torch.Tensor,
    ) -> torch.Tensor:
        """Inference: compute similarity scores without loss."""
        with torch.no_grad():
            output = self.forward(
                input_ids_1, attention_mask_1,
                input_ids_2, attention_mask_2,
            )
        return output["similarity"]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Hard Negative Mining
# ---------------------------------------------------------------------------

class HardNegativeMiner:
    """
    Online hard negative mining for contrastive/triplet training.

    Selects the hardest negatives within a batch to improve
    learning efficiency.
    """

    def __init__(self, strategy: str = "semi-hard"):
        assert strategy in {"hard", "semi-hard", "random"}
        self.strategy = strategy

    def mine_triplets(
        self,
        embeddings: torch.Tensor,  # [B, D]
        labels: torch.Tensor,      # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mine anchor-positive-negative triplets from a batch.

        Returns:
            (anchors, positives, negatives) — index tensors
        """
        device = embeddings.device
        batch_size = len(labels)

        # Compute all pairwise distances
        dist_matrix = torch.cdist(embeddings, embeddings, p=2)  # [B, B]

        triplets = []

        for i in range(batch_size):
            # Positive indices (same class, different sample)
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=device) != i)
            # Negative indices (different class)
            neg_mask = labels != labels[i]

            if not pos_mask.any() or not neg_mask.any():
                continue

            pos_dists = dist_matrix[i][pos_mask]
            neg_dists = dist_matrix[i][neg_mask]

            # Hardest positive: farthest positive
            hardest_pos_idx = pos_mask.nonzero(as_tuple=True)[0][pos_dists.argmax()]

            if self.strategy == "hard":
                # Hardest negative: closest negative
                hardest_neg_idx = neg_mask.nonzero(as_tuple=True)[0][neg_dists.argmin()]
            elif self.strategy == "semi-hard":
                # Semi-hard: negative closer than positive but far from anchor
                pos_dist = dist_matrix[i, hardest_pos_idx]
                semi_hard_mask = neg_dists > pos_dist
                if semi_hard_mask.any():
                    neg_indices = neg_mask.nonzero(as_tuple=True)[0]
                    semi_hard_negs = neg_indices[semi_hard_mask]
                    hardest_neg_idx = semi_hard_negs[
                        neg_dists[semi_hard_mask].argmin()
                    ]
                else:
                    hardest_neg_idx = neg_mask.nonzero(as_tuple=True)[0][neg_dists.argmin()]
            else:
                # Random negative
                neg_indices = neg_mask.nonzero(as_tuple=True)[0]
                hardest_neg_idx = neg_indices[torch.randint(len(neg_indices), (1,)).item()]

            triplets.append((i, hardest_pos_idx.item(), hardest_neg_idx.item()))

        if not triplets:
            return None, None, None

        anchor_ids = torch.tensor([t[0] for t in triplets], device=device)
        pos_ids = torch.tensor([t[1] for t in triplets], device=device)
        neg_ids = torch.tensor([t[2] for t in triplets], device=device)

        return anchor_ids, pos_ids, neg_ids

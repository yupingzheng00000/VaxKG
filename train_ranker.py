"""Train a vaccine-to-adjuvant ranker using PyTorch Geometric.

This script implements the modelling blueprint we discussed earlier:

* Build transductive (leave-vaccine-out) and inductive (leave-disease-out)
  splits with persisted JSONL manifests for reproducibility.
* Assemble a :class:`~torch_geometric.data.HeteroData` graph spanning Vaccine,
  Disease, Platform, and Adjuvant nodes with hashed text features derived from
  the processed training snapshot.
* Optimise a PyG-powered relational encoder with a listwise ranking loss
  (ListNet) and an auxiliary typed negative-sampling link prediction head.
* Evaluate with NDCG@K / Recall@K for ranking and filtered MRR / Hits@K for
  link prediction, mirroring common KG benchmarks.

PyTorch Geometric handles the heterogeneous message passing and aggregation,
reducing custom tensor plumbing and giving us tested CUDA kernels for faster
training on larger graphs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import APPNP, HeteroConv, SAGEConv
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "train_ranker.py now depends on PyTorch Geometric. Install it via "
        "`pip install torch-geometric` (see https://pytorch-geometric.readthedocs.io)."
    ) from exc

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

TOKEN_PATTERN = re.compile(r"[\w-]+", re.UNICODE)
ADJUVANT_CLASS_ORDER: Sequence[str] = (
    "alum",
    "emulsion",
    "saponin",
    "microbial",
    "cytokine",
    "vectorized_liposome",
    "other",
)

DEFAULT_TEXT_ENCODER_MAX_LENGTHS: Mapping[str, int] = {
    "vaccine": 64,
    "disease": 64,
    "platform": 32,
    "adjuvant": 96,
}


def set_global_seed(seed: int) -> None:
    """Seed ``random`` and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hashed_text_features(texts: Sequence[str], dim: int) -> Tensor:
    """Return L2-normalised hashed bag-of-words embeddings for ``texts``."""

    matrix = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        if not text:
            continue
        tokens = TOKEN_PATTERN.findall(text.lower())
        if not tokens:
            continue
        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            bucket = int(digest[:8], 16) % dim
            matrix[row, bucket] += 1.0
        norm = np.linalg.norm(matrix[row])
        if norm > 0:
            matrix[row] /= norm
    return torch.from_numpy(matrix)


def encode_with_transformer(
    texts: Sequence[str],
    tokenizer: "PreTrainedTokenizerBase",
    model: "PreTrainedModel",
    *,
    pooling: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    normalize: bool = True,
) -> Tensor:
    """Encode ``texts`` with a Hugging Face transformer model."""

    if not texts:
        hidden = getattr(model.config, "hidden_size")
        return torch.empty((0, hidden), dtype=torch.float32)

    outputs: List[Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            model_output = model(**encoded)
            last_hidden = getattr(model_output, "last_hidden_state", None)
            if last_hidden is None:
                last_hidden = model_output[0]  # type: ignore[index]
            if pooling == "cls":
                pooled = last_hidden[:, 0, :]
            else:
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=1)
        outputs.append(pooled.detach().cpu().to(torch.float32))

    return torch.cat(outputs, dim=0)


def _safe_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _aggregate_series(series: Optional[pd.Series]) -> str:
    """Aggregate a pandas Series of strings into a semi-colon list."""

    if series is None:
        return ""
    cleaned = series.dropna()
    if cleaned.empty:
        return ""
    return "; ".join(str(value) for value in cleaned.tolist())


def canonical_adjuvant_class(
    vo_parent: Optional[str],
    display_name: Optional[str],
    definition: Optional[str],
) -> str:
    """Collapse ontology lineage into reviewer-friendly adjuvant classes."""

    parts = [_safe_text(vo_parent), _safe_text(display_name), _safe_text(definition)]
    text = " ".join(part for part in parts if part).lower()
    if any(keyword in text for keyword in ("aluminum", "alum", "alhydrogel")):
        return "alum"
    if any(
        keyword in text
        for keyword in (
            "emulsion",
            "oil-in-water",
            "mf59",
            "as03",
            "freund",
            "squalene",
        )
    ):
        return "emulsion"
    if any(keyword in text for keyword in ("saponin", "qs-21", "as01", "matrix-m")):
        return "saponin"
    if any(
        keyword in text
        for keyword in (
            "microbial",
            "mpla",
            "tlr",
            "lipopolysaccharide",
            "cpg",
            "flagellin",
            "monophosphoryl",
            "pertussis",
        )
    ):
        return "microbial"
    if any(keyword in text for keyword in ("interleukin", "cytokine", "gm-csf", "il-")):
        return "cytokine"
    if any(
        keyword in text
        for keyword in (
            "liposome",
            "lipid",
            "nanoparticle",
            "virosome",
            "vector",
        )
    ):
        return "vectorized_liposome"
    return "other"


class PyGHeteroEncoder(nn.Module):
    """Stack message-passing layers using PyTorch Geometric primitives."""

    def __init__(
        self,
        node_feat_dims: Mapping[str, int],
        metadata: Tuple[Sequence[str], Sequence[Tuple[str, str, str]]],
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        appnp_steps: int,
        appnp_alpha: float,
        appnp_dropout: float,
    ) -> None:
        super().__init__()
        node_types, edge_types = metadata
        self.dropout = dropout
        self.input_projections = nn.ModuleDict(
            {
                node_type: nn.Linear(feat_dim, hidden_dim)
                for node_type, feat_dim in node_feat_dims.items()
            }
        )
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            hetero_conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_dim)
                    for edge_type in edge_types
                },
                aggr="mean",
            )
            self.convs.append(hetero_conv)
        self.node_types = list(node_types)
        self.appnp = (
            APPNP(K=appnp_steps, alpha=appnp_alpha, dropout=appnp_dropout)
            if appnp_steps > 0
            else None
        )

    def forward(self, data: HeteroData) -> Dict[str, Tensor]:
        x_dict = {
            node_type: F.relu(self.input_projections[node_type](data[node_type].x))
            for node_type in self.node_types
        }
        edge_index_dict = data.edge_index_dict
        for conv in self.convs:
            updated = conv(x_dict, edge_index_dict)
            new_x: Dict[str, Tensor] = {}
            for node_type in self.node_types:
                residual = x_dict[node_type]
                message = updated.get(node_type)
                if message is None:
                    message = torch.zeros_like(residual)
                activated = F.relu(residual + message)
                dropped = F.dropout(activated, p=self.dropout, training=self.training)
                new_x[node_type] = dropped
            x_dict = new_x
        if self.appnp is not None:
            homogeneous = data.to_homogeneous()
            node_type_tensor = homogeneous.node_type
            sample = next(iter(x_dict.values()))
            stacked = torch.zeros(
                (int(node_type_tensor.numel()), sample.size(1)),
                dtype=sample.dtype,
                device=sample.device,
            )
            type_indices = {
                node_type: idx for idx, node_type in enumerate(data.node_types)
            }
            index_cache: Dict[str, Tensor] = {}
            for node_type in self.node_types:
                features = x_dict[node_type]
                type_idx = type_indices[node_type]
                positions = (node_type_tensor == type_idx).nonzero(as_tuple=False).view(-1)
                positions = positions.to(features.device)
                if positions.numel() != features.size(0):
                    raise AssertionError(
                        "APPNP mapping mismatch: found "
                        f"{positions.numel()} positions for node type {node_type} "
                        f"but {features.size(0)} feature rows"
                    )
                stacked[positions] = features
                index_cache[node_type] = positions

            if index_cache:
                covered = torch.cat(list(index_cache.values()))
                covered_sorted = torch.sort(covered)[0]
                expected = torch.arange(
                    node_type_tensor.numel(), device=covered.device, dtype=covered.dtype
                )
                if not torch.equal(covered_sorted, expected):
                    raise AssertionError(
                        "APPNP mapping did not cover every homogeneous node exactly once"
                    )

            smoothed = self.appnp(stacked, homogeneous.edge_index.to(stacked.device))
            for node_type, positions in index_cache.items():
                x_dict[node_type] = smoothed[positions]
        return x_dict


class ListNetRankingHead(nn.Module):
    """ListNet objective over candidate adjuvants."""

    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        vaccine_indices: Tensor,
        candidate_indices: Tensor,
        relevance: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        vaccine_repr = embeddings["vaccine"][vaccine_indices]
        adjuvant_repr = embeddings["adjuvant"][candidate_indices]
        scores = torch.einsum("bd,bkd->bk", vaccine_repr, adjuvant_repr)

        positive_mask = (relevance > 0).float()
        positive_mass = positive_mask.sum(dim=1, keepdim=True).clamp_min(1e-9)
        target_distribution = positive_mask / positive_mass
        log_probs = F.log_softmax(scores, dim=1)
        loss = -(target_distribution * log_probs).sum(dim=1).mean()
        return loss, scores


class LinkPredictionHead(nn.Module):
    """Typed negative-sampling link prediction auxiliary head."""

    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        positive_edges: Tensor,
        negative_samples: Tensor,
    ) -> Tensor:
        vaccine_repr = embeddings["vaccine"]
        adjuvant_repr = embeddings["adjuvant"]

        pos_v = vaccine_repr[positive_edges[:, 0]]
        pos_a = adjuvant_repr[positive_edges[:, 1]]
        pos_scores = torch.sum(pos_v * pos_a, dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )

        neg_v = pos_v.unsqueeze(1)
        neg_a = adjuvant_repr[negative_samples]
        neg_scores = torch.sum(neg_v * neg_a, dim=-1)
        neg_targets = torch.zeros_like(neg_scores)
        bce = F.binary_cross_entropy_with_logits(neg_scores, neg_targets, reduction="none")
        weights = torch.softmax(neg_scores * self.temperature, dim=1)
        neg_loss = torch.sum(weights * bce, dim=1).mean()
        return pos_loss + neg_loss


class VaccineAdjuvantListDataset(torch.utils.data.Dataset):
    """Listwise ranking dataset keyed by vaccine nodes."""

    def __init__(
        self,
        vaccine_indices: Sequence[int],
        positives: Mapping[int, Sequence[int]],
        candidate_pool: Mapping[int, Sequence[int]],
        list_size: int,
        seed: int,
    ) -> None:
        self.vaccine_indices = list(vaccine_indices)
        self.positives = {idx: list(values) for idx, values in positives.items()}
        self.candidate_pool = {
            idx: list(values) for idx, values in candidate_pool.items()
        }
        self.list_size = list_size
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.vaccine_indices)

    def __getitem__(self, item: int) -> Tuple[int, Tensor, Tensor]:
        vaccine_idx = self.vaccine_indices[item]
        positives = self.positives.get(vaccine_idx, [])
        if not positives:
            raise ValueError(f"Vaccine index {vaccine_idx} has no positive adjuvants")

        desired_size = max(self.list_size, len(positives))
        pool = self.candidate_pool.get(vaccine_idx, [])
        if not pool:
            negatives_needed = 0
            desired_size = len(positives)
        else:
            negatives_needed = max(0, desired_size - len(positives))
        negatives: List[int] = []
        if negatives_needed:
            if not pool:
                raise ValueError(f"No negative candidates available for vaccine {vaccine_idx}")
            if len(pool) >= negatives_needed:
                negatives = self.rng.sample(pool, negatives_needed)
            else:
                negatives = [self.rng.choice(pool) for _ in range(negatives_needed)]

        items = list(positives) + negatives
        labels = [1.0] * len(positives) + [0.0] * len(negatives)
        permutation = list(range(len(items)))
        self.rng.shuffle(permutation)
        shuffled_candidates = torch.tensor(
            [items[i] for i in permutation], dtype=torch.long
        )
        shuffled_labels = torch.tensor(
            [labels[i] for i in permutation], dtype=torch.float32
        )
        return vaccine_idx, shuffled_candidates, shuffled_labels


def list_collate_fn(batch: Sequence[Tuple[int, Tensor, Tensor]]) -> Tuple[Tensor, Tensor, Tensor]:
    vaccine_indices, candidates, labels = zip(*batch)
    return (
        torch.tensor(vaccine_indices, dtype=torch.long),
        torch.stack(candidates, dim=0),
        torch.stack(labels, dim=0),
    )


class LinkPredictionSampler:
    """Sample positive triples with typed negative adjuvant tails."""

    def __init__(
        self,
        positive_edges: Sequence[Tuple[int, int]],
        candidate_pool: Mapping[int, Sequence[int]],
        negatives_per_positive: int,
        seed: int,
    ) -> None:
        if not positive_edges:
            raise ValueError("LinkPredictionSampler requires at least one positive edge")
        self.positive_edges = list(positive_edges)
        self.candidate_pool = {
            head: list(tails) for head, tails in candidate_pool.items()
        }
        self.negatives_per_positive = negatives_per_positive
        self.rng = random.Random(seed)

    def sample(self, batch_size: int) -> Tuple[Tensor, Tensor]:
        if len(self.positive_edges) >= batch_size:
            batch = self.rng.sample(self.positive_edges, batch_size)
        else:
            batch = [self.rng.choice(self.positive_edges) for _ in range(batch_size)]

        negative_rows: List[List[int]] = []
        for vaccine_idx, _ in batch:
            pool = self.candidate_pool.get(vaccine_idx, [])
            if not pool:
                raise ValueError(
                    f"No negatives available for vaccine index {vaccine_idx} in link sampler"
                )
            if len(pool) >= self.negatives_per_positive:
                negatives = self.rng.sample(pool, self.negatives_per_positive)
            else:
                negatives = [self.rng.choice(pool) for _ in range(self.negatives_per_positive)]
            negative_rows.append(negatives)

        positive_tensor = torch.tensor(batch, dtype=torch.long)
        negative_tensor = torch.tensor(negative_rows, dtype=torch.long)
        return positive_tensor, negative_tensor


def build_vaccine_records(df: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    records: Dict[int, Dict[str, object]] = {}
    grouped = df.groupby("vaccine_id", dropna=False)
    for vaccine_id, group in grouped:
        if pd.isna(vaccine_id):
            continue
        vaccine_id = int(vaccine_id)
        diseases = sorted({str(value) for value in group["disease_key"].dropna()})
        platforms = sorted({str(value) for value in group["platform_group"].dropna()})
        adjuvants = sorted({str(value) for value in group["adjuvant_vo_id"].dropna()})
        adjuvant_classes = sorted({str(value) for value in group["adjuvant_class"].dropna()})
        records[vaccine_id] = {
            "vaccine_id": vaccine_id,
            "sample_count": int(len(group)),
            "diseases": diseases,
            "platforms": platforms,
            "adjuvants": adjuvants,
            "adjuvant_classes": adjuvant_classes,
        }
    return records


def create_transductive_split(
    records: Mapping[int, Mapping[str, object]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    vaccine_ids = sorted(records.keys())
    rng = random.Random(seed)
    rng.shuffle(vaccine_ids)
    total = len(vaccine_ids)
    test_size = max(1, int(round(total * test_fraction)))
    val_size = max(1, int(round(total * val_fraction)))
    test_size = min(test_size, total - 1) if total > 1 else 0
    val_size = min(val_size, total - test_size - 1) if total > 2 else 0
    if total <= 2:
        val_size = max(0, total - test_size - 1)
    train_size = total - test_size - val_size
    if train_size <= 0:
        deficit = 1 - train_size
        if val_size > 1:
            adjust = min(deficit, val_size - 1)
            val_size -= adjust
            deficit -= adjust
        if deficit > 0 and test_size > 1:
            adjust = min(deficit, test_size - 1)
            test_size -= adjust
            deficit -= adjust
        train_size = total - test_size - val_size
    if train_size <= 0:
        raise ValueError("Unable to allocate at least one vaccine to the training split")
    splits = {
        "test": vaccine_ids[:test_size],
        "val": vaccine_ids[test_size : test_size + val_size],
        "train": vaccine_ids[test_size + val_size : test_size + val_size + train_size],
    }
    manifests: Dict[str, List[Dict[str, object]]] = {}
    for split_name, ids in splits.items():
        manifests[split_name] = [dict(records[v]) for v in ids]
    return manifests


def create_inductive_split(
    records: Mapping[int, Mapping[str, object]],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> Dict[str, List[Dict[str, object]]]:
    disease_to_vaccines: Dict[str, List[int]] = {}
    for vaccine_id, record in records.items():
        diseases = record.get("diseases") or ["unspecified_disease"]
        for disease in diseases:
            disease_to_vaccines.setdefault(disease, []).append(vaccine_id)

    diseases = [d for d in disease_to_vaccines if d != "unspecified_disease"]
    rng = random.Random(seed)
    rng.shuffle(diseases)
    total_diseases = len(diseases)
    test_size = max(1, int(round(total_diseases * test_fraction))) if total_diseases else 0
    val_size = max(1, int(round(total_diseases * val_fraction))) if total_diseases else 0
    if test_size + val_size > total_diseases:
        val_size = max(0, total_diseases - test_size)

    test_diseases = set(diseases[:test_size])
    val_diseases = set(diseases[test_size : test_size + val_size])

    assignment: Dict[int, str] = {}
    for vaccine_id, record in records.items():
        diseases = set(record.get("diseases") or [])
        if diseases & test_diseases:
            assignment[vaccine_id] = "test"
        elif diseases & val_diseases:
            assignment[vaccine_id] = "val"
        else:
            assignment[vaccine_id] = "train"

    manifests: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    for vaccine_id, split_name in assignment.items():
        manifests[split_name].append(dict(records[vaccine_id]))
    return manifests


def write_manifests(manifests: Mapping[str, Sequence[Mapping[str, object]]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, entries in manifests.items():
        path = output_dir / f"{split_name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for entry in entries:
                json.dump(entry, handle)
                handle.write("\n")


def ndcg_at_k(sorted_items: Sequence[int], positives: Sequence[int], k: int) -> float:
    positives_set = set(positives)
    dcg = 0.0
    for rank, item in enumerate(sorted_items[:k]):
        if item in positives_set:
            dcg += 1.0 / math.log2(rank + 2)
    ideal_hits = min(len(positives_set), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def recall_at_k(sorted_items: Sequence[int], positives: Sequence[int], k: int) -> float:
    positives_set = set(positives)
    if not positives_set:
        return 0.0
    hits = sum(1 for item in sorted_items[:k] if item in positives_set)
    return hits / len(positives_set)


def evaluate_ranking(
    embeddings: Mapping[str, Tensor],
    vaccine_indices: Sequence[int],
    positives: Mapping[int, Sequence[int]],
    candidate_ids: Sequence[int],
    ks: Sequence[int],
) -> Dict[str, float]:
    adjuvant_repr = embeddings["adjuvant"]
    vaccine_repr = embeddings["vaccine"]
    candidate_tensor = torch.tensor(candidate_ids, dtype=torch.long, device=adjuvant_repr.device)
    candidate_vectors = adjuvant_repr[candidate_tensor]

    metrics = {f"ndcg@{k}": [] for k in ks}
    metrics.update({f"recall@{k}": [] for k in ks})

    for vaccine_idx in vaccine_indices:
        if vaccine_idx not in positives:
            continue
        vaccine_vec = vaccine_repr[vaccine_idx]
        scores = torch.mv(candidate_vectors, vaccine_vec)
        order = torch.argsort(scores, descending=True)
        ranked_candidates = candidate_tensor[order].tolist()
        positive_list = list(positives[vaccine_idx])
        for k in ks:
            metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked_candidates, positive_list, k))
            metrics[f"recall@{k}"].append(recall_at_k(ranked_candidates, positive_list, k))

    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in metrics.items()
    }


def evaluate_link_prediction(
    embeddings: Mapping[str, Tensor],
    edges: Sequence[Tuple[int, int]],
    positives_lookup: Mapping[int, Sequence[int]],
    candidate_ids: Sequence[int],
    ks: Sequence[int],
) -> Dict[str, float]:
    vaccine_repr = embeddings["vaccine"]
    adjuvant_repr = embeddings["adjuvant"]
    results: Dict[str, List[float]] = {"mrr": []}
    results.update({f"hits@{k}": [] for k in ks})

    for vaccine_idx, adjuvant_idx in edges:
        vaccine_vec = vaccine_repr[vaccine_idx]
        scores = torch.mv(adjuvant_repr, vaccine_vec)
        mask = torch.zeros_like(scores, dtype=torch.bool)
        for positive_idx in positives_lookup.get(vaccine_idx, []):
            if positive_idx == adjuvant_idx:
                continue
            mask[positive_idx] = True
        scores = scores.masked_fill(mask, -1e9)
        rank = int(torch.sum((scores > scores[adjuvant_idx]).to(torch.int32)).item() + 1)
        results["mrr"].append(1.0 / rank)
        for k in ks:
            results[f"hits@{k}"].append(1.0 if rank <= k else 0.0)

    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in results.items()
    }


def build_graph(
    df: pd.DataFrame,
    feature_dim: int,
    *,
    text_encoder: Optional[Tuple["PreTrainedTokenizerBase", "PreTrainedModel"]] = None,
    text_encoder_config: Optional[Mapping[str, object]] = None,
) -> Tuple[
    HeteroData,
    Dict[str, Dict[object, int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    List[int],
]:
    df = df.copy()
    df = df.dropna(subset=["vaccine_id", "adjuvant_vo_id"])
    df["vaccine_id"] = df["vaccine_id"].astype(int)
    df["adjuvant_vo_id"] = df["adjuvant_vo_id"].astype(str)
    df["disease_key"] = df["disease_name"].fillna(df["pathogen_name"])
    df["disease_key"] = df["disease_key"].fillna("unspecified_disease")
    df["platform_group"] = df["platform_group"].fillna("unspecified")

    vaccine_ids = sorted({int(v) for v in df["vaccine_id"].unique()})
    disease_names = sorted({str(d) for d in df["disease_key"].unique()})
    platform_groups = sorted({str(p) for p in df["platform_group"].unique()})
    adjuvant_ids = sorted({str(a) for a in df["adjuvant_vo_id"].unique()})

    mappings: Dict[str, Dict[object, int]] = {
        "vaccine": {vid: idx for idx, vid in enumerate(vaccine_ids)},
        "disease": {name: idx for idx, name in enumerate(disease_names)},
        "platform": {name: idx for idx, name in enumerate(platform_groups)},
        "adjuvant": {aid: idx for idx, aid in enumerate(adjuvant_ids)},
    }

    vaccine_texts: List[str] = []
    vaccine_grouped = df.groupby("vaccine_id")
    for vaccine_id in vaccine_ids:
        group = vaccine_grouped.get_group(vaccine_id)
        parts = [
            _safe_text(group["vaccine_name"].iloc[0] if "vaccine_name" in group else ""),
            _aggregate_series(group["detail_antigens"]) if "detail_antigens" in group else "",
            _aggregate_series(group["detail_vectors"]) if "detail_vectors" in group else "",
        ]
        vaccine_texts.append(" ".join(filter(None, parts)))

    disease_texts: List[str] = []
    disease_grouped = df.groupby("disease_key")
    for disease_name in disease_names:
        group = disease_grouped.get_group(disease_name)
        parts = [
            _safe_text(disease_name),
            _aggregate_series(group["pathogen_name"]) if "pathogen_name" in group else "",
        ]
        disease_texts.append(" ".join(filter(None, parts)))

    platform_texts: List[str] = []
    platform_grouped = df.groupby("platform_group")
    for platform_name in platform_groups:
        group = platform_grouped.get_group(platform_name)
        parts = [
            _safe_text(platform_name),
            _aggregate_series(group["platform_type"]) if "platform_type" in group else "",
            _aggregate_series(group["platform_context_source"]) if "platform_context_source" in group else "",
        ]
        platform_texts.append(" ".join(filter(None, parts)))

    adjuvant_texts: List[str] = []
    adjuvant_grouped = df.groupby("adjuvant_vo_id")
    for adjuvant_id in adjuvant_ids:
        group = adjuvant_grouped.get_group(adjuvant_id)
        parts = [
            _safe_text(group["adjuvant_display_name"].iloc[0] if "adjuvant_display_name" in group else ""),
            _safe_text(group["adjuvant_description"].iloc[0] if "adjuvant_description" in group else ""),
            _aggregate_series(group["adjuvant_synonyms"]) if "adjuvant_synonyms" in group else "",
            _aggregate_series(group["adjuvant_roles"]) if "adjuvant_roles" in group else "",
            _aggregate_series(group["adjuvant_immune_profile"]) if "adjuvant_immune_profile" in group else "",
        ]
        adjuvant_texts.append(" ".join(filter(None, parts)))

    edges: Dict[Tuple[str, str, str], List[Tuple[int, int]]] = {
        ("vaccine", "for_disease", "disease"): [],
        ("disease", "rev_for_disease", "vaccine"): [],
        ("vaccine", "uses_platform", "platform"): [],
        ("platform", "rev_uses_platform", "vaccine"): [],
        ("vaccine", "contains_adjuvant", "adjuvant"): [],
        ("adjuvant", "rev_contains_adjuvant", "vaccine"): [],
    }

    positives_lookup_raw: Dict[int, set] = {}
    candidate_pool: Dict[int, List[int]] = {}

    for row in df.itertuples(index=False):
        v_idx = mappings["vaccine"][int(row.vaccine_id)]
        d_idx = mappings["disease"][row.disease_key]
        p_idx = mappings["platform"][row.platform_group]
        a_idx = mappings["adjuvant"][row.adjuvant_vo_id]

        edges[("vaccine", "for_disease", "disease")].append((v_idx, d_idx))
        edges[("disease", "rev_for_disease", "vaccine")].append((d_idx, v_idx))
        edges[("vaccine", "uses_platform", "platform")].append((v_idx, p_idx))
        edges[("platform", "rev_uses_platform", "vaccine")].append((p_idx, v_idx))
        edges[("vaccine", "contains_adjuvant", "adjuvant")].append((v_idx, a_idx))
        edges[("adjuvant", "rev_contains_adjuvant", "vaccine")].append((a_idx, v_idx))

        positives_lookup_raw.setdefault(v_idx, set()).add(a_idx)

    all_adjuvant_indices = list(range(len(adjuvant_ids)))
    positives_lookup: Dict[int, List[int]] = {}
    for vaccine_idx, positives in positives_lookup_raw.items():
        positives_list = sorted(positives)
        negatives = [idx for idx in all_adjuvant_indices if idx not in positives_list]
        candidate_pool[vaccine_idx] = negatives
        positives_lookup[vaccine_idx] = positives_list

    graph = HeteroData()

    if text_encoder is None:
        graph["vaccine"].x = hashed_text_features(vaccine_texts, feature_dim)
        graph["disease"].x = hashed_text_features(disease_texts, feature_dim)
        graph["platform"].x = hashed_text_features(platform_texts, feature_dim)
        graph["adjuvant"].x = hashed_text_features(adjuvant_texts, feature_dim)
    else:
        tokenizer, model = text_encoder
        config = dict(text_encoder_config or {})
        pooling = str(config.get("pooling", "mean"))
        batch_size = int(config.get("batch_size", 128))
        max_lengths = dict(config.get("max_lengths", DEFAULT_TEXT_ENCODER_MAX_LENGTHS))
        default_length = int(config.get("default_max_length", 64))
        device = torch.device(config.get("device", "cpu"))
        normalize = bool(config.get("normalize", True))

        def _length(node_type: str) -> int:
            return int(max_lengths.get(node_type, default_length))

        graph["vaccine"].x = encode_with_transformer(
            vaccine_texts,
            tokenizer,
            model,
            pooling=pooling,
            batch_size=batch_size,
            max_length=_length("vaccine"),
            device=device,
            normalize=normalize,
        )
        graph["disease"].x = encode_with_transformer(
            disease_texts,
            tokenizer,
            model,
            pooling=pooling,
            batch_size=batch_size,
            max_length=_length("disease"),
            device=device,
            normalize=normalize,
        )
        graph["platform"].x = encode_with_transformer(
            platform_texts,
            tokenizer,
            model,
            pooling=pooling,
            batch_size=batch_size,
            max_length=_length("platform"),
            device=device,
            normalize=normalize,
        )
        graph["adjuvant"].x = encode_with_transformer(
            adjuvant_texts,
            tokenizer,
            model,
            pooling=pooling,
            batch_size=batch_size,
            max_length=_length("adjuvant"),
            device=device,
            normalize=normalize,
        )

    for edge_type, edge_list in edges.items():
        if edge_list:
            unique_edges = list(dict.fromkeys(edge_list))
            edge_tensor = torch.tensor(unique_edges, dtype=torch.long).t().contiguous()
        else:
            edge_tensor = torch.empty((2, 0), dtype=torch.long)
        graph[edge_type].edge_index = edge_tensor

    return (
        graph,
        mappings,
        positives_lookup,
        candidate_pool,
        all_adjuvant_indices,
    )


def attach_adjuvant_classes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adjuvant_class"] = df.apply(
        lambda row: canonical_adjuvant_class(
            row.get("vo_parent"),
            row.get("adjuvant_display_name"),
            row.get("adjuvant_description"),
        ),
        axis=1,
    )
    return df


def restrict_label_edges_for_training(
    graph: HeteroData, train_vaccine_indices: Sequence[int]
) -> HeteroData:
    """Filter ``contains_adjuvant`` edges to vaccines in the training split."""

    filtered = graph.clone()
    edge_index = filtered["vaccine", "contains_adjuvant", "adjuvant"].edge_index
    if edge_index.numel() == 0:
        return filtered
    train_tensor = torch.tensor(
        sorted(set(int(idx) for idx in train_vaccine_indices)),
        dtype=torch.long,
        device=edge_index.device,
    )
    if train_tensor.numel() == 0:
        forward_empty = edge_index.new_empty((2, 0))
        reverse_empty = edge_index.new_empty((2, 0))
        filtered["vaccine", "contains_adjuvant", "adjuvant"].edge_index = forward_empty
        filtered["adjuvant", "rev_contains_adjuvant", "vaccine"].edge_index = reverse_empty
        return filtered
    keep_mask = torch.isin(edge_index[0], train_tensor)
    filtered_edge_index = edge_index[:, keep_mask]
    filtered["vaccine", "contains_adjuvant", "adjuvant"].edge_index = filtered_edge_index
    filtered["adjuvant", "rev_contains_adjuvant", "vaccine"].edge_index = (
        filtered_edge_index.flip(0)
    )
    return filtered


def restrict_context_edges_for_training(
    graph: HeteroData,
    train_vaccine_indices: Sequence[int],
    train_disease_indices: Optional[Sequence[int]] = None,
) -> HeteroData:
    """Drop context edges that touch held-out vaccines or diseases."""

    filtered = graph.clone()
    vaccine_mask = torch.zeros(filtered["vaccine"].num_nodes, dtype=torch.bool)
    if train_vaccine_indices:
        vaccine_mask[
            torch.tensor(
                sorted(set(int(v) for v in train_vaccine_indices)), dtype=torch.long
            )
        ] = True

    disease_mask: Optional[Tensor] = None
    if train_disease_indices is not None:
        disease_mask = torch.zeros(filtered["disease"].num_nodes, dtype=torch.bool)
        if train_disease_indices:
            disease_mask[
                torch.tensor(
                    sorted(set(int(d) for d in train_disease_indices)), dtype=torch.long
                )
            ] = True

    forward_fd = filtered["vaccine", "for_disease", "disease"].edge_index
    if forward_fd.numel() > 0:
        keep = vaccine_mask[forward_fd[0]]
        if disease_mask is not None:
            keep &= disease_mask[forward_fd[1]]
        filtered["vaccine", "for_disease", "disease"].edge_index = forward_fd[:, keep]

    reverse_fd = filtered["disease", "rev_for_disease", "vaccine"].edge_index
    if reverse_fd.numel() > 0:
        keep = vaccine_mask[reverse_fd[1]]
        if disease_mask is not None:
            keep &= disease_mask[reverse_fd[0]]
        filtered["disease", "rev_for_disease", "vaccine"].edge_index = reverse_fd[:, keep]

    forward_platform = filtered["vaccine", "uses_platform", "platform"].edge_index
    if forward_platform.numel() > 0:
        keep = vaccine_mask[forward_platform[0]]
        filtered["vaccine", "uses_platform", "platform"].edge_index = forward_platform[:, keep]

    reverse_platform = filtered["platform", "rev_uses_platform", "vaccine"].edge_index
    if reverse_platform.numel() > 0:
        keep = vaccine_mask[reverse_platform[1]]
        filtered["platform", "rev_uses_platform", "vaccine"].edge_index = reverse_platform[:, keep]

    return filtered


def verify_no_context_leakage(
    graph: HeteroData,
    heldout_vaccines: Sequence[int],
    heldout_diseases: Optional[Sequence[int]] = None,
) -> None:
    """Assert that training graph edges do not reference held-out nodes."""

    if heldout_vaccines:
        disallowed_v = torch.tensor(
            sorted(set(int(v) for v in heldout_vaccines)), dtype=torch.long
        )
        disallowed_v = disallowed_v.to(
            graph["vaccine", "contains_adjuvant", "adjuvant"].edge_index.device
        )

        def _count(edge_type: Tuple[str, str, str], row: int) -> int:
            edge_index = graph[edge_type].edge_index
            if edge_index.numel() == 0:
                return 0
            return int(torch.isin(edge_index[row], disallowed_v).sum().item())

        vaccine_leak = sum(
            _count(edge_type, row)
            for edge_type, row in [
                (("vaccine", "contains_adjuvant", "adjuvant"), 0),
                (("adjuvant", "rev_contains_adjuvant", "vaccine"), 1),
                (("vaccine", "for_disease", "disease"), 0),
                (("disease", "rev_for_disease", "vaccine"), 1),
                (("vaccine", "uses_platform", "platform"), 0),
                (("platform", "rev_uses_platform", "vaccine"), 1),
            ]
        )
        if vaccine_leak:
            raise AssertionError(
                f"Training graph still references {vaccine_leak} held-out vaccine nodes"
            )
        print("Verified: no held-out vaccine indices remain in training graph edges")

    if heldout_diseases:
        disallowed_d = torch.tensor(
            sorted(set(int(d) for d in heldout_diseases)), dtype=torch.long
        )
        disallowed_d = disallowed_d.to(
            graph["vaccine", "for_disease", "disease"].edge_index.device
        )

        def _count_disease(edge_type: Tuple[str, str, str], row: int) -> int:
            edge_index = graph[edge_type].edge_index
            if edge_index.numel() == 0:
                return 0
            return int(torch.isin(edge_index[row], disallowed_d).sum().item())

        disease_leak = _count_disease(("vaccine", "for_disease", "disease"), 1) + _count_disease(
            ("disease", "rev_for_disease", "vaccine"), 0
        )
        if disease_leak:
            raise AssertionError(
                f"Training graph still references {disease_leak} held-out disease nodes"
            )
        print("Verified: no held-out disease indices remain in training graph edges")

def train_one_split(
    graph: HeteroData,
    mappings: Mapping[str, Mapping[object, int]],
    positives_lookup: Mapping[int, Sequence[int]],
    candidate_pool: Mapping[int, Sequence[int]],
    candidate_ids: Sequence[int],
    manifests: Mapping[str, Sequence[Mapping[str, object]]],
    args: argparse.Namespace,
    device: torch.device,
    scheme: str,
) -> Dict[str, Dict[str, float]]:
    def _collect_diseases(entries: Sequence[Mapping[str, object]]) -> List[int]:
        indices: Set[int] = set()
        for entry in entries:
            diseases = entry.get("diseases") or []
            for disease in diseases:
                idx = mappings["disease"].get(disease)
                if idx is not None:
                    indices.add(idx)
        return sorted(indices)

    train_vaccines = [
        mappings["vaccine"][entry["vaccine_id"]]
        for entry in manifests["train"]
        if entry["vaccine_id"] in mappings["vaccine"]
    ]
    val_vaccines = [
        mappings["vaccine"][entry["vaccine_id"]]
        for entry in manifests.get("val", [])
        if entry["vaccine_id"] in mappings["vaccine"]
    ]
    test_vaccines = [
        mappings["vaccine"][entry["vaccine_id"]]
        for entry in manifests.get("test", [])
        if entry["vaccine_id"] in mappings["vaccine"]
    ]

    if not train_vaccines:
        raise ValueError("Training split is empty; cannot optimise model")

    train_diseases = _collect_diseases(manifests.get("train", []))
    val_diseases = _collect_diseases(manifests.get("val", []))
    test_diseases = _collect_diseases(manifests.get("test", []))

    train_graph = restrict_label_edges_for_training(graph, train_vaccines)
    train_graph = restrict_context_edges_for_training(
        train_graph,
        train_vaccines,
        train_diseases if scheme == "inductive" else None,
    )

    heldout_vaccines = sorted(set(val_vaccines + test_vaccines))
    heldout_diseases = (
        sorted(set(val_diseases + test_diseases)) if scheme == "inductive" else []
    )
    verify_no_context_leakage(
        train_graph,
        heldout_vaccines,
        heldout_diseases if heldout_diseases else None,
    )

    train_dataset = VaccineAdjuvantListDataset(
        train_vaccines,
        positives_lookup,
        candidate_pool,
        args.list_size,
        args.seed,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=list_collate_fn,
    )

    train_edges = [
        (v_idx, pos)
        for v_idx in train_vaccines
        for pos in positives_lookup.get(v_idx, [])
    ]
    link_sampler = (
        LinkPredictionSampler(train_edges, candidate_pool, args.negatives_per_triple, args.seed)
        if args.lambda_lp > 0 and train_edges
        else None
    )

    node_feat_dims = {
        node_type: features.size(1) for node_type, features in graph.x_dict.items()
    }
    model = PyGHeteroEncoder(
        node_feat_dims,
        graph.metadata(),
        args.hidden_dim,
        args.layers,
        args.dropout,
        args.appnp_steps,
        args.appnp_alpha,
        args.appnp_dropout,
    )
    ranking_head = ListNetRankingHead().to(device)
    link_head = LinkPredictionHead(args.self_adversarial_temperature).to(device)

    param_groups = [
        {"params": model.parameters(), "lr": args.encoder_lr},
        {
            "params": list(ranking_head.parameters()) + list(link_head.parameters()),
            "lr": args.head_lr,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    train_graph_device = train_graph.to(device)
    model = model.to(device)
    results: Dict[str, Dict[str, float]] = {}
    best_val = -float("inf")
    best_state: Optional[Dict[str, Tensor]] = None
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rank = 0.0
        total_lp = 0.0
        for vaccines, candidates, labels in train_loader:
            vaccines = vaccines.to(device)
            candidates = candidates.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = model(train_graph_device)
            rank_loss, _ = ranking_head(embeddings, vaccines, candidates, labels)
            lp_loss = torch.tensor(0.0, device=device)
            if link_sampler is not None:
                pos_edges, neg_samples = link_sampler.sample(len(vaccines))
                pos_edges = pos_edges.to(device)
                neg_samples = neg_samples.to(device)
                lp_loss = link_head(embeddings, pos_edges, neg_samples)
            loss = rank_loss + args.lambda_lp * lp_loss
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()

            total_loss += loss.item()
            total_rank += rank_loss.item()
            total_lp += lp_loss.item() if link_sampler is not None else 0.0

        avg_loss = total_loss / max(1, len(train_loader))
        avg_rank = total_rank / max(1, len(train_loader))
        avg_lp = total_lp / max(1, len(train_loader))

        model.eval()
        with torch.no_grad():
            embeddings = model(train_graph_device)
        val_metrics = (
            evaluate_ranking(embeddings, val_vaccines, positives_lookup, candidate_ids, (5, 10))
            if val_vaccines
            else {"ndcg@5": 0.0, "ndcg@10": 0.0}
        )
        score = val_metrics.get("ndcg@10", 0.0)
        print(
            f"Epoch {epoch:03d} | loss={avg_loss:.4f} rank={avg_rank:.4f} lp={avg_lp:.4f} "
            f"val_ndcg10={score:.4f}"
        )

        if score > best_val:
            best_val = score
            patience_counter = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        checkpoint_dir = args.output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{scheme}_best.pt"
        cpu_state = {key: tensor.cpu() for key, tensor in best_state.items()}
        torch.save(
            {
                "state_dict": cpu_state,
                "metadata": graph.metadata(),
                "mappings": mappings,
                "args": vars(args),
                "split": scheme,
            },
            checkpoint_path,
        )
        print(f"Saved best checkpoint to {checkpoint_path}")

    model.eval()
    with torch.no_grad():
        embeddings = model(train_graph_device)

    train_metrics = evaluate_ranking(embeddings, train_vaccines, positives_lookup, candidate_ids, (5, 10))
    val_metrics = (
        evaluate_ranking(embeddings, val_vaccines, positives_lookup, candidate_ids, (5, 10))
        if val_vaccines
        else {}
    )
    test_metrics = (
        evaluate_ranking(embeddings, test_vaccines, positives_lookup, candidate_ids, (5, 10))
        if test_vaccines
        else {}
    )

    link_metrics: Dict[str, Dict[str, float]] = {}
    if train_edges:
        link_metrics["train"] = evaluate_link_prediction(
            embeddings, train_edges, positives_lookup, candidate_ids, (1, 3, 10)
        )
    if val_vaccines:
        val_edges = [
            (idx, pos)
            for idx in val_vaccines
            for pos in positives_lookup.get(idx, [])
        ]
        link_metrics["val"] = evaluate_link_prediction(
            embeddings, val_edges, positives_lookup, candidate_ids, (1, 3, 10)
        )
    if test_vaccines:
        test_edges = [
            (idx, pos)
            for idx in test_vaccines
            for pos in positives_lookup.get(idx, [])
        ]
        link_metrics["test"] = evaluate_link_prediction(
            embeddings, test_edges, positives_lookup, candidate_ids, (1, 3, 10)
        )

    results["ranking_train"] = train_metrics
    if val_metrics:
        results["ranking_val"] = val_metrics
    if test_metrics:
        results["ranking_test"] = test_metrics
    if link_metrics:
        results["link_prediction"] = link_metrics
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train vaccine→adjuvant ranker")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/training_samples.csv"),
        help="Processed training CSV produced by prepare_training_data.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for split manifests and training artefacts",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=256,
        help="Dimension of hashed text features per node (ignored when using a transformer encoder)",
    )
    parser.add_argument(
        "--text-encoder-checkpoint",
        type=str,
        default=None,
        help="Optional Hugging Face checkpoint (e.g. cambridgeltl/SapBERT-from-PubMedBERT-fulltext-mean-token) for node text",
    )
    parser.add_argument(
        "--text-encoder-pooling",
        choices=["mean", "cls"],
        default="mean",
        help="Pooling strategy when using a transformer text encoder",
    )
    parser.add_argument(
        "--text-encoder-batch-size",
        type=int,
        default=128,
        help="Batch size for transformer inference over node texts",
    )
    parser.add_argument(
        "--text-encoder-max-length",
        type=int,
        default=None,
        help="Override maximum token length for transformer encoding (applies to all node types if set)",
    )
    parser.add_argument(
        "--no-text-encoder-normalize",
        action="store_true",
        help="Disable L2 normalisation of transformer embeddings",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the PyG message passing layers",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=2,
        help="Number of PyG hetero message passing layers",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout applied after each layer")
    parser.add_argument(
        "--appnp-steps",
        type=int,
        default=10,
        help="Propagation steps for APPNP smoothing (0 disables)",
    )
    parser.add_argument(
        "--appnp-alpha",
        type=float,
        default=0.1,
        help="Teleport (restart) probability for APPNP smoothing",
    )
    parser.add_argument(
        "--appnp-dropout",
        type=float,
        default=0.0,
        help="Dropout applied inside APPNP propagation",
    )
    parser.add_argument("--list-size", type=int, default=50, help="Candidate list size for ListNet training")
    parser.add_argument(
        "--negatives-per-triple",
        type=int,
        default=20,
        help="Negative samples per positive triple for link prediction",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Queries per optimisation step")
    parser.add_argument("--epochs", type=int, default=300, help="Maximum number of training epochs")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience on val NDCG@10")
    parser.add_argument("--encoder-lr", type=float, default=2e-3, help="Learning rate for the encoder")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for ranking/link heads")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--clip-grad", type=float, default=1.0, help="Gradient clipping norm (0 disables)")
    parser.add_argument("--lambda-lp", type=float, default=0.2, help="Weight for the link prediction auxiliary loss")
    parser.add_argument(
        "--self-adversarial-temperature",
        type=float,
        default=1.0,
        help="Temperature used in self-adversarial negative weighting",
    )
    parser.add_argument("--val-fraction", type=float, default=0.1, help="Validation fraction for splits")
    parser.add_argument("--test-fraction", type=float, default=0.1, help="Test fraction for splits")
    parser.add_argument(
        "--split-scheme",
        choices=["transductive", "inductive", "both"],
        default="both",
        help="Which split schemes to generate and (optionally) train on",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting and optimisation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device identifier",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only generate split manifests without fitting the model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    if not args.data_path.exists():
        raise FileNotFoundError(f"Training data CSV not found at {args.data_path!s}")

    df = pd.read_csv(args.data_path)
    df = attach_adjuvant_classes(df)
    records = build_vaccine_records(df)

    text_encoder: Optional[Tuple["PreTrainedTokenizerBase", "PreTrainedModel"]] = None
    text_encoder_config: Optional[Mapping[str, object]] = None
    hf_model: Optional["PreTrainedModel"] = None
    if args.text_encoder_checkpoint:
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "Using --text-encoder-checkpoint requires the `transformers` package."
            ) from exc

        print(f"Loading text encoder {args.text_encoder_checkpoint!s}")
        tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_checkpoint)
        hf_model = AutoModel.from_pretrained(args.text_encoder_checkpoint)
        encoder_device = torch.device(args.device)
        hf_model.to(encoder_device)
        hf_model.eval()

        max_lengths = dict(DEFAULT_TEXT_ENCODER_MAX_LENGTHS)
        if args.text_encoder_max_length is not None:
            for key in max_lengths:
                max_lengths[key] = args.text_encoder_max_length

        text_encoder = (tokenizer, hf_model)
        text_encoder_config = {
            "pooling": args.text_encoder_pooling,
            "batch_size": args.text_encoder_batch_size,
            "max_lengths": max_lengths,
            "default_max_length": args.text_encoder_max_length or 64,
            "device": encoder_device,
            "normalize": not args.no_text_encoder_normalize,
        }

    split_dir = args.output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    manifests_collection: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    if args.split_scheme in {"transductive", "both"}:
        transductive_manifests = create_transductive_split(
            records, args.val_fraction, args.test_fraction, args.seed
        )
        manifests_collection["transductive"] = transductive_manifests
        write_manifests(transductive_manifests, split_dir / "transductive")

    if args.split_scheme in {"inductive", "both"}:
        inductive_manifests = create_inductive_split(
            records, args.val_fraction, args.test_fraction, args.seed
        )
        manifests_collection["inductive"] = inductive_manifests
        write_manifests(inductive_manifests, split_dir / "inductive")

    if args.skip_training:
        print("Split manifests generated; skipping training as requested")
        return

    graph, mappings, positives_lookup, candidate_pool, candidate_ids = build_graph(
        df,
        args.feature_dim,
        text_encoder=text_encoder,
        text_encoder_config=text_encoder_config,
    )

    if hf_model is not None:
        hf_model.to("cpu")
        del hf_model

    device = torch.device(args.device)

    results_dir = args.output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for scheme, manifests in manifests_collection.items():
        print(f"=== Training on {scheme} split ===")
        metrics = train_one_split(
            graph,
            mappings,
            positives_lookup,
            candidate_pool,
            candidate_ids,
            manifests,
            args,
            device,
            scheme,
        )
        result_path = results_dir / f"{scheme}.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        print(f"Saved {scheme} metrics to {result_path}")


if __name__ == "__main__":
    main()

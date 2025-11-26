"""
Disease head utilities for dual-head ranking model.

Provides disease batch sampling, disease→adjuvant pair loading,
and helper functions for joint training.
"""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F


def load_disease_positives(
    csv_path: Path,
    mappings: Mapping[str, Mapping[object, int]],
) -> Tuple[Dict[int, List[int]], Dict[Tuple[int, int], int]]:
    """
    Load disease→adjuvant positives and edge weights from CSV.
    
    Args:
        csv_path: Path to disease_adjuvant_pairs.csv
        mappings: Node type → (id → index) mappings
    
    Returns:
        disease_positives_lookup: {disease_idx: [adj_idx1, adj_idx2, ...]}
        disease_edge_weights: {(disease_idx, adj_idx): edge_weight}
    """
    disease_positives_lookup: Dict[int, List[int]] = defaultdict(list)
    disease_edge_weights: Dict[Tuple[int, int], int] = {}
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Disease-adjuvant pairs not found at {csv_path}. "
            "Run `python src/build_da_pairs.py` first."
        )
    
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        disease_key = row["disease_key"]
        adjuvant_vo_id = row["adjuvant_vo_id"]
        edge_weight = int(row["edge_weight"])
        
        d_idx = mappings["disease"].get(disease_key)
        a_idx = mappings["adjuvant"].get(adjuvant_vo_id)
        
        if d_idx is not None and a_idx is not None:
            disease_positives_lookup[d_idx].append(a_idx)
            disease_edge_weights[(d_idx, a_idx)] = edge_weight
    
    return dict(disease_positives_lookup), disease_edge_weights


def sample_disease_batch(
    all_disease_indices: Sequence[int],
    disease_positives_lookup: Mapping[int, Sequence[int]],
    all_adjuvant_indices: Sequence[int],
    batch_size: int = 32,
    max_positives: int = 10,
    num_negatives: int = 40,
    seed: Optional[int] = None,
) -> List[Dict[str, Tensor]]:
    """
    Sample a batch of disease queries with their positive and negative adjuvants.
    
    Design: Uniform random sampling with replacement (simplest that works).
    
    Args:
        all_disease_indices: List of all disease indices
        disease_positives_lookup: {disease_idx: [positive_adjuvant_indices]}
        all_adjuvant_indices: List of all adjuvant indices  
        batch_size: Number of diseases to sample
        max_positives: Max positives per disease (sample if more)
        num_negatives: Number of negative adjuvants to sample per disease
        seed: Random seed (optional)
    
    Returns:
        List of dicts with keys: 'disease_idx', 'pos_indices', 'neg_indices'
    """
    if seed is not None:
        random.seed(seed)
    
    # Sample diseases uniformly with replacement
    sampled_diseases = random.choices(all_disease_indices, k=batch_size)
    
    batch = []
    all_adj_set = set(all_adjuvant_indices)
    
    for d_idx in sampled_diseases:
        pos_adjs = list(disease_positives_lookup.get(d_idx, []))
        
        # Skip if no positives (edge case)
        if not pos_adjs:
            continue
        
        # Sample positives if too many
        if len(pos_adjs) > max_positives:
            pos_adjs = random.sample(pos_adjs, max_positives)
        
        # Sample typed negatives (adjuvants not in positive set)
        pos_set = set(pos_adjs)
        neg_candidates = list(all_adj_set - pos_set)
        
        if len(neg_candidates) < num_negatives:
            # Not enough negatives (rare case), take what we have
            neg_adjs = neg_candidates
        else:
            neg_adjs = random.sample(neg_candidates, num_negatives)
        
        batch.append({
            'disease_idx': d_idx,
            'pos_indices': pos_adjs,
            'neg_indices': neg_adjs,
        })
    
    return batch


def sample_negatives_mixed(
    positive_adjs: Sequence[int],
    all_adjuvants: Sequence[int],
    vo_class_lookup: Mapping[int, str],
    n_hard: int = 10,
    n_easy: int = 30,
) -> List[int]:
    """
    Sample hard (same VO class) + easy (random typed) negatives.
    
    Hard negatives force fine-grained discrimination within adjuvant families.
    Easy negatives maintain generalization.
    
    Args:
        positive_adjs: Positive adjuvant indices for this query
        all_adjuvants: All adjuvant indices in the dataset
        vo_class_lookup: {adjuvant_idx: vo_parent_class}
        n_hard: Number of hard negatives (same VO class)
        n_easy: Number of easy negatives (random)
    
    Returns:
        List of negative adjuvant indices (hard + easy)
    """
    pos_set = set(positive_adjs)
    
    # Get VO classes of positive adjuvants
    positive_classes = {vo_class_lookup.get(adj, 'unknown') for adj in positive_adjs}
    
    # Hard negatives: same class but not in positives
    hard_candidates = [
        adj for adj in all_adjuvants
        if vo_class_lookup.get(adj, 'unknown') in positive_classes and adj not in pos_set
    ]
    
    if hard_candidates:
        hard_negs = random.sample(hard_candidates, min(n_hard, len(hard_candidates)))
    else:
        hard_negs = []
    
    # Easy negatives: random typed (not in positives or hard_negs)
    hard_set = set(hard_negs)
    easy_candidates = [
        adj for adj in all_adjuvants
        if adj not in pos_set and adj not in hard_set
    ]
    
    remaining = n_hard + n_easy - len(hard_negs)
    if easy_candidates and remaining > 0:
        easy_negs = random.sample(easy_candidates, min(remaining, len(easy_candidates)))
    else:
        easy_negs = []
    
    return hard_negs + easy_negs


def build_vo_class_lookup(
    adjuvant_metadata_csv: Path,
    mappings: Mapping[str, Mapping[object, int]],
) -> Dict[int, str]:
    """
    Build VO class lookup for hard negative sampling.
    
    Extracts parent class from adjuvant_metadata_enriched.csv.
    Falls back to extracting from VO_ID prefix if no explicit parent.
    
    Args:
        adjuvant_metadata_csv: Path to adjuvant_metadata_enriched.csv
        mappings: Node type → (id → index) mappings
    
    Returns:
        {adjuvant_idx: vo_parent_class}
    """
    vo_class_lookup: Dict[int, str] = {}
    
    if not adjuvant_metadata_csv.exists():
        print(f"Warning: {adjuvant_metadata_csv} not found, using VO_ID prefix fallback")
        # Fallback: use VO_ID prefix as class
        inv_adj_map = {idx: vo_id for vo_id, idx in mappings["adjuvant"].items()}
        for adj_idx, vo_id in inv_adj_map.items():
            # Extract parent from VO:0001234 → VO:000123X
            if isinstance(vo_id, str) and vo_id.startswith("VO:"):
                prefix = vo_id[:9]  # VO:00012XX
                vo_class_lookup[adj_idx] = prefix
            else:
                vo_class_lookup[adj_idx] = "unknown"
        return vo_class_lookup
    
    # Read from CSV
    df = pd.read_csv(adjuvant_metadata_csv)
    
    for _, row in df.iterrows():
        vo_id = row.get("adjuvant_vo_id") or row.get("vo_term_id")
        
        if pd.isna(vo_id):
            continue
        
        adj_idx = mappings["adjuvant"].get(str(vo_id))
        if adj_idx is None:
            continue
        
        # Try to get parent class (multiple potential columns)
        parent_class = None
        for col in ["vo_parent", "vo_proposed_parent", "adjuvant_label"]:
            if col in df.columns and not pd.isna(row.get(col)):
                parent_class = str(row[col])
                break
        
        # Fallback to VO_ID prefix
        if parent_class is None:
            if isinstance(vo_id, str) and vo_id.startswith("VO:"):
                parent_class = vo_id[:9]
            else:
                parent_class = "unknown"
        
        vo_class_lookup[adj_idx] = parent_class
    
    return vo_class_lookup


def _approx_ndcg_loss(scores: Tensor, gains: Tensor, tau: float = 1.0) -> Tensor:
    """Return ``1 - ApproxNDCG`` for the provided ``scores`` and ``gains``."""

    if gains.sum() <= 0:
        return scores.new_zeros(())

    # Pairwise sigmoid approximation of the rank ("soft" rank)
    # rank_i ≈ 1 + Σ_j sigmoid((s_j - s_i) / τ)
    diff = (scores.unsqueeze(0) - scores.unsqueeze(1)) / tau
    pairwise = torch.sigmoid(diff)
    mask = torch.ones_like(pairwise) - torch.eye(
        pairwise.size(0), device=pairwise.device, dtype=pairwise.dtype
    )
    pairwise = pairwise * mask
    approx_rank = 1.0 + pairwise.sum(dim=1)

    discounts = torch.log2(approx_rank + 1.0).reciprocal()
    approx_dcg = (gains * discounts).sum()

    # Ideal DCG computed on sorted gains for the same slate length
    ideal_gains = torch.sort(gains, descending=True).values
    positions = torch.arange(
        2, gains.numel() + 2, device=gains.device, dtype=gains.dtype
    )
    ideal_discounts = torch.log2(positions).reciprocal()
    ideal_dcg = (ideal_gains * ideal_discounts).sum()

    if ideal_dcg.item() <= 0:
        return scores.new_zeros(())

    ndcg = approx_dcg / ideal_dcg.clamp_min(1e-9)
    return (1.0 - ndcg).clamp_min(0.0)


def disease_ranking_losses(
    embeddings: Mapping[str, Tensor],
    disease_batch: Sequence[Dict[str, object]],
    dual_ranker: torch.nn.Module,
    device: torch.device,
    *,
    ndcg_tau: float = 1.0,
    ndcg_topk: Optional[int] = None,
    adjuvant_mechanism: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Compute ApproxNDCG and ListNet losses for a disease batch."""

    if not disease_batch:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    ndcg_losses: List[Tensor] = []
    listnet_losses: List[Tensor] = []

    for item in disease_batch:
        pos_indices: Sequence[int] = item["pos_indices"]
        if not pos_indices:
            continue

        candidate_indices: List[int] = list(pos_indices) + list(item["neg_indices"])
        if not candidate_indices:
            continue

        d_idx = int(item["disease_idx"])
        h_dis = embeddings["disease"][d_idx].unsqueeze(0)
        h_adj = embeddings["adjuvant"][candidate_indices].unsqueeze(0)

        mech_vec = None
        if adjuvant_mechanism is not None and adjuvant_mechanism.numel() > 0:
            mech_vec = adjuvant_mechanism[candidate_indices].unsqueeze(0)
        scores = dual_ranker.score_dis(h_dis, h_adj, mech_vec).squeeze(0)
        num_candidates = scores.numel()
        num_pos = len(pos_indices)

        if num_candidates == 0 or num_pos == 0:
            continue

        target_dist = scores.new_zeros(num_candidates)
        target_dist[:num_pos] = 1.0 / num_pos
        log_probs = F.log_softmax(scores, dim=0)
        listnet_losses.append(-(target_dist * log_probs).sum())

        gains = scores.new_zeros(num_candidates)
        gains[:num_pos] = 1.0

        if ndcg_topk is not None:
            k = min(int(ndcg_topk), num_candidates)
        else:
            k = num_candidates

        if k <= 0:
            continue

        if k < num_candidates:
            top_scores, top_indices = torch.topk(scores, k=k)
            top_gains = gains[top_indices]
        else:
            top_scores = scores
            top_gains = gains

        ndcg_losses.append(_approx_ndcg_loss(top_scores, top_gains, tau=ndcg_tau))

    if not ndcg_losses and not listnet_losses:
        zero = torch.tensor(0.0, device=device)
        return zero, zero

    ndcg_mean = (
        torch.stack(ndcg_losses).mean() if ndcg_losses else torch.tensor(0.0, device=device)
    )
    listnet_mean = (
        torch.stack(listnet_losses).mean()
        if listnet_losses
        else torch.tensor(0.0, device=device)
    )
    return ndcg_mean, listnet_mean


def listnet_loss_disease(
    embeddings: Mapping[str, Tensor],
    disease_batch: Sequence[Dict[str, object]],
    dual_ranker: torch.nn.Module,
    device: torch.device,
    adjuvant_mechanism: Optional[Tensor] = None,
) -> Tensor:
    """Backward-compatible wrapper returning only the ListNet loss."""

    _, listnet_loss = disease_ranking_losses(
        embeddings,
        disease_batch,
        dual_ranker,
        device,
        ndcg_tau=1.0,
        ndcg_topk=0,
        adjuvant_mechanism=adjuvant_mechanism,
    )
    return listnet_loss

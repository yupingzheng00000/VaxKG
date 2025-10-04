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


def listnet_loss_disease(
    embeddings: Mapping[str, Tensor],
    disease_batch: Sequence[Dict[str, object]],
    dual_ranker: torch.nn.Module,
    device: torch.device,
) -> Tensor:
    """
    Compute ListNet ranking loss for disease→adjuvant queries.
    
    Args:
        embeddings: {node_type: embeddings_tensor}
        disease_batch: List of {disease_idx, pos_indices, neg_indices}
        dual_ranker: DualRanker model with score_dis() method
        device: torch device
    
    Returns:
        Scalar loss tensor
    """
    if not disease_batch:
        return torch.tensor(0.0, device=device)
    
    batch_losses = []
    
    for item in disease_batch:
        d_idx = item['disease_idx']
        pos_indices = item['pos_indices']
        neg_indices = item['neg_indices']
        
        # Combine positives and negatives
        candidate_indices = pos_indices + neg_indices
        num_pos = len(pos_indices)
        num_candidates = len(candidate_indices)
        
        # Get disease embedding (single query)
        h_dis = embeddings["disease"][d_idx].unsqueeze(0)  # [1, H]
        
        # Get adjuvant embeddings (all candidates)
        h_adj = embeddings["adjuvant"][candidate_indices]  # [K, H]
        h_adj = h_adj.unsqueeze(0)  # [1, K, H]
        
        # Score using disease head
        scores = dual_ranker.score_dis(h_dis, h_adj).squeeze(0)  # [K]
        
        # Build target distribution (uniform over positives, zero over negatives)
        target_dist = torch.zeros(num_candidates, device=device)
        target_dist[:num_pos] = 1.0 / num_pos
        
        # ListNet loss
        log_probs = torch.nn.functional.log_softmax(scores, dim=0)
        loss = -(target_dist * log_probs).sum()
        
        batch_losses.append(loss)
    
    return torch.stack(batch_losses).mean() if batch_losses else torch.tensor(0.0, device=device)

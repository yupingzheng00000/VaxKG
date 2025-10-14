"""Export disease→adjuvant ranking data for case study figures."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from train_disease_ranker import (  # type: ignore
    DEFAULT_TEXT_ENCODER_MAX_LENGTHS,
    DualRanker,
    PyGHeteroEncoder,
    attach_adjuvant_classes,
    build_graph,
)


@dataclass
class RunOutputs:
    """Container for per-run scores and rankings."""

    label: str
    scores: np.ndarray
    order: np.ndarray
    rank_map: Dict[int, int]


@dataclass
class ReliabilityBin:
    """Reliability diagram bin statistics."""

    lower: float
    upper: float
    confidence: float
    precision: float
    count: int


class LegacyListNetRanker(nn.Module):
    """Dot-product ranking head used by the original ``train_ranker.py`` script."""

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _dot_product(query: Tensor, candidates: Tensor) -> Tensor:
        if query.dim() == 2 and candidates.dim() == 3:
            expanded = query.unsqueeze(1).expand_as(candidates)
            return (expanded * candidates).sum(dim=-1)
        return (query * candidates).sum(dim=-1)

    def score_vax(self, h_vax: Tensor, h_adj: Tensor) -> Tensor:
        return self._dot_product(h_vax, h_adj)

    def score_dis(
        self, h_dis: Tensor, h_adj: Tensor, mech_vec: Optional[Tensor] = None
    ) -> Tensor:
        del mech_vec  # Legacy head ignores mechanism cues entirely.
        return self._dot_product(h_dis, h_adj)

    def forward(
        self,
        embeddings: Mapping[str, Tensor],
        vaccine_indices: Tensor,
        candidate_indices: Tensor,
        relevance: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        vaccine_repr = embeddings["vaccine"][vaccine_indices]
        adjuvant_repr = embeddings["adjuvant"][candidate_indices]
        scores = self.score_vax(vaccine_repr, adjuvant_repr)

        positive_mask = (relevance > 0).float()
        positive_mass = positive_mask.sum(dim=1, keepdim=True).clamp_min(1e-9)
        target_distribution = positive_mask / positive_mass
        log_probs = F.log_softmax(scores, dim=1)
        loss = -(target_distribution * log_probs).sum(dim=1).mean()
        return loss, scores


RankerModule = Union[DualRanker, LegacyListNetRanker]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export disease→adjuvant ranking details for figure generation. "
            "The script recreates the heterogeneous graph, loads the trained "
            "encoder + ranking head, and writes a CSV with per-candidate scores "
            "alongside a JSON summary containing Top-K metrics, DCG curves, and "
            "reliability statistics."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the trained checkpoint saved by train_disease_ranker.py",
    )
    parser.add_argument(
        "--comparison-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint for a baseline run (enables rank-swap analysis).",
    )
    parser.add_argument(
        "--disease-key",
        type=str,
        required=True,
        help="Disease key (matches the 'disease_key' column produced during training).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Destination CSV containing per-adjuvant scores, ranks, and metadata.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Optional path for metrics summary JSON (defaults to output-csv with .json).",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Processed training CSV (defaults to the path stored in the checkpoint).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for inference (default: cpu).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        nargs="*",
        default=(5, 10),
        help="Top-k cut-offs for reporting precision/recall/NDCG metrics.",
    )
    parser.add_argument(
        "--curve-topk",
        type=int,
        default=10,
        help="Length of the DCG/IDCG curve (Panel B).",
    )
    parser.add_argument(
        "--reliability-bins",
        type=int,
        default=10,
        help="Number of equal-width bins for the reliability diagram.",
    )
    parser.add_argument(
        "--score-normalization",
        choices=["minmax", "sigmoid"],
        default="minmax",
        help="Normalisation applied before calibration (Panel C).",
    )
    parser.add_argument(
        "--primary-label",
        type=str,
        default="target",
        help="Label for the primary checkpoint (used in JSON + column names).",
    )
    parser.add_argument(
        "--comparison-label",
        type=str,
        default="baseline",
        help="Label for the optional comparison checkpoint.",
    )
    parser.add_argument(
        "--disease-stage-csv",
        type=Path,
        default=Path("data/processed/disease_adjuvant_pairs.csv"),
        help="Optional disease-adjuvant summary CSV (used to attach stage metadata if present).",
    )
    return parser.parse_args()


def _invert(mapping: Mapping[object, int]) -> Dict[int, object]:
    return {idx: key for key, idx in mapping.items()}


def _prepare_text_encoder(ckpt_args: Mapping[str, object], device: torch.device):
    checkpoint = ckpt_args.get("text_encoder_checkpoint")
    if not checkpoint:
        return None, None
    from transformers import AutoModel, AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model = AutoModel.from_pretrained(str(checkpoint))
    model.to(device)
    model.eval()

    max_lengths = dict(DEFAULT_TEXT_ENCODER_MAX_LENGTHS)
    override = ckpt_args.get("text_encoder_max_length")
    if override is not None:
        for key in max_lengths:
            max_lengths[key] = int(override)
    config = {
        "pooling": ckpt_args.get("text_encoder_pooling", "mean"),
        "batch_size": int(ckpt_args.get("text_encoder_batch_size", 128)),
        "max_lengths": max_lengths,
        "default_max_length": int(override or 64),
        "device": device,
        "normalize": not bool(ckpt_args.get("no_text_encoder_normalize", False)),
    }
    return (tokenizer, model), config


def _looks_like_dual_ranker_checkpoint(args: Mapping[str, object]) -> bool:
    """Heuristically determine if ``train_disease_ranker.py`` produced the checkpoint."""

    # The dual-ranker script introduces several disease-specific hyperparameters that
    # never existed in the original ``train_ranker.py`` CLI.  Presence of any of these
    # keys therefore implies the checkpoint *should* contain a dedicated ranking head.
    dual_ranker_keys: Iterable[str] = (
        "lambda_disease",
        "disease_ndcg_weight",
        "disease_listnet_weight",
        "disease_batch_size",
    )
    return any(key in args for key in dual_ranker_keys)


def _load_checkpoint(path: Path) -> MutableMapping[str, object]:
    checkpoint = torch.load(path, map_location="cpu")

    # Backwards compatibility for checkpoints produced before DualRanker support.
    # Older files bundled every parameter under ``state_dict`` (including the
    # ranking head) or stored the model weights under ``model_state_dict``.
    if "state_dict" not in checkpoint and "model_state_dict" in checkpoint:
        checkpoint["state_dict"] = checkpoint.pop("model_state_dict")

    if "ranking_head_state_dict" not in checkpoint:
        state = checkpoint.get("state_dict")
        if isinstance(state, Mapping):
            prefix = "ranking_head."
            ranking_keys = [key for key in state if key.startswith(prefix)]
            if ranking_keys:
                checkpoint["ranking_head_state_dict"] = {
                    key[len(prefix) :]: state[key]
                    for key in ranking_keys
                }
                for key in ranking_keys:
                    del state[key]

    required_keys = {"state_dict", "args"}
    missing = required_keys - checkpoint.keys()
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(
            "Checkpoint at "
            f"{path} is missing required keys: {missing_str}. "
            "The file was likely produced by an outdated training script."
        )
    args = checkpoint.get("args", {})
    if not isinstance(args, Mapping):
        args = {}

    legacy_head = "ranking_head_state_dict" not in checkpoint
    if legacy_head and _looks_like_dual_ranker_checkpoint(args):
        raise KeyError(
            "Checkpoint at "
            f"{path} was produced by train_disease_ranker.py but is missing "
            "'ranking_head_state_dict'. The run likely failed before saving the "
            "disease head; please re-train to generate a complete checkpoint."
        )

    checkpoint["_legacy_listnet_head"] = legacy_head

    if not legacy_head and "ranking_head_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint at "
            f"{path} is missing 'ranking_head_state_dict' despite recovery attempts."
        )
    return checkpoint


def _build_model(
    graph: "HeteroData",
    ckpt_args: Mapping[str, object],
    checkpoint: Mapping[str, object],
    device: torch.device,
) -> Tuple[PyGHeteroEncoder, RankerModule, Optional[Tensor]]:
    node_feat_dims = {nt: graph[nt].x.size(1) for nt in graph.node_types}
    model = PyGHeteroEncoder(
        node_feat_dims,
        graph.metadata(),
        int(ckpt_args.get("hidden_dim", 128)),
        int(ckpt_args.get("layers", 2)),
        float(ckpt_args.get("dropout", 0.3)),
        int(ckpt_args.get("appnp_steps", 10)),
        float(ckpt_args.get("appnp_alpha", 0.1)),
        float(ckpt_args.get("appnp_dropout", 0.0)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    structured = getattr(graph["adjuvant"], "structured", None)
    mech_dim = int(structured.size(1)) if isinstance(structured, Tensor) else 0
    use_mech = bool(ckpt_args.get("enable_disease_mechanism_cues", False)) and mech_dim > 0
    mech_in_dim = mech_dim if use_mech else None
    legacy_head = bool(checkpoint.get("_legacy_listnet_head", False))
    if legacy_head:
        print(
            "Warning: checkpoint lacks a dedicated ranking head; using legacy "
            "dot-product scores (disease ranking quality may degrade)."
        )
        ranking_head: RankerModule = LegacyListNetRanker()
    else:
        ranking_head = DualRanker(
            int(ckpt_args.get("hidden_dim", 128)),
            mech_in_dim,
            float(ckpt_args.get("gamma_mech", 0.3)),
        )
        ranking_head.load_state_dict(checkpoint["ranking_head_state_dict"])
    ranking_head = ranking_head.to(device)
    ranking_head.eval()

    if use_mech and isinstance(structured, Tensor):
        adjuvant_mechanism = structured.to(device)
    else:
        adjuvant_mechanism = None
    return model, ranking_head, adjuvant_mechanism


def _score_disease(
    disease_idx: int,
    candidate_ids: Sequence[int],
    model: PyGHeteroEncoder,
    ranking_head: RankerModule,
    graph: "HeteroData",
    adjuvant_mechanism: Optional[Tensor],
    device: torch.device,
) -> np.ndarray:
    graph_device = graph.to(device)
    with torch.no_grad():
        embeddings = model(graph_device)
        disease_vec = embeddings["disease"][disease_idx].unsqueeze(0)
        candidates = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        adjuvant_vec = embeddings["adjuvant"][candidates].unsqueeze(0)
        mech = None
        if adjuvant_mechanism is not None:
            mech = adjuvant_mechanism[candidates].unsqueeze(0)
        scores = ranking_head.score_dis(disease_vec, adjuvant_vec, mech)
    return scores.squeeze(0).cpu().numpy()


def _order_from_scores(
    scores: np.ndarray, candidate_ids: Sequence[int]
) -> Tuple[np.ndarray, Dict[int, int]]:
    order = np.argsort(-scores)
    rank_map = {
        int(candidate_ids[idx]): int(rank) for rank, idx in enumerate(order, start=1)
    }
    return order, rank_map


def _precision_at_k(ranked: Sequence[int], positives: Sequence[int], k: int) -> float:
    if k <= 0:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for item in top if item in positives)
    return hits / k


def _recall_at_k(ranked: Sequence[int], positives: Sequence[int], k: int) -> float:
    if not positives:
        return 0.0
    top = ranked[:k]
    hits = sum(1 for item in top if item in positives)
    return hits / len(positives)


def _dcg_at_k(ranked: Sequence[int], gains: Mapping[int, float], k: int) -> float:
    total = 0.0
    for idx, item in enumerate(ranked[:k], start=1):
        gain = float(gains.get(item, 0.0))
        if gain <= 0:
            continue
        total += gain / math.log2(idx + 1)
    return total


def _compute_reliability(
    scores: np.ndarray,
    candidate_ids: Sequence[int],
    positives: Sequence[int],
    bins: int,
    normalisation: str,
) -> Tuple[List[ReliabilityBin], float]:
    if bins <= 0 or scores.size == 0:
        return [], 0.0

    if normalisation == "sigmoid":
        norm_scores = 1.0 / (1.0 + np.exp(-scores))
    else:  # minmax
        min_score = float(scores.min())
        max_score = float(scores.max())
        if math.isclose(max_score, min_score):
            norm_scores = np.full_like(scores, 0.5, dtype=float)
        else:
            norm_scores = (scores - min_score) / (max_score - min_score)

    positives_set = set(positives)
    counts = np.zeros(bins, dtype=int)
    confs = np.zeros(bins, dtype=float)
    precs = np.zeros(bins, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_indices = np.digitize(norm_scores, edges, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, bins - 1)

    for idx, bin_id in enumerate(bin_indices):
        counts[bin_id] += 1
        confs[bin_id] += norm_scores[idx]
        candidate = int(candidate_ids[idx])
        precs[bin_id] += 1.0 if candidate in positives_set else 0.0

    reliability: List[ReliabilityBin] = []
    total = scores.size
    ece = 0.0
    for bin_id in range(bins):
        count = int(counts[bin_id])
        lower = float(edges[bin_id])
        upper = float(edges[bin_id + 1])
        if count == 0:
            reliability.append(
                ReliabilityBin(lower=lower, upper=upper, confidence=0.0, precision=0.0, count=0)
            )
            continue
        confidence = confs[bin_id] / count
        precision = precs[bin_id] / count
        reliability.append(
            ReliabilityBin(
                lower=lower,
                upper=upper,
                confidence=float(confidence),
                precision=float(precision),
                count=count,
            )
        )
        ece += (count / total) * abs(precision - confidence)
    return reliability, float(ece)


def _compute_run_metrics(
    run: RunOutputs,
    candidate_ids: Sequence[int],
    positives: Sequence[int],
    gains: Mapping[int, float],
    topk: Sequence[int],
    curve_k: int,
    reliability_bins: int,
    normalisation: str,
) -> Dict[str, object]:
    ranked_indices = [int(candidate_ids[i]) for i in run.order]
    metrics: Dict[str, object] = {
        "label": run.label,
        "num_candidates": len(candidate_ids),
        "num_positives": len(positives),
    }

    per_k: Dict[str, Dict[str, float]] = {}
    for k in topk:
        ranked_slice = ranked_indices[:k]
        per_k[str(k)] = {
            "precision": _precision_at_k(ranked_slice, positives, k),
            "recall": _recall_at_k(ranked_slice, positives, k),
            "ndcg": _ndcg_at_k(ranked_slice, gains, positives, k),
        }
    metrics["topk"] = per_k

    curve_limit = min(curve_k, len(candidate_ids))
    dcg_curve = []
    idcg_curve = []
    sorted_gains = sorted(gains.values(), reverse=True)
    if not sorted_gains:
        sorted_gains = []
    cumulative_dcg = 0.0
    for idx, candidate in enumerate(ranked_indices[:curve_limit], start=1):
        cumulative_dcg += float(gains.get(candidate, 0.0)) / math.log2(idx + 1)
        dcg_curve.append(cumulative_dcg)
    cumulative_idcg = 0.0
    for idx in range(1, curve_limit + 1):
        gain = float(sorted_gains[idx - 1]) if idx - 1 < len(sorted_gains) else 0.0
        cumulative_idcg += gain / math.log2(idx + 1)
        idcg_curve.append(cumulative_idcg)
    metrics["dcg_curve"] = dcg_curve
    metrics["idcg_curve"] = idcg_curve
    ndcg_at_curve = _ndcg_value(cumulative_dcg, cumulative_idcg)
    metrics["ndcg@curve_k"] = ndcg_at_curve

    reliability, ece = _compute_reliability(
        run.scores,
        candidate_ids,
        positives,
        reliability_bins,
        normalisation,
    )
    metrics["ece"] = ece
    metrics["reliability_bins"] = [
        {
            "lower": bin.lower,
            "upper": bin.upper,
            "confidence": bin.confidence,
            "precision": bin.precision,
            "count": bin.count,
        }
        for bin in reliability
    ]
    return metrics


def _ndcg_value(dcg: float, idcg: float) -> float:
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def _ndcg_at_k(
    ranked: Sequence[int],
    gains: Mapping[int, float],
    positives: Sequence[int],
    k: int,
) -> float:
    if k <= 0:
        return 0.0
    dcg = _dcg_at_k(ranked, gains, k)
    ideal_gains = sorted((gains.get(idx, 1.0) for idx in positives), reverse=True)
    ideal = 0.0
    for idx in range(1, min(k, len(ideal_gains)) + 1):
        ideal += ideal_gains[idx - 1] / math.log2(idx + 1)
    return _ndcg_value(dcg, ideal)


def _collect_display_labels(df: pd.DataFrame) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    columns = [
        "adjuvant_display_name",
        "vo_preferred_label",
        "adjuvant_name",
    ]
    for adjuvant_id, group in df.groupby("adjuvant_vo_id"):
        label = None
        for column in columns:
            if column in group and column in df.columns:
                series = group[column].dropna()
                if not series.empty:
                    value = str(series.iloc[0]).strip()
                    if value:
                        label = value
                        break
        labels[str(adjuvant_id)] = label or str(adjuvant_id)
    return labels


def main() -> None:
    args = parse_args()
    summary_path = args.summary_json or args.output_csv.with_suffix(".json")
    device = torch.device(args.device)
    topk_values = sorted({int(k) for k in args.topk if int(k) > 0})
    if not topk_values:
        raise ValueError("At least one positive --topk value is required")

    checkpoint = _load_checkpoint(args.checkpoint)
    ckpt_args: Mapping[str, object] = checkpoint["args"]

    data_path = args.data_path or Path(ckpt_args.get("data_path", ""))
    if not data_path:
        raise FileNotFoundError(
            "Data path not provided and checkpoint metadata is missing 'data_path'."
        )
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Training data CSV not found at {data_path}")

    df = pd.read_csv(data_path)
    df = attach_adjuvant_classes(df)
    text_encoder, encoder_config = _prepare_text_encoder(ckpt_args, device)
    graph, mappings, _, _, candidate_ids, disease_positives_lookup, disease_edge_weights = build_graph(
        df,
        int(ckpt_args.get("feature_dim", 256)),
        text_encoder=text_encoder,
        text_encoder_config=encoder_config,
        enable_disease_structured=bool(ckpt_args.get("enable_disease_mechanism_cues", False)),
    )

    if text_encoder is not None:
        tokenizer, model = text_encoder
        model.to("cpu")
        del tokenizer
        del model

    disease_mapping = mappings["disease"]
    disease_key = args.disease_key
    disease_idx = disease_mapping.get(disease_key)
    if disease_idx is None:
        lowered = {key.lower(): value for key, value in disease_mapping.items()}
        disease_idx = lowered.get(disease_key.lower())
    if disease_idx is None:
        available = ", ".join(sorted(disease_mapping.keys())[:20])
        raise KeyError(
            f"Disease key '{args.disease_key}' not found. Example keys: {available}..."
        )

    candidate_list = list(candidate_ids)
    adjuvant_mapping = _invert(mappings["adjuvant"])
    positive_indices = disease_positives_lookup.get(disease_idx, [])
    gains: Dict[int, float] = {
        int(adj_idx): float(disease_edge_weights.get((disease_idx, adj_idx), 1.0))
        for adj_idx in positive_indices
    }

    model, ranking_head, adjuvant_mechanism = _build_model(
        graph,
        ckpt_args,
        checkpoint,
        device,
    )
    primary_scores = _score_disease(
        disease_idx,
        candidate_list,
        model,
        ranking_head,
        graph,
        adjuvant_mechanism,
        device,
    )
    primary_order, primary_ranks = _order_from_scores(primary_scores, candidate_list)
    primary_score_map = {
        candidate_list[i]: float(primary_scores[i]) for i in range(len(candidate_list))
    }
    runs = [
        RunOutputs(
            label=args.primary_label,
            scores=primary_scores,
            order=primary_order,
            rank_map=primary_ranks,
        )
    ]

    comparison_data: Optional[RunOutputs] = None
    comp_score_map: Dict[int, float] = {}
    if args.comparison_checkpoint is not None:
        comparison_ckpt = _load_checkpoint(args.comparison_checkpoint)
        comparison_args: Mapping[str, object] = comparison_ckpt["args"]
        if comparison_args.get("data_path") != ckpt_args.get("data_path"):
            print("Warning: comparison checkpoint was trained on a different data path.")
        comp_model, comp_head, comp_mech = _build_model(
            graph,
            comparison_args,
            comparison_ckpt,
            device,
        )
        comp_scores = _score_disease(
            disease_idx,
            candidate_list,
            comp_model,
            comp_head,
            graph,
            comp_mech,
            device,
        )
        comp_order, comp_ranks = _order_from_scores(comp_scores, candidate_list)
        comp_score_map = {
            candidate_list[i]: float(comp_scores[i]) for i in range(len(candidate_list))
        }
        comparison_data = RunOutputs(
            label=args.comparison_label,
            scores=comp_scores,
            order=comp_order,
            rank_map=comp_ranks,
        )
        runs.append(comparison_data)

    class_lookup = (
        df.groupby("adjuvant_vo_id")["adjuvant_class"].first().to_dict()
        if "adjuvant_class" in df.columns
        else {}
    )
    display_lookup = _collect_display_labels(df)

    stage_lookup: Dict[Tuple[str, str], str] = {}
    if args.disease_stage_csv and args.disease_stage_csv.exists():
        stage_df = pd.read_csv(args.disease_stage_csv)
        if {"disease_key", "adjuvant_vo_id", "stage_mode"}.issubset(stage_df.columns):
            stage_lookup = {
                (str(row["disease_key"]), str(row["adjuvant_vo_id"])): str(row["stage_mode"])
                for _, row in stage_df.iterrows()
            }

    primary_ranked_indices = [candidate_list[idx] for idx in primary_order]
    rows: List[Dict[str, object]] = []
    for candidate_idx in primary_ranked_indices:
        adjuvant_id = str(adjuvant_mapping[candidate_idx])
        row: Dict[str, object] = {
            "disease_key": disease_key,
            "adjuvant_vo_id": adjuvant_id,
            "adjuvant_label": display_lookup.get(adjuvant_id, adjuvant_id),
            "adjuvant_class": class_lookup.get(adjuvant_id, "unknown"),
            f"score_{args.primary_label}": primary_score_map[candidate_idx],
            f"rank_{args.primary_label}": int(primary_ranks[candidate_idx]),
            "relevance_binary": int(candidate_idx in positive_indices),
            "relevance_gain": float(gains.get(candidate_idx, 0.0)),
            "edge_weight": float(disease_edge_weights.get((disease_idx, candidate_idx), 0.0)),
            "stage_mode": stage_lookup.get((disease_key, adjuvant_id), ""),
        }
        if comparison_data is not None:
            row[f"score_{args.comparison_label}"] = comp_score_map[candidate_idx]
            row[f"rank_{args.comparison_label}"] = int(
                comparison_data.rank_map[candidate_idx]
            )
        rows.append(row)

    output_df = pd.DataFrame(rows)
    output_df.sort_values(by=f"rank_{args.primary_label}", inplace=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(args.output_csv, index=False)
    print(f"Wrote candidate table to {args.output_csv}")

    metrics_summary = {
        "disease_key": disease_key,
        "num_candidates": len(candidate_list),
        "num_positives": len(positive_indices),
        "runs": {},
    }
    for run in runs:
        metrics_summary["runs"][run.label] = _compute_run_metrics(
            run,
            candidate_list,
            positive_indices,
            gains,
            tuple(topk_values),
            int(args.curve_topk),
            int(args.reliability_bins),
            args.score_normalization,
        )

    if comparison_data is not None:
        deltas: Dict[str, float] = {}
        for k in topk_values:
            primary_ndcg = metrics_summary["runs"][args.primary_label]["topk"][str(k)]["ndcg"]
            baseline_ndcg = metrics_summary["runs"][args.comparison_label]["topk"][str(k)][
                "ndcg"
            ]
            deltas[f"ndcg@{k}"] = primary_ndcg - baseline_ndcg
        metrics_summary["delta_ndcg"] = deltas

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_summary, handle, indent=2)
    print(f"Wrote summary metrics to {summary_path}")


if __name__ == "__main__":
    main()

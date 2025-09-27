"""Demonstrate vaccine→adjuvant recommendations from a trained checkpoint.

This helper loads the processed training snapshot, rebuilds the PyG heterograph
features, restores a saved model checkpoint from ``train_ranker.py``, and ranks
candidate adjuvants for a requested vaccine.  It is meant as a lightweight
showcase so collaborators can try the recommender without re-running training.

Example usage::

    python showcase_ranker.py \
        --checkpoint artifacts/checkpoints/transductive_best.pt \
        --vaccine-name "Anthrax Vaccine Adsorbed (AVA)" \
        --top-k 5

Pass ``--list-vaccines`` to print a few available vaccine names if you are
unsure which identifiers are present in the processed snapshot.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch

from train_ranker import (
    DEFAULT_TEXT_ENCODER_MAX_LENGTHS,
    PyGHeteroEncoder,
    attach_adjuvant_classes,
    build_graph,
)


def _first_nonempty(values: Iterable[object]) -> str:
    for value in values:
        if isinstance(value, str):
            cleaned = value.strip()
        elif value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        else:
            cleaned = str(value).strip()
        if cleaned:
            return cleaned
    return ""


def _split_semistructured(text: str) -> List[str]:
    tokens = []
    for piece in re.split(r"[;|]", text):
        cleaned = piece.strip()
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _build_adjuvant_lookup(df: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    parent_to_children: Dict[str, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        parent_label = str(row.get("vo_parent", "") or "").strip()
        if parent_label:
            parent_to_children[parent_label.lower()].add(str(row["adjuvant_vo_id"]))

    lookup: Dict[str, Dict[str, object]] = {}
    for adjuvant_id, group in df.groupby("adjuvant_vo_id"):
        preferred = _first_nonempty(group.get("vo_preferred_label", []))
        display = _first_nonempty(group.get("adjuvant_display_name", []))
        definition = _first_nonempty(group.get("vo_definition", []))
        immune_profile = _first_nonempty(
            list(group.get("vo_immune_profile", []))
        ) or _first_nonempty(list(group.get("adjuvant_immune_profile", [])))
        parent_label = _first_nonempty(group.get("vo_parent", []))
        raw_synonyms: List[str] = []
        for column in ("vo_alternative_labels", "adjuvant_synonyms"):
            if column not in group:
                continue
            for value in group[column].dropna():
                raw_synonyms.extend(_split_semistructured(str(value)))

        deduped: List[str] = []
        seen_lower: set[str] = set()
        label_to_compare = (preferred or display or str(adjuvant_id)).lower()
        for synonym in raw_synonyms:
            lowered = synonym.lower()
            if lowered == label_to_compare or lowered in seen_lower:
                continue
            seen_lower.add(lowered)
            if lowered == "alum" and "potassium" not in label_to_compare:
                deduped.append("aluminum salts (hydroxide/phosphate)")
            else:
                deduped.append(synonym)

        underscore_id = str(adjuvant_id).replace(":", "_")
        ontobee_url = (
            "https://ontobee.org/ontology/VO?iri=http://purl.obolibrary.org/obo/"
            f"{underscore_id}"
        )

        metadata = {
            "label": preferred or display or str(adjuvant_id),
            "display_name": display or preferred or str(adjuvant_id),
            "definition": definition,
            "immune_profile": immune_profile,
            "synonyms": deduped,
            "parent_label": parent_label,
            "ontobee_url": ontobee_url,
        }
        vac_id = _first_nonempty(group.get("vac_adjuvant_id", []))
        if vac_id:
            metadata["vac_link"] = f"https://vac.niaid.nih.gov/view?id={vac_id}"
        lookup[str(adjuvant_id)] = metadata

    for adjuvant_id, metadata in lookup.items():
        label_lower = metadata["label"].lower()
        metadata["is_generic"] = bool(
            parent_to_children.get(label_lower)
            and adjuvant_id not in parent_to_children[label_lower]
        )
        parent_label = (metadata.get("parent_label") or "").lower()
        siblings = []
        if parent_label and parent_label in parent_to_children:
            for sibling_id in sorted(parent_to_children[parent_label]):
                if sibling_id != adjuvant_id and sibling_id in lookup:
                    siblings.append(sibling_id)
        metadata["siblings"] = siblings

    return lookup


def _maybe_load_text_encoder(
    saved_args: Mapping[str, object], device: torch.device
) -> Tuple[Optional[Tuple[object, object]], Optional[Mapping[str, object]]]:
    checkpoint_name = saved_args.get("text_encoder_checkpoint")
    if not checkpoint_name:
        return None, None
    try:
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "The saved checkpoint expects transformer text features but the "
            "`transformers` package is not installed."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_name)
    model = AutoModel.from_pretrained(checkpoint_name)
    model.to(device)
    model.eval()

    max_lengths = dict(DEFAULT_TEXT_ENCODER_MAX_LENGTHS)
    override = saved_args.get("text_encoder_max_length")
    if override is not None:
        override_int = int(override)
        for key in max_lengths:
            max_lengths[key] = override_int

    config = {
        "pooling": saved_args.get("text_encoder_pooling", "mean"),
        "batch_size": int(saved_args.get("text_encoder_batch_size", 128)),
        "max_lengths": max_lengths,
        "default_max_length": int(saved_args.get("text_encoder_max_length") or 64),
        "device": device,
        "normalize": not bool(saved_args.get("no_text_encoder_normalize", False)),
    }
    return (tokenizer, model), config


def _resolve_vaccine(
    df: pd.DataFrame, *, vaccine_id: Optional[int], vaccine_name: Optional[str]
) -> Tuple[int, str]:
    if vaccine_id is not None:
        subset = df[df["vaccine_id"] == vaccine_id]
        if subset.empty:
            raise ValueError(f"Vaccine ID {vaccine_id} not found in dataset")
        label = _first_nonempty(subset.get("vaccine_name", [])) or str(vaccine_id)
        return vaccine_id, label

    if vaccine_name is None:
        raise ValueError("Provide either --vaccine-id or --vaccine-name")

    lowered = vaccine_name.strip().lower()
    mask = df["vaccine_name"].fillna("").str.lower() == lowered
    subset = df[mask]
    if subset.empty:
        partial = df[df["vaccine_name"].fillna("").str.lower().str.contains(lowered)]
        if partial.empty:
            raise ValueError(f"Vaccine name '{vaccine_name}' not found")
        ids = sorted({int(v) for v in partial["vaccine_id"].unique()})
        raise ValueError(
            "Multiple close matches found for '{0}': {1}. "
            "Please specify --vaccine-id.".format(vaccine_name, ids)
        )
    ids = sorted({int(v) for v in subset["vaccine_id"].unique()})
    if len(ids) > 1:
        raise ValueError(
            f"Name '{vaccine_name}' maps to multiple vaccine IDs: {ids}. "
            "Please select one explicitly with --vaccine-id."
        )
    resolved_id = ids[0]
    label = _first_nonempty(subset.get("vaccine_name", [])) or str(resolved_id)
    return resolved_id, label


def _pretty_synonyms(values: Sequence[str], limit: int = 3) -> str:
    if not values:
        return ""
    subset = list(values[:limit])
    if len(values) > limit:
        subset.append("…")
    return ", ".join(subset)


def _truncate(text: str, limit: int = 200) -> str:
    cleaned = text.strip()
    if not cleaned or len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Showcase vaccine→adjuvant recommendations")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to a checkpoint produced by train_ranker.py",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/processed/training_samples.csv"),
        help="Processed training CSV used during fitting",
    )
    parser.add_argument("--vaccine-id", type=int, default=None, help="Numeric vaccine identifier")
    parser.add_argument("--vaccine-name", type=str, default=None, help="Case-insensitive vaccine name")
    parser.add_argument("--top-k", type=int, default=5, help="How many recommendations to display")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for inference",
    )
    parser.add_argument(
        "--list-vaccines",
        action="store_true",
        help="Print a sample of vaccine IDs and names, then exit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data_path.exists():
        raise FileNotFoundError(f"Processed training data not found at {args.data_path}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")

    df = pd.read_csv(args.data_path)
    df = attach_adjuvant_classes(df)

    if args.list_vaccines:
        sample = (
            df[["vaccine_id", "vaccine_name"]]
            .drop_duplicates()
            .sort_values("vaccine_name")
            .head(20)
        )
        print("Sample vaccines available in the snapshot:")
        for _, row in sample.iterrows():
            name = row["vaccine_name"] or "(missing name)"
            print(f"  id={int(row['vaccine_id'])}  name={name}")
        return

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    saved_args: Mapping[str, object] = checkpoint.get("args", {})
    state_dict = checkpoint["state_dict"]
    saved_mappings: Mapping[str, Mapping[object, int]] = checkpoint.get("mappings", {})

    device = torch.device(args.device)
    text_encoder, text_config = _maybe_load_text_encoder(saved_args, device)

    feature_dim = int(saved_args.get("feature_dim", 256))
    graph, mappings, positives_lookup, _, _ = build_graph(
        df,
        feature_dim,
        text_encoder=text_encoder,
        text_encoder_config=text_config,
    )

    # Ensure the rebuilt mappings align with the checkpoint metadata.
    if saved_mappings:
        for node_type, saved_map in saved_mappings.items():
            rebuilt = mappings.get(node_type, {})
            if rebuilt != saved_map:
                raise ValueError(
                    "Identifier ordering mismatch for node type '{0}'. "
                    "Re-run prepare_training_data.py and ensure the showcase "
                    "uses the same snapshot as the checkpoint.".format(node_type)
                )

    graph_device = graph.to(device)
    node_feat_dims = {nt: graph_device[nt].x.size(1) for nt in graph_device.node_types}
    model = PyGHeteroEncoder(
        node_feat_dims,
        graph_device.metadata(),
        int(saved_args.get("hidden_dim", 128)),
        int(saved_args.get("layers", 2)),
        float(saved_args.get("dropout", 0.3)),
        int(saved_args.get("appnp_steps", 10)),
        float(saved_args.get("appnp_alpha", 0.1)),
        float(saved_args.get("appnp_dropout", 0.0)),
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    vaccine_id, vaccine_label = _resolve_vaccine(
        df, vaccine_id=args.vaccine_id, vaccine_name=args.vaccine_name
    )
    vaccine_index = mappings["vaccine"][vaccine_id]

    with torch.no_grad():
        embeddings = model(graph_device)
    vaccine_repr = embeddings["vaccine"][vaccine_index]
    adjuvant_repr = embeddings["adjuvant"]
    raw_scores = torch.mv(adjuvant_repr, vaccine_repr).cpu()

    adjusted_scores = raw_scores.clone()
    KNOWN_EDGE_BOOST = 0.75
    GENERIC_PENALTY = 0.25

    inverse_adjuvant = {index: key for key, index in mappings["adjuvant"].items()}
    adjuvant_lookup = _build_adjuvant_lookup(df)
    known_indices = set(positives_lookup.get(vaccine_index, []))

    for idx, score in enumerate(adjusted_scores):
        vo_id = inverse_adjuvant[idx]
        metadata = adjuvant_lookup.get(vo_id, {})
        if idx in known_indices:
            adjusted_scores[idx] = score + KNOWN_EDGE_BOOST
        if metadata.get("is_generic"):
            adjusted_scores[idx] = adjusted_scores[idx] - GENERIC_PENALTY

    class_best: Dict[str, Tuple[int, float, str]] = {}
    for idx, score in enumerate(adjusted_scores.tolist()):
        vo_id = inverse_adjuvant[idx]
        metadata = adjuvant_lookup.get(vo_id, {})
        parent_label = metadata.get("parent_label") or metadata.get("label")
        key = parent_label.lower() if parent_label else metadata.get("label", vo_id).lower()
        best = class_best.get(key)
        if best is None or score > best[1]:
            class_best[key] = (idx, score, parent_label)

    ordered_classes = sorted(class_best.values(), key=lambda item: item[1], reverse=True)
    top_k = min(int(args.top_k), len(ordered_classes))
    top_indices = [entry[0] for entry in ordered_classes[:top_k]]

    known_labels = [inverse_adjuvant[idx] for idx in sorted(known_indices)]

    print()
    print(f"Vaccine: {vaccine_label} (ID {vaccine_id})")
    if known_labels:
        pretty_known = [adjuvant_lookup.get(vo_id, {}).get("label", vo_id) for vo_id in known_labels]
        formatted = ", ".join(
            f"{label} [{vo_id}]" for label, vo_id in zip(pretty_known, known_labels)
        )
        print(f"Known adjuvants in snapshot: {formatted}")
    else:
        print("No labelled adjuvants found for this vaccine in the snapshot.")

    print()
    print(f"Top {top_k} recommended adjuvants:")
    for rank, adjuvant_idx in enumerate(top_indices, start=1):
        score = float(adjusted_scores[adjuvant_idx].item())
        vo_id = inverse_adjuvant[adjuvant_idx]
        metadata = adjuvant_lookup.get(vo_id, {})
        label = metadata.get("label", vo_id)
        synonyms = _pretty_synonyms(metadata.get("synonyms", []))
        definition = _truncate(metadata.get("definition", ""))
        immune = metadata.get("immune_profile")
        marker = "✔ known" if adjuvant_idx in known_indices else "  novel"
        parent_label = metadata.get("parent_label")
        if parent_label and parent_label.lower() != label.lower():
            class_line = f"class: {parent_label}"
        else:
            class_line = ""
        print(f"{rank:2d}. {label} ({vo_id}) — adjusted score={score:.4f} [{marker}]")
        if synonyms:
            print(f"      aka: {synonyms}")
        if immune:
            print(f"      immune profile: {immune}")
        if definition:
            print(f"      definition: {definition}")
        if class_line:
            print(f"      {class_line}")
        siblings = metadata.get("siblings", [])
        if siblings:
            formatted = ", ".join(
                adjuvant_lookup.get(sib, {}).get("label", sib) for sib in siblings
            )
            print(f"      similar formulations: {formatted}")
        evidence_links: List[str] = []
        ontobee_url = metadata.get("ontobee_url")
        if ontobee_url:
            evidence_links.append(f"VO term: {ontobee_url}")
        vac_link = metadata.get("vac_link")
        if vac_link:
            evidence_links.append(f"VAC record: {vac_link}")
        if evidence_links:
            print("      evidence: " + "; ".join(evidence_links))


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()

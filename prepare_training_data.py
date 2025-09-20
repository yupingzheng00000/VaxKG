"""Assemble vaccine-adjuvant training data from the VaxKG CSV exports.

The script rebuilds the joins that the Neo4j loader performs so we can
prototype models straight from the relational snapshots.  It produces:

* ``training_samples.csv`` – one row per curated vaccine/adjuvant pair with
  disease, platform, and metadata fields that are useful for feature
  engineering.
* ``adjuvant_metadata_enriched.csv`` – the adjuvant usage table with missing
  labels/descriptions filled from the VO term editing sheet plus ontology
  annotations such as immune profiles and receptor targets.
* ``disease_adjuvant_candidates.json`` – candidate adjuvant VO IDs grouped by
  disease, preserving the order in which they appear in the curated data.
* ``disease_platform_adjuvant_candidates.json`` – similar to the above but
  conditioned on the vaccine platform/type string so the model can draw
  disease-by-platform priors.

Example usage::

    python prepare_training_data.py --data-dir data --output-dir data/processed

"""
from __future__ import annotations

import argparse
import json
import numbers
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

PATHOGEN_DISEASE_CORRECTIONS: Dict[int, Dict[str, str]] = {
    # c_pathogen_id 113 exists in t_pathogen.csv but its c_disease_name is blank.
    113: {
        "disease_name": "Chlamydia muridarum infection",
        "note": "Pathogen record is missing c_disease_name in t_pathogen.csv",
    },
    # c_pathogen_id 164 also lacks a disease label in the exported pathogen table.
    164: {
        "disease_name": "Avian paramyxovirus infection",
        "note": "Pathogen record is missing c_disease_name in t_pathogen.csv",
    },
    # Vaccines reference pathogen_id 266, which is absent from t_pathogen.csv.
    266: {
        "disease_name": "Giardiasis (Beaver fever)",
        "pathogen_name": "Giardia lamblia",
        "note": "Pathogen ID is not present in t_pathogen.csv; mapped using vaccine metadata",
    },
}

MISSING_VALUE_SUMMARY_COLUMNS: Sequence[str] = (
    "vaccine_id",
    "vaccine_name",
    "pathogen_id",
    "pathogen_name",
    "disease_name",
    "platform_group",
    "adjuvant_vo_id",
    "adjuvant_label",
    "adjuvant_description",
)

DEFAULT_VO_TERMS_PATH = Path(__file__).with_name(
    "VO ID term editing - vaccine adjvant.csv"
)

PLATFORM_CATEGORY_ORDER: Sequence[str] = (
    "subunit",
    "inactivated",
    "live_attenuated",
    "toxoid",
    "conjugate",
    "dna_rna",
    "vector",
    "other",
    "unspecified",
)

MOJIBAKE_MARKERS: Sequence[str] = ("Ã", "Â", "â", "Ê", "¤")


def _repair_mojibake(text: str) -> str:
    """Attempt to fix common UTF-8/latin-1 mojibake sequences."""

    if not any(marker in text for marker in MOJIBAKE_MARKERS):
        return text
    try:
        candidate = text.encode("latin1").decode("utf-8")
    except UnicodeError:
        return text
    if any(marker in candidate for marker in MOJIBAKE_MARKERS):
        return text
    return candidate


def read_csv_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """Load a CSV, falling back to latin-1 encoding when UTF-8 fails."""

    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", **kwargs)


def load_source_tables(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load the core relational tables we need from ``data_dir``."""

    file_map = {
        "vaccines": "t_vaccine.csv",
        "adjuvants": "t_adjuvant.csv",
        "vaxjo": "t_vaxjo.csv",
        "vaxvec": "t_vaxvec.csv",
        "pathogens": "t_pathogen.csv",
        "vaccine_detail": "t_vaccine_detail.csv",
    }
    tables: Dict[str, pd.DataFrame] = {}

    for key, filename in file_map.items():
        csv_path = data_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected CSV '{filename}' in {data_dir!s}")
        tables[key] = read_csv_with_fallback(csv_path)
    return tables


def _normalize_text_value(value: object) -> object:
    """Normalize whitespace-only strings to ``pd.NA`` and coerce numbers."""

    if isinstance(value, str):
        fixed = _repair_mojibake(value)
        normalized = " ".join(fixed.replace("\r", " ").split())
        return normalized or pd.NA
    if pd.isna(value):
        return pd.NA
    if isinstance(value, numbers.Integral):
        return str(int(value))
    if isinstance(value, numbers.Real):
        real_value = float(value)
        if real_value.is_integer():
            return str(int(real_value))
        return str(real_value)
    text = str(value).strip()
    return text or pd.NA


def _normalize_text_series(series: pd.Series) -> pd.Series:
    """Apply :func:`_normalize_text_value` element-wise."""

    return series.map(_normalize_text_value)


def _normalize_pipe_separated(value: object) -> object:
    """Convert ``|``-delimited lists into semi-colon separated strings."""

    normalized = _normalize_text_value(value)
    if normalized is pd.NA or pd.isna(normalized):
        return pd.NA
    parts: List[str] = []
    for chunk in str(normalized).split("|"):
        cleaned = " ".join(chunk.replace("\r", " ").split())
        if cleaned:
            parts.append(cleaned)
    if not parts:
        return pd.NA
    return "; ".join(parts)


VO_ID_PATTERN = re.compile(r"VO[:_]?([0-9]+)", re.IGNORECASE)


def canonicalize_vo_identifier(value: object) -> object:
    """Return a canonical ``VO:######`` identifier when possible."""

    if pd.isna(value):
        return pd.NA
    if isinstance(value, numbers.Integral):
        return f"VO:{int(value):07d}"
    if isinstance(value, numbers.Real):
        real_value = float(value)
        if real_value.is_integer():
            return f"VO:{int(real_value):07d}"
    text = str(value).strip()
    if not text:
        return pd.NA
    match = VO_ID_PATTERN.search(text.replace(" ", ""))
    if match:
        return f"VO:{int(match.group(1)):07d}"
    digits = re.sub(r"\D", "", text)
    if digits:
        return f"VO:{int(digits):07d}"
    return text


def underscore_vo_identifier(value: object) -> object:
    """Return the underscore form (``VO_######``) of a VO identifier."""

    canonical = canonicalize_vo_identifier(value)
    if pd.isna(canonical):
        return pd.NA
    return str(canonical).replace(":", "_")


PLATFORM_SPLIT_PATTERN = re.compile(
    r"\s*(?:/|\+|;|,|&|\band\b|\bor\b)\s*", re.IGNORECASE
)


def _dedupe_preserving_order(values: Iterable[str]) -> List[str]:
    """Deduplicate a sequence while preserving element order."""

    seen: OrderedDict[str, None] = OrderedDict()
    for value in values:
        if not value:
            continue
        if value not in seen:
            seen[value] = None
    return list(seen.keys())


def _sorted_platform_categories(categories: Iterable[str]) -> List[str]:
    """Sort platform categories using ``PLATFORM_CATEGORY_ORDER``."""

    order = {name: idx for idx, name in enumerate(PLATFORM_CATEGORY_ORDER)}
    unique = _dedupe_preserving_order(categories)
    return sorted(unique, key=lambda value: order.get(value, len(order)))


def _classify_platform_chunk(chunk: str) -> List[str]:
    """Return canonical platform categories inferred from ``chunk``."""

    normalized = chunk.lower().strip()
    if not normalized:
        return []

    categories: List[str] = []

    def contains_all(*parts: str) -> bool:
        return all(part in normalized for part in parts)

    def contains_any(*parts: str) -> bool:
        return any(part in normalized for part in parts)

    if "toxoid" in normalized:
        categories.append("toxoid")

    if "conjugate" in normalized:
        categories.append("conjugate")

    if contains_any("mrna", " rna vaccine", "dna vaccine", " dna ", " rna ", "plasmid", "nucleic acid"):
        categories.append("dna_rna")

    if contains_any(
        "vector",
        "vectored",
        "viral vector",
        "recombinant vector",
        "carrier virus",
        "vrp",
        "replicon",
        "vaccinia",
        "adenovirus",
        "poxvirus",
        "canarypox",
        "vesicular stomatitis",
    ):
        categories.append("vector")

    if (
        contains_all("live", "attenuated")
        or "modified live" in normalized
        or "live virus" in normalized
        or "avirulent live" in normalized
        or "mlv" in normalized
    ):
        categories.append("live_attenuated")

    if contains_any(
        "inactivated",
        "killed",
        "bacterin",
        "split virion",
        "whole virion",
        "heat killed",
        "chemically inactivated",
        "protozoa",
    ):
        categories.append("inactivated")

    if contains_any(
        "subunit",
        "protein",
        "peptide",
        "virus-like particle",
        "vlp",
        "polysaccharide",
        "glycoprotein",
        "glycolipid",
        "recombinant",
        "synthetic",
        "capsular",
        "outer membrane",
        "surface antigen",
    ):
        categories.append("subunit")

    return _sorted_platform_categories(categories)


def _clean_platform_text(value: object) -> List[str]:
    """Split a raw platform string into analysable chunks."""

    normalized = _normalize_text_value(value)
    if pd.isna(normalized):
        return []
    text = str(normalized)
    text = text.replace("(", " ").replace(")", " ")
    text = text.replace("[", " ").replace("]", " ")
    chunks = PLATFORM_SPLIT_PATTERN.split(text)
    cleaned: List[str] = []
    for chunk in chunks:
        stripped = " ".join(chunk.split())
        if stripped:
            cleaned.append(stripped)
    if not cleaned:
        stripped = " ".join(text.split())
        return [stripped] if stripped else []
    return cleaned


def infer_platform_categories(row: pd.Series) -> Tuple[List[str], List[str]]:
    """Infer canonical platform categories and describe their provenance."""

    categories: List[str] = []
    sources: List[str] = []
    text_observed = False

    def extend_from_chunks(chunks: Iterable[str], source: str) -> None:
        nonlocal categories, sources, text_observed
        chunk_list = list(chunks)
        if chunk_list:
            text_observed = True
        new_categories: List[str] = []
        for chunk in chunk_list:
            new_categories.extend(_classify_platform_chunk(chunk))
        new_categories = _sorted_platform_categories(new_categories)
        if new_categories:
            categories.extend(new_categories)
            sources.append(source)
        elif chunk_list:
            sources.append(f"{source}: no canonical match")

    extend_from_chunks(_clean_platform_text(row.get("platform_type")), "platform_type")

    if not categories:
        extend_from_chunks(
            _clean_platform_text(row.get("vector_label")), "vector_label"
        )

    if not categories:
        extend_from_chunks(
            _clean_platform_text(row.get("detail_vectors")), "detail_vectors"
        )

    if not categories:
        extend_from_chunks(_clean_platform_text(row.get("vaccine_name")), "vaccine_name")

    if not categories:
        if text_observed:
            categories.append("other")
            if not sources:
                sources.append("platform metadata present but unclassified")
        else:
            categories.append("unspecified")
            sources.append("missing: no platform keywords identified")

    categories = _sorted_platform_categories(categories or ["unspecified"])
    sources = _dedupe_preserving_order(sources or ["platform_type"])

    return categories, sources
def load_vo_term_metadata(csv_path: Path) -> pd.DataFrame:
    """Load curated VO adjuvant metadata exported from term editing sheets."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Expected VO term metadata CSV at {csv_path!s}")

    vo_terms = read_csv_with_fallback(csv_path, skiprows=[1])

    rename_map = {
        "ID": "vo_term_id",
        "LABEL": "vo_preferred_label",
        "inSubset": "vo_subset",
        "Parent": "vo_parent",
        "Proposed parent for those under root": "vo_proposed_parent",
        "alternative label": "vo_alternative_labels",
        "definition": "vo_definition",
        "definition source": "vo_definition_source",
        "term editor": "vo_term_editors",
        "seeAlso": "vo_see_also",
        "editor note": "vo_editor_notes",
        "VAC adjuvant ID": "vac_adjuvant_id",
        "VAC adjuvant site": "vac_adjuvant_site",
        "term tracker item": "term_tracker_item",
        "has molecular receptor": "vo_molecular_receptors",
        "has role": "vo_roles",
        "induces immune profile": "vo_immune_profile",
        "Imm profile ref.": "vo_immune_profile_refs",
        "Imm profile editor": "vo_immune_profile_editor",
        "derives from": "vo_derives_from",
        "equivalent axioms": "vo_equivalent_axioms",
    }

    vo_terms = vo_terms.rename(columns=rename_map)

    for column in vo_terms.columns:
        vo_terms[column] = _normalize_text_series(vo_terms[column])

    pipe_columns = {
        "vo_alternative_labels",
        "vo_definition_source",
        "vo_term_editors",
        "vo_see_also",
        "vo_editor_notes",
        "vac_adjuvant_id",
        "vac_adjuvant_site",
        "term_tracker_item",
        "vo_molecular_receptors",
        "vo_roles",
        "vo_immune_profile",
        "vo_immune_profile_refs",
        "vo_immune_profile_editor",
        "vo_derives_from",
        "vo_equivalent_axioms",
    }

    for column in pipe_columns.intersection(vo_terms.columns):
        vo_terms[column] = vo_terms[column].map(_normalize_pipe_separated)

    vo_terms["vo_term_id"] = vo_terms["vo_term_id"].map(canonicalize_vo_identifier)
    vo_terms["adjuvant_vo_id"] = vo_terms["vo_term_id"]
    vo_terms["adjuvant_vo_id_underscore"] = vo_terms["adjuvant_vo_id"].map(
        underscore_vo_identifier
    )

    vo_terms = vo_terms.dropna(subset=["adjuvant_vo_id"])
    vo_terms = vo_terms.drop_duplicates(subset=["adjuvant_vo_id"], keep="first")

    ordered_columns = [
        "adjuvant_vo_id",
        "vo_term_id",
        "vo_preferred_label",
        "vo_alternative_labels",
        "vo_definition",
        "vo_definition_source",
        "vo_term_editors",
        "vo_see_also",
        "vo_editor_notes",
        "vo_subset",
        "vo_parent",
        "vo_proposed_parent",
        "vac_adjuvant_id",
        "vac_adjuvant_site",
        "term_tracker_item",
        "vo_molecular_receptors",
        "vo_roles",
        "vo_immune_profile",
        "vo_immune_profile_refs",
        "vo_immune_profile_editor",
        "vo_derives_from",
        "vo_equivalent_axioms",
    ]

    existing_columns = [column for column in ordered_columns if column in vo_terms.columns]
    remaining_columns = [
        column for column in vo_terms.columns if column not in existing_columns
    ]
    return vo_terms[existing_columns + remaining_columns]


def _unique_preserving_order(series: pd.Series) -> List[str]:
    """Return unique, non-empty strings in first-observed order."""

    seen: OrderedDict[str, None] = OrderedDict()
    for value in series:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text not in seen:
            seen[text] = None
    return list(seen.keys())


def _aggregate_text(series: pd.Series) -> str | pd.NA:
    """Aggregate a string column into a semi-colon separated list."""

    values = _unique_preserving_order(series)
    if not values:
        return pd.NA
    return "; ".join(values)


def build_vaccine_context_table(
    vaccines: pd.DataFrame,
    pathogens: pd.DataFrame,
    vaccine_detail: pd.DataFrame,
) -> pd.DataFrame:
    """Compile per-vaccine context fields (disease, platform, vectors)."""

    detail = (
        vaccine_detail.groupby("c_vaccine_id")
        .agg(
            {
                "c_antigen": _aggregate_text,
                "c_vector": _aggregate_text,
                "c_model_host": _aggregate_text,
            }
        )
        .rename(
            columns={
                "c_antigen": "detail_antigens",
                "c_vector": "detail_vectors",
                "c_model_host": "detail_model_hosts",
            }
        )
        .reset_index()
    )

    context = (
        vaccines[
            [
                "c_vaccine_id",
                "c_vaccine_name",
                "c_type",
                "c_vector",
                "c_pathogen_id",
                "c_vo_id",
            ]
        ]
        .merge(
            pathogens[
                ["c_pathogen_id", "c_disease_name", "c_pathogen_name"]
            ],
            how="left",
            on="c_pathogen_id",
        )
        .merge(detail, how="left", on="c_vaccine_id")
        .rename(
            columns={
                "c_vaccine_id": "vaccine_id",
                "c_vaccine_name": "vaccine_name",
                "c_type": "platform_type",
                "c_vector": "vector_label",
                "c_pathogen_id": "pathogen_id",
                "c_vo_id": "vaccine_vo_id",
                "c_disease_name": "disease_name",
                "c_pathogen_name": "pathogen_name",
            }
        )
    )

    for column in [
        "vaccine_name",
        "platform_type",
        "vector_label",
        "disease_name",
        "pathogen_name",
    ]:
        if column in context.columns:
            context[column] = _normalize_text_series(context[column])

    return context


def build_adjuvant_metadata(
    adjuvants: pd.DataFrame,
    vaxjo: pd.DataFrame,
    vaxvec: pd.DataFrame,
    vo_terms: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge adjuvant usage rows with Vaxjo/Vaxvec descriptors."""

    vaxjo_subset = (
        vaxjo[
            [
                "c_vaxjo_vo_id",
                "c_vaxjo_id",
                "c_vaxjo_name",
                "c_function",
                "c_description",
                "c_components",
                "c_stage_dev",
            ]
        ]
        .rename(
            columns={
                "c_vaxjo_vo_id": "vaxjo_vo_id",
                "c_vaxjo_id": "vaxjo_id",
                "c_vaxjo_name": "vaxjo_name",
                "c_function": "vaxjo_function",
                "c_description": "vaxjo_description",
                "c_components": "vaxjo_components",
                "c_stage_dev": "vaxjo_stage",
            }
        )
        .assign(
            adjuvant_vo_id=lambda df: df["vaxjo_vo_id"].map(canonicalize_vo_identifier),
            vaxjo_vo_id_underscore=lambda df: df["adjuvant_vo_id"].map(
                underscore_vo_identifier
            ),
        )
    )
    vaxjo_subset = vaxjo_subset.dropna(subset=["adjuvant_vo_id"])
    vaxjo_subset = vaxjo_subset.drop_duplicates(
        subset=["adjuvant_vo_id"], keep="first"
    )

    vaxvec_subset = (
        vaxvec[
            [
                "c_vaxvec_vo_id",
                "c_vaxvec_id",
                "c_vaxvec_name",
                "c_function",
                "c_description",
            ]
        ]
        .rename(
            columns={
                "c_vaxvec_vo_id": "vaxvec_vo_id",
                "c_vaxvec_id": "vaxvec_id",
                "c_vaxvec_name": "vaxvec_name",
                "c_function": "vaxvec_function",
                "c_description": "vaxvec_description",
            }
        )
        .assign(
            adjuvant_vo_id=lambda df: df["vaxvec_vo_id"].map(canonicalize_vo_identifier),
            vaxvec_vo_id_underscore=lambda df: df["adjuvant_vo_id"].map(
                underscore_vo_identifier
            ),
        )
    )
    vaxvec_subset = vaxvec_subset.dropna(subset=["adjuvant_vo_id"])
    vaxvec_subset = vaxvec_subset.drop_duplicates(
        subset=["adjuvant_vo_id"], keep="first"
    )

    adjuvant_meta = (
        adjuvants[
            [
                "c_adjuvant_id",
                "c_adjuvant_vo_id",
                "c_adjuvant_label",
                "c_adjuvant_description",
                "c_vaccine_id",
                "c_curation_flag",
            ]
        ]
        .rename(
            columns={
                "c_adjuvant_id": "adjuvant_record_id",
                "c_adjuvant_vo_id": "adjuvant_vo_id",
                "c_adjuvant_label": "adjuvant_label",
                "c_adjuvant_description": "adjuvant_description",
                "c_vaccine_id": "vaccine_id",
                "c_curation_flag": "adjuvant_curation_flag",
            }
        )
        .assign(
            adjuvant_vo_id=lambda df: df["adjuvant_vo_id"].map(
                canonicalize_vo_identifier
            ),
        )
        .assign(
            adjuvant_vo_id_underscore=lambda df: df["adjuvant_vo_id"].map(
                underscore_vo_identifier
            )
        )
        .merge(vaxjo_subset, how="left", on="adjuvant_vo_id")
        .merge(
            vaxvec_subset,
            how="left",
            on="adjuvant_vo_id",
            suffixes=("", "_vector"),
        )
    )

    text_columns = [
        "adjuvant_label",
        "adjuvant_description",
        "vaxjo_name",
        "vaxjo_description",
        "vaxvec_name",
        "vaxvec_description",
    ]
    for column in text_columns:
        if column in adjuvant_meta.columns:
            adjuvant_meta[column] = _normalize_text_series(adjuvant_meta[column])

    if vo_terms is not None:
        adjuvant_meta = adjuvant_meta.merge(
            vo_terms, on="adjuvant_vo_id", how="left"
        )

    if "vo_preferred_label" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_label"] = adjuvant_meta["adjuvant_label"].combine_first(
            adjuvant_meta["vo_preferred_label"]
        )

    if "vaxjo_name" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_label"] = adjuvant_meta["adjuvant_label"].combine_first(
            adjuvant_meta["vaxjo_name"]
        )
    if "vaxvec_name" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_label"] = adjuvant_meta["adjuvant_label"].combine_first(
            adjuvant_meta["vaxvec_name"]
        )

    if "vo_definition" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_description"] = adjuvant_meta[
            "adjuvant_description"
        ].combine_first(adjuvant_meta["vo_definition"])
    if "vaxjo_description" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_description"] = adjuvant_meta[
            "adjuvant_description"
        ].combine_first(adjuvant_meta["vaxjo_description"])
    if "vaxvec_description" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_description"] = adjuvant_meta[
            "adjuvant_description"
        ].combine_first(adjuvant_meta["vaxvec_description"])

    adjuvant_meta["adjuvant_display_name"] = adjuvant_meta["adjuvant_label"]
    if "vo_preferred_label" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_display_name"] = adjuvant_meta[
            "adjuvant_display_name"
        ].combine_first(adjuvant_meta["vo_preferred_label"])
    if "vaxjo_name" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_display_name"] = adjuvant_meta[
            "adjuvant_display_name"
        ].combine_first(adjuvant_meta["vaxjo_name"])
    if "vaxvec_name" in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_display_name"] = adjuvant_meta[
            "adjuvant_display_name"
        ].combine_first(adjuvant_meta["vaxvec_name"])

    if "vo_alternative_labels" in adjuvant_meta.columns and "adjuvant_synonyms" not in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_synonyms"] = adjuvant_meta["vo_alternative_labels"]
    if "vo_immune_profile" in adjuvant_meta.columns and "adjuvant_immune_profile" not in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_immune_profile"] = adjuvant_meta["vo_immune_profile"]
    if "vo_roles" in adjuvant_meta.columns and "adjuvant_roles" not in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_roles"] = adjuvant_meta["vo_roles"]
    if "vo_molecular_receptors" in adjuvant_meta.columns and "adjuvant_molecular_receptors" not in adjuvant_meta.columns:
        adjuvant_meta["adjuvant_molecular_receptors"] = adjuvant_meta[
            "vo_molecular_receptors"
        ]

    underscore_cols = [
        column
        for column in adjuvant_meta.columns
        if column.startswith("adjuvant_vo_id_underscore")
    ]
    if underscore_cols:
        combined = adjuvant_meta[underscore_cols[0]].copy()
        for column in underscore_cols[1:]:
            combined = combined.combine_first(adjuvant_meta[column])
        adjuvant_meta["adjuvant_vo_id_underscore"] = combined
        for column in underscore_cols:
            if column != "adjuvant_vo_id_underscore":
                adjuvant_meta = adjuvant_meta.drop(columns=column)

    return adjuvant_meta


def build_training_dataset(
    tables: Dict[str, pd.DataFrame], vo_terms: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, Dict[str, int], pd.DataFrame]:
    """Return the fully-joined training dataset and summary statistics."""

    context = build_vaccine_context_table(
        tables["vaccines"], tables["pathogens"], tables["vaccine_detail"]
    )
    adjuvant_meta = build_adjuvant_metadata(
        tables["adjuvants"], tables["vaxjo"], tables["vaxvec"], vo_terms
    )

    dataset = (
        adjuvant_meta.merge(context, on="vaccine_id", how="left")
        .sort_values(["pathogen_id", "vaccine_id", "adjuvant_vo_id"])
        .reset_index(drop=True)
    )

    dataset["disease_context_source"] = "pathogen_table"
    missing_initial = dataset["disease_name"].isna()
    dataset.loc[missing_initial, "disease_context_source"] = (
        "missing: pathogen table lacks disease_name"
    )

    duplicate_mask = dataset["adjuvant_vo_id"].notna() & dataset.duplicated(
        subset=["vaccine_id", "adjuvant_vo_id"], keep="first"
    )
    duplicate_pairs = int(duplicate_mask.sum())
    if duplicate_pairs:
        dataset = dataset.loc[~duplicate_mask].reset_index(drop=True)

    platform_info = dataset.apply(infer_platform_categories, axis=1, result_type="expand")
    dataset["_platform_categories"] = platform_info[0]
    dataset["platform_context_source"] = platform_info[1].map(
        lambda values: "; ".join(_dedupe_preserving_order(values)) if values else pd.NA
    )

    dataset["_platform_categories"] = dataset["_platform_categories"].map(
        lambda values: values if isinstance(values, list) and values else ["unspecified"]
    )

    pre_explode_rows = len(dataset)
    dataset = dataset.explode("_platform_categories").reset_index(drop=True)
    dataset = dataset.rename(columns={"_platform_categories": "platform_group"})
    dataset["platform_group"] = dataset["platform_group"].astype(str)
    dataset = dataset.sort_values(
        ["pathogen_id", "vaccine_id", "adjuvant_vo_id", "platform_group"],
        kind="mergesort",
    ).reset_index(drop=True)

    summary = {
        "deduplicated_pairs": duplicate_pairs,
        "rows_before_platform_split": pre_explode_rows,
        "rows_after_platform_split": len(dataset),
    }

    return dataset, summary, adjuvant_meta


def analyze_missing_diseases(
    dataset: pd.DataFrame, pathogens: pd.DataFrame
) -> List[Dict[str, object]]:
    """Summarize the rows where ``disease_name`` is missing."""

    missing = dataset[dataset["disease_name"].isna()]
    if missing.empty:
        return []

    report: List[Dict[str, object]] = []
    for pathogen_id, group in missing.groupby("pathogen_id", dropna=False):
        entry: Dict[str, object] = {"rows": int(len(group))}
        if pd.isna(pathogen_id):
            entry.update(
                {
                    "pathogen_id": None,
                    "pathogen_name": None,
                    "reason": (
                        "Vaccine metadata missing from t_vaccine.csv; the adjuvant row "
                        "does not provide pathogen context."
                    ),
                }
            )
        else:
            entry["pathogen_id"] = int(pathogen_id)
            pathogen_names = group["pathogen_name"].dropna().unique()
            entry["pathogen_name"] = pathogen_names[0] if len(pathogen_names) else None
            source_row = pathogens[pathogens["c_pathogen_id"] == int(pathogen_id)]
            if source_row.empty:
                entry["reason"] = "Pathogen ID not present in t_pathogen.csv."
            elif source_row["c_disease_name"].notna().any():
                entry["reason"] = (
                    "Upstream pathogen record includes a disease name but the join "
                    "returned NaN."
                )
            else:
                entry["reason"] = "Pathogen record exists but c_disease_name is blank."
        report.append(entry)

    def _sort_key(item: Dict[str, object]) -> Tuple[int, int]:
        pid = item.get("pathogen_id")
        if pid is None:
            return (1, 0)
        return (0, int(pid))

    report.sort(key=_sort_key)
    return report


def normalize_disease_annotations(
    dataset: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, object]]]:
    """Fill in disease names using curated overrides and fallbacks."""

    dataset = dataset.copy()
    applied: List[Dict[str, object]] = []
    context_col = "disease_context_source" if "disease_context_source" in dataset.columns else None

    for pathogen_id, fix in PATHOGEN_DISEASE_CORRECTIONS.items():
        mask = dataset["pathogen_id"] == pathogen_id
        if not mask.any():
            continue

        disease_mask = mask & dataset["disease_name"].isna()
        pathogen_mask = mask & dataset["pathogen_name"].isna()

        disease_updates = 0
        pathogen_updates = 0

        if "disease_name" in fix and disease_mask.any():
            dataset.loc[disease_mask, "disease_name"] = fix["disease_name"]
            disease_updates = int(disease_mask.sum())
            if context_col:
                dataset.loc[disease_mask, context_col] = "curated_override"

        if "pathogen_name" in fix and pathogen_mask.any():
            dataset.loc[pathogen_mask, "pathogen_name"] = fix["pathogen_name"]
            pathogen_updates = int(pathogen_mask.sum())

        if disease_updates or pathogen_updates:
            applied.append(
                {
                    "pathogen_id": pathogen_id,
                    "rows": int(mask.sum()),
                    "filled_disease_name": disease_updates,
                    "filled_pathogen_name": pathogen_updates,
                    "note": fix.get("note"),
                }
            )

    fallback_mask = dataset["disease_name"].isna() & dataset["pathogen_name"].notna()
    if fallback_mask.any():
        dataset.loc[fallback_mask, "disease_name"] = dataset.loc[
            fallback_mask, "pathogen_name"
        ]
        if context_col:
            dataset.loc[fallback_mask, context_col] = "pathogen_name_fallback"
        applied.append(
            {
                "pathogen_id": None,
                "rows": int(fallback_mask.sum()),
                "filled_disease_name": int(fallback_mask.sum()),
                "filled_pathogen_name": 0,
                "note": "Filled disease_name from pathogen_name fallback.",
            }
        )

    dataset["disease_name"] = dataset["disease_name"].apply(
        lambda value: value.strip() if isinstance(value, str) else value
    )

    return dataset, applied


def _normalize_group_value(value: object) -> object:
    """Prepare grouping keys for JSON serialization."""

    if pd.isna(value):
        return None
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, float):
        if value.is_integer():
            return int(value)
        return float(value)
    return value


def report_missing_values(
    dataset: pd.DataFrame, columns: Sequence[str]
) -> Dict[str, int]:
    """Print and return missing-value counts for selected columns."""

    missing_counts = dataset[list(columns)].isna().sum()
    summary = {column: int(missing_counts[column]) for column in columns}

    print("Missing values in training_samples.csv (selected columns):")
    for column in columns:
        print(f"  {column}: {summary[column]}")

    total_missing_cells = int(dataset.isna().sum().sum())
    print(f"  Total missing cells across dataset: {total_missing_cells}")

    return summary


def _candidate_records(
    dataset: pd.DataFrame, group_cols: Iterable[str]
) -> List[Dict[str, object]]:
    """Helper to build candidate lists keyed by ``group_cols``."""

    records: List[Dict[str, object]] = []
    grouping = dataset.groupby(list(group_cols), dropna=False)

    for keys, group in grouping:
        if not isinstance(keys, tuple):
            keys = (keys,)
        seen: OrderedDict[str, Dict[str, object]] = OrderedDict()
        for _, row in group.iterrows():
            adjuvant_vo = row["adjuvant_vo_id"]
            if pd.isna(adjuvant_vo):
                continue
            adjuvant_vo = str(adjuvant_vo)
            if adjuvant_vo not in seen:
                display = row.get("adjuvant_display_name")
                if pd.isna(display):
                    display = None
                elif isinstance(display, str):
                    display = display.strip() or None
                seen[adjuvant_vo] = {
                    "adjuvant_vo_id": adjuvant_vo,
                    "display_name": display,
                }
        if not seen:
            continue
        record = {
            col: _normalize_group_value(keys[idx]) for idx, col in enumerate(group_cols)
        }
        if all(value is None for value in record.values()):
            continue
        record["adjuvant_candidates"] = list(seen.values())
        records.append(record)
    return records


def export_adjuvant_metadata(
    adjuvant_meta: pd.DataFrame, output_dir: Path
) -> Path:
    """Write an enriched adjuvant metadata table to ``output_dir``."""

    columns = [
        "adjuvant_record_id",
        "adjuvant_vo_id",
        "adjuvant_vo_id_underscore",
        "adjuvant_label",
        "adjuvant_description",
        "adjuvant_display_name",
        "adjuvant_synonyms",
        "adjuvant_immune_profile",
        "adjuvant_roles",
        "adjuvant_molecular_receptors",
        "vo_term_id",
        "vo_preferred_label",
        "vo_definition",
        "vo_alternative_labels",
        "vo_immune_profile",
        "vo_roles",
        "vo_molecular_receptors",
        "vac_adjuvant_id",
        "vac_adjuvant_site",
    ]

    export_columns = [column for column in columns if column in adjuvant_meta.columns]
    export_frame = adjuvant_meta[export_columns].copy()
    export_frame = export_frame.sort_values(
        ["adjuvant_vo_id", "adjuvant_record_id"], kind="mergesort"
    ).reset_index(drop=True)

    path = output_dir / "adjuvant_metadata_enriched.csv"
    export_frame.to_csv(path, index=False)
    return path


def export_outputs(
    dataset: pd.DataFrame, adjuvant_meta: pd.DataFrame, output_dir: Path
) -> None:
    """Persist the assembled dataset, candidate lists, and adjuvant metadata."""

    output_dir.mkdir(parents=True, exist_ok=True)

    training_path = output_dir / "training_samples.csv"
    dataset.to_csv(training_path, index=False)

    export_adjuvant_metadata(adjuvant_meta, output_dir)

    disease_candidates = _candidate_records(dataset, ["pathogen_id", "disease_name"])
    disease_platform_candidates = _candidate_records(
        dataset, ["pathogen_id", "disease_name", "platform_group"]
    )

    (output_dir / "disease_adjuvant_candidates.json").write_text(
        json.dumps(
            disease_candidates, indent=2, ensure_ascii=False, allow_nan=False
        )
    )
    (output_dir / "disease_platform_adjuvant_candidates.json").write_text(
        json.dumps(
            disease_platform_candidates,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the exported CSV tables (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory where derived datasets will be written",
    )
    parser.add_argument(
        "--vo-terms-path",
        type=Path,
        default=DEFAULT_VO_TERMS_PATH,
        help=(
            "CSV file containing curated VO adjuvant metadata used to enrich "
            "labels and descriptions (default: %(default)s)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = load_source_tables(args.data_dir)
    vo_terms: Optional[pd.DataFrame] = None
    if args.vo_terms_path is not None:
        vo_terms = load_vo_term_metadata(args.vo_terms_path)
        print(
            f"Loaded {len(vo_terms):,} VO adjuvant term rows from {args.vo_terms_path}"
        )

    dataset, dataset_summary, adjuvant_meta = build_training_dataset(tables, vo_terms)

    if dataset_summary.get("deduplicated_pairs"):
        print(
            f"Dropped {dataset_summary['deduplicated_pairs']} duplicate vaccine/adjuvant pair(s) before export."
        )
    else:
        print("No duplicate vaccine/adjuvant pairs detected in source tables.")

    rows_before = dataset_summary.get("rows_before_platform_split")
    rows_after = dataset_summary.get("rows_after_platform_split")
    if rows_before is not None and rows_after is not None and rows_after != rows_before:
        print(
            "Normalized multi-platform annotations into canonical categories: "
            f"{rows_before} → {rows_after} training rows."
        )

    missing_before = analyze_missing_diseases(dataset, tables["pathogens"])
    if missing_before:
        print("Missing disease_name diagnostics before normalization:")
        for entry in missing_before:
            pid = entry["pathogen_id"]
            label = f"pathogen_id={pid}" if pid is not None else "pathogen_id=<missing>"
            pathogen_name = entry.get("pathogen_name")
            if pathogen_name:
                label = f"{label} ({pathogen_name})"
            print(f"  - {label}: {entry['rows']} row(s) -> {entry['reason']}")
    else:
        print("No missing disease_name values detected before normalization.")

    dataset, applied_fixes = normalize_disease_annotations(dataset)
    if applied_fixes:
        print("Applied disease name corrections:")
        for fix in applied_fixes:
            pid = fix["pathogen_id"]
            label = (
                "pathogen_name fallback" if pid is None else f"pathogen_id={pid}"
            )
            details = []
            disease_count = fix.get("filled_disease_name", 0)
            if disease_count:
                details.append(f"disease_name ({disease_count} row(s))")
            pathogen_count = fix.get("filled_pathogen_name", 0)
            if pathogen_count:
                details.append(f"pathogen_name ({pathogen_count} row(s))")
            detail_text = "; ".join(details) if details else "no updates applied"
            note = fix.get("note")
            if note:
                detail_text = f"{detail_text} [{note}]"
            print(f"  - {label}: {detail_text}")

    missing_after = analyze_missing_diseases(dataset, tables["pathogens"])
    if missing_after:
        print("Remaining disease_name gaps after normalization:")
        for entry in missing_after:
            pid = entry["pathogen_id"]
            label = f"pathogen_id={pid}" if pid is not None else "pathogen_id=<missing>"
            pathogen_name = entry.get("pathogen_name")
            if pathogen_name:
                label = f"{label} ({pathogen_name})"
            print(f"  - {label}: {entry['rows']} row(s) -> {entry['reason']}")
            if "disease_context_source" in dataset.columns:
                if pid is None:
                    mask = dataset["pathogen_id"].isna()
                else:
                    mask = dataset["pathogen_id"] == pid
                mask &= dataset["disease_name"].isna()
                dataset.loc[mask, "disease_context_source"] = (
                    f"missing: {entry['reason']}"
                )
    else:
        print("All disease_name values resolved after normalization.")

    export_outputs(dataset, adjuvant_meta, args.output_dir)

    report_missing_values(dataset, MISSING_VALUE_SUMMARY_COLUMNS)

    print(f"Wrote {len(dataset):,} training rows to {args.output_dir/'training_samples.csv'}")
    print(
        "Wrote enriched adjuvant metadata snapshot to "
        f"{args.output_dir/'adjuvant_metadata_enriched.csv'}"
    )
    print(
        "Candidate JSON files prepared for disease-level and disease+platform-level recommendations."
    )


if __name__ == "__main__":
    main()

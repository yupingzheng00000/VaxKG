"""Assemble vaccine-adjuvant training data from the VaxKG CSV exports.

The script rebuilds the joins that the Neo4j loader performs so we can
prototype models straight from the relational snapshots.  It produces:

* ``training_samples.csv`` – one row per curated vaccine/adjuvant pair with
  disease, platform, and metadata fields that are useful for feature
  engineering.
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
        normalized = " ".join(value.replace("\r", " ").split())
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

    vo_terms["adjuvant_vo_id"] = vo_terms["vo_term_id"].map(
        lambda value: value.replace(":", "_") if isinstance(value, str) else pd.NA
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

    return context


def build_adjuvant_metadata(
    adjuvants: pd.DataFrame,
    vaxjo: pd.DataFrame,
    vaxvec: pd.DataFrame,
    vo_terms: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge adjuvant usage rows with Vaxjo/Vaxvec descriptors."""

    vaxjo_subset = vaxjo[
        [
            "c_vaxjo_vo_id",
            "c_vaxjo_id",
            "c_vaxjo_name",
            "c_function",
            "c_description",
            "c_components",
            "c_stage_dev",
        ]
    ].rename(
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

    vaxvec_subset = vaxvec[
        [
            "c_vaxvec_vo_id",
            "c_vaxvec_id",
            "c_vaxvec_name",
            "c_function",
            "c_description",
        ]
    ].rename(
        columns={
            "c_vaxvec_vo_id": "vaxvec_vo_id",
            "c_vaxvec_id": "vaxvec_id",
            "c_vaxvec_name": "vaxvec_name",
            "c_function": "vaxvec_function",
            "c_description": "vaxvec_description",
        }
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
        .merge(
            vaxjo_subset,
            how="left",
            left_on="adjuvant_vo_id",
            right_on="vaxjo_vo_id",
        )
        .merge(
            vaxvec_subset,
            how="left",
            left_on="adjuvant_vo_id",
            right_on="vaxvec_vo_id",
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

    return adjuvant_meta


def build_training_dataset(
    tables: Dict[str, pd.DataFrame], vo_terms: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """Return the fully-joined training dataset."""

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

    dataset["platform_group"] = (
        dataset["platform_type"].fillna("Unspecified").astype(str).str.strip()
    )
    dataset.loc[dataset["platform_group"] == "", "platform_group"] = "Unspecified"

    return dataset


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


def export_outputs(dataset: pd.DataFrame, output_dir: Path) -> None:
    """Persist the assembled dataset and candidate lists."""

    output_dir.mkdir(parents=True, exist_ok=True)

    training_path = output_dir / "training_samples.csv"
    dataset.to_csv(training_path, index=False)

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

    dataset = build_training_dataset(tables, vo_terms)

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
    else:
        print("All disease_name values resolved after normalization.")

    export_outputs(dataset, args.output_dir)

    report_missing_values(dataset, MISSING_VALUE_SUMMARY_COLUMNS)

    print(f"Wrote {len(dataset):,} training rows to {args.output_dir/'training_samples.csv'}")
    print(
        "Candidate JSON files prepared for disease-level and disease+platform-level recommendations."
    )


if __name__ == "__main__":
    main()

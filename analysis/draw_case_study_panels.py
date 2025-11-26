"""Generate the four-panel case-study figure from exporter outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence

import matplotlib

# Use a non-interactive backend so the script can run on headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_run(summary: Mapping[str, object], label: str) -> Mapping[str, object]:
    runs = summary.get("runs")
    if not isinstance(runs, Mapping):
        raise KeyError("Summary JSON is missing the 'runs' mapping.")
    run = runs.get(label)
    if run is None:
        available = ", ".join(sorted(map(str, runs.keys())))
        raise KeyError(f"Run label '{label}' not found. Available labels: {available}")
    if not isinstance(run, Mapping):
        raise TypeError(f"Run entry for label '{label}' must be a mapping.")
    return run


def _sorted_topk_keys(run_metrics: Mapping[str, object]) -> List[int]:
    topk = run_metrics.get("topk", {})
    if not isinstance(topk, Mapping):
        return []
    result: List[int] = []
    for key in topk.keys():
        try:
            result.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(result)


def _plot_topk_panel(ax: plt.Axes, run_metrics: Mapping[str, object]) -> None:
    topk_list = run_metrics.get("topk_list", [])
    if not isinstance(topk_list, Sequence) or len(topk_list) == 0:
        ax.text(0.5, 0.5, "No Top-10 entries available", ha="center", va="center")
        ax.set_axis_off()
        return

    labels: List[str] = []
    scores: List[float] = []
    relevance: List[int] = []
    for entry in topk_list:
        if not isinstance(entry, Mapping):
            continue
        labels.append(str(entry.get("adjuvant_label", entry.get("adjuvant_id", "?"))))
        try:
            scores.append(float(entry.get("score", 0.0)))
        except (TypeError, ValueError):
            scores.append(0.0)
        relevance.append(int(entry.get("relevance_binary", 0)))

    if not labels:
        ax.text(0.5, 0.5, "No Top-10 entries available", ha="center", va="center")
        ax.set_axis_off()
        return

    y_positions = np.arange(len(labels))
    colors = ["#1b9e77" if rel else "#cccccc" for rel in relevance]
    ax.barh(y_positions, scores, color=colors)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"#{rank + 1} {label}" for rank, label in enumerate(labels)])
    ax.invert_yaxis()
    ax.set_xlabel("Model score")
    ax.set_title("Panel A – Top-10 ranking")

    for idx, (score, rel) in enumerate(zip(scores, relevance)):
        if rel:
            ax.text(
                score,
                idx,
                " ✓",
                ha="left",
                va="center",
                color="#1b9e77",
                fontsize=10,
                fontweight="bold",
            )

    topk_metrics = run_metrics.get("topk", {})
    if isinstance(topk_metrics, Mapping):
        lines: List[str] = []
        for k in _sorted_topk_keys(run_metrics):
            metrics = topk_metrics.get(str(k), {})
            if not isinstance(metrics, Mapping):
                continue
            precision = metrics.get("precision")
            recall = metrics.get("recall")
            ndcg = metrics.get("ndcg")
            if isinstance(precision, (int, float)):
                lines.append(f"P@{k}: {precision:.3f}")
            if isinstance(recall, (int, float)):
                lines.append(f"R@{k}: {recall:.3f}")
            if isinstance(ndcg, (int, float)):
                lines.append(f"NDCG@{k}: {ndcg:.3f}")
        if lines:
            ax.text(
                0.98,
                0.02,
                "\n".join(lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8),
            )


def _plot_dcg_panel(ax: plt.Axes, run_metrics: Mapping[str, object]) -> None:
    dcg_curve = run_metrics.get("dcg_curve", [])
    idcg_curve = run_metrics.get("idcg_curve", [])
    ndcg_curve = run_metrics.get("ndcg_curve_k", [])
    if not isinstance(dcg_curve, Sequence) or len(dcg_curve) == 0:
        ax.text(0.5, 0.5, "No DCG data", ha="center", va="center")
        ax.set_axis_off()
        return

    ranks = np.arange(1, len(dcg_curve) + 1)
    ax.plot(ranks, dcg_curve, label="DCG", color="#4c72b0", marker="o")
    if isinstance(idcg_curve, Sequence) and len(idcg_curve) == len(ranks):
        ax.plot(ranks, idcg_curve, label="IDCG", color="#55a868", marker="s")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Cumulative gain")
    ax.set_title("Panel B – DCG / IDCG / NDCG")

    ax2 = ax.twinx()
    if isinstance(ndcg_curve, Sequence) and len(ndcg_curve) == len(ranks):
        ax2.plot(ranks, ndcg_curve, label="NDCG", color="#c44e52", linestyle="--")
        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("NDCG")
        final_ndcg = run_metrics.get("ndcg@curve_k")
        if isinstance(final_ndcg, (int, float)):
            ax2.text(
                0.98,
                0.05,
                f"NDCG@{len(ranks)} = {final_ndcg:.3f}",
                transform=ax2.transAxes,
                ha="right",
                va="bottom",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8),
            )

    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    if handles2:
        handles.extend(handles2)
        labels.extend(labels2)
    if handles:
        ax.legend(handles, labels, loc="upper left")


def _plot_reliability_panel(ax: plt.Axes, run_metrics: Mapping[str, object]) -> None:
    reliability = run_metrics.get("reliability", {})
    bins = reliability.get("bins") if isinstance(reliability, Mapping) else None
    mode = reliability.get("mode", "unknown") if isinstance(reliability, Mapping) else "unknown"
    if not isinstance(bins, Sequence) or len(bins) == 0:
        ax.text(0.5, 0.5, "Reliability data unavailable", ha="center", va="center")
        ax.set_axis_off()
        return

    confidences: List[float] = []
    precisions: List[float] = []
    counts: List[int] = []
    for entry in bins:
        if not isinstance(entry, Mapping):
            continue
        confidences.append(float(entry.get("confidence", 0.0)))
        precisions.append(float(entry.get("precision", 0.0)))
        counts.append(int(entry.get("count", 0)))

    if not confidences:
        ax.text(0.5, 0.5, "Reliability data unavailable", ha="center", va="center")
        ax.set_axis_off()
        return

    sizes = 30 + 5 * np.array(counts)
    ax.plot([0, 1], [0, 1], color="#444444", linestyle="--", linewidth=1, label="Perfect calibration")
    ax.scatter(confidences, precisions, s=sizes, color="#4c72b0", alpha=0.8, label="Bins")
    for conf, prec, count in zip(confidences, precisions, counts):
        ax.text(conf, prec, str(count), fontsize=8, ha="center", va="bottom")

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Panel C – Reliability diagram")
    ece = run_metrics.get("ece")
    if isinstance(ece, (int, float)):
        ax.text(
            0.02,
            0.98,
            f"ECE = {ece:.3f}\nMode = {mode}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8),
        )

    ax.legend(loc="upper left")


def _compute_rank_swaps(
    df: pd.DataFrame, primary_label: str, baseline_label: Optional[str]
) -> List[Dict[str, object]]:
    if baseline_label is None:
        return []
    primary_col = f"rank_{primary_label}"
    baseline_col = f"rank_{baseline_label}"
    if primary_col not in df.columns or baseline_col not in df.columns:
        return []

    relevant = df[df.get("relevance_binary", 0) > 0]
    swaps: List[Dict[str, object]] = []
    for _, row in relevant.iterrows():
        baseline_rank = row.get(baseline_col)
        target_rank = row.get(primary_col)
        if pd.isna(baseline_rank) or pd.isna(target_rank):
            continue
        baseline_rank = int(baseline_rank)
        target_rank = int(target_rank)
        label = row.get("adjuvant_label")
        if pd.isna(label) or label is None or str(label).strip() == "":
            label = row.get("adjuvant_vo_id", "Adjuvant")
        swaps.append(
            {
                "label": str(label),
                "baseline_rank": baseline_rank,
                "target_rank": target_rank,
                "delta": baseline_rank - target_rank,
            }
        )

    swaps.sort(key=lambda item: (item["delta"], -item["target_rank"]), reverse=True)
    return swaps


def _plot_rank_swap_panel(
    ax: plt.Axes,
    summary: Mapping[str, object],
    df: Optional[pd.DataFrame],
    primary_label: str,
    baseline_label: Optional[str],
) -> None:
    if df is None or baseline_label is None:
        ax.text(0.5, 0.5, "No comparison data available", ha="center", va="center")
        ax.set_axis_off()
        return

    swaps = _compute_rank_swaps(df, primary_label, baseline_label)
    if not swaps:
        ax.text(0.5, 0.5, "No relevant adjuvants with baseline ranks", ha="center", va="center")
        ax.set_axis_off()
        return

    labels = [item["label"] for item in swaps]
    deltas = [item["delta"] for item in swaps]
    y_positions = np.arange(len(labels))
    colours = ["#1b9e77" if delta > 0 else ("#d95f02" if delta < 0 else "#7570b3") for delta in deltas]

    ax.barh(y_positions, deltas, color=colours)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Δ rank (baseline − target)")
    ax.axvline(0, color="#444444", linewidth=1)
    ax.invert_yaxis()
    ax.set_title("Panel D – Rank swap ablation")

    for idx, delta in enumerate(deltas):
        ax.text(
            delta,
            idx,
            f" {delta:+d}",
            ha="left" if delta >= 0 else "right",
            va="center",
            color="black",
        )

    delta_ndcg = summary.get("delta_ndcg")
    if isinstance(delta_ndcg, Mapping) and delta_ndcg:
        lines = [f"{key}: {value:+.3f}" for key, value in sorted(delta_ndcg.items())]
        ax.text(
            0.98,
            0.02,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="none", alpha=0.8),
        )


def create_case_study_panels(
    summary_path: Path,
    output_path: Path,
    candidates_csv: Optional[Path] = None,
    primary_label: str = "target",
    baseline_label: Optional[str] = "baseline",
) -> Path:
    with Path(summary_path).open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if not isinstance(summary, MutableMapping):
        raise TypeError("Summary JSON must contain a mapping at the top level.")

    run_metrics = _ensure_run(summary, primary_label)
    if baseline_label is not None:
        try:
            _ensure_run(summary, baseline_label)
        except KeyError:
            baseline_label = None

    df: Optional[pd.DataFrame] = None
    if candidates_csv is not None and Path(candidates_csv).exists():
        df = pd.read_csv(candidates_csv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax_topk, ax_dcg, ax_rel, ax_rank = axes.flatten()

    _plot_topk_panel(ax_topk, run_metrics)
    _plot_dcg_panel(ax_dcg, run_metrics)
    _plot_reliability_panel(ax_rel, run_metrics)
    _plot_rank_swap_panel(ax_rank, summary, df, primary_label, baseline_label)

    disease = summary.get("disease_key", "Unknown disease")
    fig.suptitle(f"Case study panels – {disease}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-json",
        type=Path,
        required=True,
        help="Metrics summary JSON produced by export_case_study.py",
    )
    parser.add_argument(
        "--candidates-csv",
        type=Path,
        default=None,
        help="Candidate table CSV produced by export_case_study.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination image path (PNG, PDF, etc.)",
    )
    parser.add_argument(
        "--primary-label",
        type=str,
        default="target",
        help="Label of the primary run inside the summary JSON",
    )
    parser.add_argument(
        "--baseline-label",
        type=str,
        default="baseline",
        help="Optional label of the comparison run for Panel D",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    baseline_label: Optional[str] = args.baseline_label
    if baseline_label is not None and baseline_label.strip() == "":
        baseline_label = None
    create_case_study_panels(
        summary_path=args.summary_json,
        output_path=args.output,
        candidates_csv=args.candidates_csv,
        primary_label=args.primary_label,
        baseline_label=baseline_label,
    )


if __name__ == "__main__":
    main()


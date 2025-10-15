from __future__ import annotations

import json
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from analysis.draw_case_study_panels import _compute_rank_swaps, create_case_study_panels


def _build_sample_summary(tmp_path: Path) -> Path:
    summary = {
        "disease_key": "COVID-19",
        "runs": {
            "target": {
                "topk": {
                    "5": {"precision": 0.2, "recall": 0.2, "ndcg": 0.3},
                    "10": {"precision": 0.1, "recall": 0.5, "ndcg": 0.4},
                },
                "dcg_curve": [1.0, 1.2, 1.2, 1.2, 1.2],
                "idcg_curve": [1.0, 1.5, 1.7, 1.8, 1.9],
                "ndcg@curve_k": 0.632,
                "ndcg_curve_k": [1.0, 0.8, 0.71, 0.66, 0.63],
                "ece": 0.073,
                "reliability": {
                    "mode": "equal_width",
                    "bins": [
                        {"lower": 0.0, "upper": 0.33, "confidence": 0.1, "precision": 0.0, "count": 2},
                        {"lower": 0.33, "upper": 0.66, "confidence": 0.55, "precision": 0.5, "count": 3},
                        {"lower": 0.66, "upper": 1.0, "confidence": 0.8, "precision": 1.0, "count": 1},
                    ],
                },
                "topk_list": [
                    {
                        "adjuvant_id": "Adj-1",
                        "adjuvant_label": "Adj 1",
                        "score": 0.91,
                        "relevance_binary": 1,
                    },
                    {
                        "adjuvant_id": "Adj-2",
                        "adjuvant_label": "Adj 2",
                        "score": 0.65,
                        "relevance_binary": 0,
                    },
                    {
                        "adjuvant_id": "Adj-3",
                        "adjuvant_label": "Adj 3",
                        "score": 0.5,
                        "relevance_binary": 0,
                    },
                ],
            },
            "baseline": {
                "topk": {
                    "5": {"precision": 0.0, "recall": 0.0, "ndcg": 0.1},
                    "10": {"precision": 0.1, "recall": 0.3, "ndcg": 0.2},
                },
                "dcg_curve": [0.5, 0.5, 0.5, 0.5, 0.5],
                "idcg_curve": [1.0, 1.5, 1.7, 1.8, 1.9],
                "ndcg@curve_k": 0.26,
                "ndcg_curve_k": [0.5, 0.33, 0.29, 0.27, 0.26],
                "ece": 0.15,
                "reliability": {
                    "mode": "equal_width",
                    "bins": [
                        {"lower": 0.0, "upper": 0.33, "confidence": 0.15, "precision": 0.0, "count": 2},
                        {"lower": 0.33, "upper": 0.66, "confidence": 0.5, "precision": 0.3, "count": 3},
                        {"lower": 0.66, "upper": 1.0, "confidence": 0.7, "precision": 0.6, "count": 1},
                    ],
                },
                "topk_list": [],
            },
        },
        "delta_ndcg": {"ndcg@5": 0.2, "ndcg@10": 0.2},
    }
    summary_path = tmp_path / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle)
    return summary_path


def _build_sample_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "adjuvant_vo_id": "Adj-1",
                "adjuvant_label": "Adj 1",
                "rank_target": 1,
                "rank_baseline": 4,
                "relevance_binary": 1,
            },
            {
                "adjuvant_vo_id": "Adj-2",
                "adjuvant_label": "Adj 2",
                "rank_target": 2,
                "rank_baseline": 2,
                "relevance_binary": 1,
            },
            {
                "adjuvant_vo_id": "Adj-3",
                "adjuvant_label": "Adj 3",
                "rank_target": 3,
                "rank_baseline": 3,
                "relevance_binary": 0,
            },
        ]
    )
    csv_path = tmp_path / "candidates.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_compute_rank_swaps_orders_by_improvement(tmp_path: Path) -> None:
    csv_path = _build_sample_csv(tmp_path)
    df = pd.read_csv(csv_path)
    swaps = _compute_rank_swaps(df, primary_label="target", baseline_label="baseline")
    assert [entry["label"] for entry in swaps] == ["Adj 1", "Adj 2"]
    assert swaps[0]["delta"] == 3
    assert swaps[1]["delta"] == 0


def test_create_case_study_panels_writes_image(tmp_path: Path) -> None:
    summary_path = _build_sample_summary(tmp_path)
    csv_path = _build_sample_csv(tmp_path)
    output_path = tmp_path / "figure.png"

    create_case_study_panels(
        summary_path=summary_path,
        output_path=output_path,
        candidates_csv=csv_path,
        primary_label="target",
        baseline_label="baseline",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

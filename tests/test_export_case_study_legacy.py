"""Regression check for loading legacy train_ranker checkpoints."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

# Ensure the repository root and src directory are on the import path.
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import export_case_study as ecs  # noqa: E402

np = ecs.np


class DummyGraph:
    """Minimal stand-in for the PyG HeteroData object."""

    node_types = ("disease", "adjuvant")

    def __init__(self, feat_dim: int = 8, mech_dim: int = 0) -> None:
        adjuvant_structured = (
            torch.zeros(3, mech_dim, dtype=torch.float32) if mech_dim > 0 else None
        )
        self._nodes = {
            "disease": SimpleNamespace(x=torch.zeros(2, feat_dim, dtype=torch.float32)),
            "adjuvant": SimpleNamespace(
                x=torch.zeros(4, feat_dim, dtype=torch.float32), structured=adjuvant_structured
            ),
        }

    def __getitem__(self, key: str) -> SimpleNamespace:
        return self._nodes[key]

    def metadata(self) -> tuple:
        return (list(self.node_types), [("disease", "rel", "adjuvant")])


class DummyEncoder:
    """Tracks whether ``load_state_dict`` was called."""

    last_instance: "DummyEncoder | None" = None

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        self.state_loaded = False
        DummyEncoder.last_instance = self

    def load_state_dict(self, state):  # type: ignore[no-untyped-def]
        self.state_loaded = True
        self.state = state

    def to(self, device):  # type: ignore[no-untyped-def]
        return self

    def eval(self):  # type: ignore[no-untyped-def]
        return self


class FailingDualRanker:
    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        raise AssertionError("Modern DualRanker should not be constructed for legacy checkpoints")


def test_legacy_checkpoint_falls_back_to_dot_product(monkeypatch, tmp_path, capsys):
    """Exporter should accept the legacy checkpoint layout and emit a warning."""

    ckpt_path = tmp_path / "legacy.pt"
    legacy_args = {"data_path": "dummy.csv", "hidden_dim": 8}
    legacy_state = {"encoder.weight": torch.zeros(1)}
    torch.save({"state_dict": legacy_state, "args": legacy_args}, ckpt_path)

    checkpoint = ecs._load_checkpoint(ckpt_path)
    assert checkpoint["_legacy_listnet_head"] is True

    monkeypatch.setattr(ecs, "PyGHeteroEncoder", DummyEncoder)
    monkeypatch.setattr(ecs, "DualRanker", FailingDualRanker)

    graph = DummyGraph()
    model, ranking_head, adjuvant_mechanism = ecs._build_model(
        graph,
        checkpoint["args"],
        checkpoint,
        torch.device("cpu"),
    )

    captured = capsys.readouterr()
    assert "legacy dot-product" in captured.out
    assert isinstance(ranking_head, ecs.LegacyListNetRanker)
    assert isinstance(model, DummyEncoder)
    assert model.state_loaded is True
    assert adjuvant_mechanism is None


def test_dual_ranker_checkpoint_without_head_errors(tmp_path):
    """Modern checkpoints should refuse to load if the ranking head is missing."""

    ckpt_path = tmp_path / "broken.pt"
    dual_args = {
        "data_path": "dummy.csv",
        "hidden_dim": 8,
        "lambda_disease": 0.5,
        "disease_ndcg_weight": 1.0,
    }
    torch.save({"state_dict": {}, "args": dual_args}, ckpt_path)

    with pytest.raises(KeyError) as err:
        ecs._load_checkpoint(ckpt_path)

    message = str(err.value)
    assert "train_disease_ranker.py" in message
    assert "ranking_head_state_dict" in message


def test_compute_reliability_quantile_and_equal_width():
    scores = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    candidate_ids = [0, 1, 2, 3]
    positives = [0, 2]

    quantile_bins, quantile_ece, mode = ecs._compute_reliability(
        scores,
        candidate_ids,
        positives,
        bins=2,
        normalisation="minmax",
        binning="quantile",
    )
    assert mode == "quantile"
    assert [bin.count for bin in quantile_bins] == [2, 2]
    assert pytest.approx(quantile_ece, rel=1e-4) == 1 / 3

    width_bins, width_ece, mode_width = ecs._compute_reliability(
        scores,
        candidate_ids,
        positives,
        bins=2,
        normalisation="minmax",
        binning="equal_width",
    )
    assert mode_width == "equal_width"
    assert [bin.count for bin in width_bins] == [2, 2]
    assert pytest.approx(width_ece, rel=1e-4) == pytest.approx(quantile_ece)


def test_compute_reliability_quantile_fallback():
    scores = np.array([0.5, 0.5, 0.5], dtype=float)
    candidate_ids = [0, 1, 2]
    positives = [1]

    bins, ece, mode = ecs._compute_reliability(
        scores,
        candidate_ids,
        positives,
        bins=3,
        normalisation="minmax",
        binning="quantile",
    )
    assert mode == "equal_width"
    assert sum(bin.count for bin in bins) == len(scores)
    assert ece == pytest.approx(0.0)


def test_compute_run_metrics_enrichments():
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=float)
    candidate_ids = [0, 1, 2, 3]
    order = np.array([0, 1, 2, 3], dtype=int)
    rank_map = {candidate_ids[idx]: idx + 1 for idx in range(len(candidate_ids))}
    run = ecs.RunOutputs(label="primary", scores=scores, order=order, rank_map=rank_map)

    positives = [0, 2]
    gains = {0: 1.0, 2: 1.0}
    adjuvant_mapping = {idx: f"A{idx}" for idx in candidate_ids}
    display_lookup = {f"A{idx}": f"Adj {idx}" for idx in candidate_ids}
    class_lookup = {f"A{idx}": f"cls{idx}" for idx in candidate_ids}

    metrics = ecs._compute_run_metrics(
        run,
        candidate_ids,
        positives,
        gains,
        topk=(1, 3),
        curve_k=4,
        reliability_bins=2,
        normalisation="minmax",
        reliability_binning="equal_width",
        adjuvant_mapping=adjuvant_mapping,
        display_lookup=display_lookup,
        class_lookup=class_lookup,
    )

    assert metrics["first_hit_rank"] == 1
    assert metrics["mrr"] == pytest.approx(1.0)
    assert metrics["reliability"]["mode"] == "equal_width"
    assert len(metrics["ndcg_curve_k"]) == 4
    assert metrics["topk_list"][0]["adjuvant_id"] == "A0"
    assert metrics["topk_list"][0]["relevance_binary"] == 1
    assert metrics["topk_list"][1]["relevance_binary"] == 0
    assert metrics["random_recall_multiplier"]["3"] == pytest.approx(4 / 3)

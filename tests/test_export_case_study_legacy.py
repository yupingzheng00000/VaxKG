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

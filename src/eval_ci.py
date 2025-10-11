"""Compute bootstrap confidence intervals and paired randomization tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Mapping as MappingABC
from typing import Dict, List, Mapping, Sequence

import numpy as np


def percentile_bootstrap_ci(
    per_query_scores: Sequence[float],
    *,
    iterations: int,
    alpha: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return percentile bootstrap confidence interval for the mean."""

    scores = np.asarray(per_query_scores, dtype=float)
    if scores.size == 0:
        raise ValueError("Cannot bootstrap confidence interval without scores")

    boot_means = np.empty(iterations, dtype=float)
    n = scores.size
    for i in range(iterations):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = float(scores[indices].mean())
    lo, hi = np.percentile(
        boot_means,
        [100 * alpha / 2.0, 100 * (1.0 - alpha / 2.0)],
    )
    return float(lo), float(hi)


def paired_randomization_test(
    system_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    iterations: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Return mean difference and paired randomization p-value."""

    system = np.asarray(system_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    if system.shape != baseline.shape:
        raise ValueError("Scores must share shape for paired randomization test")

    diffs = system - baseline
    observed = float(diffs.mean())
    n = diffs.size
    ge = 0
    for _ in range(iterations):
        signs = rng.integers(0, 2, size=n, dtype=np.int8)
        signs = np.where(signs == 0, -1.0, 1.0)
        sampled = float((diffs * signs).mean())
        if abs(sampled) >= abs(observed):
            ge += 1
    p_value = (ge + 1) / (iterations + 1)
    return observed, float(p_value)


def parse_system_specs(specs: Sequence[str]) -> List[tuple[str, Path]]:
    parsed: List[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"System specification '{spec}' must be of the form name=/path/to/results.json"
            )
        name, path_str = spec.split("=", 1)
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        parsed.append((name, path))
    return parsed


def load_per_query_metrics(
    path: Path,
    *,
    head: str,
    split: str,
) -> tuple[Dict[str, Dict[str, float]], Mapping[str, float]]:
    prefix = "ranking" if head == "vaccine" else "disease_ranking"
    per_query_key = f"{prefix}_{split}_per_query"
    macro_key = f"{prefix}_{split}"

    with path.open("r", encoding="utf-8") as handle:
        raw_text = handle.read()

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        # Fall back to JSON Lines: take the last record containing the per-query key.
        records: List[Mapping[str, object]] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                records = []
                break
            records.append(record)

        payload = None
        if records:
            for record in reversed(records):
                if isinstance(record, MappingABC) and per_query_key in record:
                    payload = record
                    break

        if payload is None:
            hint = ""
            lines = [line for line in raw_text.splitlines() if line.strip()]
            if path.suffix == ".jsonl" or all(
                line.lstrip().startswith("{") for line in lines[:3]
            ):
                hint = (
                    " Detected a JSON Lines file but none of the entries contained "
                    f"'{per_query_key}'. Pass the aggregate metrics JSON (e.g., "
                    "artifacts/results/<scheme>.json) or regenerate metrics with per-query "
                    "payloads."
                )
            raise ValueError(
                f"Failed to parse metrics file {path}: {exc}.{hint}"
            ) from exc

    if per_query_key not in payload:
        available = ", ".join(sorted(payload.keys()))
        raise KeyError(
            f"Key '{per_query_key}' not found in {path}. Available keys: {available}"
        )

    raw_per_query: Mapping[str, Mapping[str, float]] = payload[per_query_key]
    per_query: Dict[str, Dict[str, float]] = {}
    for query_id, metrics in raw_per_query.items():
        per_query[str(query_id)] = {metric: float(value) for metric, value in metrics.items()}

    macro_metrics: Mapping[str, float] = {
        metric: float(value) for metric, value in payload.get(macro_key, {}).items()
    }
    return per_query, macro_metrics


def intersect_metric_names(per_query_data: Mapping[str, Dict[str, float]]) -> set[str]:
    names: set[str] = set()
    for metrics in per_query_data.values():
        if not names:
            names = set(metrics.keys())
        else:
            names &= set(metrics.keys())
    return names


def common_query_ids(
    systems: Mapping[str, Dict[str, Dict[str, float]]],
    metric: str,
) -> List[str]:
    shared: set[str] | None = None
    for per_query in systems.values():
        present = {qid for qid, metrics in per_query.items() if metric in metrics}
        shared = present if shared is None else shared & present
    if not shared:
        raise ValueError(f"No common queries contain metric '{metric}' across systems")
    return sorted(shared)


def format_ci(lo: float, hi: float) -> str:
    return f"[{lo:.4f}, {hi:.4f}]"


def format_p_value(p: float, alpha: float) -> str:
    stars = ""
    if p < 0.01:
        stars = "**"
    elif p < alpha:
        stars = "*"
    return f"{p:.4f}{(' ' + stars) if stars else ''}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap CIs and paired randomization tests from stored metrics",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        required=True,
        help="System specifiers of the form name=/path/to/metrics.json",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Name of the baseline system (must match one of the --systems names)",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Split to evaluate (e.g., train, val, test)",
    )
    parser.add_argument(
        "--head",
        choices=["vaccine", "disease"],
        default="vaccine",
        help="Ranking head to evaluate",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        help="Specific metrics to evaluate (default: intersection across systems)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for confidence intervals",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100_000,
        help="Iterations for the paired randomization test",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for CIs and star annotations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Base random seed for bootstrap/permutation procedures",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    specs = parse_system_specs(args.systems)
    system_order = [name for name, _ in specs]
    if args.baseline not in system_order:
        raise ValueError("--baseline must match one of the provided system names")

    per_query_data: Dict[str, Dict[str, Dict[str, float]]] = {}
    macro_data: Dict[str, Mapping[str, float]] = {}
    for name, path in specs:
        per_query, macro = load_per_query_metrics(path, head=args.head, split=args.split)
        per_query_data[name] = per_query
        macro_data[name] = macro

    baseline_macro = macro_data[args.baseline]

    if len(system_order) == 1:
        print(
            "[info] Only one system provided. Add more entries to --systems to compare "
            "against the baseline and run paired randomization tests."
        )
        print()

    if args.metrics:
        metrics_to_use = args.metrics
    else:
        available = intersect_metric_names(per_query_data[args.baseline])
        if not available:
            raise ValueError("No metrics found in baseline per-query data")
        metrics_to_use = sorted(available)

    rng_master = np.random.default_rng(args.seed)

    print(
        f"Split: {args.split} | Head: {args.head} | Baseline: {args.baseline} | Systems: "
        f"{', '.join(system_order)}"
    )
    print()

    for metric in metrics_to_use:
        ids = common_query_ids(per_query_data, metric)
        n_queries = len(ids)
        if n_queries == 0:
            continue

        metric_arrays: Dict[str, np.ndarray] = {}
        for name in system_order:
            per_query = per_query_data[name]
            metric_arrays[name] = np.array(
                [per_query[qid][metric] for qid in ids],
                dtype=float,
            )

        baseline_values = metric_arrays[args.baseline]
        baseline_mean = float(baseline_values.mean())
        baseline_ci_lo, baseline_ci_hi = percentile_bootstrap_ci(
            baseline_values,
            iterations=args.bootstrap,
            alpha=args.alpha,
            rng=np.random.default_rng(rng_master.integers(0, 2**32)),
        )

        print(f"Metric: {metric} (n={n_queries})")
        print(f"{'System':<20}{'Mean':>12}{'95% CI':>20}{('Δ vs ' + args.baseline):>18}{'p-value':>12}")

        for name in system_order:
            values = metric_arrays[name]
            mean = float(values.mean())
            ci_lo, ci_hi = percentile_bootstrap_ci(
                values,
                iterations=args.bootstrap,
                alpha=args.alpha,
                rng=np.random.default_rng(rng_master.integers(0, 2**32)),
            )
            ci_str = format_ci(ci_lo, ci_hi)
            if name == args.baseline:
                diff_str = f"{0.0:+.4f}"
                p_str = "-"
            else:
                diff, p_value = paired_randomization_test(
                    values,
                    baseline_values,
                    iterations=args.permutations,
                    rng=np.random.default_rng(rng_master.integers(0, 2**32)),
                )
                diff_str = f"{diff:+.4f}"
                p_str = format_p_value(p_value, args.alpha)

            print(f"{name:<20}{mean:>12.4f}{ci_str:>20}{diff_str:>18}{p_str:>12}")

        if metric in baseline_macro:
            macro_mean = float(baseline_macro[metric])
            if abs(macro_mean - baseline_mean) > 1e-6:
                print(
                    f"  [warn] Stored macro mean for baseline ({macro_mean:.4f}) "
                    f"differs from per-query mean ({baseline_mean:.4f})."
                )

        print()


if __name__ == "__main__":
    main()

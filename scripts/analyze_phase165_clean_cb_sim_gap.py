#!/usr/bin/env python3
"""Analyze Phase164 clean GPU benchmark rows against cb_sim budget diagnostics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import NamedTuple


MANIFEST_SOURCE = "phase164_clean_gpu_benchmark"
CANDIDATE_SOURCE = "phase164_clean_gpu_benchmark"

EXPECTED_MANIFEST = {
    "tp8ep8-bt8000": {
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "max_num_batched_tokens": 8000,
        "breakdown_name": "K2.5-tp8ep8-8k2k",
        "topology_key": "tp8_dp1_ep8",
    },
    "tp8ep8-bt65536": {
        "tp": 8,
        "dp": 1,
        "ep": 8,
        "max_num_batched_tokens": 65536,
        "breakdown_name": "K2.5-tp8ep8-8k2k-bt65536",
        "topology_key": "tp8_dp1_ep8",
    },
    "tp4dp2ep8-bt8000": {
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "max_num_batched_tokens": 8000,
        "breakdown_name": "K2.5-tp4ep8dp2-8k2k",
        "topology_key": "tp4_dp2_ep8",
    },
    "tp4dp2ep8-bt65536": {
        "tp": 4,
        "dp": 2,
        "ep": 8,
        "max_num_batched_tokens": 65536,
        "breakdown_name": "K2.5-tp4ep8dp2-8k2k-bt65536",
        "topology_key": "tp4_dp2_ep8",
    },
}

EXPECTED_BREAKDOWN_NAMES = {
    "K2.5-tp8ep8-8k2k",
    "K2.5-tp8ep8-32k3k",
    "K2.5-tp4ep8dp2-8k2k",
    "K2.5-tp4ep8dp2-32k3k",
    "K2.5-tp8ep8-8k2k-bt65536",
    "K2.5-tp4ep8dp2-8k2k-bt65536",
}

TOPOLOGY_PAIRS = [
    ("tp8_dp1_ep8", "tp8ep8-bt8000", "tp8ep8-bt65536"),
    ("tp4_dp2_ep8", "tp4dp2ep8-bt8000", "tp4dp2ep8-bt65536"),
]

GAP_FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "baseline_scenario",
    "budget_scenario",
    "baseline_breakdown_name",
    "budget_breakdown_name",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "batch_size",
    "baseline_max_num_batched_tokens",
    "budget_max_num_batched_tokens",
    "clean_baseline_output_tok_s_gpu",
    "clean_budget_output_tok_s_gpu",
    "sim_baseline_output_tok_s_gpu",
    "sim_budget_output_tok_s_gpu",
    "clean_budget_effect",
    "sim_budget_effect",
    "budget_gap",
    "budget_rank",
    "budget_real_rank",
    "baseline_peak_tokens_per_iter",
    "budget_peak_tokens_per_iter",
    "baseline_steady_state_time_ms",
    "budget_steady_state_time_ms",
    "candidate_key",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


class ManifestRow(NamedTuple):
    scenario: str
    tp: int
    dp: int
    ep: int
    isl: int
    osl: int
    batch_size: int
    max_num_batched_tokens: int
    real_output_tok_s_gpu: float


class BreakdownRow(NamedTuple):
    name: str
    tp: int
    dp: int
    ep: int
    max_bt: int
    sim_output_tok_s_gpu: float
    rank: int
    real_rank: int
    peak_tokens_per_iter: float
    steady_state_time_ms: float


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty csv: {path}")
    return rows


def _required(row: dict[str, str], field: str, path: Path) -> str:
    if field not in row or row[field] == "":
        raise ValueError(f"missing {field} in {path}")
    return row[field]


def _as_int(value: str, field: str, path: Path) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be int in {path}: {value!r}") from exc


def _as_float(value: str, field: str, path: Path) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be float in {path}: {value!r}") from exc


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _expect_flag(row: dict[str, str], key: str, expected: str, path: Path) -> None:
    value = _required(row, key, path).lower()
    if value != expected:
        raise ValueError(f"{key} must be {expected} in {path}: {value!r}")


def _load_manifest(path: Path) -> dict[str, ManifestRow]:
    rows = _read_csv(path)
    scenarios = [_required(row, "scenario", path) for row in rows]
    duplicates = sorted({scenario for scenario in scenarios if scenarios.count(scenario) > 1})
    if duplicates:
        raise ValueError(f"duplicate manifest scenario: {duplicates}")
    seen = set(scenarios)
    expected = set(EXPECTED_MANIFEST)
    if seen != expected:
        raise ValueError(
            "manifest scenario set mismatch: "
            f"missing={sorted(expected - seen)} extra={sorted(seen - expected)}"
        )
    if len(rows) != len(EXPECTED_MANIFEST):
        raise ValueError(f"manifest must contain 4 rows: got {len(rows)}")

    parsed: dict[str, ManifestRow] = {}
    for row in rows:
        scenario = _required(row, "scenario", path)
        expected_meta = EXPECTED_MANIFEST[scenario]
        if _required(row, "source", path) != MANIFEST_SOURCE:
            raise ValueError(f"source must be {MANIFEST_SOURCE} for {scenario}")
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        if _as_int(_required(row, "request_success_count", path), "request_success_count", path) != 128:
            raise ValueError(f"request_success_count must be 128 for {scenario}")
        if _as_int(_required(row, "request_fail_count", path), "request_fail_count", path) != 0:
            raise ValueError(f"request_fail_count must be 0 for {scenario}")

        tp = _as_int(_required(row, "tp", path), "tp", path)
        dp = _as_int(_required(row, "dp", path), "dp", path)
        ep = _as_int(_required(row, "ep", path), "ep", path)
        max_bt = _as_int(
            _required(row, "max_num_batched_tokens", path),
            "max_num_batched_tokens",
            path,
        )
        if (tp, dp, ep, max_bt) != (
            expected_meta["tp"],
            expected_meta["dp"],
            expected_meta["ep"],
            expected_meta["max_num_batched_tokens"],
        ):
            raise ValueError(f"manifest shape mismatch for {scenario}")

        parsed[scenario] = ManifestRow(
            scenario=scenario,
            tp=tp,
            dp=dp,
            ep=ep,
            isl=_as_int(_required(row, "isl", path), "isl", path),
            osl=_as_int(_required(row, "osl", path), "osl", path),
            batch_size=_as_int(_required(row, "batch_size", path), "batch_size", path),
            max_num_batched_tokens=max_bt,
            real_output_tok_s_gpu=_as_float(
                _required(row, "real_output_tok_s_gpu", path),
                "real_output_tok_s_gpu",
                path,
            ),
        )
    return parsed


def _load_breakdown(path: Path) -> dict[str, BreakdownRow]:
    rows = _read_csv(path)
    if len(rows) != 6:
        raise ValueError(f"budget breakdown must contain 6 rows: got {len(rows)}")

    names = [_required(row, "name", path) for row in rows]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate breakdown name: {duplicates}")
    seen = set(names)
    if seen != EXPECTED_BREAKDOWN_NAMES:
        raise ValueError(
            "breakdown scenario set mismatch: "
            f"missing={sorted(EXPECTED_BREAKDOWN_NAMES - seen)} "
            f"extra={sorted(seen - EXPECTED_BREAKDOWN_NAMES)}"
        )

    parsed: dict[str, BreakdownRow] = {}
    expected_by_name = {
        value["breakdown_name"]: value for value in EXPECTED_MANIFEST.values()
    }
    for row in rows:
        name = _required(row, "name", path)
        _expect_flag(row, "diagnostic_only", "true", path)
        _expect_flag(row, "valid_for_default", "false", path)
        _expect_flag(row, "perf_database", "false", path)
        parsed_row = BreakdownRow(
            name=name,
            tp=_as_int(_required(row, "tp", path), "tp", path),
            dp=_as_int(_required(row, "dp", path), "dp", path),
            ep=_as_int(_required(row, "ep", path), "ep", path),
            max_bt=_as_int(_required(row, "max_bt", path), "max_bt", path),
            sim_output_tok_s_gpu=_as_float(
                _required(row, "sim_output_tok_s_gpu", path),
                "sim_output_tok_s_gpu",
                path,
            ),
            rank=_as_int(_required(row, "rank", path), "rank", path),
            real_rank=_as_int(_required(row, "real_rank", path), "real_rank", path),
            peak_tokens_per_iter=_as_float(
                _required(row, "peak_tokens_per_iter", path),
                "peak_tokens_per_iter",
                path,
            ),
            steady_state_time_ms=_as_float(
                _required(row, "steady_state_time_ms", path),
                "steady_state_time_ms",
                path,
            ),
        )
        expected = expected_by_name.get(name)
        if expected is not None and (
            parsed_row.tp,
            parsed_row.dp,
            parsed_row.ep,
            parsed_row.max_bt,
        ) != (
            expected["tp"],
            expected["dp"],
            expected["ep"],
            expected["max_num_batched_tokens"],
        ):
            raise ValueError(f"breakdown shape mismatch for {name}")
        parsed[name] = parsed_row
    return parsed


def _ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator == 0:
        raise ValueError(f"zero denominator for {label}")
    return numerator / denominator


def analyze_gap(manifest_path: Path, breakdown_path: Path) -> list[dict[str, str]]:
    manifest = _load_manifest(Path(manifest_path))
    breakdown = _load_breakdown(Path(breakdown_path))
    rows: list[dict[str, str]] = []

    for topology_key, baseline_scenario, budget_scenario in TOPOLOGY_PAIRS:
        baseline = manifest[baseline_scenario]
        budget = manifest[budget_scenario]
        baseline_breakdown_name = EXPECTED_MANIFEST[baseline_scenario]["breakdown_name"]
        budget_breakdown_name = EXPECTED_MANIFEST[budget_scenario]["breakdown_name"]
        baseline_sim = breakdown[baseline_breakdown_name]
        budget_sim = breakdown[budget_breakdown_name]
        shape_key = f"isl{baseline.isl}_osl{baseline.osl}_batch{baseline.batch_size}"

        clean_budget_effect = _ratio(
            budget.real_output_tok_s_gpu,
            baseline.real_output_tok_s_gpu,
            f"clean {topology_key}",
        )
        sim_budget_effect = _ratio(
            budget_sim.sim_output_tok_s_gpu,
            baseline_sim.sim_output_tok_s_gpu,
            f"sim {topology_key}",
        )
        budget_gap = _ratio(clean_budget_effect, sim_budget_effect, f"gap {topology_key}")
        candidate_key = f"{topology_key}:{shape_key}:max_bt{budget.max_num_batched_tokens}"

        rows.append(
            {
                "source": CANDIDATE_SOURCE,
                "topology_key": topology_key,
                "shape_key": shape_key,
                "baseline_scenario": baseline_scenario,
                "budget_scenario": budget_scenario,
                "baseline_breakdown_name": baseline_breakdown_name,
                "budget_breakdown_name": budget_breakdown_name,
                "tp": str(budget.tp),
                "dp": str(budget.dp),
                "ep": str(budget.ep),
                "isl": str(budget.isl),
                "osl": str(budget.osl),
                "batch_size": str(budget.batch_size),
                "baseline_max_num_batched_tokens": str(baseline.max_num_batched_tokens),
                "budget_max_num_batched_tokens": str(budget.max_num_batched_tokens),
                "clean_baseline_output_tok_s_gpu": _format_float(
                    baseline.real_output_tok_s_gpu
                ),
                "clean_budget_output_tok_s_gpu": _format_float(budget.real_output_tok_s_gpu),
                "sim_baseline_output_tok_s_gpu": _format_float(
                    baseline_sim.sim_output_tok_s_gpu
                ),
                "sim_budget_output_tok_s_gpu": _format_float(
                    budget_sim.sim_output_tok_s_gpu
                ),
                "clean_budget_effect": _format_float(clean_budget_effect),
                "sim_budget_effect": _format_float(sim_budget_effect),
                "budget_gap": _format_float(budget_gap),
                "budget_rank": str(budget_sim.rank),
                "budget_real_rank": str(budget_sim.real_rank),
                "baseline_peak_tokens_per_iter": _format_float(
                    baseline_sim.peak_tokens_per_iter
                ),
                "budget_peak_tokens_per_iter": _format_float(
                    budget_sim.peak_tokens_per_iter
                ),
                "baseline_steady_state_time_ms": _format_float(
                    baseline_sim.steady_state_time_ms
                ),
                "budget_steady_state_time_ms": _format_float(
                    budget_sim.steady_state_time_ms
                ),
                "candidate_key": candidate_key,
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def write_gap_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != len(TOPOLOGY_PAIRS):
        raise ValueError(f"gap csv must contain {len(TOPOLOGY_PAIRS)} rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=GAP_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_doc(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase165: Clean cb_sim Budget Candidate",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Scope | Diagnostic keyed model item design only |",
        "| Source | phase164_clean_gpu_benchmark |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| GPU benchmark | No-Go in Phase165 |",
        "",
        "## Observed Budget Effects",
        "",
        "| topology_key | shape_key | clean_budget_effect | sim_budget_effect | budget_gap |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {topology_key} | {shape_key} | {clean_budget_effect} | "
            "{sim_budget_effect} | {budget_gap} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Candidate Key",
            "",
            "Use `topology_key + shape_key + max_num_batched_tokens` as the diagnostic key.",
            "",
            "| Field | Value |",
            "|---|---|",
            "| source | phase164_clean_gpu_benchmark |",
            "| diagnostic_only | true |",
            "| valid_for_default | false |",
            "| perf_database | false |",
            "",
            "## Boundary",
            "",
            "This candidate is a design artifact. It is not a default cb_sim formula change, "
            "not a PerfDatabase row, and not a claim that benchmark metrics are globally "
            "portable across other models or hardware.",
            "",
            "Next gate: Phase166 may implement a callable diagnostic-only model item. "
            "Default AIC integration still requires a separate holdout gate.",
            "",
            "diagnostic_only=true valid_for_default=false perf_database=false",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze Phase164 clean GPU benchmark against cb_sim budget diagnostics."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--budget-breakdown-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidate-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase165_keyed_budget_candidate.md"),
    )
    args = parser.parse_args()

    rows = analyze_gap(args.manifest, args.budget_breakdown_csv)
    write_gap_csv(args.out, rows)
    write_candidate_doc(args.candidate_out, rows)
    print(f"wrote_phase165_gap={args.out}")
    print(f"phase165_gap_rows={len(rows)}")
    print(f"wrote_phase165_candidate={args.candidate_out}")


if __name__ == "__main__":
    main()

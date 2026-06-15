#!/usr/bin/env python3
"""Audit whether mixed phase changes explain Phase264 throughput spread."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


SOURCE = "phase267_mixed_phase_partial_audit"
INPUT_SOURCE = "phase264_phase_mix_decode_batch"
INPUT_MECHANISM = "phase_mix_decode_batch_diagnostic"
DEFAULT_READINESS = "No-Go"
NEXT_MECHANISM = "boundary_timeline_diagnostic"
MAX_HOLDOUT_FILL_FOR_CEILING_GUARD = 0.50

FIELDNAMES = [
    "source",
    "pair_key",
    "topology_key",
    "shape_key",
    "control_scenario",
    "holdout_scenario",
    "output_ratio",
    "mixed_count_delta",
    "mixed_p99_delta",
    "mixed_max_delta",
    "pure_decode_p99_max_invariant",
    "budget_ceiling_not_linear",
    "verdict",
    "next_mechanism",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


EXPECTED_PAIRS = [
    {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "control_scenario": "tp8ep8-4k2k-bt4000",
        "holdout_scenario": "tp8ep8-4k2k-bt65536",
        "control_bt": 4000,
        "mixed_count_delta": 1,
        "mixed_p99_delta": -480.0,
        "mixed_max_delta": -480,
        "verdict": "partial",
    },
    {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl4000_osl2000_batch128",
        "control_scenario": "tp4dp2ep8-4k2k-bt4000",
        "holdout_scenario": "tp4dp2ep8-4k2k-bt65536",
        "control_bt": 4000,
        "mixed_count_delta": -1,
        "mixed_p99_delta": 0.0,
        "mixed_max_delta": 0,
        "verdict": "explains",
    },
    {
        "topology_key": "tp8_dp1_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "control_scenario": "tp8ep8-12k2k-bt12000",
        "holdout_scenario": "tp8ep8-12k2k-bt65536",
        "control_bt": 12000,
        "mixed_count_delta": -1,
        "mixed_p99_delta": 224.0,
        "mixed_max_delta": 224,
        "verdict": "partial",
    },
    {
        "topology_key": "tp4_dp2_ep8",
        "shape_key": "isl12000_osl2000_batch128",
        "control_scenario": "tp4dp2ep8-12k2k-bt12000",
        "holdout_scenario": "tp4dp2ep8-12k2k-bt65536",
        "control_bt": 12000,
        "mixed_count_delta": 0,
        "mixed_p99_delta": 570.0,
        "mixed_max_delta": 570,
        "verdict": "does_not_explain",
    },
]


def _format_float(value: float) -> str:
    return f"{value:.6f}"


def _read_phase264_csv(path: Path | str) -> list[dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 8:
        raise ValueError(f"phase264 csv must contain 8 rows: got {len(rows)}")
    scenarios = [row.get("scenario") for row in rows]
    expected = []
    for pair in EXPECTED_PAIRS:
        expected.extend([str(pair["control_scenario"]), str(pair["holdout_scenario"])])
    if scenarios != expected:
        raise ValueError(f"phase264 scenario order mismatch: {scenarios}")
    for row in rows:
        _validate_phase264_row(row)
    return rows


def _validate_phase264_row(row: dict[str, str]) -> None:
    if row.get("source") != INPUT_SOURCE:
        raise ValueError("phase264 source mismatch")
    if row.get("mechanism_hypothesis") != INPUT_MECHANISM:
        raise ValueError("phase264 mechanism_hypothesis mismatch")
    if row.get("default_readiness") != DEFAULT_READINESS:
        raise ValueError("phase264 default_readiness mismatch")
    if (
        row.get("diagnostic_only") != "true"
        or row.get("valid_for_default") != "false"
        or row.get("perf_database") != "false"
    ):
        raise ValueError("phase264 flag mismatch")
    for field in (
        "max_bt",
        "mixed_iterations",
        "mixed_rank_sum_p99_total_tokens",
        "mixed_rank_sum_max_total_tokens",
        "pure_decode_rank_sum_p99_decode_tokens",
        "pure_decode_rank_sum_max_decode_tokens",
        "rank_sum_p99_fill_ratio",
        "rank_sum_max_fill_ratio",
        "output_ratio_vs_control",
    ):
        if row.get(field) in (None, ""):
            raise ValueError(f"phase264 missing field: {field}")


def _pair_key(topology_key: str, shape_key: str, control_bt: int) -> str:
    return f"{topology_key}:{shape_key}:control_bt{control_bt}:holdout_bt65536"


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise ValueError(f"{message}: got {actual!r}, expected {expected!r}")


def _analyze_pair(
    control: dict[str, str],
    holdout: dict[str, str],
    expected: dict[str, object],
) -> dict[str, str]:
    topology_key = str(expected["topology_key"])
    shape_key = str(expected["shape_key"])
    control_bt = int(expected["control_bt"])
    _assert_equal(control["topology_key"], topology_key, "control topology mismatch")
    _assert_equal(holdout["topology_key"], topology_key, "holdout topology mismatch")
    _assert_equal(control["shape_key"], shape_key, "control shape mismatch")
    _assert_equal(holdout["shape_key"], shape_key, "holdout shape mismatch")
    _assert_equal(control["role"], "control", "control role mismatch")
    _assert_equal(holdout["role"], "holdout", "holdout role mismatch")
    _assert_equal(control["scenario"], expected["control_scenario"], "control scenario mismatch")
    _assert_equal(holdout["scenario"], expected["holdout_scenario"], "holdout scenario mismatch")
    _assert_equal(int(control["max_bt"]), control_bt, "control max_bt mismatch")
    _assert_equal(int(holdout["max_bt"]), 65536, "holdout max_bt mismatch")

    mixed_count_delta = int(holdout["mixed_iterations"]) - int(control["mixed_iterations"])
    mixed_p99_delta = float(holdout["mixed_rank_sum_p99_total_tokens"]) - float(
        control["mixed_rank_sum_p99_total_tokens"]
    )
    mixed_max_delta = int(holdout["mixed_rank_sum_max_total_tokens"]) - int(
        control["mixed_rank_sum_max_total_tokens"]
    )
    if mixed_count_delta != int(expected["mixed_count_delta"]):
        raise ValueError(
            "mixed_count_delta mismatch: "
            f"{_pair_key(topology_key, shape_key, control_bt)} got {mixed_count_delta}"
        )
    if mixed_p99_delta != float(expected["mixed_p99_delta"]):
        raise ValueError(
            "mixed_p99_delta mismatch: "
            f"{_pair_key(topology_key, shape_key, control_bt)} got {mixed_p99_delta}"
        )
    if mixed_max_delta != int(expected["mixed_max_delta"]):
        raise ValueError(
            "mixed_max_delta mismatch: "
            f"{_pair_key(topology_key, shape_key, control_bt)} got {mixed_max_delta}"
        )

    pure_decode_values = {
        control["pure_decode_rank_sum_p99_decode_tokens"],
        control["pure_decode_rank_sum_max_decode_tokens"],
        holdout["pure_decode_rank_sum_p99_decode_tokens"],
        holdout["pure_decode_rank_sum_max_decode_tokens"],
    }
    pure_decode_invariant = pure_decode_values == {"128.000000", "128"}
    if not pure_decode_invariant:
        raise ValueError("pure decode p99/max must stay invariant at 128/128")

    holdout_p99_fill = float(holdout["rank_sum_p99_fill_ratio"])
    holdout_max_fill = float(holdout["rank_sum_max_fill_ratio"])
    budget_ceiling_not_linear = (
        holdout_p99_fill < MAX_HOLDOUT_FILL_FOR_CEILING_GUARD
        and holdout_max_fill < MAX_HOLDOUT_FILL_FOR_CEILING_GUARD
    )
    if not budget_ceiling_not_linear:
        raise ValueError("holdout budget fill must stay far below aggregate configured budget")

    return {
        "source": SOURCE,
        "pair_key": _pair_key(topology_key, shape_key, control_bt),
        "topology_key": topology_key,
        "shape_key": shape_key,
        "control_scenario": control["scenario"],
        "holdout_scenario": holdout["scenario"],
        "output_ratio": _format_float(float(holdout["output_ratio_vs_control"])),
        "mixed_count_delta": str(mixed_count_delta),
        "mixed_p99_delta": _format_float(mixed_p99_delta),
        "mixed_max_delta": str(mixed_max_delta),
        "pure_decode_p99_max_invariant": _bool_text(pure_decode_invariant),
        "budget_ceiling_not_linear": _bool_text(budget_ceiling_not_linear),
        "verdict": str(expected["verdict"]),
        "next_mechanism": NEXT_MECHANISM,
        "default_readiness": DEFAULT_READINESS,
        "diagnostic_only": "true",
        "valid_for_default": "false",
        "perf_database": "false",
    }


def analyze_mixed_phase_partial_audit(input_csv: Path | str) -> list[dict[str, str]]:
    rows = _read_phase264_csv(input_csv)
    by_scenario = {row["scenario"]: row for row in rows}
    if len(by_scenario) != 8:
        raise ValueError("phase264 scenarios must be unique")

    output_rows: list[dict[str, str]] = []
    for expected in EXPECTED_PAIRS:
        control = by_scenario[str(expected["control_scenario"])]
        holdout = by_scenario[str(expected["holdout_scenario"])]
        output_rows.append(_analyze_pair(control, holdout, expected))
    verdicts = [row["verdict"] for row in output_rows]
    if verdicts.count("partial") != 2 or verdicts.count("explains") != 1 or verdicts.count("does_not_explain") != 1:
        raise ValueError(f"unexpected verdict distribution: {verdicts}")
    return output_rows


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 4:
        raise ValueError(f"phase267 audit must contain 4 rows: got {len(rows)}")
    expected_keys = [
        _pair_key(str(pair["topology_key"]), str(pair["shape_key"]), int(pair["control_bt"]))
        for pair in EXPECTED_PAIRS
    ]
    if [row.get("pair_key") for row in rows] != expected_keys:
        raise ValueError("phase267 pair key order mismatch")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("phase267 source mismatch")
        if row.get("next_mechanism") != NEXT_MECHANISM:
            raise ValueError("phase267 next_mechanism mismatch")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("phase267 default_readiness mismatch")
        if (
            row.get("diagnostic_only") != "true"
            or row.get("valid_for_default") != "false"
            or row.get("perf_database") != "false"
        ):
            raise ValueError("phase267 flag mismatch")
        if row.get("pure_decode_p99_max_invariant") != "true":
            raise ValueError("phase267 pure decode invariant mismatch")
        if row.get("budget_ceiling_not_linear") != "true":
            raise ValueError("phase267 budget ceiling guard mismatch")


def write_mixed_phase_partial_audit_csv(path: Path | str, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with out.open(newline="", encoding="utf-8") as f:
        _validate_output_rows(list(csv.DictReader(f)))


def write_mixed_phase_partial_audit_doc(path: Path | str, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase267: Mixed Phase Partial-Only Audit",
        "",
        "## Decision",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Mixed phase | Mixed phase is partial-only |",
        "| Pure decode | Invariant at p99/max 128/128 |",
        "| Budget ceiling | Still not a linear configured-budget cost |",
        "| Next mechanism | boundary_timeline_diagnostic |",
        "| Default AIC | No-Go |",
        "| PerfDatabase | No-Go |",
        "| Boundary flags | diagnostic_only=true valid_for_default=false perf_database=false |",
        "",
        "## Pair Audit",
        "",
        "| Pair | output ratio | mixed count delta | mixed p99 delta | mixed max delta | verdict |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_key']} | {row['output_ratio']} | {row['mixed_count_delta']} | "
            f"{row['mixed_p99_delta']} | {row['mixed_max_delta']} | {row['verdict']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Phase267 makes the Phase266 readout machine-checkable.",
            "Pure decode batch size is not the source of the spread because every pair keeps p99/max at 128/128.",
            "The high-budget rows still schedule far below the aggregate configured budget, so max_num_batched_tokens remains a ceiling rather than a direct cost.",
            "Mixed phase changes explain only one pair cleanly and are partial or insufficient for the others.",
            "The next diagnostic layer should inspect boundary/timeline behavior instead of turning mixed phase into a default model.",
            "This remains diagnostic-only evidence. Do not wire it into VLLMBackend.run_agg, default AIC, or PerfDatabase.",
            "",
        ]
    )
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase264_phase_mix_decode_batch.csv"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase267_mixed_phase_partial_audit.csv"),
    )
    parser.add_argument(
        "--doc-out",
        type=Path,
        default=Path("docs/iter_gap_investigation/phase267_mixed_phase_partial_audit.md"),
    )
    args = parser.parse_args()

    rows = analyze_mixed_phase_partial_audit(args.input)
    write_mixed_phase_partial_audit_csv(args.out, rows)
    write_mixed_phase_partial_audit_doc(args.doc_out, rows)
    print(f"wrote_phase267_mixed_phase_partial_audit={args.out}")
    print(f"phase267_rows={len(rows)}")
    print("verdict_partial=2")
    print("verdict_explains=1")
    print("verdict_does_not_explain=1")
    print(f"default_readiness={DEFAULT_READINESS}")


if __name__ == "__main__":
    main()

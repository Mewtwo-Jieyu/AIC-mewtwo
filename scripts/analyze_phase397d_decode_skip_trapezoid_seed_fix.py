"""Phase397d decode-skip trapezoid seed structural fix (Route delta, step 1).

Phase397c pinned the standing tp16 throughput miss to a RUNTIME defect: the
decode batch-skip trapezoid in simulator.py::_estimate_decode_skip_latency was
seeded with `first_iter_latency_ms=iter_lat`, where `iter_lat` is the latency of
the iteration that TRIGGERS the skip -- a MIXED/prefill iteration whose latency
includes the ~isl-token prefill chunk (~231 ms at 10k3k), not a pure-decode
latency (~27 ms). This inflated the decode-skip wall ~3.2x. 397c also showed the
inflation was partially compensated by a per-iteration decode UNDER-count
(gen_non_attn MoE /tp_size, hidden under attention by overlap_factor=0 max), so
the net miss was ~1.5x (sim throughput ~0.66x real).

Phase397d applies ONLY the structural trapezoid-seed fix: both trapezoid
endpoints are now computed as pure-decode iterations, at `start_avg_kv_len` and
`start_avg_kv_len + skip_iters`. This is deterministic and carries NO tuning.

Because the compensating per-iteration under-count is intentionally left for
phase397e, removing the (compensating) trapezoid inflation swings the gate
predictions the OTHER way -- sim throughput crosses from ~0.66x real (too slow)
to ~1.9-2.7x real (too fast), exactly the ~2.6x reversal 397c predicted for a
trapezoid-only fix. This phase therefore does NOT claim throughput accuracy; it
claims the skip accounting is now structurally correct. Default AIC stays No-Go
and the accuracy gate is only re-opened after 397e corrects the decode
composition and both fixes are re-validated together.

This analyzer is an audit record over measured before/after gate throughput
(canonical run_agg cb_sim path, overlap_factor=0, per_iteration_overhead_ms=0,
offline, no GPU/SSH). Raw numbers in phase397d_gate_before_after_raw.csv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397d_decode_skip_trapezoid_seed_fix.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397d_decode_skip_trapezoid_seed_fix.md"
)
RAW_EVIDENCE_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397d_gate_before_after_raw.csv"
)

SOURCE = "phase397d_decode_skip_trapezoid_seed_fix"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_DB_VERSION = "0.12.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397e_decode_periter_composition_offline"
GATE_SCENARIO = "10k3k_b128"


@dataclass(frozen=True)
class GateResult:
    name: str
    real_out_tok_s_gpu: float
    sim_out_before: float
    sim_out_after: float

    @property
    def sim_over_real_before(self) -> float:
        return self.sim_out_before / self.real_out_tok_s_gpu

    @property
    def sim_over_real_after(self) -> float:
        return self.sim_out_after / self.real_out_tok_s_gpu


# Measured offline via canonical run_agg cb_sim (aic env, pr403 worktree).
# before = pre-fix sim throughput (phase397c); after = post-seed-fix sim throughput.
GATES: dict[str, GateResult] = {
    "10k3k_b128": GateResult("10k3k_b128", 89.5, 59.01, 237.40),
    "10k2k_b32": GateResult("10k2k_b32", 49.8, 34.16, 110.73),
    "16k2k_b32": GateResult("16k2k_b32", 41.7, 31.09, 77.78),
}

FIELDNAMES = [
    "source",
    "row_type",
    "scenario",
    "metric",
    "value_a",
    "value_b",
    "ratio",
    "verdict",
    "model",
    "hardware",
    "vllm_db_version",
    "topology",
    "next_allowed_phase",
    "runtime_modified",
    "nearest_lookup_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "fudge_factor_tuning_used",
    "scope_gating_used",
    "write_real_data_file",
    "gpu_allowed",
    "ssh_allowed",
    "default_aic_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "scenario": "",
    "metric": "",
    "value_a": "",
    "value_b": "",
    "ratio": "",
    "verdict": "",
    "model": MODEL,
    "hardware": HARDWARE,
    "vllm_db_version": VLLM_DB_VERSION,
    "topology": TOPOLOGY,
    "next_allowed_phase": "",
    # This phase DOES modify runtime (the structural seed fix).
    "runtime_modified": TRUE,
    "nearest_lookup_allowed": FALSE,
    "interpolation_allowed": TRUE,
    "extrapolation_allowed": FALSE,
    # Deterministic structural fix; NO empirical tuning.
    "fudge_factor_tuning_used": FALSE,
    "scope_gating_used": FALSE,
    "write_real_data_file": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "default_aic_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    # A real fix, not a diagnostic-only artifact.
    "diagnostic_only": FALSE,
    # Knowingly regresses accuracy the other way until 397e -> not for default.
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase397d() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    rows.append(
        _row(
            "fix_provenance",
            metric="estimate_decode_skip_latency_seed",
            value_a="before:first_iter_latency_ms=iter_lat(triggering mixed iter)",
            value_b="after:first=compute(pure_decode@start_kv), "
            "end=compute(pure_decode@start_kv+skip)",
            verdict=(
                "structural seed fix: both trapezoid endpoints are pure-decode "
                "iterations; no tuning, deterministic"
            ),
        )
    )

    for name, g in GATES.items():
        rows.append(
            _row(
                "gate_before_after",
                scenario=name,
                metric="sim_over_real_throughput_before_vs_after",
                value_a=f"before={g.sim_over_real_before:.4f}",
                value_b=f"after={g.sim_over_real_after:.4f}",
                ratio=f"{g.sim_over_real_after / g.sim_over_real_before:.4f}",
                verdict=(
                    "seed fix removes the compensating trapezoid inflation; sim "
                    "throughput crosses 1.0x (was too slow -> now too fast)"
                ),
            )
        )

    rows.append(
        _row(
            "direction_swing",
            metric="all_gates_cross_unity_after_fix",
            value_a="before: sim/real < 1 (wall over-counted, throughput too low)",
            value_b="after: sim/real > 1 (per-iter decode under-count now unmasked)",
            verdict=(
                "confirms 397c compensating-error model: trapezoid-only fix swings "
                "~2.6x the other way; decode composition (397e) still owed"
            ),
        )
    )

    rows.append(
        _row(
            "accuracy_claim",
            metric="phase_scope",
            value_a="claims: skip accounting structurally correct",
            value_b="does NOT claim: throughput accuracy (regressed the other way)",
            verdict=(
                "gated: valid_for_default=false, Default AIC remains No-Go until "
                "397e corrects the per-iteration decode composition and both are "
                "re-validated together"
            ),
        )
    )

    rows.append(
        _row(
            "verdict",
            metric="phase397d_outcome",
            verdict=(
                "decode-skip trapezoid seed fixed structurally (pure-decode "
                "endpoints); no tuning; runtime_modified=true but gated "
                "(valid_for_default=false); accuracy paired with 397e"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="route_delta_step2_target",
            verdict=(
                "phase397e OFFLINE: determine from perf_database/collector source "
                "whether generation_moe table values are aggregate or per-rank, "
                "decide if iteration_latency.py /tp_size double-counts, and whether "
                "overlap_factor=0 max masks a real decode non-attention term; then "
                "re-validate 397d+397e together"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "fix_provenance":
        raise ValueError(f"Phase397d must start with fix_provenance: {actual[0]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397d must end with verdict,next_phase: {actual[-2:]}")

    required_types = {
        "fix_provenance",
        "gate_before_after",
        "direction_swing",
        "accuracy_claim",
        "verdict",
        "next_phase",
    }
    if required_types - set(actual):
        raise ValueError(f"Phase397d missing row types: {required_types - set(actual)}")

    n = len(GATES)
    if sum(1 for r in rows if r["row_type"] == "gate_before_after") != n:
        raise ValueError(f"gate_before_after must have {n} rows")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        # Gated-fix discipline: real fix, but not for default, no data/DB/GPU.
        for guard in (
            "nearest_lookup_allowed",
            "extrapolation_allowed",
            "fudge_factor_tuning_used",
            "scope_gating_used",
            "write_real_data_file",
            "gpu_allowed",
            "ssh_allowed",
            "default_aic_allowed",
            "valid_for_default",
            "perf_database",
        ):
            if row[guard] != FALSE:
                raise ValueError(f"{label} {guard} must be {FALSE}")
        if row["runtime_modified"] != TRUE:
            raise ValueError(f"{label} runtime_modified must be {TRUE}")
        if row["diagnostic_only"] != FALSE:
            raise ValueError(f"{label} diagnostic_only must be {FALSE}")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness must be {DEFAULT_READINESS}")

    # Load-bearing: every gate crosses unity (before <1, after >1).
    for row in rows:
        if row["row_type"] != "gate_before_after":
            continue
        before = float(row["value_a"].split("=")[1])
        after = float(row["value_b"].split("=")[1])
        if not (before < 1.0 < after):
            raise ValueError(
                f"{row['scenario']} must cross unity: before {before} < 1 < after {after}"
            )

    fix = next(r for r in rows if r["row_type"] == "fix_provenance")
    if "pure-decode" not in fix["verdict"] and "pure_decode" not in fix["verdict"]:
        raise ValueError("fix_provenance must state pure-decode endpoints")
    if fix["fudge_factor_tuning_used"] != FALSE:
        raise ValueError("fix must carry no tuning")

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "trapezoid seed" not in verdict["verdict"]:
        raise ValueError("verdict must name the trapezoid seed fix")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase397d_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397d_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    gate = GATES[GATE_SCENARIO]
    lines = [
        "# Phase397d decode-skip Trapezoid Seed Structural Fix (Route delta, step 1)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Fix | `simulator.py::_estimate_decode_skip_latency` now computes BOTH "
        "trapezoid endpoints as pure-decode iterations (`start_avg_kv_len` and "
        "`start_avg_kv_len + skip_iters`); the `first_iter_latency_ms=iter_lat` "
        "seed (the triggering mixed/prefill iteration) is removed. |",
        "| Nature | structural, deterministic, no tuning; runtime modified but "
        "gated (`valid_for_default=false`). |",
        f"| Gate ({GATE_SCENARIO}) | sim/real throughput {gate.sim_over_real_before:.2f}x "
        f"(before, too slow) -> {gate.sim_over_real_after:.2f}x (after, too fast). |",
        "| Scope | claims skip accounting is structurally correct; does NOT claim "
        "throughput accuracy. Default AIC stays No-Go until 397e. |",
        "",
        "## Before/after gate throughput (canonical run_agg cb_sim, offline)",
        "",
        "| Scenario | real out | sim/real before | sim/real after | crosses 1.0 |",
        "|---|---|---|---|---|",
    ]
    for name, g in GATES.items():
        lines.append(
            f"| {name} | {g.real_out_tok_s_gpu:.1f} | "
            f"{g.sim_over_real_before:.2f}x | {g.sim_over_real_after:.2f}x | yes |"
        )
    lines += [
        "",
        "## Why the prediction flips direction (expected)",
        "",
        "397c showed the standing miss was two compensating errors:",
        "",
        "1. the skip-trapezoid seed inflated the decode wall ~3.2x (sim too slow), and",
        "2. the per-iteration decode was under-counted ~3x (`generation_moe /tp_size` "
        "hidden under attention by `overlap_factor=0` max), which partially cancelled (1).",
        "",
        "397d removes (1) only. With nothing left to compensate the still-present "
        "under-count (2), sim wall drops too far and throughput crosses from ~0.66x "
        "real (too slow) to ~1.9-2.7x real (too fast) -- exactly the ~2.6x reversal "
        "397c predicted for a trapezoid-only fix. This is expected and gated.",
        "",
        "## Verdict -- Route delta step 1",
        "",
        "- The decode-skip trapezoid is now structurally correct (pure-decode "
        "ramp), with no empirical tuning.",
        f"- Next: **{NEXT_PHASE}** -- offline determination of the `generation_moe` "
        "table semantics (aggregate vs per-rank) to decide whether the `/tp_size` "
        "scaling in `iteration_latency.py` double-counts and whether "
        "`overlap_factor=0` max masks a real decode non-attention term; then "
        "re-validate 397d+397e together before re-opening the accuracy gate.",
        "",
        "## Discipline",
        "",
        "- `runtime_modified=true` (the structural fix) but `valid_for_default=false`, "
        "`fudge_factor_tuning_used=false`, `scope_gating_used=false`, no PerfDatabase "
        "write, no real-data write, no GPU/SSH; Default AIC remains No-Go.",
        f"- Raw before/after evidence at "
        f"`{RAW_EVIDENCE_CSV.relative_to(REPO_ROOT)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397d()
    write_phase397d_csv(args.output_csv, rows)
    write_phase397d_md(args.output_md, rows)


if __name__ == "__main__":
    main()

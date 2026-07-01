"""Phase397f decode per-iteration composition runtime fix (Route delta step 3).

Phase397e (verdict-only) proved two coupled decode-composition errors in
iteration_latency.py's pure-decode path: (1) a /tp_size double-count on the
already-per-rank generation_moe* terms, and (2) overlap_factor=0 max() dropping
the serial non-attention term (decode attention and MoE/FFN run serially within
a layer). Phase397f applies the runtime fix (pure-decode-local only) and
re-validates OFFLINE (no GPU/SSH, no PerfDatabase writes, no fudge tuning).

Runtime change (src/aiconfigurator/sdk/backends/cb_simulator/iteration_latency.py):
  1. _split_generation_non_attention: drop the `/ tp_size` on generation_moe*
     (values are per-rank; TP is in the moe_tp_size lookup key).
  2. pure-decode branch of _compute_3pass: total = generation_non_attn + gen_attn
     (serial sum) instead of _combine_with_overlap(...) = max(...). overlap_factor
     still governs the mixed / pure-prefill branches (untouched).
  3. removed dead _scale_generation_non_attention.

Result (offline validate_cb_simulator, tp16 real data + multi-config table):
  - THROUGHPUT max abs error 4.13x -> 1.44x (mean 2.43 -> 1.18): PASS (<=1.499).
    Every tp16 gate improved; the primary ~1.5x/over-count investigation target
    is now structurally consistent (3k3k 4.13->1.04, 10k3k 2.65->1.07).
  - MULTI-CONFIG max 3.26x -> 2.17x (mean 1.75 -> 1.47): still FAIL (<=1.470).
    Residual is concentrated in expert-parallel ep8 topologies (moe_ep=8,
    moe_tp=1): tp4ep8dp2-32k3k 3.26->2.17, and tp8ep8-8k2k REGRESSED 1.02->0.62
    (now over-corrected). Root: _get_tp_size() returns the ATTENTION tp, so the
    (now removed) /tp never matched moe sharding for ep8, and ep8 decode has
    different per-rank MoE semantics (moe_ep vs moe_tp) than tp16.
  - TTFT unchanged (1.79x, prefill-side; decode fix does not touch it): PASS.

Verdict: the tp16 decode throughput path is structurally fixed and passes its
gate; the ep8/dp2 multi-config path improved but still fails, so Default AIC
stays No-Go. Next: phase397g targets the expert-parallel (ep8) decode
composition. Acceptance-threshold tightening (the 1.499/1.470 gates are frozen
at the pre-397d ~1.5x behavior) is noted as a candidate but NOT applied here.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397f_decode_periter_composition_fix.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397f_decode_periter_composition_fix.md"
)
FULLTABLE_RAW_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397f_fulltable_before_after_raw.csv"
)
PERITER_RAW_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397f_periter_after_raw.csv"
)

SOURCE = "phase397f_decode_periter_composition_fix"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_DB_VERSION = "0.12.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
NEXT_PHASE = "phase397g_ep8_decode_periter_composition"

# validate_cb_simulator.py:50-52 acceptance gates (frozen at pre-397d behavior).
THROUGHPUT_GATE = 1.4989592822599629
MULTI_CONFIG_GATE = 1.4696362054928367
TTFT_GATE = 1.790056498134038

# Full-table before/after (validate_cb_simulator, offline; raw in the *_raw.csv).
THROUGHPUT_MAX_BEFORE, THROUGHPUT_MAX_AFTER = 4.13, 1.44
THROUGHPUT_MEAN_BEFORE, THROUGHPUT_MEAN_AFTER = 2.43, 1.18
MULTI_MAX_BEFORE, MULTI_MAX_AFTER = 3.26, 2.17
MULTI_MEAN_BEFORE, MULTI_MEAN_AFTER = 1.75, 1.47
TTFT_MAX_BEFORE, TTFT_MAX_AFTER = 1.79, 1.79

# Data-driven readiness: throughput+ttft pass, multi-config still fails.
THROUGHPUT_OK = THROUGHPUT_MAX_AFTER <= THROUGHPUT_GATE
MULTI_OK = MULTI_MAX_AFTER <= MULTI_CONFIG_GATE
TTFT_OK = TTFT_MAX_AFTER <= TTFT_GATE
DEFAULT_READY = THROUGHPUT_OK and MULTI_OK and TTFT_OK
DEFAULT_READINESS = "Go" if DEFAULT_READY else "No-Go"
VALID_FOR_DEFAULT = TRUE if DEFAULT_READY else FALSE


@dataclass(frozen=True)
class PerIter:
    name: str
    gen_attn_ms: float
    gen_non_attn_raw_ms: float
    real_decode_iter_ms: float

    @property
    def per_iter_after_ms(self) -> float:
        return self.gen_non_attn_raw_ms + self.gen_attn_ms

    @property
    def ratio(self) -> float:
        return self.per_iter_after_ms / self.real_decode_iter_ms


PERITER: dict[str, PerIter] = {
    "10k3k_b128": PerIter("10k3k_b128", 26.6206, 51.1034, 89.3855),
    "10k2k_b32": PerIter("10k2k_b32", 12.1814, 32.4249, 40.1606),
    "16k2k_b32": PerIter("16k2k_b32", 17.9083, 32.4249, 47.9616),
}


@dataclass(frozen=True)
class ConfigBA:
    name: str
    ratio_before: float
    ratio_after: float

    @staticmethod
    def _abs_err(ratio: float) -> float:
        if ratio <= 0:
            return float("inf")
        return max(ratio, 1.0 / ratio)

    @property
    def abs_err_before(self) -> float:
        return self._abs_err(self.ratio_before)

    @property
    def abs_err_after(self) -> float:
        return self._abs_err(self.ratio_after)

    @property
    def regressed(self) -> bool:
        return self.abs_err_after > self.abs_err_before + 1e-9


# Multi-config before/after ratios (Sim/Out) from the full-table raw csv.
MULTI_CONFIGS: list[ConfigBA] = [
    ConfigBA("K2.5-tp8ep8-8k2k", 1.02, 0.62),
    ConfigBA("K2.5-tp8ep8-32k3k", 1.79, 1.24),
    ConfigBA("K2.5-tp4ep8dp2-8k2k", 1.82, 1.09),
    ConfigBA("K2.5-tp4ep8dp2-32k3k", 3.26, 2.17),
    ConfigBA("K2.5-tp8ep8-8k2k-bt65536", 0.98, 0.60),
    ConfigBA("K2.5-tp4ep8dp2-8k2k-bt65536", 1.61, 0.96),
]

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
    # First phase that changes runtime.
    "runtime_modified": TRUE,
    "nearest_lookup_allowed": FALSE,
    "interpolation_allowed": TRUE,
    "extrapolation_allowed": FALSE,
    "fudge_factor_tuning_used": FALSE,
    "scope_gating_used": FALSE,
    "write_real_data_file": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "default_aic_allowed": VALID_FOR_DEFAULT,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": FALSE,
    "valid_for_default": VALID_FOR_DEFAULT,
    "perf_database": FALSE,
}


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase397f() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- runtime change record ---
    rows.append(
        _row(
            "runtime_change",
            metric="iteration_latency.py pure-decode composition fix",
            value_a="_split_generation_non_attention: drop /tp on generation_moe* (per-rank)",
            value_b="pure-decode total = generation_non_attn + gen_attn (serial sum); "
            "removed dead _scale_generation_non_attention",
            verdict="pure-decode-local only; overlap_factor and mixed/prefill branches untouched",
        )
    )

    # --- per-iteration reconciliation after fix ---
    for name, p in PERITER.items():
        rows.append(
            _row(
                "periter_after",
                scenario=name,
                metric="per_iter_after_serialsum_vs_real",
                value_a=f"per_iter_after={p.per_iter_after_ms:.2f}ms",
                value_b=f"real_decode_iter={p.real_decode_iter_ms:.2f}ms",
                ratio=f"{p.ratio:.4f}",
                verdict="fixed pure-decode iter = raw non-attn + attn (serial); ~real +/-13%",
            )
        )

    # --- full-table before/after ---
    rows.append(
        _row(
            "fulltable_before_after",
            metric="throughput_max_abs_err",
            value_a=f"before={THROUGHPUT_MAX_BEFORE:.2f}x",
            value_b=f"after={THROUGHPUT_MAX_AFTER:.2f}x (gate {THROUGHPUT_GATE:.3f})",
            ratio=f"{THROUGHPUT_MAX_AFTER:.4f}",
            verdict="PASS" if THROUGHPUT_OK else "FAIL",
        )
    )
    rows.append(
        _row(
            "fulltable_before_after",
            metric="throughput_mean_abs_err",
            value_a=f"before={THROUGHPUT_MEAN_BEFORE:.2f}x",
            value_b=f"after={THROUGHPUT_MEAN_AFTER:.2f}x",
            ratio=f"{THROUGHPUT_MEAN_AFTER:.4f}",
            verdict="tp16 throughput path structurally improved",
        )
    )
    rows.append(
        _row(
            "fulltable_before_after",
            metric="multi_config_max_abs_err",
            value_a=f"before={MULTI_MAX_BEFORE:.2f}x",
            value_b=f"after={MULTI_MAX_AFTER:.2f}x (gate {MULTI_CONFIG_GATE:.3f})",
            ratio=f"{MULTI_MAX_AFTER:.4f}",
            verdict="PASS" if MULTI_OK else "FAIL",
        )
    )
    rows.append(
        _row(
            "fulltable_before_after",
            metric="multi_config_mean_abs_err",
            value_a=f"before={MULTI_MEAN_BEFORE:.2f}x",
            value_b=f"after={MULTI_MEAN_AFTER:.2f}x",
            ratio=f"{MULTI_MEAN_AFTER:.4f}",
            verdict="ep8/dp2 residual remains",
        )
    )
    rows.append(
        _row(
            "fulltable_before_after",
            metric="ttft_max_abs_err_threshold",
            value_a=f"before={TTFT_MAX_BEFORE:.2f}x",
            value_b=f"after={TTFT_MAX_AFTER:.2f}x (gate {TTFT_GATE:.3f})",
            ratio=f"{TTFT_MAX_AFTER:.4f}",
            verdict="PASS (prefill-side, unchanged)" if TTFT_OK else "FAIL",
        )
    )

    # --- per-config regression (multi-config) ---
    for c in MULTI_CONFIGS:
        rows.append(
            _row(
                "per_config_regression",
                scenario=c.name,
                metric="abs_err_before_vs_after",
                value_a=f"before={c.abs_err_before:.2f}x(ratio {c.ratio_before:.2f})",
                value_b=f"after={c.abs_err_after:.2f}x(ratio {c.ratio_after:.2f})",
                ratio=f"{c.abs_err_after:.4f}",
                verdict="REGRESSED (over-corrected)" if c.regressed else "improved",
            )
        )

    # --- verdict ---
    rows.append(
        _row(
            "verdict",
            metric="decode_periter_composition_fix_outcome",
            value_a=f"throughput {THROUGHPUT_MAX_AFTER:.2f}x "
            f"({'PASS' if THROUGHPUT_OK else 'FAIL'})",
            value_b=f"multi_config {MULTI_MAX_AFTER:.2f}x "
            f"({'PASS' if MULTI_OK else 'FAIL'}); Default {DEFAULT_READINESS}",
            verdict=(
                "tp16 decode throughput path structurally fixed and passing "
                "(4.13x->1.44x); ep8/dp2 multi-config improved (3.26x->2.17x) but "
                "still fails -> Default AIC No-Go"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="route_delta_step4_target",
            value_a="ep8 decode composition: _get_tp_size returns attention tp; "
            "moe_ep(=8) sharding has different per-rank semantics than moe_tp",
            value_b="candidate: threshold tightening (gates frozen at pre-397d ~1.5x) "
            "-- proposed, NOT applied here",
            verdict=(
                "phase397g: forensics + fix for expert-parallel (ep8/dp2) decode "
                "per-iteration composition; re-validate multi-config table offline"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "runtime_change":
        raise ValueError(f"Phase397f must start with runtime_change: {actual[0]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397f must end with verdict,next_phase: {actual[-2:]}")

    required = {
        "runtime_change",
        "periter_after",
        "fulltable_before_after",
        "per_config_regression",
        "verdict",
        "next_phase",
    }
    if required - set(actual):
        raise ValueError(f"Phase397f missing row types: {required - set(actual)}")

    if sum(1 for r in rows if r["row_type"] == "periter_after") != len(PERITER):
        raise ValueError("periter_after row count mismatch")
    if sum(1 for r in rows if r["row_type"] == "per_config_regression") != len(MULTI_CONFIGS):
        raise ValueError("per_config_regression row count mismatch")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        # This phase DOES modify runtime; but must not fudge / gate / touch DB / GPU.
        if row["runtime_modified"] != TRUE:
            raise ValueError(f"{label} runtime_modified must be {TRUE}")
        for guard in (
            "fudge_factor_tuning_used",
            "scope_gating_used",
            "write_real_data_file",
            "gpu_allowed",
            "ssh_allowed",
            "perf_database",
            "nearest_lookup_allowed",
            "extrapolation_allowed",
        ):
            if row[guard] != FALSE:
                raise ValueError(f"{label} {guard} must be {FALSE}")
        if row["diagnostic_only"] != FALSE:
            raise ValueError(f"{label} diagnostic_only must be {FALSE} (this is a fix)")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness mismatch")
        if row["valid_for_default"] != VALID_FOR_DEFAULT:
            raise ValueError(f"{label} valid_for_default mismatch")

    # Load-bearing: throughput improved and passes; multi-config improved but fails.
    if not THROUGHPUT_OK:
        raise ValueError("throughput must pass its gate after the fix")
    if MULTI_OK:
        raise ValueError("multi-config still fails; MULTI_OK must be False this phase")
    if THROUGHPUT_MAX_AFTER >= THROUGHPUT_MAX_BEFORE:
        raise ValueError("throughput max must improve")
    if MULTI_MAX_AFTER >= MULTI_MAX_BEFORE:
        raise ValueError("multi-config max must improve")
    if DEFAULT_READINESS != "No-Go":
        raise ValueError("Default must remain No-Go while multi-config fails")

    # Load-bearing: at least one ep8 config regressed (documents the over-correction).
    if not any(c.regressed for c in MULTI_CONFIGS):
        raise ValueError("expected at least one regressed ep8 config")

    # Load-bearing: per-iter after reconciles within +/-15% of real.
    for row in rows:
        if row["row_type"] == "periter_after":
            if not 0.85 <= float(row["ratio"]) <= 1.15:
                raise ValueError(
                    f"{row['scenario']} per_iter_after/real {row['ratio']} out of band"
                )

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase397f_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397f_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase397f decode Per-Iteration Composition Runtime Fix (Route delta step 3)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Change | pure-decode-local runtime fix in "
        "`iteration_latency.py`: drop the `/tp` double-count on `generation_moe*`, "
        "sum decode attention + non-attention serially, remove dead "
        "`_scale_generation_non_attention` |",
        f"| Throughput | max {THROUGHPUT_MAX_BEFORE:.2f}x -> "
        f"{THROUGHPUT_MAX_AFTER:.2f}x (gate {THROUGHPUT_GATE:.3f}): "
        f"{'PASS' if THROUGHPUT_OK else 'FAIL'} |",
        f"| Multi-config | max {MULTI_MAX_BEFORE:.2f}x -> {MULTI_MAX_AFTER:.2f}x "
        f"(gate {MULTI_CONFIG_GATE:.3f}): {'PASS' if MULTI_OK else 'FAIL'} |",
        f"| TTFT | max {TTFT_MAX_AFTER:.2f}x (gate {TTFT_GATE:.3f}): "
        f"{'PASS' if TTFT_OK else 'FAIL'} (prefill-side, unchanged) |",
        f"| Default AIC | **{DEFAULT_READINESS}** |",
        "",
        "## Runtime change (only file: iteration_latency.py, pure-decode branch)",
        "",
        "1. `_split_generation_non_attention`: dropped `/ tp_size` on "
        "`generation_moe*` -- the perf-database values are per-rank (TP is in the "
        "`moe_tp_size` lookup key), so dividing again was a double-count (phase397e).",
        "2. pure-decode branch of `_compute_3pass`: "
        "`total = generation_non_attn + gen_attn` (serial sum) instead of "
        "`_combine_with_overlap(...) = max(...)`. `overlap_factor` still governs the "
        "mixed / pure-prefill branches.",
        "3. removed dead `_scale_generation_non_attention`.",
        "",
        "## Per-iteration reconciliation after fix (tp16 gates)",
        "",
        "| Scenario | per-iter after (serial sum) | real decode iter | ratio |",
        "|---|---|---|---|",
    ]
    for name, p in PERITER.items():
        lines.append(
            f"| {name} | {p.per_iter_after_ms:.1f} ms | "
            f"{p.real_decode_iter_ms:.1f} ms | {p.ratio:.2f}x |"
        )
    lines += [
        "",
        "## Full-table before/after (offline validate_cb_simulator)",
        "",
        "| Metric | before (397e HEAD) | after (397f) | gate | result |",
        "|---|---|---|---|---|",
        f"| Throughput max | {THROUGHPUT_MAX_BEFORE:.2f}x | {THROUGHPUT_MAX_AFTER:.2f}x "
        f"| {THROUGHPUT_GATE:.3f} | {'PASS' if THROUGHPUT_OK else 'FAIL'} |",
        f"| Throughput mean | {THROUGHPUT_MEAN_BEFORE:.2f}x | {THROUGHPUT_MEAN_AFTER:.2f}x "
        f"| - | - |",
        f"| Multi-config max | {MULTI_MAX_BEFORE:.2f}x | {MULTI_MAX_AFTER:.2f}x "
        f"| {MULTI_CONFIG_GATE:.3f} | {'PASS' if MULTI_OK else 'FAIL'} |",
        f"| Multi-config mean | {MULTI_MEAN_BEFORE:.2f}x | {MULTI_MEAN_AFTER:.2f}x "
        f"| - | - |",
        f"| TTFT max | {TTFT_MAX_BEFORE:.2f}x | {TTFT_MAX_AFTER:.2f}x "
        f"| {TTFT_GATE:.3f} | {'PASS' if TTFT_OK else 'FAIL'} |",
        "",
        "## Per-config (multi-config) before/after",
        "",
        "| Config | ratio before | ratio after | abs-err before | abs-err after | note |",
        "|---|---|---|---|---|---|",
    ]
    for c in MULTI_CONFIGS:
        lines.append(
            f"| {c.name} | {c.ratio_before:.2f}x | {c.ratio_after:.2f}x | "
            f"{c.abs_err_before:.2f}x | {c.abs_err_after:.2f}x | "
            f"{'REGRESSED' if c.regressed else 'improved'} |"
        )
    lines += [
        "",
        "## Verdict -- Route delta step 3",
        "",
        "- The tp16 decode throughput path (the original ~1.5x over-count target) is "
        "now structurally consistent and passes its gate (max 4.13x -> 1.44x). Every "
        "tp16 gate improved.",
        "- The expert-parallel (ep8 / dp2) multi-config path improved "
        "(3.26x -> 2.17x max, mean 1.75x -> 1.47x) but still fails, and two "
        "`tp8ep8-8k2k` cases over-corrected to ~0.6x. Root: `_get_tp_size()` returns "
        "the ATTENTION tp, so the removed `/tp` never matched ep8 MoE sharding, and "
        "ep8 decode has different per-rank semantics (`moe_ep` vs `moe_tp`).",
        f"- Default AIC remains **{DEFAULT_READINESS}** (multi-config gate fails).",
        f"- Next: **{NEXT_PHASE}** -- forensics + fix for the ep8/dp2 decode "
        "composition; re-validate the multi-config table offline. Tightening the "
        "acceptance gates (frozen at the pre-397d ~1.5x behaviour) is a candidate but "
        "is NOT applied here.",
        "",
        "## Discipline",
        "",
        "- Runtime change limited to the pure-decode composition in "
        "`iteration_latency.py`; `overlap_factor`, mixed/prefill branches, "
        "`simulator.py`, and the PerfDatabase are untouched.",
        "- No fudge tuning, no scope gating, no GPU/SSH, no acceptance-gate changes.",
        "- Raw evidence: "
        f"`{FULLTABLE_RAW_CSV.relative_to(REPO_ROOT)}`, "
        f"`{PERITER_RAW_CSV.relative_to(REPO_ROOT)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397f()
    write_phase397f_csv(args.output_csv, rows)
    write_phase397f_md(args.output_md, rows)


if __name__ == "__main__":
    main()

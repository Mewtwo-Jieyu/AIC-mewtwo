"""Phase397g ep8/dp2 decode composition residual attribution (Route delta step 4).

Phase397f fixed the tp16 decode composition (throughput table PASS, 4.13x->1.44x)
but the expert-parallel multi-config table still FAILs (max 3.26x->2.17x), and the
residual is bidirectional: tp8ep8-8k2k over-corrected to 0.62x while
tp4ep8dp2-32k3k stays 2.17x. This phase is OFFLINE and verdict-only: it fully
attributes the ep8/dp2 residual and hands phase397h concrete structural fixes +
targets. It does NOT modify runtime, does NOT tune the ep8 overhead, needs no
GPU/SSH, and does not touch the PerfDatabase.

Attribution (offline probe: overhead sweep {0,90}, budget breakdown, pure-decode
ep8 probe, dual num_gpus convention; raw in the *_raw.csv files):

  1. ep8_per_iteration_overhead_ms=90 is a MISCALIBRATED BLUNT FUDGE.
     validate_cb_simulator.py feeds a flat 90ms/iter overhead ONLY to the ep8
     multi-config path (default 90.0; _make_cb_config -> CBSimConfig; charged in
     iteration_latency when decode_bs>0). At overhead=0 EVERY ep8 config
     OVER-predicts (1.12x-3.22x), so the 90ms was calibrated to the pre-397f
     under-count. Post-397f it over-corrects decode-heavy tp8ep8-8k2k
     (155.2->82.8, ratio 1.16x->0.62x; abs err 1.16->1.61). The physically owed
     per-iter EP comm for that config is real 119.8ms - compose 100.2ms ~= 20ms,
     NOT 90ms.

  2. dp2 PER-GPU NORMALIZATION BUG. _run_agg_cb_sim normalizes throughput with
     num_gpus=tp, but attention-dp configs run on tp*dp physical GPUs. The
     pure-decode compose for tp4ep8dp2-8k2k (116.39ms) matches the real decode
     iter at num_gpus=tp*dp=8 (116.18ms, 1.00x), NOT at num_gpus=tp=4 (232.36ms,
     0.50x). Dividing dp2 sim throughput by dp lands tp4ep8dp2-8k2k at 0.94x.

  3. LONG-CONTEXT (32k3k) MIXED/PREFILL OVER-PREDICTION. Even at overhead=0 with
     correct normalization, the 32k3k configs over-predict (tp8ep8-32k3k 1.95x;
     tp4ep8dp2-32k3k ~1.61x after dp fix). These are mixed/prefill-dominated
     (long isl), a residual separate from pure decode.

Verdict: the ep8 residual is NOT a single decode scaling error. 397h should
(i) replace the flat 90ms ep8 overhead with a structural per-rank EP
dispatch/combine comm term (decode gap ~20ms for tp8ep8-8k2k), (ii) fix the dp
per-GPU normalization (num_gpus=tp*dp), then (iii) hand the residual long-context
32k3k mixed/prefill over-prediction to a follow-up. Default AIC stays No-Go.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397g_ep8_decode_composition_offline.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397g_ep8_decode_composition_offline.md"
)
OVERHEAD_RAW_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_overhead_sensitivity_raw.csv"
)
ATTRIBUTION_RAW_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_residual_attribution_raw.csv"
)

SOURCE = "phase397g_ep8_decode_composition_offline"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_DB_VERSION = "0.12.0"
TOPOLOGY = "ep8_multi_config"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397h_ep8_decode_composition_fix"
EP8_OVERHEAD_MS = 90.0


@dataclass(frozen=True)
class Attr:
    name: str
    tp: int
    dp: int
    shape: str
    per_iter_no_ovh_ms: float
    real_iter_ng_tp_ms: float
    real_iter_ng_tp_dp_ms: float
    sim_ovh0_ratio: float
    sim_ovh90_ratio: float
    dominant_cause: str

    @property
    def abs_err_ovh0(self) -> float:
        return max(self.sim_ovh0_ratio, 1.0 / self.sim_ovh0_ratio)

    @property
    def abs_err_ovh90(self) -> float:
        return max(self.sim_ovh90_ratio, 1.0 / self.sim_ovh90_ratio)

    @property
    def overhead_regressed(self) -> bool:
        return self.abs_err_ovh90 > self.abs_err_ovh0 + 1e-9

    @property
    def per_iter_over_real_ng_tp_dp(self) -> float:
        return self.per_iter_no_ovh_ms / self.real_iter_ng_tp_dp_ms


# From the offline probe (aic env, pr403 worktree). batch=128 all configs.
ATTR: dict[str, Attr] = {
    "tp8ep8-8k2k": Attr(
        "tp8ep8-8k2k", 8, 1, "8k2k",
        100.1540, 119.83, 119.83, 1.1624, 0.6203, "ep8_overhead_90ms_overcorrects",
    ),
    "tp8ep8-32k3k": Attr(
        "tp8ep8-32k3k", 8, 1, "32k3k",
        149.7794, 304.94, 304.94, 1.9454, 1.2357, "longcontext_mixed_prefill",
    ),
    "tp4ep8dp2-8k2k": Attr(
        "tp4ep8dp2-8k2k", 4, 2, "8k2k",
        116.3865, 232.36, 116.18, 1.8784, 1.0869, "dp2_pergpu_normalization",
    ),
    "tp4ep8dp2-32k3k": Attr(
        "tp4ep8dp2-32k3k", 4, 2, "32k3k",
        166.5988, 600.63, 300.31, 3.2229, 2.1731, "dp2_normalization_plus_longcontext",
    ),
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
    "runtime_modified": FALSE,
    "nearest_lookup_allowed": FALSE,
    "interpolation_allowed": TRUE,
    "extrapolation_allowed": FALSE,
    "fudge_factor_tuning_used": FALSE,
    "scope_gating_used": FALSE,
    "write_real_data_file": FALSE,
    "gpu_allowed": FALSE,
    "ssh_allowed": FALSE,
    "default_aic_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}


def _row(row_type: str, **overrides: str) -> dict[str, str]:
    row = dict(COMMON_FIELDS)
    row["row_type"] = row_type
    row.update(overrides)
    return row


def analyze_phase397g() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- ep8 overhead semantics ---
    rows.append(
        _row(
            "ep8_overhead_semantics",
            metric="ep8_per_iteration_overhead_ms",
            value_a=f"flat {EP8_OVERHEAD_MS}ms/iter on ep8 multi-config only "
            "(validate_cb_simulator.py:1096; _make_cb_config->CBSimConfig)",
            value_b="charged in iteration_latency when decode_bs>0; tp16 main table uses 0",
            verdict="blunt empirical fudge calibrated to the pre-397f under-count",
        )
    )

    # --- overhead sensitivity per config ---
    for name, a in ATTR.items():
        rows.append(
            _row(
                "overhead_sensitivity",
                scenario=name,
                metric="abs_err_overhead_0_vs_90",
                value_a=f"ovh0 ratio={a.sim_ovh0_ratio:.2f}x (err {a.abs_err_ovh0:.2f})",
                value_b=f"ovh90 ratio={a.sim_ovh90_ratio:.2f}x (err {a.abs_err_ovh90:.2f})",
                ratio=f"{a.abs_err_ovh90:.4f}",
                verdict="90ms REGRESSES this config" if a.overhead_regressed
                else "90ms helps but does not fix",
            )
        )

    # --- pure-decode ep8 probe + dp normalization ---
    for name, a in ATTR.items():
        rows.append(
            _row(
                "puredecode_ep8_probe",
                scenario=name,
                metric="per_iter_no_ovh_vs_real_iter",
                value_a=f"per_iter(no ovh)={a.per_iter_no_ovh_ms:.2f}ms",
                value_b=f"real_iter(ng=tp)={a.real_iter_ng_tp_ms:.2f}ms; "
                f"real_iter(ng=tp*dp)={a.real_iter_ng_tp_dp_ms:.2f}ms",
                ratio=f"{a.per_iter_over_real_ng_tp_dp:.4f}",
                verdict="pure-decode compose vs real decode iter (num_gpus=tp*dp)",
            )
        )

    # --- dp normalization check ---
    rows.append(
        _row(
            "dp_normalization_check",
            metric="num_gpus_convention",
            value_a="sim uses num_gpus=tp (_run_agg_cb_sim sim.run num_gpus=tp)",
            value_b="attention-dp physical GPUs = tp*dp; tp4ep8dp2-8k2k compose "
            "116.39ms == real_iter(ng=tp*dp)=116.18ms (1.00x), NOT ng=tp (0.50x)",
            ratio=f"{ATTR['tp4ep8dp2-8k2k'].per_iter_over_real_ng_tp_dp:.4f}",
            verdict="dp2 throughput normalized by tp not tp*dp -> ~dp x over-prediction",
        )
    )

    # --- per-config residual attribution ---
    for name, a in ATTR.items():
        rows.append(
            _row(
                "residual_attribution",
                scenario=name,
                metric="dominant_cause",
                value_a=f"tp={a.tp} dp={a.dp} shape={a.shape}",
                value_b=f"ovh0={a.sim_ovh0_ratio:.2f}x ovh90={a.sim_ovh90_ratio:.2f}x",
                ratio=f"{a.sim_ovh90_ratio:.4f}",
                verdict=a.dominant_cause,
            )
        )

    # --- candidate fixes ---
    rows.append(
        _row(
            "candidate_fixes",
            metric="phase397h_change_set",
            value_a="(i) replace flat 90ms ep8 overhead with structural per-rank EP "
            "dispatch/combine comm (decode gap ~20ms for tp8ep8-8k2k); "
            "(ii) fix dp per-GPU normalization num_gpus=tp -> tp*dp",
            value_b="(iii) hand residual long-context 32k3k mixed/prefill "
            "over-prediction (~1.95x dp1) to a follow-up mixed-path phase",
            verdict="enumerated for 397h; not implemented here",
        )
    )

    # --- verdict + next phase ---
    rows.append(
        _row(
            "verdict",
            metric="ep8_residual_root_cause",
            value_a="bidirectional residual = 90ms fudge over-correction (decode-heavy) "
            "+ dp2 num_gpus=tp normalization (~dp x) + long-context mixed over-predict",
            value_b=f"Default {DEFAULT_READINESS}",
            verdict=(
                "not a single decode scaling error; the flat 90ms ep8 overhead and the "
                "dp per-GPU normalization are structural, the 32k3k residual is mixed-path"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="route_delta_step5_target",
            value_a="structuralize ep8 comm (drop flat 90ms) + fix num_gpus=tp*dp",
            value_b="targets: tp8ep8-8k2k owed ~20ms EP comm; tp4ep8dp2-8k2k -> 0.94x "
            "after dp fix; re-validate multi-config offline",
            verdict=(
                "phase397h: structural ep8 comm + dp normalization fix; then a mixed-path "
                "phase for the 32k3k long-context residual"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "ep8_overhead_semantics":
        raise ValueError(f"Phase397g must start with ep8_overhead_semantics: {actual[0]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397g must end with verdict,next_phase: {actual[-2:]}")

    required = {
        "ep8_overhead_semantics",
        "overhead_sensitivity",
        "puredecode_ep8_probe",
        "dp_normalization_check",
        "residual_attribution",
        "candidate_fixes",
        "verdict",
        "next_phase",
    }
    if required - set(actual):
        raise ValueError(f"Phase397g missing row types: {required - set(actual)}")

    n = len(ATTR)
    for per_scenario_type in ("overhead_sensitivity", "puredecode_ep8_probe", "residual_attribution"):
        if sum(1 for r in rows if r["row_type"] == per_scenario_type) != n:
            raise ValueError(f"{per_scenario_type} must have {n} rows")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        for guard in (
            "runtime_modified",
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
        if row["diagnostic_only"] != TRUE:
            raise ValueError(f"{label} diagnostic_only must be {TRUE}")
        if row["default_readiness"] != DEFAULT_READINESS:
            raise ValueError(f"{label} default_readiness must be {DEFAULT_READINESS}")

    # Load-bearing 1: at overhead=0 every ep8 config over-predicts (ratio > 1).
    for name, a in ATTR.items():
        if a.sim_ovh0_ratio <= 1.0:
            raise ValueError(f"{name} expected over-prediction at overhead=0")

    # Load-bearing 2: the 90ms overhead REGRESSES the decode-heavy tp8ep8-8k2k.
    if not ATTR["tp8ep8-8k2k"].overhead_regressed:
        raise ValueError("tp8ep8-8k2k must regress under the 90ms overhead")

    # Load-bearing 3: dp2 pure-decode compose matches real at num_gpus=tp*dp (~1.0x).
    dp2 = ATTR["tp4ep8dp2-8k2k"]
    if not 0.9 <= dp2.per_iter_over_real_ng_tp_dp <= 1.1:
        raise ValueError("tp4ep8dp2-8k2k compose must match real_iter(ng=tp*dp) ~1.0x")
    # ...and does NOT match at num_gpus=tp (should be ~0.5x for dp2).
    if dp2.per_iter_no_ovh_ms / dp2.real_iter_ng_tp_ms >= 0.75:
        raise ValueError("tp4ep8dp2-8k2k should mismatch badly at num_gpus=tp")

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "single" not in verdict["verdict"]:
        raise ValueError("verdict must state it is not a single scaling error")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase397g_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397g_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase397g ep8/dp2 decode Composition Residual Attribution (Route delta step 4)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | after 397f, why does the ep8 multi-config table still FAIL "
        "(2.17x) with a bidirectional residual (tp8ep8-8k2k 0.62x under, "
        "tp4ep8dp2-32k3k 2.17x over)? |",
        "| Answer | three structural causes: (1) the flat 90ms ep8 overhead is a "
        "miscalibrated fudge that over-corrects decode-heavy configs; (2) dp2 "
        "throughput is normalized by num_gpus=tp instead of tp*dp; (3) the 32k3k "
        "long-context configs over-predict in the mixed/prefill path. |",
        "| Runtime / table / Default AIC | not modified (verdict-only, offline, "
        "no GPU/SSH, no overhead tuning) |",
        "",
        "## 1. ep8 overhead is a miscalibrated blunt fudge",
        "",
        "`validate_cb_simulator.py` feeds a flat **90ms/iter** overhead only to the "
        "ep8 multi-config path (`ep8_per_iteration_overhead_ms` default 90.0, line "
        "1096; `_make_cb_config` -> `CBSimConfig`; charged in `iteration_latency` when "
        "`decode_bs>0`). The tp16 main table uses 0.",
        "",
        "| Config | ratio @ovh0 | ratio @ovh90 | abs-err @0 | abs-err @90 | note |",
        "|---|---|---|---|---|---|",
    ]
    for name, a in ATTR.items():
        lines.append(
            f"| {name} | {a.sim_ovh0_ratio:.2f}x | {a.sim_ovh90_ratio:.2f}x | "
            f"{a.abs_err_ovh0:.2f}x | {a.abs_err_ovh90:.2f}x | "
            f"{'REGRESSED' if a.overhead_regressed else 'helps'} |"
        )
    lines += [
        "",
        "At `overhead=0` EVERY ep8 config over-predicts (1.12x-3.22x), so the 90ms was "
        "calibrated to the pre-397f under-count. Post-397f it over-corrects "
        "decode-heavy `tp8ep8-8k2k` (1.16x -> 0.62x). The physically owed per-iter EP "
        "comm there is `real 119.8ms - compose 100.2ms ~= 20ms`, not 90ms.",
        "",
        "## 2. dp2 per-GPU normalization bug",
        "",
        "`_run_agg_cb_sim` normalizes with `num_gpus=tp`, but attention-dp configs run "
        "on `tp*dp` physical GPUs.",
        "",
        "| Config | compose (no ovh) | real_iter(ng=tp) | real_iter(ng=tp*dp) | compose/real(ng=tp*dp) |",
        "|---|---|---|---|---|",
    ]
    for name, a in ATTR.items():
        lines.append(
            f"| {name} | {a.per_iter_no_ovh_ms:.1f} ms | {a.real_iter_ng_tp_ms:.1f} ms | "
            f"{a.real_iter_ng_tp_dp_ms:.1f} ms | {a.per_iter_over_real_ng_tp_dp:.2f}x |"
        )
    lines += [
        "",
        "For `tp4ep8dp2-8k2k` the pure-decode compose (116.39ms) matches the real "
        "decode iter at `num_gpus=tp*dp=8` (116.18ms, **1.00x**), not at `num_gpus=tp=4` "
        "(0.50x). Dividing dp2 sim throughput by `dp` lands `tp4ep8dp2-8k2k` at 0.94x.",
        "",
        "## 3. Per-config attribution",
        "",
        "| Config | tp | dp | shape | dominant cause |",
        "|---|---|---|---|---|",
    ]
    for name, a in ATTR.items():
        lines.append(
            f"| {name} | {a.tp} | {a.dp} | {a.shape} | {a.dominant_cause} |"
        )
    lines += [
        "",
        "## Verdict -- Route delta step 4",
        "",
        "- The ep8 residual is NOT a single decode scaling error.",
        "- (i) The flat 90ms ep8 overhead is a miscalibrated fudge (owed comm ~20ms, "
        "not 90ms). (ii) dp2 throughput is normalized by `num_gpus=tp` instead of "
        "`tp*dp`. (iii) The 32k3k long-context configs over-predict in the "
        "mixed/prefill path.",
        f"- Default AIC remains **{DEFAULT_READINESS}**.",
        f"- Next: **{NEXT_PHASE}** -- replace the flat 90ms ep8 overhead with a "
        "structural per-rank EP dispatch/combine comm term and fix the dp per-GPU "
        "normalization (`num_gpus=tp*dp`); then a mixed-path phase for the 32k3k "
        "long-context residual. Re-validate the multi-config table offline.",
        "",
        "## Discipline",
        "",
        "- Verdict-only, offline: runtime / PerfDatabase not modified; the ep8 "
        "overhead was only SWEPT ({0,90}) as a diagnostic via the existing CLI switch, "
        "not tuned or persisted; no scope gating; no GPU/SSH; Default AIC No-Go.",
        f"- Raw evidence: `{OVERHEAD_RAW_CSV.relative_to(REPO_ROOT)}`, "
        f"`{ATTRIBUTION_RAW_CSV.relative_to(REPO_ROOT)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397g()
    write_phase397g_csv(args.output_csv, rows)
    write_phase397g_md(args.output_md, rows)


if __name__ == "__main__":
    main()

"""Phase397g ep8/dp2 decode composition residual attribution (Route delta step 4).

CORRECTION NOTE (supersedes the first 397g cut, commit c935c9e5): the first pass
claimed a "dp2 per-GPU normalization bug". That was WRONG. Re-reading the
throughput path, dp cancels exactly in the per-GPU normalization
(vllm_backend.py:699-723): tokens_s_gpu = throughput_tok_s * (pp*dp) / (tp*pp*dp)
= throughput_tok_s / tp, and throughput_tok_s is the aggregate one-replica value
(simulator.py:303). So the normalization is correct. This corrected cut re-runs a
read-only per-op probe and re-attributes the residual to EP all-to-all
communication that the simulator under-models.

Phase397f fixed the tp16 decode composition (throughput table PASS, 4.13x->1.44x)
but the expert-parallel multi-config table still FAILs (max 3.26x->2.17x), and the
residual is bidirectional: tp8ep8-8k2k over-corrected to 0.62x while
tp4ep8dp2-32k3k stays 2.17x. This phase is OFFLINE and verdict-only: it fully
attributes the ep8/dp2 residual and hands phase397h concrete structural fixes +
targets. It does NOT modify runtime, does NOT tune the ep8 overhead, needs no
GPU/SSH, and does not touch the PerfDatabase.

Corrected attribution (offline probe: overhead sweep {0,90}, budget breakdown,
pure-decode ep8 probe, per-op tp8-vs-tp4 dump; raw in the *_raw.csv files):

  1. ep8_per_iteration_overhead_ms=90 is a MISCALIBRATED BLUNT FUDGE.
     validate_cb_simulator.py feeds a flat 90ms/iter overhead ONLY to the ep8
     multi-config path (default 90.0; _make_cb_config -> CBSimConfig; charged in
     iteration_latency when decode_bs>0). At overhead=0 EVERY ep8 config
     OVER-predicts (1.12x-3.22x), so the 90ms was calibrated to the pre-397f
     under-count. Post-397f it over-corrects decode-heavy tp8ep8-8k2k
     (155.2->82.8, ratio 1.16x->0.62x). A single constant cannot cover a cost
     that scales with the topology: tp8ep8-8k2k owes ~20ms, tp4ep8dp2-8k2k owes
     ~116ms.

  2. EP ALL-TO-ALL COMMUNICATION IS UNDER-MODELED (the corrected root cause; NOT
     a normalization bug). dp cancels in the per-GPU normalization, yet the real
     per-replica decode iter roughly DOUBLES from tp8dp1 (real 119.8ms) to
     tp4dp2 (real 232.4ms) while the simulator barely moves (100.2ms -> 116.4ms,
     1.16x). The per-op tp8-vs-tp4 dump shows why the sim misses it:
       - generation_moe dominates (71.8ms -> 84.8ms, ~72% of the iter) and is
         nearly flat across tp (EP topology fixed at moe_ep=8; only num_tokens
         128->256 moves it).
       - generation_attention is FLAT (21.7ms -> 21.5ms, 0.99x). For MLA the
         latent KV is replicated across TP, so flat attention is plausibly
         CORRECT, not the gap.
       - dense gemms are tiny (~4-5ms total).
       - the ONLY comm proxy, generation_moe_pre/post_dispatch, is ~2-5ms total
         -- far too small to cover the owed 20ms (tp8dp1) / 116ms (tp4dp2).
     So the missing per-iter cost is un-modeled EP dispatch/combine all-to-all
     communication, which grows with attention_dp * decode_bs tokens and with
     cross-dp-group network hops (tp4dp2 routes 256 tokens across 8 EP ranks
     spanning two dp groups; tp8dp1 routes 128 within one group).

  3. LONG-CONTEXT (32k3k) MIXED/PREFILL OVER-PREDICTION. Even at overhead=0 the
     32k3k configs over-predict (tp8ep8-32k3k 1.95x) -- mixed/prefill-dominated
     (long isl), a residual separate from the pure-decode EP-comm gap.

Verdict: the ep8 residual is NOT a single decode scaling error and NOT a
normalization bug. 397h should (i) model EP dispatch/combine all-to-all comm
structurally (per-rank, scaling with attention_dp*decode_bs and cross-group hops)
in place of the flat 90ms, (ii) verify the generation_moe token count under
attention_dp, then (iii) hand the residual long-context 32k3k mixed/prefill
over-prediction to a follow-up. Default AIC stays No-Go.
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
TP_SCALING_RAW_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397g_tp_scaling_probe_raw.csv"
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
    def sim_too_fast_ratio(self) -> float:
        # Correct comparison: sim per-iter vs the real per-replica decode iter,
        # which is normalized by num_gpus=tp (dp cancels). <1 means sim too fast.
        return self.per_iter_no_ovh_ms / self.real_iter_ng_tp_ms

    @property
    def owed_comm_ms(self) -> float:
        return self.real_iter_ng_tp_ms - self.per_iter_no_ovh_ms


@dataclass(frozen=True)
class TpOp:
    op_name: str
    tp8_ms: float
    tp4_ms: float

    @property
    def ratio(self) -> float:
        return self.tp4_ms / self.tp8_ms if self.tp8_ms > 0 else float("inf")

    @property
    def scales_with_tp(self) -> bool:
        return 1.7 <= self.ratio <= 2.3


# From the offline probe (aic env, pr403 worktree). batch=128 all configs.
ATTR: dict[str, Attr] = {
    "tp8ep8-8k2k": Attr(
        "tp8ep8-8k2k", 8, 1, "8k2k",
        100.1540, 119.83, 119.83, 1.1624, 0.6203, "ep_comm_undermodeled_owes~20ms",
    ),
    "tp8ep8-32k3k": Attr(
        "tp8ep8-32k3k", 8, 1, "32k3k",
        149.7794, 304.94, 304.94, 1.9454, 1.2357, "longcontext_mixed_prefill",
    ),
    "tp4ep8dp2-8k2k": Attr(
        "tp4ep8dp2-8k2k", 4, 2, "8k2k",
        116.3865, 232.36, 116.18, 1.8784, 1.0869, "ep_comm_undermodeled_owes~116ms",
    ),
    "tp4ep8dp2-32k3k": Attr(
        "tp4ep8dp2-32k3k", 4, 2, "32k3k",
        166.5988, 600.63, 300.31, 3.2229, 2.1731, "ep_comm_plus_longcontext",
    ),
}

# Per-op generation latency, tp8ep8dp1 vs tp4ep8dp2 (8k2k, kv=9000, bs=128).
# The load-bearing ops for the EP-comm story (full 17-op dump in the raw csv).
TP_SCALING: dict[str, TpOp] = {
    "generation_moe": TpOp("generation_moe", 71.7758, 84.7642),
    "generation_attention": TpOp("generation_attention", 21.7362, 21.5231),
    "generation_moe_pre_dispatch": TpOp("generation_moe_pre_dispatch", 1.0213, 2.4552),
    "generation_moe_post_dispatch": TpOp("generation_moe_post_dispatch", 1.0213, 2.3336),
    "dense_gemm_sum": TpOp("dense_gemm_sum", 4.1593, 4.8631),
    "SIM_TOTAL": TpOp("SIM_TOTAL", 99.7139, 115.9392),
    "REAL_DECODE_ITER": TpOp("REAL_DECODE_ITER", 119.8300, 232.3600),
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

    # --- pure-decode ep8 probe: sim per-iter vs real decode iter (num_gpus=tp) ---
    for name, a in ATTR.items():
        rows.append(
            _row(
                "puredecode_ep8_probe",
                scenario=name,
                metric="per_iter_no_ovh_vs_real_iter_ng_tp",
                value_a=f"per_iter(no ovh)={a.per_iter_no_ovh_ms:.2f}ms",
                value_b=f"real_iter(ng=tp)={a.real_iter_ng_tp_ms:.2f}ms "
                f"(owed comm ~{a.owed_comm_ms:.0f}ms)",
                ratio=f"{a.sim_too_fast_ratio:.4f}",
                verdict="sim too fast: pure-decode compose under-covers real iter",
            )
        )

    # --- tp-degree check: dp cancels; real tp4 ~2x tp8; gap is EP comm ---
    rows.append(
        _row(
            "tp_degree_check",
            metric="dp_cancels_real_tp4_is_2x_tp8",
            value_a="dp CANCELS in per-GPU normalization: tokens_s_gpu = "
            "throughput_tok_s*(pp*dp)/(tp*pp*dp) = throughput_tok_s/tp "
            "(vllm_backend.py:699-723; simulator.py:303) -> NOT a normalization bug",
            value_b="real per-replica decode iter tp8dp1=119.8ms vs tp4dp2=232.4ms "
            "(~1.94x) while sim barely moves 100.2->116.4ms (1.16x)",
            ratio=f"{TP_SCALING['REAL_DECODE_ITER'].ratio:.4f}",
            verdict="the ~2x real gap is un-modeled EP all-to-all comm, not "
            "attention/dense tp-scaling and not normalization",
        )
    )

    # --- per-op tp8-vs-tp4 scaling probe ---
    for key in (
        "generation_moe",
        "generation_attention",
        "generation_moe_pre_dispatch",
        "generation_moe_post_dispatch",
        "dense_gemm_sum",
    ):
        op = TP_SCALING[key]
        if key == "generation_moe":
            note = "dominant ~72% of iter; flat across tp (EP fixed, only tokens 128->256)"
        elif key == "generation_attention":
            note = "FLAT (0.99x) - MLA latent KV replicated across tp, plausibly correct"
        elif "dispatch" in key:
            note = "only comm proxy; ~2-5ms total, far below owed 20/116ms -> under-modeled"
        else:
            note = "tiny (~4-5ms); partial tp-scaling"
        rows.append(
            _row(
                "tp_scaling_probe",
                scenario=op.op_name,
                metric="tp4_over_tp8_ratio",
                value_a=f"tp8={op.tp8_ms:.3f}ms",
                value_b=f"tp4={op.tp4_ms:.3f}ms",
                ratio=f"{op.ratio:.4f}",
                verdict=note,
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
                value_b=f"ovh0={a.sim_ovh0_ratio:.2f}x ovh90={a.sim_ovh90_ratio:.2f}x "
                f"owed~{a.owed_comm_ms:.0f}ms",
                ratio=f"{a.sim_ovh90_ratio:.4f}",
                verdict=a.dominant_cause,
            )
        )

    # --- candidate fixes ---
    rows.append(
        _row(
            "candidate_fixes",
            metric="phase397h_change_set",
            value_a="(i) model EP dispatch/combine all-to-all comm structurally "
            "(per-rank, scaling with attention_dp*decode_bs tokens and cross-dp-group "
            "hops) in place of the flat 90ms; (ii) verify generation_moe token count "
            "under attention_dp (128->256)",
            value_b="(iii) drop the flat 90ms ep8 overhead once comm is structural; "
            "(iv) hand residual long-context 32k3k mixed/prefill over-prediction "
            "(~1.95x dp1) to a follow-up mixed-path phase",
            verdict="enumerated for 397h; not implemented here",
        )
    )

    # --- verdict + next phase ---
    rows.append(
        _row(
            "verdict",
            metric="ep8_residual_root_cause",
            value_a="bidirectional residual = 90ms fudge over-correction (decode-heavy) "
            "+ un-modeled EP all-to-all comm (~20ms tp8dp1 to ~116ms tp4dp2) "
            "+ long-context mixed over-predict; dp normalization is CORRECT",
            value_b=f"Default {DEFAULT_READINESS}",
            verdict=(
                "not a single decode scaling error and not a normalization bug; the flat "
                "90ms ep8 overhead stands in for topology-dependent EP dispatch/combine "
                "communication that the simulator under-models"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="route_delta_step5_target",
            value_a="structuralize ep8 EP all-to-all comm (drop flat 90ms)",
            value_b="targets: tp8ep8-8k2k owed ~20ms comm; tp4ep8dp2-8k2k owed ~116ms; "
            "re-validate multi-config offline to max <= 1.470",
            verdict=(
                "phase397h: structural EP dispatch/combine comm term; then a mixed-path "
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
        "tp_degree_check",
        "tp_scaling_probe",
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
    if sum(1 for r in rows if r["row_type"] == "tp_scaling_probe") != 5:
        raise ValueError("tp_scaling_probe must have 5 rows")

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

    # Load-bearing 3: dp1 configs have equal num_gpus conventions (dp cancels).
    for name, a in ATTR.items():
        if a.dp == 1 and abs(a.real_iter_ng_tp_ms - a.real_iter_ng_tp_dp_ms) > 1e-6:
            raise ValueError(f"{name} dp1 num_gpus conventions must match")

    # Load-bearing 4: real tp4 decode iter is ~2x tp8 (per-GPU throughput near-equal).
    real_ratio = TP_SCALING["REAL_DECODE_ITER"].ratio
    if not 1.8 <= real_ratio <= 2.2:
        raise ValueError("real tp4 decode iter must be ~2x tp8")
    if TP_SCALING["SIM_TOTAL"].ratio >= 1.4:
        raise ValueError("sim total must barely move tp8->tp4 (<1.4x)")

    # Load-bearing 5: attention is flat across tp (MLA), NOT the gap.
    if not 0.85 <= TP_SCALING["generation_attention"].ratio <= 1.15:
        raise ValueError("generation_attention must be flat across tp")

    # Load-bearing 6: modeled EP dispatch is far below the owed comm.
    modeled_dispatch_tp4 = (
        TP_SCALING["generation_moe_pre_dispatch"].tp4_ms
        + TP_SCALING["generation_moe_post_dispatch"].tp4_ms
    )
    owed_tp4 = ATTR["tp4ep8dp2-8k2k"].owed_comm_ms
    if modeled_dispatch_tp4 >= 0.15 * owed_tp4:
        raise ValueError("modeled EP dispatch must be far below owed comm")

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "not a normalization bug" not in verdict["verdict"]:
        raise ValueError("verdict must explicitly retract the normalization claim")
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
        "> CORRECTION (supersedes commit c935c9e5): the first cut claimed a \"dp2 "
        "per-GPU normalization bug\". That was WRONG -- dp cancels exactly in the "
        "per-GPU normalization. This corrected cut re-attributes the residual to "
        "under-modeled EP all-to-all communication.",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | after 397f, why does the ep8 multi-config table still FAIL "
        "(2.17x) with a bidirectional residual (tp8ep8-8k2k 0.62x under, "
        "tp4ep8dp2-32k3k 2.17x over)? |",
        "| Answer | three causes: (1) the flat 90ms ep8 overhead is a miscalibrated "
        "fudge that over-corrects decode-heavy configs; (2) EP all-to-all comm is "
        "under-modeled (owed ~20ms tp8dp1 to ~116ms tp4dp2); (3) the 32k3k "
        "long-context configs over-predict in the mixed/prefill path. dp "
        "normalization is CORRECT. |",
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
        "calibrated to the pre-397f under-count. A single constant cannot cover a cost "
        "that scales with the topology: `tp8ep8-8k2k` owes ~20ms, `tp4ep8dp2-8k2k` owes "
        "~116ms.",
        "",
        "## 2. dp normalization is correct; the gap is un-modeled EP comm",
        "",
        "The per-GPU throughput normalizes as `tokens_s_gpu = throughput_tok_s*(pp*dp)"
        "/(tp*pp*dp) = throughput_tok_s/tp` (`vllm_backend.py`:699-723; "
        "`simulator.py`:303), so **dp cancels** -- there is no normalization bug. Yet "
        "the real per-replica decode iter roughly doubles from tp8dp1 to tp4dp2 while "
        "the sim barely moves:",
        "",
        "| op (8k2k, kv=9000, bs=128) | tp8 ms | tp4 ms | tp4/tp8 | reading |",
        "|---|---|---|---|---|",
    ]
    for key in (
        "generation_moe",
        "generation_attention",
        "generation_moe_pre_dispatch",
        "generation_moe_post_dispatch",
        "dense_gemm_sum",
        "SIM_TOTAL",
        "REAL_DECODE_ITER",
    ):
        op = TP_SCALING[key]
        reading = {
            "generation_moe": "dominant ~72%; flat (EP fixed, tokens 128->256)",
            "generation_attention": "flat -- MLA latent KV, plausibly correct",
            "generation_moe_pre_dispatch": "comm proxy, tiny",
            "generation_moe_post_dispatch": "comm proxy, tiny",
            "dense_gemm_sum": "tiny",
            "SIM_TOTAL": "sim barely moves (1.16x)",
            "REAL_DECODE_ITER": "real ~2x -> owed cost is EP comm",
        }[key]
        lines.append(
            f"| {op.op_name} | {op.tp8_ms:.3f} | {op.tp4_ms:.3f} | {op.ratio:.2f}x | {reading} |"
        )
    lines += [
        "",
        "The only comm proxy (`generation_moe_pre/post_dispatch`) totals ~2-5ms -- far "
        "below the owed 20ms (tp8dp1) / 116ms (tp4dp2). The missing per-iter cost is "
        "un-modeled EP dispatch/combine all-to-all communication, which grows with "
        "`attention_dp * decode_bs` tokens and with cross-dp-group hops.",
        "",
        "## 3. Per-config attribution",
        "",
        "| Config | tp | dp | shape | owed comm | dominant cause |",
        "|---|---|---|---|---|---|",
    ]
    for name, a in ATTR.items():
        lines.append(
            f"| {name} | {a.tp} | {a.dp} | {a.shape} | ~{a.owed_comm_ms:.0f}ms | {a.dominant_cause} |"
        )
    lines += [
        "",
        "## Verdict -- Route delta step 4 (corrected)",
        "",
        "- The ep8 residual is NOT a single decode scaling error and NOT a "
        "normalization bug (dp cancels).",
        "- (i) The flat 90ms ep8 overhead is a miscalibrated fudge standing in for "
        "topology-dependent EP dispatch/combine communication. (ii) That EP all-to-all "
        "comm is under-modeled (modeled ~2-5ms vs owed 20-116ms). (iii) The 32k3k "
        "long-context configs over-predict in the mixed/prefill path.",
        f"- Default AIC remains **{DEFAULT_READINESS}**.",
        f"- Next: **{NEXT_PHASE}** -- model EP dispatch/combine all-to-all comm "
        "structurally (per-rank, scaling with `attention_dp*decode_bs` and cross-group "
        "hops) in place of the flat 90ms; verify the `generation_moe` token count under "
        "`attention_dp`; then a mixed-path phase for the 32k3k residual. Re-validate "
        "the multi-config table offline.",
        "",
        "## Discipline",
        "",
        "- Verdict-only, offline: runtime / PerfDatabase not modified; the ep8 "
        "overhead was only SWEPT ({0,90}) as a diagnostic via the existing CLI switch, "
        "not tuned or persisted; no scope gating; no GPU/SSH; Default AIC No-Go.",
        f"- Raw evidence: `{OVERHEAD_RAW_CSV.relative_to(REPO_ROOT)}`, "
        f"`{ATTRIBUTION_RAW_CSV.relative_to(REPO_ROOT)}`, "
        f"`{TP_SCALING_RAW_CSV.relative_to(REPO_ROOT)}`.",
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

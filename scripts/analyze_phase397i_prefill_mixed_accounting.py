"""Phase397i ep8 baseline realignment + real per-iter TPOT (Route delta step 6).

VERDICT-ONLY. Simulator runtime is NOT modified. GPU/SSH authorized this round
for a READ-ONLY per-iteration TPOT capture; nothing is written to the validation
real-data source or the PerfDatabase.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs/iter_gap_investigation"
DEFAULT_OUTPUT_CSV = DOCS / "phase397i_prefill_mixed_accounting.csv"
DEFAULT_OUTPUT_MD = DOCS / "phase397i_prefill_mixed_accounting.md"
MAXDROP_RAW_CSV = DOCS / "phase397i_maxdrop_attn_raw.csv"
WALLSHARE_RAW_CSV = DOCS / "phase397i_prefill_wallshare_raw.csv"
REAL_TPOT_RAW_CSV = DOCS / "phase397i_real_tpot_raw.csv"

SOURCE = "phase397i_prefill_mixed_accounting"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_DB_VERSION = "0.12.0"
REAL_BASELINE_VERSION = "0.19.0"
STALE_BASELINE_VERSION = "0.17"
TOPOLOGY = "ep8_multi_config"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397j_decode_iter_baseline_realign"


@dataclass(frozen=True)
class BaselineRealign:
    """One ep8 config: sim vs stale-0.17 vs fresh-0.19.0 real (in-repo Phase164)."""
    name: str
    tp: int
    dp: int
    real_0p17: float
    real_0p19: float
    sim_ovh0: float

    @property
    def ratio_over_0p17(self) -> float:
        return self.sim_ovh0 / self.real_0p17

    @property
    def ratio_over_0p19(self) -> float:
        return self.sim_ovh0 / self.real_0p19

    @property
    def flips_sign(self) -> bool:
        # "over-prediction" vs 0.17 but "under-prediction" vs 0.19.0
        return self.ratio_over_0p17 > 1.0 and self.ratio_over_0p19 < 1.0


@dataclass(frozen=True)
class RealTpot:
    """Measured 0.19.0 pure-decode iter vs sim steady decode iter."""
    name: str
    tp: int
    dp: int
    real_decode_bs_per_replica: int
    real_decode_iter_ms: float   # measured pure-decode iter (mean)
    real_decode_iter_p99_ms: float
    sim_steady_iter_ms: float
    real_output_tok_s_gpu: float
    sim_output_tok_s_gpu: float

    @property
    def sim_iter_over_real(self) -> float:
        return self.sim_steady_iter_ms / self.real_decode_iter_ms

    @property
    def sim_out_over_real(self) -> float:
        return self.sim_output_tok_s_gpu / self.real_output_tok_s_gpu


@dataclass(frozen=True)
class MechA:
    """Mechanism A (max() drops context attention) magnitude, per config."""
    name: str
    isl: int
    dropped_frac_of_steady: float


# --- Baseline realignment: stale 0.17 gate vs in-repo Phase164 0.19.0 real ---
# 0.19.0 reals are the Phase164 clean manifest (real_output_tok_s_gpu); the
# tp8ep8/tp4dp2ep8 8k2k reruns this phase reproduced them (434.96 / 348.50).
REALIGN: dict[str, BaselineRealign] = {
    "tp8ep8-8k2k": BaselineRealign("tp8ep8-8k2k", 8, 1, 133.53, 437.89, 155.21),
    "tp8ep8-8k2k-bt65536": BaselineRealign(
        "tp8ep8-8k2k-bt65536", 8, 1, 138.47, 432.36, 154.95),
    "tp4ep8dp2-8k2k": BaselineRealign("tp4ep8dp2-8k2k", 4, 2, 137.72, 343.64, 258.69),
    "tp4ep8dp2-8k2k-bt65536": BaselineRealign(
        "tp4ep8dp2-8k2k-bt65536", 4, 2, 155.95, 401.20, 256.70),
}

# --- Real per-iter TPOT (this phase, authorized GPU/SSH, vLLM 0.19.0) ---
# tp4dp2 iter is the mean of DP0 (44.09, bs72) and DP1 (44.15, bs56).
REAL_TPOT: dict[str, RealTpot] = {
    "tp8ep8-8k2k": RealTpot(
        "tp8ep8-8k2k", 8, 1, 128, 34.29, 36.90, 102.994, 434.96, 155.21),
    "tp4ep8dp2-8k2k": RealTpot(
        "tp4ep8dp2-8k2k", 4, 2, 64, 44.12, 46.46, 123.590, 348.50, 258.69),
}

# --- Mechanism A magnitude (offline probe, dropped context attention) ---
MECH_A: dict[str, MechA] = {
    "tp8ep8-8k2k": MechA("tp8ep8-8k2k", 8000, 0.042),
    "tp8ep8-32k3k": MechA("tp8ep8-32k3k", 32000, 0.109),
    "tp4ep8dp2-8k2k": MechA("tp4ep8dp2-8k2k", 8000, 0.048),
    "tp4ep8dp2-32k3k": MechA("tp4ep8dp2-32k3k", 32000, 0.170),
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
    "real_baseline_version",
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
    "real_baseline_version": REAL_BASELINE_VERSION,
    "topology": TOPOLOGY,
    "next_allowed_phase": "",
    "runtime_modified": FALSE,
    "nearest_lookup_allowed": FALSE,
    "interpolation_allowed": TRUE,
    "extrapolation_allowed": FALSE,
    "fudge_factor_tuning_used": FALSE,
    "scope_gating_used": FALSE,
    "write_real_data_file": FALSE,
    "gpu_allowed": TRUE,
    "ssh_allowed": TRUE,
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


def analyze_phase397i() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- hypothesis under test (first row) ---
    rows.append(
        _row(
            "hypothesis_under_test",
            metric="is_ep8_overprediction_real_or_stale_0p17_artifact",
            value_a="397c-397h chased an ep8 'over-prediction' (1.16x-3.22x) measured "
            "against a stale vLLM 0.17 real baseline",
            value_b="397i realigns to in-repo Phase164 0.19.0 real (same shapes, same "
            "H200 box) and takes a real per-iter TPOT capture (authorized)",
            verdict="RESOLVED below: vs 0.19.0 the sim UNDER-predicts; its decode iter "
            "is ~3x too slow. The 'over-prediction' was a stale-0.17 artifact.",
        )
    )

    # --- Step A: baseline realignment (0.17 -> 0.19.0) flips the sign ---
    for name, b in REALIGN.items():
        rows.append(
            _row(
                "baseline_realign",
                scenario=name,
                metric="sim_vs_0p17_vs_0p19_real",
                value_a=f"real 0.17={b.real_0p17:.2f} -> sim/0.17={b.ratio_over_0p17:.2f}x "
                "(looked like over-prediction)",
                value_b=f"real 0.19.0={b.real_0p19:.2f} -> sim/0.19.0={b.ratio_over_0p19:.3f}x "
                "(actually under-prediction)",
                ratio=f"{b.ratio_over_0p19:.4f}",
                verdict="SIGN FLIPS: over vs 0.17, under vs 0.19.0"
                if b.flips_sign
                else "no flip",
            )
        )

    # --- Step B: real per-iter TPOT -> sim decode iter is ~3x too slow ---
    for name, t in REAL_TPOT.items():
        rows.append(
            _row(
                "real_tpot",
                scenario=name,
                metric="real_pure_decode_iter_vs_sim_steady_iter",
                value_a=f"real 0.19.0 pure-decode iter={t.real_decode_iter_ms:.2f}ms "
                f"(p99={t.real_decode_iter_p99_ms:.2f}, bs~{t.real_decode_bs_per_replica}/replica)",
                value_b=f"sim steady decode iter={t.sim_steady_iter_ms:.1f}ms "
                f"(overhead=0)",
                ratio=f"{t.sim_iter_over_real:.3f}",
                verdict=f"sim decode iter {t.sim_iter_over_real:.1f}x TOO SLOW vs 0.19.0 "
                f"-> under-predicts throughput ({t.sim_out_over_real:.2f}x real)",
            )
        )

    # --- Step B2: generation_moe alone exceeds the whole real decode iter ---
    tp8 = REAL_TPOT["tp8ep8-8k2k"]
    rows.append(
        _row(
            "decode_moe_oversized",
            scenario="tp8ep8-8k2k",
            metric="sim_generation_moe_vs_real_full_decode_iter",
            value_a="sim generation_moe (0.12.0 DB) ~= 71.8ms all-layer at tp8dp1 "
            "(decode token-count is CORRECT here: 128 = real bs)",
            value_b=f"real 0.19.0 FULL decode iter (MoE+attn+comm+dense) "
            f"={tp8.real_decode_iter_ms:.2f}ms",
            ratio=f"{71.8 / tp8.real_decode_iter_ms:.3f}",
            verdict="sim MoE alone > 2x the entire real iter -> decode-MoE magnitude "
            "(0.12.0 DB) is the primary over-cost, present even at correct tp8 token-count",
        )
    )

    # --- Step B3: dp batch-split / MoE token-count secondary error (tp4dp2) ---
    rows.append(
        _row(
            "dp_tokencount_error",
            scenario="tp4ep8dp2-8k2k",
            metric="sim_decode_tokens_per_replica_vs_real",
            value_a="sim charges decode/MoE at bs*attention_dp=256 tokens on ONE bs=128 "
            "replica",
            value_b="real 0.19.0 splits concurrency across 2 DP replicas: ~64 "
            "tokens/replica (DP0 bs72 / DP1 bs56)",
            ratio="4.0",
            verdict="secondary dp modeling error (per-replica bs and MoE token-count "
            "over-stated ~4x); but decode-MoE magnitude dominates even at tp8dp1",
        )
    )

    # --- Step C: refute 397h's 'sim decode composition ~124ms matches measured' ---
    tp4 = REAL_TPOT["tp4ep8dp2-8k2k"]
    rows.append(
        _row(
            "refute_397h_composition",
            scenario="tp4ep8dp2-8k2k",
            metric="397h_claimed_decode_iter_124ms_matches_measured",
            value_a="397h reconciled sim 123.6ms with a per-call fusedmoe module x ~45 "
            "layers (~124ms) and declared the decode composition correct",
            value_b=f"real 0.19.0 pure-decode iter={tp4.real_decode_iter_ms:.1f}ms "
            "/replica -- ~2.8x smaller than sim 123.6ms",
            ratio=f"{tp4.sim_iter_over_real:.3f}",
            verdict="REFUTED: the per-call x layer-count reconstruction over-counted "
            "MoE; the measured decode iter is ~3x below the sim",
        )
    )

    # --- Step D: mechanism A/B are minor AND help (not the fix target vs 0.19) ---
    a8 = MECH_A["tp8ep8-8k2k"]
    a32 = MECH_A["tp8ep8-32k3k"]
    a4_8 = MECH_A["tp4ep8dp2-8k2k"]
    a4_32 = MECH_A["tp4ep8dp2-32k3k"]
    rows.append(
        _row(
            "mechanism_a_minor",
            metric="maxdrop_context_attention_frac_of_steady",
            value_a=f"dropped/steady grows with isl: tp8 {a8.dropped_frac_of_steady:.3f}"
            f"@8k -> {a32.dropped_frac_of_steady:.3f}@32k; tp4 "
            f"{a4_8.dropped_frac_of_steady:.3f}@8k -> {a4_32.dropped_frac_of_steady:.3f}@32k",
            value_b="real and isl-growing, but only ~4-17% of wall; and it makes the sim "
            "FASTER, which vs 0.19.0 REDUCES the under-prediction gap",
            verdict="mechanism A is minor and NOT a fix target vs 0.19.0 (would worsen "
            "under-prediction); the decode-iter magnitude dominates",
        )
    )
    rows.append(
        _row(
            "mechanism_b_confounded",
            metric="prefill_wallshare_in_steady_window",
            value_a="batch-skip wall spans warmup+steady so raw skip/steady frac>1 is "
            "confounded; steady throughput is dominated by pure-decode skip wall",
            value_b="prefill/mixed under-charge has limited leverage on steady "
            "throughput -> mechanism B is not the dominant lever either",
            verdict="mechanism B minor/confounded; decode-iter magnitude is the lever",
        )
    )

    # --- Step E: candidate fixes for 397j ---
    rows.append(
        _row(
            "candidate_fixes",
            metric="phase397j_change_set",
            value_a="(1) MIGRATE multi-config real baseline from stale 0.17 to 0.19.0 "
            "(user-directed; drop 0.17); (2) FIX decode-iter magnitude: the 0.12.0-DB "
            "generation_moe (72-85ms) over-states 0.19.0 decode MoE ~2-3x -> refresh MoE "
            "latency to 0.19.0 or fix per-call->all-layer / token-count lookup",
            value_b="(3) FIX dp batch-split: per-replica bs=bs/dp and MoE token-count "
            "per replica (tp4dp2 uses 256, real ~64); (4) DECIDE DB version alignment "
            "(0.12.0 DB vs 0.19.0 real); target sim decode iter -> real band "
            "(tp8 ~34ms, tp4dp2 ~44ms/replica)",
            verdict="397j realigns baseline to 0.19.0 and fixes decode-iter magnitude "
            "(primary) + dp token-count (secondary); mechanisms A/B are NOT fixed",
        )
    )

    # --- verdict + next phase ---
    rows.append(
        _row(
            "verdict",
            metric="ep8_direction_reversed_against_0p19",
            value_a="vs in-repo 0.19.0 real the sim UNDER-predicts (tp8 0.36x, tp4dp2 "
            "0.74x); real per-iter decode is 34ms (tp8) / 44ms/replica (tp4dp2) vs sim "
            "103ms / 124ms -> sim decode iter ~3x too slow",
            value_b=f"Default {DEFAULT_READINESS}; the 'ep8 over-prediction' of 397c-h "
            "was a stale-0.17 artifact; 397h's 124ms decode composition is refuted",
            verdict="realign the real baseline to 0.19.0 and fix the over-sized decode "
            "iteration (decode-MoE magnitude + dp token-count); mechanisms A/B are minor",
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="route_delta_step7_target",
            value_a="phase397j: migrate multi-config real to 0.19.0, fix decode-iter "
            "magnitude and dp token-count, decide DB-version alignment",
            value_b="target: sim decode iter within the measured 0.19.0 band and "
            "multi-config error vs 0.19.0 within the validation gate",
            verdict="baseline realignment + decode-iter magnitude fix is the primary "
            "lead; prefill/mixed accounting (A/B) is minor and deferred",
            next_allowed_phase=NEXT_PHASE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "hypothesis_under_test":
        raise ValueError(f"Phase397i must start with hypothesis_under_test: {actual[0]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397i must end with verdict,next_phase: {actual[-2:]}")

    required = {
        "hypothesis_under_test",
        "baseline_realign",
        "real_tpot",
        "decode_moe_oversized",
        "dp_tokencount_error",
        "refute_397h_composition",
        "mechanism_a_minor",
        "mechanism_b_confounded",
        "candidate_fixes",
        "verdict",
        "next_phase",
    }
    if required - set(actual):
        raise ValueError(f"Phase397i missing row types: {required - set(actual)}")

    if sum(1 for r in rows if r["row_type"] == "baseline_realign") != len(REALIGN):
        raise ValueError("baseline_realign must have one row per config")
    if sum(1 for r in rows if r["row_type"] == "real_tpot") != len(REAL_TPOT):
        raise ValueError("real_tpot must have one row per measured config")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        # This round GPU/SSH were authorized for a read-only capture.
        if row["gpu_allowed"] != TRUE or row["ssh_allowed"] != TRUE:
            raise ValueError(f"{label} gpu_allowed/ssh_allowed must be {TRUE}")
        for guard in (
            "runtime_modified",
            "nearest_lookup_allowed",
            "extrapolation_allowed",
            "fudge_factor_tuning_used",
            "scope_gating_used",
            "write_real_data_file",
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

    # Load-bearing 1: the baseline realignment FLIPS the sign for every config
    # (over-prediction vs 0.17, under-prediction vs 0.19.0).
    for name, b in REALIGN.items():
        if not b.flips_sign:
            raise ValueError(f"{name} must flip sign from 0.17 to 0.19.0")
        if b.ratio_over_0p19 >= 1.0:
            raise ValueError(f"{name} sim must under-predict vs 0.19.0")

    # Load-bearing 2: the real per-iter decode iter is far below the sim steady
    # iter (sim ~3x too slow) for every measured config.
    for name, t in REAL_TPOT.items():
        if t.sim_iter_over_real <= 2.0:
            raise ValueError(f"{name} sim decode iter must be >2x the real 0.19.0 iter")
        if t.sim_out_over_real >= 1.0:
            raise ValueError(f"{name} sim throughput must be below 0.19.0 real")

    # Load-bearing 3: even at tp8dp1 (correct decode token-count 128), the sim MoE
    # alone exceeds the entire real decode iter -> magnitude, not just dp count.
    if 71.8 <= REAL_TPOT["tp8ep8-8k2k"].real_decode_iter_ms:
        raise ValueError("sim MoE (71.8ms) must exceed the real tp8 decode iter")

    # Load-bearing 4: mechanism A is minor (<20% of wall) and grows with isl.
    if MECH_A["tp8ep8-32k3k"].dropped_frac_of_steady <= MECH_A["tp8ep8-8k2k"].dropped_frac_of_steady:
        raise ValueError("mechanism A must grow with isl (tp8)")
    if MECH_A["tp4ep8dp2-32k3k"].dropped_frac_of_steady > 0.25:
        raise ValueError("mechanism A must stay minor (<25% of wall)")

    # Load-bearing 5: verdict reverses direction and names the decode-iter fix.
    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "UNDER-predicts" not in verdict["value_a"]:
        raise ValueError("verdict must state under-prediction vs 0.19.0")
    if "decode iter" not in verdict["verdict"]:
        raise ValueError("verdict must name the decode-iter fix")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")

    # Load-bearing 6: 397h's 124ms composition is explicitly refuted.
    refute = next(r for r in rows if r["row_type"] == "refute_397h_composition")
    if "REFUTED" not in refute["verdict"]:
        raise ValueError("refute_397h_composition must mark REFUTED")


def write_phase397i_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397i_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase397i ep8 Baseline Realignment + Real per-iter TPOT (Route delta step 6)",
        "",
        "> REVERSAL: 397c-397h chased an ep8 'over-prediction' (1.16x-3.22x) that was "
        "measured against a **stale vLLM 0.17 real baseline**. Realigning to the in-repo "
        "Phase164 **0.19.0** real (same shapes, same H200 box) flips the sign: the sim "
        "**under-predicts**. An authorized read-only per-iteration TPOT capture confirms "
        "the sim's decode iteration is **~3x too slow** vs 0.19.0.",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | is the ep8 over-prediction real, or an artifact of the stale 0.17 "
        "baseline? |",
        "| Answer | ARTIFACT. vs 0.19.0 the sim under-predicts (tp8 0.36x, tp4dp2 0.74x); "
        "its decode iter is ~3x too slow. |",
        "| Runtime / table / Default AIC | not modified (verdict-only). GPU/SSH used ONLY "
        "for a read-only 0.19.0 per-iter TPOT capture; nothing written to the validation "
        "real source or PerfDatabase. |",
        "",
        "## 1. Baseline realignment: 0.17 -> 0.19.0 flips the sign",
        "",
        "In-repo Phase164 (vLLM 0.19.0, H200, identical shapes) measured much higher real "
        "throughput than the 0.17 gate. This phase reproduced the two 8k2k reruns "
        "(tp8ep8 434.96 / tp4dp2ep8 348.50 tok/s/gpu), matching Phase164.",
        "",
        "| config | tp | dp | real 0.17 | real 0.19.0 | sim @ovh0 | sim/0.17 | sim/0.19.0 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, b in REALIGN.items():
        lines.append(
            f"| {name} | {b.tp} | {b.dp} | {b.real_0p17:.2f} | {b.real_0p19:.2f} | "
            f"{b.sim_ovh0:.2f} | {b.ratio_over_0p17:.2f}x | {b.ratio_over_0p19:.3f}x |"
        )
    lines += [
        "",
        "Every config flips: apparent over-prediction vs 0.17, actual under-prediction "
        "vs 0.19.0. The 0.19.0 ordering (tp8 437 > tp4dp2 343 per-gpu) is also the "
        "opposite of the sim (tp4dp2 259 > tp8 155).",
        "",
        "## 2. Real per-iter TPOT: sim decode iter is ~3x too slow",
        "",
        "`--enable-logging-iteration-details` emits, per scheduler step: "
        "`Iteration(N): C context requests, C context tokens, G generation requests, "
        "G generation tokens, iteration elapsed time: X ms`. Pure-decode iters "
        "(0 context, full batch) are the TPOT ground truth (tp4dp2 split by DP rank).",
        "",
        "| config | real decode iter (0.19.0) | p99 | sim steady iter | sim/real | sim out/real |",
        "|---|---|---|---|---|---|",
    ]
    for name, t in REAL_TPOT.items():
        lines.append(
            f"| {name} | {t.real_decode_iter_ms:.2f}ms (bs~{t.real_decode_bs_per_replica}"
            f"/replica) | {t.real_decode_iter_p99_ms:.2f}ms | {t.sim_steady_iter_ms:.1f}ms | "
            f"{t.sim_iter_over_real:.2f}x | {t.sim_out_over_real:.2f}x |"
        )
    lines += [
        "",
        "tp8ep8 pure-decode = 34.3ms at bs128 (p99 36.9); tp4dp2ep8 = ~44ms/replica "
        "(DP0 44.09 @bs72, DP1 44.15 @bs56). The sim's steady decode iter (103ms / "
        "123.6ms) is ~3x larger.",
        "",
        "## 3. Root cause: over-sized decode iteration",
        "",
        "- **Decode-MoE magnitude (primary).** At tp8dp1 the decode token-count is "
        "correct (128 = real bs), yet the sim `generation_moe` (~71.8ms, 0.12.0 DB) "
        "ALONE exceeds the entire real 0.19.0 decode iter (34.3ms). The 0.12.0-DB MoE "
        "latency over-states 0.19.0 decode MoE by ~2-3x.",
        "- **dp batch-split / token-count (secondary).** tp4dp2 charges decode/MoE at "
        "bs*attention_dp=256 tokens on one bs=128 replica, while real splits concurrency "
        "across 2 DP replicas (~64 tokens/replica).",
        "- This **refutes 397h**'s reconciliation that the sim decode composition (~124ms) "
        "matched the measured modules: that used a per-call fusedmoe module x ~45 layers, "
        "which over-counts. The measured decode iter is ~3x below the sim.",
        "",
        "## 4. Mechanisms A/B are minor (and not the fix target vs 0.19.0)",
        "",
        "- **Mechanism A** (`max()` drops context attention at overlap_factor=0): real "
        "and isl-growing, but only ~4-17% of steady wall (tp8 0.042@8k -> 0.109@32k; "
        "tp4 0.048@8k -> 0.170@32k). It makes the sim FASTER, so vs 0.19.0 it REDUCES "
        "the under-prediction gap -- fixing it would worsen the error.",
        "- **Mechanism B** (prefill wall diluted by batch-skip): the raw skip/steady "
        "fraction is confounded (skip wall spans warmup); steady throughput is dominated "
        "by pure-decode skip wall, so prefill under-charge has limited leverage.",
        "",
        "## Verdict -- Route delta step 6",
        "",
        "- The ep8 'over-prediction' of 397c-397h was a **stale-0.17 baseline artifact**. "
        "Against in-repo 0.19.0 real the sim **under-predicts** (tp8 0.36x, tp4dp2 0.74x).",
        "- The sim's **decode iteration is ~3x too slow** vs 0.19.0 (real 34ms/44ms vs "
        "sim 103ms/124ms), dominated by decode-MoE magnitude (0.12.0 DB) with a secondary "
        "dp token-count error.",
        "- Mechanisms A/B are minor and are NOT fix targets vs 0.19.0.",
        f"- Default AIC remains **{DEFAULT_READINESS}**.",
        f"- Next: **{NEXT_PHASE}** -- migrate the multi-config real baseline to 0.19.0 "
        "(drop 0.17), fix the over-sized decode iteration (decode-MoE magnitude + dp "
        "token-count), and decide 0.12.0-DB vs 0.19.0 version alignment.",
        "",
        "## Discipline",
        "",
        "- Verdict-only: simulator runtime / PerfDatabase NOT modified; no overhead/fudge "
        "tuning; no scope gating; Default AIC No-Go.",
        "- The ONLY runtime-adjacent change is the collector benchmark harness adding the "
        f"`--enable-logging-iteration-details` measurement flag (v{REAL_BASELINE_VERSION} "
        "serve). It changes logging only, not vLLM compute or the simulator.",
        "- GPU/SSH authorized this round for a READ-ONLY per-iter TPOT capture; no real "
        "data written into the validation source or PerfDatabase.",
        "- Note: the 0.19.0 weights (`models--moonshotai--Kimi-K2.5`) load as "
        "compressed-tensors WNA16-Marlin MoE; the runner's recorded `quantization=fp8` is "
        "its hardcoded default. Aggregate throughput still matches Phase164.",
        "- Raw evidence: `phase397i_maxdrop_attn_raw.csv`, "
        "`phase397i_prefill_wallshare_raw.csv`, `phase397i_real_tpot_raw.csv`, "
        "`phase397i_tpot/*/decode_iter_extract.txt`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397i()
    write_phase397i_csv(args.output_csv, rows)
    write_phase397i_md(args.output_md, rows)


if __name__ == "__main__":
    main()

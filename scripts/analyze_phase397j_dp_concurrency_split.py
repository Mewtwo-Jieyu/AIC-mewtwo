"""Phase397j DP concurrency / MoE token-count modeling patch (cb_sim agg).

RUNTIME MODIFIED. This phase edits `_run_agg_cb_sim` so the deployment's GLOBAL
concurrency is split across dp attention replicas (per_replica = ceil(b/dp)),
matching a real DP benchmark that specifies a global max-concurrency. It is a
version-independent structural fix; the absolute under-prediction residual is a
separate DB-version / overhead issue and is out of scope here. No GPU/SSH.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "docs/iter_gap_investigation"
DEFAULT_OUTPUT_CSV = DOCS / "phase397j_dp_concurrency_split.csv"
DEFAULT_OUTPUT_MD = DOCS / "phase397j_dp_concurrency_split.md"
MULTI_RAW_CSV = DOCS / "phase397j_multi_config_before_after_raw.csv"
TOKENCOUNT_RAW_CSV = DOCS / "phase397j_dp_tokencount_raw.csv"

SOURCE = "phase397j_dp_concurrency_split"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_DB_VERSION = "0.12.0"
REAL_BASELINE_VERSION = "0.17"
TOPOLOGY = "ep8_multi_config"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397k_decode_iter_db_version_realign"


@dataclass(frozen=True)
class TokenCount:
    """DP concurrency / MoE token-count bug state (before) and fix (after).

    per_replica_bs is the concurrency PASSED to CBSimulator.run (the decode
    batch size on one tp replica). moe_tokens is that batch after the MoE op's
    x*attention_dp gather. global_bs is the reported global batch.
    """
    name: str
    tp: int
    dp: int
    per_replica_bs_before: int
    per_replica_bs_after: int
    moe_tokens_before: int
    moe_tokens_after: int
    global_bs_before: int
    global_bs_after: int
    real_gathered_moe_tokens: int  # measured DP0+DP1 gathered decode tokens

    @property
    def is_dp1(self) -> bool:
        return self.dp == 1

    @property
    def unchanged(self) -> bool:
        return (
            self.per_replica_bs_before == self.per_replica_bs_after
            and self.moe_tokens_before == self.moe_tokens_after
            and self.global_bs_before == self.global_bs_after
        )

    @property
    def moe_matches_real_after(self) -> bool:
        return self.moe_tokens_after == self.real_gathered_moe_tokens


@dataclass(frozen=True)
class MultiConfig:
    """One multi-config point: real (0.17) vs sim before/after the dp fix."""
    name: str
    tp: int
    dp: int
    max_bt: int
    real: float
    sim_before: float
    sim_after: float

    @staticmethod
    def _abs_err(sim: float, real: float) -> float:
        if sim <= 0 or real <= 0:
            return float("inf")
        ratio = sim / real
        return max(ratio, 1.0 / ratio)

    @property
    def err_before(self) -> float:
        return self._abs_err(self.sim_before, self.real)

    @property
    def err_after(self) -> float:
        return self._abs_err(self.sim_after, self.real)

    @property
    def dp1_bitforbit_unchanged(self) -> bool:
        return self.dp == 1 and abs(self.sim_before - self.sim_after) < 1e-6


@dataclass(frozen=True)
class RatioPair:
    """tp8 (dp1) vs tp4dp2 per-GPU throughput ratio at one shape."""
    shape: str
    real_tp8: float
    real_tp4dp2: float
    sim_tp8_before: float
    sim_tp4dp2_before: float
    sim_tp8_after: float
    sim_tp4dp2_after: float

    @property
    def real_ratio(self) -> float:
        return self.real_tp8 / self.real_tp4dp2

    @property
    def sim_ratio_before(self) -> float:
        return self.sim_tp8_before / self.sim_tp4dp2_before

    @property
    def sim_ratio_after(self) -> float:
        return self.sim_tp8_after / self.sim_tp4dp2_after

    @property
    def dist_before(self) -> float:
        return abs(self.sim_ratio_before - self.real_ratio)

    @property
    def dist_after(self) -> float:
        return abs(self.sim_ratio_after - self.real_ratio)

    @property
    def improves(self) -> bool:
        return self.dist_after < self.dist_before


# --- DP concurrency / MoE token-count: bug state (before) vs fix (after) ---
# Captured by a read-only probe on the cb_sim agg path (overlap=0, overhead=90):
# per_replica_bs = concurrency passed to CBSimulator.run; moe_tokens = that batch
# after the MoE op's x*attention_dp gather; global_bs = reported global batch.
# real_gathered_moe_tokens is the measured 0.19.0 decode gather (DP0 72 + DP1 56).
TOKENCOUNT: dict[str, TokenCount] = {
    "tp8ep8-8k2k": TokenCount(
        "tp8ep8-8k2k", 8, 1, 128, 128, 128, 128, 128, 128, 128),
    "tp4ep8dp2-8k2k": TokenCount(
        "tp4ep8dp2-8k2k", 4, 2, 128, 64, 256, 128, 256, 128, 128),
}

# --- Multi-config throughput: real (0.17 gate) vs sim before/after the fix ---
# tok/s/GPU, output-only; overlap_factor=0.0, ep8 overhead=90ms; 0.12.0 DB.
# sim_before is the pre-fix runtime (b per replica); sim_after is post-fix
# (ceil(b/dp) per replica). dp=1 rows are byte-identical before/after.
MULTI: list[MultiConfig] = [
    MultiConfig("K2.5-tp8ep8-8k2k", 8, 1, 8000, 133.528, 82.8288, 82.8288),
    MultiConfig("K2.5-tp8ep8-32k3k", 8, 1, 32000, 52.469, 64.8351, 64.8351),
    MultiConfig("K2.5-tp4ep8dp2-8k2k", 4, 2, 8000, 137.716, 149.6837, 82.7432),
    MultiConfig("K2.5-tp4ep8dp2-32k3k", 4, 2, 32000, 53.277, 115.7772, 57.0600),
    MultiConfig("K2.5-tp8ep8-8k2k-bt65536", 8, 1, 65536, 138.470, 82.7803, 82.7803),
    MultiConfig(
        "K2.5-tp4ep8dp2-8k2k-bt65536", 4, 2, 65536, 155.952, 149.0570, 82.5529),
]

# --- tp8-vs-tp4dp2 per-GPU ratio (the version-independent fix target) ---
RATIOS: list[RatioPair] = [
    RatioPair("8k2k", 133.528, 137.716, 82.8288, 149.6837, 82.8288, 82.7432),
    RatioPair("32k3k", 52.469, 53.277, 64.8351, 115.7772, 64.8351, 57.0600),
    RatioPair("bt65536", 138.470, 155.952, 82.7803, 149.0570, 82.7803, 82.5529),
]


def _multi_max_before() -> float:
    return max(m.err_before for m in MULTI)


def _multi_max_after() -> float:
    return max(m.err_after for m in MULTI)


def _multi_mean_before() -> float:
    return sum(m.err_before for m in MULTI) / len(MULTI)


def _multi_mean_after() -> float:
    return sum(m.err_after for m in MULTI) / len(MULTI)


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
    # phase397j edits the simulator runtime (_run_agg_cb_sim dp split).
    "runtime_modified": TRUE,
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


def analyze_phase397j() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- hypothesis under test (first row) ---
    rows.append(
        _row(
            "hypothesis_under_test",
            metric="does_dp_concurrency_split_fix_the_tp_vs_tpdp_ranking",
            value_a="the cb_sim agg path ran the deployment's full concurrency b on ONE "
            "dp replica (concurrency=b) and charged the MoE op at b*attention_dp tokens, "
            "so a tp4dp2 config modeled 2x the real global concurrency and 2x MoE tokens",
            value_b="phase397j interprets b as GLOBAL concurrency and splits it per "
            "replica (per_replica=ceil(b/dp)); a real DP benchmark specifies a global "
            "max-concurrency (tp4dp2 128 total -> ~64/replica; measured DP0 72 / DP1 56)",
            verdict="RESOLVED below: the fix restores the tp8-vs-tp4dp2 per-GPU ranking "
            "to match real (both ~equal) and is bit-for-bit identical for dp=1; the "
            "absolute under-prediction residual is a separate DB-version issue",
        )
    )

    # --- Step A: DP token-count bug state (before) vs fix (after) ---
    for name, t in TOKENCOUNT.items():
        rows.append(
            _row(
                "dp_tokencount_fix",
                scenario=name,
                metric="per_replica_bs__moe_tokens__global_bs__before_to_after",
                value_a=f"BEFORE: per_replica_bs={t.per_replica_bs_before}, "
                f"moe_tokens={t.moe_tokens_before}, global_bs={t.global_bs_before}",
                value_b=f"AFTER: per_replica_bs={t.per_replica_bs_after}, "
                f"moe_tokens={t.moe_tokens_after}, global_bs={t.global_bs_after} "
                f"(real gathered decode tokens ~= {t.real_gathered_moe_tokens})",
                ratio=f"{t.moe_tokens_before / t.moe_tokens_after:.2f}",
                verdict="dp=1 UNCHANGED (no-op)"
                if t.unchanged
                else "dp>1: per-replica bs and MoE token-count halved -> match real",
            )
        )

    # --- Step B: multi-config throughput, before vs after (baseline stays 0.17) ---
    for m in MULTI:
        rows.append(
            _row(
                "multi_config_before_after",
                scenario=m.name,
                metric="sim_tok_s_gpu_before_vs_after_vs_real_0p17",
                value_a=f"real={m.real:.2f}, sim_before={m.sim_before:.2f} "
                f"({m.err_before:.2f}x)",
                value_b=f"sim_after={m.sim_after:.2f} ({m.err_after:.2f}x)",
                ratio=f"{m.sim_after / m.real:.4f}",
                verdict="dp=1 bit-for-bit unchanged"
                if m.dp1_bitforbit_unchanged
                else ("dp>1 err improves" if m.err_after < m.err_before
                      else "dp>1 err now shows the tp8-shared under-prediction"),
            )
        )

    # --- Step C: the tp8-vs-tp4dp2 ratio fix (version-independent win) ---
    for r in RATIOS:
        rows.append(
            _row(
                "ratio_fix",
                scenario=r.shape,
                metric="tp8_over_tp4dp2_per_gpu_ratio_vs_real",
                value_a=f"real ratio={r.real_ratio:.3f}; sim BEFORE={r.sim_ratio_before:.3f} "
                f"(|err|={r.dist_before:.3f}, tp4dp2 looked ~1.8x faster than tp8)",
                value_b=f"sim AFTER={r.sim_ratio_after:.3f} (|err|={r.dist_after:.3f}, "
                f"tp8 ~= tp4dp2 like real)",
                ratio=f"{r.sim_ratio_after:.4f}",
                verdict="RATIO FIXED: sim ranking now matches real"
                if r.improves
                else "no improvement",
            )
        )

    # --- Step D: aggregate error before/after (max improves, mean ~flat) ---
    rows.append(
        _row(
            "multi_config_aggregate",
            metric="multi_config_abs_err_max_and_mean_before_to_after",
            value_a=f"BEFORE: max={_multi_max_before():.2f}x mean={_multi_mean_before():.2f}x "
            "(worst = tp4dp2-32k3k 2.17x over-prediction from the dp double-count)",
            value_b=f"AFTER: max={_multi_max_after():.2f}x mean={_multi_mean_after():.2f}x "
            "(worst-config error improves; mean rises slightly as the two coincidentally "
            "accurate tp4dp2 configs now share the tp8 under-prediction)",
            ratio=f"{_multi_max_after() / _multi_max_before():.3f}",
            verdict="max multi-config error IMPROVES 2.17x -> 1.89x; the fix removes a "
            "real over-count, not tunes a fudge",
        )
    )

    # --- Step E: residual is a DB-version issue, not a dp bug (out of scope) ---
    rows.append(
        _row(
            "residual_out_of_scope",
            metric="absolute_under_prediction_is_shared_by_tp8_dp1",
            value_a="tp8 (dp=1) is 0.62x vs 0.17 and UNCHANGED by this fix -> the absolute "
            "under-prediction is not a dp bug; after the fix tp4dp2 shares the same ~0.6x",
            value_b="closing the absolute gap needs a 0.19.0 DB re-collection (0.12.0-DB "
            "vs 0.19.0-engine version gap; older/slower kernels); a separate data phase, "
            "not a sim edit",
            verdict="the pre-fix tp4dp2 'accuracy' was two errors cancelling (dp over-count "
            "+ version under-prediction); the fix exposes the honest version residual",
        )
    )

    # --- Step F: cross-backend note (batch_sync / trtllm share the assumption) ---
    rows.append(
        _row(
            "cross_backend_followup",
            metric="batch_sync_and_trtllm_carry_the_same_per_replica_assumption",
            value_a="batch_sync agg (vllm_backend scale_factor=pp*dp, global_bs=b*dp) and "
            "trtllm_backend (identical pattern) also treat b as per-replica",
            value_b="phase397j scopes the fix to the cb_sim agg path (the validated path); "
            "batch_sync/trtllm alignment is a recorded follow-up, not changed here",
            verdict="scoped to cb_sim agg to keep blast radius minimal; dp=1 unaffected",
        )
    )

    # --- verdict + next phase ---
    rows.append(
        _row(
            "verdict",
            metric="dp_concurrency_split_fixes_ranking_residual_is_version",
            value_a="the dp concurrency / MoE token-count over-count is FIXED: tp4dp2 now "
            "models b/dp per replica and MoE at the real gathered token-count; the "
            "tp8-vs-tp4dp2 per-GPU ranking matches real and dp=1 is bit-for-bit unchanged",
            value_b=f"Default {DEFAULT_READINESS}: the remaining absolute under-prediction "
            "(~0.6x, shared by tp8 dp=1) is a 0.12.0-DB-vs-0.19.0-engine version gap, out "
            "of scope; multi-config max error improves 2.17x -> 1.89x",
            verdict="version-independent dp modeling patch landed; MULTI_CONFIG gate left "
            "unchanged (still No-Go); absolute alignment deferred to a DB-version phase",
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="phase397k_target",
            value_a="phase397k: re-collect a vLLM 0.19.0 full perf DB (attention/gemm/moe) "
            "so the absolute decode-iter magnitude matches the 0.19.0 engine",
            value_b="target: with the dp ranking already correct, a version-matched DB "
            "brings the absolute multi-config error within the validation gate",
            verdict="dp modeling patch is done; the absolute residual is a data/DB-version "
            "re-collection, tracked as the next phase",
            next_allowed_phase=NEXT_PHASE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "hypothesis_under_test":
        raise ValueError(f"Phase397j must start with hypothesis_under_test: {actual[0]}")
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397j must end with verdict,next_phase: {actual[-2:]}")

    required = {
        "hypothesis_under_test",
        "dp_tokencount_fix",
        "multi_config_before_after",
        "ratio_fix",
        "multi_config_aggregate",
        "residual_out_of_scope",
        "cross_backend_followup",
        "verdict",
        "next_phase",
    }
    if required - set(actual):
        raise ValueError(f"Phase397j missing row types: {required - set(actual)}")

    if sum(1 for r in rows if r["row_type"] == "dp_tokencount_fix") != len(TOKENCOUNT):
        raise ValueError("dp_tokencount_fix must have one row per probed config")
    if sum(1 for r in rows if r["row_type"] == "multi_config_before_after") != len(MULTI):
        raise ValueError("multi_config_before_after must have one row per config")
    if sum(1 for r in rows if r["row_type"] == "ratio_fix") != len(RATIOS):
        raise ValueError("ratio_fix must have one row per shape")

    for row in rows:
        label = row["row_type"]
        if set(row) != set(FIELDNAMES):
            raise ValueError(f"{label} fields do not match FIELDNAMES")
        if row["source"] != SOURCE:
            raise ValueError(f"{label} bad source")
        # phase397j edits the simulator runtime.
        if row["runtime_modified"] != TRUE:
            raise ValueError(f"{label} runtime_modified must be {TRUE}")
        # no GPU/SSH this round.
        if row["gpu_allowed"] != FALSE or row["ssh_allowed"] != FALSE:
            raise ValueError(f"{label} gpu_allowed/ssh_allowed must be {FALSE}")
        for guard in (
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

    # Load-bearing 1: dp=1 configs are bit-for-bit unchanged by the fix.
    for m in MULTI:
        if m.dp == 1 and not m.dp1_bitforbit_unchanged:
            raise ValueError(f"{m.name} (dp=1) must be bit-for-bit unchanged")
    for name, t in TOKENCOUNT.items():
        if t.is_dp1 and not t.unchanged:
            raise ValueError(f"{name} (dp=1) token-count must be unchanged")

    # Load-bearing 2: the dp>1 fix halves per-replica bs and MoE tokens, and the
    # post-fix MoE token-count matches the real gathered decode tokens.
    tp4 = TOKENCOUNT["tp4ep8dp2-8k2k"]
    if tp4.per_replica_bs_before != 128 or tp4.per_replica_bs_after != 64:
        raise ValueError("tp4dp2 per-replica bs must go 128 -> 64")
    if tp4.moe_tokens_before != 256 or tp4.moe_tokens_after != 128:
        raise ValueError("tp4dp2 MoE tokens must go 256 -> 128")
    if not tp4.moe_matches_real_after:
        raise ValueError("tp4dp2 post-fix MoE tokens must match real gathered (~128)")

    # Load-bearing 3: the tp8-vs-tp4dp2 ratio moves toward real for every shape.
    for r in RATIOS:
        if not r.improves:
            raise ValueError(f"ratio_fix {r.shape} must move sim ratio toward real")
    # And specifically the ranking inverts: before tp4dp2 faster (ratio<1),
    # after tp8 >= tp4dp2 (ratio ~>=1) at 8k2k.
    r8 = next(r for r in RATIOS if r.shape == "8k2k")
    if not (r8.sim_ratio_before < 0.75 <= r8.sim_ratio_after):
        raise ValueError("8k2k ranking must invert from tp4dp2-faster to tp8>=tp4dp2")

    # Load-bearing 4: the worst-config multi-config error improves.
    if not (_multi_max_after() < _multi_max_before()):
        raise ValueError("multi-config max error must improve after the fix")

    # Load-bearing 5: the residual is shared by tp8 (dp=1) -> a version issue.
    tp8 = next(m for m in MULTI if m.name == "K2.5-tp8ep8-8k2k")
    if tp8.err_after < 1.4:
        raise ValueError("tp8 (dp=1) must still under-predict (version residual)")

    # Load-bearing 6: verdict names the dp fix and the version residual + No-Go.
    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "FIXED" not in verdict["value_a"]:
        raise ValueError("verdict must state the dp fix landed")
    if "version" not in verdict["value_b"]:
        raise ValueError("verdict must attribute the residual to the DB version")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase397j_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_multi_raw_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "tp", "dp", "max_bt", "real_output_tok_s_gpu",
        "sim_before", "sim_after", "err_before", "err_after",
        "dp1_bitforbit_unchanged",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for m in MULTI:
            writer.writerow({
                "name": m.name, "tp": m.tp, "dp": m.dp, "max_bt": m.max_bt,
                "real_output_tok_s_gpu": f"{m.real:.4f}",
                "sim_before": f"{m.sim_before:.4f}",
                "sim_after": f"{m.sim_after:.4f}",
                "err_before": f"{m.err_before:.4f}",
                "err_after": f"{m.err_after:.4f}",
                "dp1_bitforbit_unchanged": TRUE if m.dp1_bitforbit_unchanged else FALSE,
            })


def write_tokencount_raw_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "tp", "dp", "per_replica_bs_before", "per_replica_bs_after",
        "moe_tokens_before", "moe_tokens_after", "global_bs_before",
        "global_bs_after", "real_gathered_moe_tokens", "moe_matches_real_after",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for t in TOKENCOUNT.values():
            writer.writerow({
                "name": t.name, "tp": t.tp, "dp": t.dp,
                "per_replica_bs_before": t.per_replica_bs_before,
                "per_replica_bs_after": t.per_replica_bs_after,
                "moe_tokens_before": t.moe_tokens_before,
                "moe_tokens_after": t.moe_tokens_after,
                "global_bs_before": t.global_bs_before,
                "global_bs_after": t.global_bs_after,
                "real_gathered_moe_tokens": t.real_gathered_moe_tokens,
                "moe_matches_real_after": TRUE if t.moe_matches_real_after else FALSE,
            })


def write_phase397j_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    r8 = next(r for r in RATIOS if r.shape == "8k2k")
    lines = [
        "# Phase397j DP Concurrency / MoE Token-count Modeling Patch (cb_sim agg)",
        "",
        "> RUNTIME MODIFIED. The cb_sim agg path (`_run_agg_cb_sim`) ran the "
        "deployment's full concurrency `b` on ONE dp replica and charged the MoE op at "
        "`b*attention_dp` tokens, so a `dp>1` config modeled 2x the real global "
        "concurrency and 2x the MoE token-count -- inverting the tp8-vs-tp4dp2 ranking. "
        "This phase interprets `b` as GLOBAL concurrency and splits it per replica "
        "(`per_replica = ceil(b/dp)`). Version-independent structural fix; the absolute "
        "under-prediction residual is a separate DB-version issue.",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | does the dp concurrency split fix the tp8-vs-tp4dp2 ranking? |",
        "| Answer | YES. Ranking now matches real (both ~equal); dp=1 bit-for-bit "
        "unchanged; multi-config max error improves 2.17x -> 1.89x. |",
        "| Runtime / table / Default AIC | runtime modified (dp split only); PerfDatabase "
        "and the 0.17 real baseline NOT changed; no GPU/SSH; Default remains No-Go. |",
        "",
        "## 1. The DP token-count over-count (before -> after)",
        "",
        "A read-only probe on the agg path (concurrency passed to `CBSimulator.run`; MoE "
        "tokens after the op's `x*attention_dp` gather; reported `global_bs`):",
        "",
        "| config | tp | dp | per-replica bs | MoE tokens | global_bs | real gathered |",
        "|---|---|---|---|---|---|---|",
    ]
    for t in TOKENCOUNT.values():
        lines.append(
            f"| {t.name} | {t.tp} | {t.dp} | "
            f"{t.per_replica_bs_before} -> {t.per_replica_bs_after} | "
            f"{t.moe_tokens_before} -> {t.moe_tokens_after} | "
            f"{t.global_bs_before} -> {t.global_bs_after} | "
            f"{t.real_gathered_moe_tokens} |"
        )
    lines += [
        "",
        "tp8 (dp=1) is a no-op (`ceil(b/1)=b`). tp4dp2 now decodes 64/replica and gathers "
        "128 MoE tokens -- matching the measured 0.19.0 decode gather (DP0 72 + DP1 56).",
        "",
        "## 2. Multi-config throughput (baseline stays 0.17)",
        "",
        "tok/s/GPU output-only; overlap_factor=0.0, ep8 overhead=90ms, 0.12.0 DB.",
        "",
        "| config | tp | dp | real | sim before | err | sim after | err |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in MULTI:
        lines.append(
            f"| {m.name} | {m.tp} | {m.dp} | {m.real:.1f} | {m.sim_before:.1f} | "
            f"{m.err_before:.2f}x | {m.sim_after:.1f} | {m.err_after:.2f}x |"
        )
    lines += [
        "",
        f"Max error {_multi_max_before():.2f}x -> {_multi_max_after():.2f}x (worst config "
        f"improves); mean {_multi_mean_before():.2f}x -> {_multi_mean_after():.2f}x. dp=1 "
        "rows are byte-identical before/after.",
        "",
        "## 3. The ranking fix (version-independent win)",
        "",
        "tp8 (dp1) vs tp4dp2 per-GPU throughput ratio, sim vs real:",
        "",
        "| shape | real ratio | sim before | sim after |",
        "|---|---|---|---|",
    ]
    for r in RATIOS:
        lines.append(
            f"| {r.shape} | {r.real_ratio:.3f} | {r.sim_ratio_before:.3f} | "
            f"{r.sim_ratio_after:.3f} |"
        )
    lines += [
        "",
        f"At 8k2k the sim went from tp4dp2 looking ~1.8x faster than tp8 "
        f"(ratio {r8.sim_ratio_before:.3f}) to tp8 ~= tp4dp2 (ratio {r8.sim_ratio_after:.3f}), "
        f"matching real ({r8.real_ratio:.3f}). Every shape moves toward real.",
        "",
        "## 4. Residual is a DB-version issue, not a dp bug",
        "",
        "- tp8 (dp=1) is 0.62x vs 0.17 and UNCHANGED by this fix. After the fix tp4dp2 "
        "shares the same ~0.6x. The absolute under-prediction is therefore not a dp bug.",
        "- The pre-fix tp4dp2 'accuracy' (1.09x / 0.96x) was two errors cancelling: the dp "
        "over-count (over-prediction) masked the tp8-shared version under-prediction.",
        "- Closing the absolute gap needs a vLLM 0.19.0 full DB re-collection (the 0.12.0 "
        "DB models older/slower kernels) -- a separate data phase, not a sim edit.",
        "",
        "## 5. Cross-backend follow-up (not changed here)",
        "",
        "`batch_sync` agg (`scale_factor = pp*dp`, `global_bs = b*dp`) and `trtllm_backend` "
        "carry the identical per-replica assumption. phase397j scopes the fix to the "
        "cb_sim agg path (the validated path); their alignment is a recorded follow-up.",
        "",
        "## Verdict",
        "",
        "- The dp concurrency / MoE token-count over-count is **FIXED**: tp4dp2 models "
        "`b/dp` per replica and MoE at the real gathered token-count; the tp8-vs-tp4dp2 "
        "per-GPU ranking matches real; dp=1 is bit-for-bit unchanged.",
        f"- Multi-config max error improves {_multi_max_before():.2f}x -> "
        f"{_multi_max_after():.2f}x.",
        "- The remaining absolute under-prediction (~0.6x, shared by tp8 dp=1) is a "
        "0.12.0-DB-vs-0.19.0-engine **version** gap, out of scope.",
        f"- `MULTI_CONFIG_MAX_ACCEPTANCE` left unchanged; Default AIC remains "
        f"**{DEFAULT_READINESS}**.",
        f"- Next: **{NEXT_PHASE}** -- re-collect a vLLM 0.19.0 full perf DB so the "
        "absolute decode-iter magnitude matches the 0.19.0 engine.",
        "",
        "## Discipline",
        "",
        "- Runtime change is scoped to `_run_agg_cb_sim`'s dp concurrency split; no MoE "
        "op or PerfDatabase change; no overhead/fudge tuning; no scope gating.",
        "- The 0.17 multi-config real baseline is unchanged; the acceptance gate is not "
        "loosened; Default AIC stays No-Go.",
        "- dp=1 configs verified bit-for-bit identical before/after.",
        "- Raw evidence: `phase397j_multi_config_before_after_raw.csv`, "
        "`phase397j_dp_tokencount_raw.csv`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397j()
    write_phase397j_csv(args.output_csv, rows)
    write_phase397j_md(args.output_md, rows)
    write_multi_raw_csv(MULTI_RAW_CSV)
    write_tokencount_raw_csv(TOKENCOUNT_RAW_CSV)


if __name__ == "__main__":
    main()

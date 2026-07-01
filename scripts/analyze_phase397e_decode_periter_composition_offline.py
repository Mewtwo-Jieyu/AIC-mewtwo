"""Phase397e decode per-iteration composition offline verdict (Route delta step 2).

Phase397d fixed the decode skip-trapezoid seed; that unmasked a decode
per-iteration NON-ATTENTION under-count (3 gates swung from ~0.66x to ~1.9-2.7x
real). Phase397e is OFFLINE and verdict-only: it determines the `generation_moe`
table semantics from source, proves the `/tp_size` in iteration_latency.py is a
double-count, and reconciles the per-iteration decode composition against the
real per-iteration budget to hand phase397f a physically-grounded target. It
does NOT modify runtime, does NOT write the PerfDatabase, needs no GPU/SSH.

Findings (source refs are pr403 worktree; raw probe in
phase397e_periter_reconciliation_raw.csv):

  A. generation_moe table values are PER-RANK / single-GPU wall time, with TP
     encoded in the `moe_tp_size` lookup key (and topology key for the vLLM
     module table), NOT applied as a post-lookup divisor:
       - perf_database.py load_moe_data / query_moe / query_vllm_module: no /tp
         at load or query; TP is a key dimension.
       - collector/vllm/collect_moe.py: single GPU (cuda:0), weights sharded by
         (moe_tp_size, moe_ep_size), FULL num_tokens fed to that rank; log_perf
         records single-GPU latency. Empirical: same 32 tokens, moe_tp=1 ~0.688ms
         vs moe_tp=16 ~0.094ms -> TP shrinks per-rank latency via the KEY.
       - operations.py MoE.query: generation MoE passes moe_tp_size as a key and
         scales only by scale_factor (num_layers * mtp); no /tp divisor.

  B. Double-count: iteration_latency.py::_split_generation_non_attention (live;
     compute() pure-decode branch) divides every op whose name contains
     "generation_moe" by tp_size AGAIN. On the tp16 gates (moe_tp_size=16) this
     turns a per-rank generation_moe(+dispatch) of ~29-38 ms into ~2-3 ms.
     `_scale_generation_non_attention` is dead code (no caller).

  C. Reconciliation vs the real per-iteration budget
     (real_decode_iter_ms = batch / (real_out_tok_s_gpu * num_gpus) * 1000):
       - current (`/tp` then overlap_factor=0 max) ~= gen_attn only -> ~3x UNDER
         real (non-attention is both crushed by /tp AND masked by max()).
       - raw non-attention + max(gen_attn) -> still UNDER real.
       - raw non-attention SERIAL-SUM gen_attn -> within ~13% of real at all 3
         gates (0.87x / 1.11x / 1.05x).
     So the decode composition has TWO COUPLED errors: (1) the /tp double-count
     and (2) overlap_factor=0 max() (attention and MoE/FFN run serially within a
     decode layer, so they should SUM, not max). Both must be fixed together;
     397c's "overlap ruled out" was narrowly true only because the /tp
     under-count had already masked the non-attention term.

Verdict: the owed decode per-iteration term is the per-rank non-attention
(generation_moe + dispatch + gemms) added SERIALLY to generation_attention. The
397f target is per-iter ~= gen_non_attn_raw + gen_attn ~= real +/-13%. 397f
(runtime fix) must (i) drop the /tp double-count on generation_moe* and (ii) make
the pure-decode composition a serial sum, then re-validate 397d+397e+397f offline.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397e_decode_periter_composition_offline.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397e_decode_periter_composition_offline.md"
)
RAW_EVIDENCE_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397e_periter_reconciliation_raw.csv"
)

SOURCE = "phase397e_decode_periter_composition_offline"
MODEL = "kimi-k2.5"
HARDWARE = "h200_sxm"
VLLM_DB_VERSION = "0.12.0"
TOPOLOGY = "tp16dp1"
TRUE = "true"
FALSE = "false"
DEFAULT_READINESS = "No-Go"
NEXT_PHASE = "phase397f_decode_periter_composition_fix"
GATE_SCENARIO = "10k3k_b128"


@dataclass(frozen=True)
class Recon:
    name: str
    real_out_tok_s_gpu: float
    num_gpus: int
    batch: int
    gen_attn_ms: float
    gen_non_attn_raw_ms: float
    gen_non_attn_div_tp_ms: float

    @property
    def real_decode_iter_ms(self) -> float:
        return self.batch / (self.real_out_tok_s_gpu * self.num_gpus) * 1000.0

    @property
    def per_iter_current_ms(self) -> float:
        # /tp then overlap_factor=0 max(non_attn_div, attn)
        return max(self.gen_non_attn_div_tp_ms, self.gen_attn_ms)

    @property
    def per_iter_raw_max_ms(self) -> float:
        return max(self.gen_non_attn_raw_ms, self.gen_attn_ms)

    @property
    def per_iter_raw_sum_ms(self) -> float:
        return self.gen_non_attn_raw_ms + self.gen_attn_ms


# Measured offline via canonical run_static(static_gen) (aic env, pr403 worktree),
# tp16 moe_tp=16 moe_ep=1, representative decode KV = isl + osl//2.
RECON: dict[str, Recon] = {
    "10k3k_b128": Recon("10k3k_b128", 89.5, 16, 128, 26.6206, 51.1034, 6.8851),
    "10k2k_b32": Recon("10k2k_b32", 49.8, 16, 32, 12.1814, 32.4249, 5.2816),
    "16k2k_b32": Recon("16k2k_b32", 41.7, 16, 32, 17.9083, 32.4249, 5.2816),
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
    # Pure verdict-only: no runtime change this phase.
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


def analyze_phase397e() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    # --- A: table semantics ---
    rows.append(
        _row(
            "table_semantics_verdict",
            metric="generation_moe_table_value_scope",
            value_a="per_rank_single_gpu_wall_time",
            value_b="TP encoded in moe_tp_size/topology lookup key, no post-lookup /tp",
            verdict=(
                "perf_database.py load_moe_data/query_moe/query_vllm_module do NOT "
                "divide by tp; collector/vllm/collect_moe.py runs 1 GPU with sharded "
                "weights + full num_tokens; operations.py MoE.query passes moe_tp_size "
                "as key and scales only by num_layers*mtp"
            ),
        )
    )

    # --- B: double-count proof ---
    for name, r in RECON.items():
        rows.append(
            _row(
                "double_count_proof",
                scenario=name,
                metric="gen_non_attn_raw_vs_div_tp",
                value_a=f"raw={r.gen_non_attn_raw_ms:.4f}ms",
                value_b=f"div_tp={r.gen_non_attn_div_tp_ms:.4f}ms",
                ratio=f"{r.gen_non_attn_raw_ms / r.gen_non_attn_div_tp_ms:.3f}",
                verdict=(
                    "_split_generation_non_attention divides per-rank generation_moe* "
                    "by tp_size AGAIN -> double-count (per-rank value crushed ~16x)"
                ),
            )
        )
    rows.append(
        _row(
            "dead_code",
            metric="_scale_generation_non_attention",
            value_a="defined iteration_latency.py:92-109",
            value_b="no caller (only _split_generation_non_attention runs, line 266)",
            verdict="dead code; 397f should remove alongside the /tp fix",
        )
    )

    # --- C: per-iteration reconciliation ---
    for name, r in RECON.items():
        rows.append(
            _row(
                "periter_reconciliation",
                scenario=name,
                metric="current_div_max__vs__raw_sum__vs__real_decode_iter",
                value_a=(
                    f"current(/tp,max)={r.per_iter_current_ms:.2f}ms"
                    f"({r.per_iter_current_ms / r.real_decode_iter_ms:.2f}x); "
                    f"raw_max={r.per_iter_raw_max_ms:.2f}ms"
                    f"({r.per_iter_raw_max_ms / r.real_decode_iter_ms:.2f}x)"
                ),
                value_b=(
                    f"raw_sum={r.per_iter_raw_sum_ms:.2f}ms"
                    f"({r.per_iter_raw_sum_ms / r.real_decode_iter_ms:.2f}x); "
                    f"real_decode_iter={r.real_decode_iter_ms:.2f}ms"
                ),
                ratio=f"{r.per_iter_raw_sum_ms / r.real_decode_iter_ms:.4f}",
                verdict=(
                    "current under-counts ~3x; raw+max still under; raw+SUM lands "
                    "within ~13% of real -> decode attn and non-attn are serial"
                ),
            )
        )

    # --- coupled-error summary + 397c nuance ---
    rows.append(
        _row(
            "coupled_errors",
            metric="two_coupled_decode_composition_errors",
            value_a="(1) /tp double-count on generation_moe* (per-rank crushed ~16x)",
            value_b="(2) overlap_factor=0 max() drops the smaller of attn/non_attn "
            "(pure-decode attn+MoE are SERIAL -> should sum)",
            verdict=(
                "both must be fixed together; 397c 'overlap ruled out' was narrow -- "
                "the /tp under-count had already masked the non-attention term so "
                "max-vs-sum did not matter then"
            ),
        )
    )

    # --- 397f target + candidate fixes ---
    for name, r in RECON.items():
        target = r.gen_non_attn_raw_ms + r.gen_attn_ms
        rows.append(
            _row(
                "target_397f",
                scenario=name,
                metric="decode_per_iter_target",
                value_a=f"target=gen_non_attn_raw+gen_attn={target:.2f}ms",
                value_b=f"real_decode_iter={r.real_decode_iter_ms:.2f}ms",
                ratio=f"{target / r.real_decode_iter_ms:.4f}",
                verdict="397f should land pure-decode iter at ~this target (raw non-attn "
                "+ attn, serial)",
            )
        )
    rows.append(
        _row(
            "candidate_fixes",
            metric="phase397f_change_set",
            value_a="(i) drop /tp on generation_moe* in _split_generation_non_attention; "
            "(ii) pure-decode composition = serial sum(non_attn, attn)",
            value_b="(iii) remove dead _scale_generation_non_attention; "
            "(iv) WATCH decode num_tokens/topk & moe_tp-vs-ep routing (raw already "
            "near target, likely no change)",
            verdict="enumerated for 397f; not implemented here",
        )
    )

    # --- verdict + next phase ---
    rows.append(
        _row(
            "verdict",
            metric="decode_periter_composition_root_cause",
            verdict=(
                "generation_moe table is per-rank; the /tp_size in "
                "_split_generation_non_attention is a double-count, and "
                "overlap_factor=0 max() wrongly drops the serial non-attention term; "
                "owed decode per-iter = gen_non_attn_raw + gen_attn (serial), ~real "
                "+/-13%"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )
    rows.append(
        _row(
            "next_phase",
            metric="route_delta_step3_target",
            verdict=(
                "phase397f runtime fix: drop the generation_moe* /tp double-count AND "
                "make pure-decode composition a serial sum; remove dead "
                "_scale_generation_non_attention; then re-validate 397d+397e+397f "
                "offline on the 3 gates and the validate throughput table"
            ),
            next_allowed_phase=NEXT_PHASE,
        )
    )

    _validate_rows(rows)
    return rows


def _validate_rows(rows: list[dict[str, str]]) -> None:
    actual = [r["row_type"] for r in rows]
    if actual[0] != "table_semantics_verdict":
        raise ValueError(
            f"Phase397e must start with table_semantics_verdict: {actual[0]}"
        )
    if actual[-2:] != ["verdict", "next_phase"]:
        raise ValueError(f"Phase397e must end with verdict,next_phase: {actual[-2:]}")

    required_types = {
        "table_semantics_verdict",
        "double_count_proof",
        "dead_code",
        "periter_reconciliation",
        "coupled_errors",
        "target_397f",
        "candidate_fixes",
        "verdict",
        "next_phase",
    }
    if required_types - set(actual):
        raise ValueError(f"Phase397e missing row types: {required_types - set(actual)}")

    n = len(RECON)
    for per_scenario_type in ("double_count_proof", "periter_reconciliation", "target_397f"):
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

    # Load-bearing: the /tp is a heavy double-count (raw >> div).
    for row in rows:
        if row["row_type"] == "double_count_proof":
            if float(row["ratio"]) <= 4.0:
                raise ValueError(
                    f"{row['scenario']} raw/div {row['ratio']} must be >4 (double-count)"
                )

    # Load-bearing: raw+SUM lands within +/-15% of real at every gate.
    for row in rows:
        if row["row_type"] == "periter_reconciliation":
            if not 0.85 <= float(row["ratio"]) <= 1.15:
                raise ValueError(
                    f"{row['scenario']} raw_sum/real {row['ratio']} must be within 0.85-1.15"
                )

    # Load-bearing: current (/tp,max) badly under-counts (<0.5x real) everywhere.
    for name, r in RECON.items():
        if r.per_iter_current_ms / r.real_decode_iter_ms >= 0.5:
            raise ValueError(f"{name} current must be <0.5x real (under-count)")

    verdict = next(r for r in rows if r["row_type"] == "verdict")
    if "double-count" not in verdict["verdict"]:
        raise ValueError("verdict must name the double-count")
    if "serial" not in verdict["verdict"]:
        raise ValueError("verdict must name the serial (sum) composition")
    if verdict["next_allowed_phase"] != NEXT_PHASE:
        raise ValueError("verdict next phase mismatch")


def write_phase397e_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase397e_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase397e decode Per-Iteration Composition Offline Verdict (Route delta step 2)",
        "",
        "| Item | Result |",
        "|---|---|",
        "| Question | after 397d unmasked a decode non-attention under-count, is the "
        "`generation_moe` table aggregate or per-rank, and is `iteration_latency.py`'s "
        "`/tp_size` a double-count? |",
        "| Answer | table is **per-rank** (TP in the lookup key); the `/tp_size` is a "
        "**double-count**; and `overlap_factor=0` `max()` wrongly drops the serial "
        "non-attention term. Both are coupled decode-composition errors. |",
        "| Runtime / table / Default AIC | not modified (verdict-only, offline, "
        "no GPU/SSH) |",
        "",
        "## A. `generation_moe` table semantics = per-rank",
        "",
        "- `perf_database.py` `load_moe_data` / `query_moe` / `query_vllm_module`: no "
        "`/tp` at load or query; TP is encoded in the `moe_tp_size` (and topology) key.",
        "- `collector/vllm/collect_moe.py`: one GPU, weights sharded by "
        "`(moe_tp_size, moe_ep_size)`, FULL `num_tokens` fed to that rank; `log_perf` "
        "records single-GPU latency (empirical: 32 tokens, moe_tp=1 ~0.688ms vs "
        "moe_tp=16 ~0.094ms -> TP shrinks per-rank latency via the KEY).",
        "- `operations.py` `MoE.query`: generation MoE passes `moe_tp_size` as a key "
        "and scales only by `num_layers*mtp`; no `/tp` divisor.",
        "",
        "## B. Double-count",
        "",
        "`iteration_latency.py::_split_generation_non_attention` (live; pure-decode "
        "branch) divides every op whose name contains `generation_moe` by `tp_size` "
        "AGAIN. `_scale_generation_non_attention` (lines 92-109) is dead code.",
        "",
        "| Scenario | gen_non_attn raw | gen_non_attn /tp | raw/div |",
        "|---|---|---|---|",
    ]
    for name, r in RECON.items():
        lines.append(
            f"| {name} | {r.gen_non_attn_raw_ms:.2f} ms | "
            f"{r.gen_non_attn_div_tp_ms:.2f} ms | "
            f"{r.gen_non_attn_raw_ms / r.gen_non_attn_div_tp_ms:.1f}x |"
        )
    lines += [
        "",
        "## C. Per-iteration reconciliation (vs real decode-iter budget)",
        "",
        "`real_decode_iter_ms = batch / (real_out_tok_s_gpu * num_gpus) * 1000`.",
        "",
        "| Scenario | current (/tp,max) | raw+max | raw+SUM | real decode iter |",
        "|---|---|---|---|---|",
    ]
    for name, r in RECON.items():
        lines.append(
            f"| {name} | {r.per_iter_current_ms:.1f} ms "
            f"({r.per_iter_current_ms / r.real_decode_iter_ms:.2f}x) | "
            f"{r.per_iter_raw_max_ms:.1f} ms "
            f"({r.per_iter_raw_max_ms / r.real_decode_iter_ms:.2f}x) | "
            f"{r.per_iter_raw_sum_ms:.1f} ms "
            f"({r.per_iter_raw_sum_ms / r.real_decode_iter_ms:.2f}x) | "
            f"{r.real_decode_iter_ms:.1f} ms |"
        )
    lines += [
        "",
        "Current (`/tp` then `max`) under-counts ~3x (non-attention is both crushed "
        "by `/tp` AND masked by `max()`). Raw non-attention + `max(attn)` is still "
        "under. Raw non-attention **serial-summed** with attention lands within ~13% "
        "of real at all three gates -- decode attention and MoE/FFN run serially "
        "within a layer, so they should SUM.",
        "",
        "## Two coupled errors (both owed to 397f)",
        "",
        "1. **`/tp` double-count** on `generation_moe*` (per-rank value divided by TP "
        "again).",
        "2. **`overlap_factor=0` `max()`** for pure decode drops the smaller of "
        "attention / non-attention; the two are serial and should sum.",
        "",
        "397c's \"overlap ruled out\" was narrowly true only because the `/tp` "
        "under-count had already masked the non-attention term, so max-vs-sum did not "
        "matter at the time.",
        "",
        "## Verdict -- Route delta step 2",
        "",
        "- Owed decode per-iteration term = `gen_non_attn_raw + gen_attn` (serial), "
        "which reconciles to real +/-13% across the gates.",
        f"- Next: **{NEXT_PHASE}** -- runtime fix that (i) drops the `generation_moe*` "
        "`/tp` double-count, (ii) makes pure-decode composition a serial sum, and "
        "(iii) removes the dead `_scale_generation_non_attention`; then re-validate "
        "397d+397e+397f offline on the 3 gates and the validate throughput table.",
        "",
        "## Discipline",
        "",
        "- Verdict-only, offline: runtime / PerfDatabase not modified; no fudge tuning; "
        "no scope gating; no GPU/SSH; Default AIC remains No-Go.",
        f"- Raw probe evidence at `{RAW_EVIDENCE_CSV.relative_to(REPO_ROOT)}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase397e()
    write_phase397e_csv(args.output_csv, rows)
    write_phase397e_md(args.output_md, rows)


if __name__ == "__main__":
    main()

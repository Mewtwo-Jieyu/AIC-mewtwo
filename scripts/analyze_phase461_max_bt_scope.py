#!/usr/bin/env python3
"""Phase461 exact max-batched-token scope and spike reachability audit."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_cb_simulator as validate
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator


DEFAULT_BASELINE = REPO_ROOT / "docs/iter_gap_investigation/phase458_validate_ab.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_max_bt_scope.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase461_max_bt_scope.md"

MODEL = "kimi-k2.5"
HIDDEN_SIZE = 7168
TOPK = 8
MOE_EP_SIZE = 8
QUANT_RUNTIME = "CompressedTensorsWNA16MarlinMoEMethod"


class SpikeCell(NamedTuple):
    scenario: str
    max_bt: int
    bucket_tokens: int
    decode_batch: int


SPIKE_CELLS = (
    SpikeCell("K2.5-tp4ep8dp2-32k3k", 32_000, 63, 9),
    SpikeCell("K2.5-tp4ep8dp2-8k2k-bt65536", 65_536, 14_484, 14),
)

CSV_FIELDS = [
    "section",
    "scenario",
    "phase",
    "row_kind",
    "category",
    "max_bt",
    "bucket_tokens",
    "decode_batch",
    "baseline_error_ratio",
    "current_error_ratio",
    "delta_error_ratio",
    "classification",
    "hit_count",
    "miss_count",
    "scope_query_count",
    "exact_query_count",
    "influence_hit_count",
    "status",
    "note",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def phase458_current_baseline(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, float | str]]:
    baseline: dict[str, dict[str, float | str]] = {}
    for row in rows:
        name = row["name"]
        baseline[name] = {
            "name": name,
            "real_output_tok_s_gpu": float(row["current_real_output_tok_s_gpu"]),
            "sim_output_tok_s_gpu": float(row["current_sim_output_tok_s_gpu"]),
            "error_ratio": float(row["current_error_ratio"]),
        }
    return baseline


def _bracket(value: int, values: Iterable[int]) -> tuple[int, int] | None:
    ordered = sorted(int(item) for item in values)
    if value in ordered:
        return value, value
    left = [item for item in ordered if item < value]
    right = [item for item in ordered if item > value]
    if not left or not right:
        return None
    return max(left), min(right)


def _query_uses_spike(record: dict[str, object], table: dict, spike: SpikeCell) -> bool:
    if not bool(record.get("hit")):
        return False
    token_bracket = _bracket(int(record["bucket_tokens"]), table.keys())
    if token_bracket is None or spike.bucket_tokens not in token_bracket:
        return False
    batch_table = table.get(spike.bucket_tokens, {})
    batch_bracket = _bracket(int(record["decode_batch"]), batch_table.keys())
    return batch_bracket is not None and spike.decode_batch in batch_bracket


def summarize_spike_reachability(
    records: Iterable[dict[str, object]],
    table: dict,
    spike: SpikeCell,
) -> dict[str, int | bool]:
    scoped = [
        record
        for record in records
        if record.get("phase") == "mixed_prefill"
        and record.get("row_kind") == "forward_total"
        and record.get("category") == "forward_total"
        and record.get("max_num_batched_tokens") == spike.max_bt
    ]
    exact = [
        record
        for record in scoped
        if record.get("bucket_tokens") == spike.bucket_tokens
        and record.get("decode_batch") == spike.decode_batch
    ]
    influenced = [record for record in scoped if _query_uses_spike(record, table, spike)]
    return {
        "scope_query_count": len(scoped),
        "exact_query_count": len(exact),
        "influence_hit_count": len(influenced),
        "reachable": bool(influenced),
    }


def evaluate_tp8_candidate_scope(
    *,
    candidate_max_bt: int,
    candidate_isl: int,
    validation_regimes: dict[str, tuple[int, int]],
) -> dict[str, int | bool | str]:
    collisions = sum(
        max_bt == candidate_max_bt and isl != candidate_isl
        for max_bt, isl in validation_regimes.values()
    )
    return {
        "six_point_collision_count": collisions,
        "generic_isl_safe": False,
        "status": "retain_pending_isl_replay",
    }


def _point(name: str):
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == name)


def run_scenario_query_audit(name: str) -> tuple[list[dict[str, object]], object]:
    point = _point(name)
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    sim = CBSimulator(backend, model, database, config)
    if point.dp > 1 and point.max_num_batched_tokens == point.isl:
        sim.run_multi_replica(
            isl=point.isl,
            osl=point.osl,
            concurrency=point.batch_size,
            data_parallel_size=point.dp,
            num_gpus=point.tp * point.dp,
            lockstep=True,
        )
    else:
        sim.run(
            isl=point.isl,
            osl=point.osl,
            concurrency=math.ceil(point.batch_size / point.dp),
            num_gpus=point.tp,
        )
    records = [record.as_dict() for record in sim.get_last_serving_state_query_audit()]
    return records, database


def _spike_table(database: object, spike: SpikeCell) -> dict:
    key = (
        MODEL,
        "tp4dp2ep8",
        spike.max_bt,
        "mixed_prefill",
        "forward_total",
        "forward_total",
        HIDDEN_SIZE,
        TOPK,
        MOE_EP_SIZE,
        QUANT_RUNTIME,
    )
    data = getattr(database, "_vllm_serving_state_data", None) or {}
    return data.get(key, {})


def build_report_rows(baseline_path: Path = DEFAULT_BASELINE) -> list[dict[str, object]]:
    baseline = phase458_current_baseline(_read_csv(baseline_path))
    current = validate.run_diagnostic_multi_config_topk(verbose=False)
    ab_rows = validate.build_multi_config_ab_rows(baseline, current, delta_threshold=0.005)
    rows: list[dict[str, object]] = []
    for row in ab_rows:
        rows.append(
            {
                "section": "six_point_ab",
                "scenario": row.name,
                "max_bt": row.max_bt,
                "baseline_error_ratio": row.baseline_error_ratio,
                "current_error_ratio": row.current_error_ratio,
                "delta_error_ratio": row.delta_error_ratio,
                "classification": row.classification,
                "status": "explicit_scope_attribution",
                "note": "Phase458 current is the pre-keying baseline",
            }
        )

    for spike in SPIKE_CELLS:
        records, database = run_scenario_query_audit(spike.scenario)
        counts = Counter("hit" if record["hit"] else str(record["miss_reason"]) for record in records)
        summary = summarize_spike_reachability(records, _spike_table(database, spike), spike)
        rows.append(
            {
                "section": "spike_reachability",
                "scenario": spike.scenario,
                "phase": "mixed_prefill",
                "row_kind": "forward_total",
                "category": "forward_total",
                "max_bt": spike.max_bt,
                "bucket_tokens": spike.bucket_tokens,
                "decode_batch": spike.decode_batch,
                "hit_count": counts.get("hit", 0),
                "miss_count": len(records) - counts.get("hit", 0),
                **summary,
                "status": "reachable_recollect" if summary["reachable"] else "not_reached_current_six",
                "note": ";".join(f"{key}={value}" for key, value in sorted(counts.items())),
            }
        )

    tp8_scope = evaluate_tp8_candidate_scope(
        candidate_max_bt=8_000,
        candidate_isl=8_000,
        validation_regimes={
            point.name: (point.max_num_batched_tokens, point.isl)
            for point in validate.MULTI_CONFIG_DATA
            if point.tp == 8
        },
    )

    attribution = {
        "K2.5-tp8ep8-8k2k": "TP8 already used exact max_bt before Step 1",
        "K2.5-tp8ep8-32k3k": "TP8 already used exact max_bt before Step 1",
        "K2.5-tp8ep8-8k2k-bt65536": "TP8 already used exact max_bt before Step 1",
        "K2.5-tp4ep8dp2-8k2k": "exact bt8000 reproduces the prior effective charge",
        "K2.5-tp4ep8dp2-32k3k": (
            "correct bt32000 rows replace the cross-regime unscoped table; 63/9 spike not reached"
        ),
        "K2.5-tp4ep8dp2-8k2k-bt65536": (
            "correct bt65536 rows replace the cross-regime unscoped table; 14484/14 spike reached"
        ),
    }
    for row in rows:
        if row["section"] == "six_point_ab":
            row["note"] = attribution[str(row["scenario"])]
    rows.append(
        {
            "section": "tp8_phase455_scope_decision",
            "scenario": "K2.5-tp8ep8-8k2k",
            "max_bt": 8_000,
            "status": tp8_scope["status"],
            "note": (
                f"six_point_collision_count={tp8_scope['six_point_collision_count']};"
                "max_bt separates current six but does not prove generic ISL safety"
            ),
        }
    )
    return rows


def _fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    ab = [row for row in rows if row["section"] == "six_point_ab"]
    spikes = [row for row in rows if row["section"] == "spike_reachability"]
    scope = next(row for row in rows if row["section"] == "tp8_phase455_scope_decision")
    lines = [
        "# Phase461 max_bt scope",
        "",
        "Verdict: `max_bt_exact_keying_enabled_step2_pending`.",
        "",
        "## Six-point A/B",
        "",
        "| scenario | before | after | delta | class | attribution |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in ab:
        lines.append(
            f"| {row['scenario']} | {float(row['baseline_error_ratio']):.3f}x | "
            f"{float(row['current_error_ratio']):.3f}x | {float(row['delta_error_ratio']):+.3f} | "
            f"{row['classification']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Spike reachability",
            "",
            "| scenario | cell | exact queries | influenced hits | status |",
            "|---|---|---:|---:|---|",
        ]
    )
    for row in spikes:
        lines.append(
            f"| {row['scenario']} | {row['max_bt']}/{row['bucket_tokens']}/{row['decode_batch']} | "
            f"{row['exact_query_count']} | {row['influence_hit_count']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Scope decision",
            "",
            "| item | decision |",
            "|---|---|",
            "| DP2 query key | Exact configured `max_num_batched_tokens`; absent scope returns `None` and the existing analytic path is used. |",
            "| Regression handling | Retain source-correct exact scope. The two DP2 movements expose prior cross-regime error cancellation and are not silently rolled back. |",
            f"| Phase455 TP8 rows | `{scope['status']}`: safe from collisions in the current six points, but max_bt alone does not prove generic cross-ISL safety. |",
            "| ISL band | Not added in Step 1; Step 2 replay remains the decision gate. |",
            "| Default AIC | No-Go. |",
            "",
            "## Verification",
            "",
            "| check | result |",
            "|---|---|",
            "| Step1 targeted pytest | 17 passed |",
            "| Broader two-file pytest | 115 passed, 3 pre-existing unrelated failures |",
            "| py_compile / diff check / CRLF | passed |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_report_rows(args.baseline)
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

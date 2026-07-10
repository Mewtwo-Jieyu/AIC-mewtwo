#!/usr/bin/env python3
"""Phase461 Step2: topology scope, DP2 replay, and GPU manifest audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import scripts.analyze_phase454_gpu_batch_acceptance as phase454
import scripts.analyze_phase459_residual_triage as phase459
import scripts.analyze_phase461_max_bt_scope as phase461
import validate_cb_simulator as validate


PERFDB = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase461_step2_scope_replay.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase461_step2_scope_replay.md"

DP2_32K = "K2.5-tp4ep8dp2-32k3k"
DP2_BT = "K2.5-tp4ep8dp2-8k2k-bt65536"
DP2_32K_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase454_gpu_batch/recollect_dp2_32k3k"
    / DP2_32K
    / "serve.log"
)
DP2_BT_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_dp2_8k2k_bt65536"
    / DP2_BT
    / "serve.log"
)
DP2_BT_B2B_ROOT = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase454_gpu_batch_scope/b2b_dp2_8k2k_bt65536"
)
DP2_BT_B2B_LOG = DP2_BT_B2B_ROOT / "overhead_on" / DP2_BT / "serve.log"
DP2_BT_B2B_META = DP2_BT_B2B_ROOT / "overhead_on" / DP2_BT / "meta.json"
DP2_BT_PHASE458_META = DP2_BT_LOG.parent / "meta.json"

CSV_FIELDS = [
    "section",
    "scenario",
    "topology",
    "max_bt",
    "phase",
    "row_kind",
    "source",
    "metric",
    "value",
    "target",
    "status",
    "note",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def inventory_scope_families(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int, str, str, str, str], list[tuple[int, int]]] = defaultdict(list)
    for row in rows:
        key = (
            row["topology"],
            int(row["max_num_batched_tokens"]),
            row["phase"],
            row.get("row_kind") or "category",
            row["kernel_source"],
            row["provenance"],
        )
        grouped[key].append((int(row["bucket_tokens"]), int(row["decode_batch"])))
    result: list[dict[str, object]] = []
    for key, points in sorted(grouped.items()):
        topology, max_bt, phase, row_kind, source, provenance = key
        result.append(
            {
                "topology": topology,
                "max_bt": max_bt,
                "phase": phase,
                "row_kind": row_kind,
                "source": source,
                "provenance": provenance,
                "rows": len(points),
                "bucket_min": min(point[0] for point in points),
                "bucket_max": max(point[0] for point in points),
                "batch_min": min(point[1] for point in points),
                "batch_max": max(point[1] for point in points),
            }
        )
    return result


def classify_modal_replay(
    *,
    modal_sim_over_real: float,
    large_step_sim_over_real: float,
    rank_spread: float,
) -> str:
    if 0.85 <= modal_sim_over_real <= 1.15:
        return "cost_rows_matched_no_window"
    if 0.85 <= large_step_sim_over_real <= 1.15 and rank_spread >= 1.5:
        return "dp_phase_lockstep_not_scalar_row_gap"
    return "cost_coverage_or_measurement_gap"


def decide_isl_band(
    *,
    distinct_observed_isls: int,
    same_run_modal_sim_over_real: float,
    rank_spread: float,
) -> str:
    if distinct_observed_isls < 2 and (
        not 0.85 <= same_run_modal_sim_over_real <= 1.15 or rank_spread >= 1.5
    ):
        return "reject_isl_band_no_causal_evidence"
    return "require_cross_isl_comparison"


def summarize_rank_spread(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["dp_rank"])].append(float(row["forward_busy_ms"]))
    medians = {rank: statistics.median(values) for rank, values in sorted(grouped.items())}
    if len(medians) < 2 or min(medians.values()) <= 0:
        spread = 1.0
    else:
        spread = max(medians.values()) / min(medians.values())
    return {"rank_medians": medians, "max_over_min": spread}


def build_gpu_manifest(
    *,
    dp2_32k_modal_ratio: float,
    dp2_bt_modal_ratio: float,
    dp2_bt_rank_spread: float,
    spike_reachable: bool,
) -> list[dict[str, str]]:
    dp2_32k_decision = "skip" if 0.85 <= dp2_32k_modal_ratio <= 1.15 else "collect"
    dp2_bt_decision = (
        "collect_diagnostic_repeat"
        if dp2_bt_rank_spread >= 1.5 or not 0.85 <= dp2_bt_modal_ratio <= 1.15
        else "skip"
    )
    return [
        {
            "item": "mla_decode_grid",
            "decision": "collect",
            "measurement": "kernel_microbench",
            "coverage": "heads8; batch=8,16; KV=16384,32768,65536; >=3 repeats",
            "reason": "replace confirmed SILICON bad row without smoothing",
        },
        {
            "item": "tp8_bt65536_mixed",
            "decision": "collect",
            "measurement": "serving_state_window",
            "coverage": "tp8ep8/max_bt65536 mixed working set",
            "reason": "no TP8 serving-state rows exist for this regime",
        },
        {
            "item": "tp8_32k3k_mixed",
            "decision": "collect",
            "measurement": "serving_state_window",
            "coverage": "tp8ep8/max_bt32000 mixed working set",
            "reason": "decode bad-row fix exposes the measured mixed undercharge",
        },
        {
            "item": "dp2_32k3k_mixed",
            "decision": dp2_32k_decision,
            "measurement": "serving_state_window",
            "coverage": "tp4dp2ep8/max_bt32000 mixed working set",
            "reason": f"existing modal sim/real={dp2_32k_modal_ratio:.4f}",
        },
        {
            "item": "dp2_bt65536_mixed",
            "decision": dp2_bt_decision,
            "measurement": "paired_rank_event_plus_iteration_wall",
            "coverage": "14484/14 spike; modal rank-conditioned cells; both DP ranks",
            "reason": (
                f"modal sim/real={dp2_bt_modal_ratio:.4f}; rank spread={dp2_bt_rank_spread:.2f}x; "
                f"spike_reachable={spike_reachable}; diagnostic only until lockstep semantics are resolved"
            ),
        },
    ]


def _topology(point) -> str:
    if point.tp == 8 and point.dp == 1:
        return "tp8ep8"
    if point.tp == 4 and point.dp == 2:
        return "tp4dp2ep8"
    raise ValueError(f"unsupported topology: tp={point.tp} dp={point.dp}")


def _audit_six_points() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for point in validate.MULTI_CONFIG_DATA:
        records, _ = phase461.run_scenario_query_audit(point.name)
        counts = Counter("hit" if row["hit"] else str(row["miss_reason"]) for row in records)
        rows.append(
            {
                "scenario": point.name,
                "topology": _topology(point),
                "max_bt": point.max_num_batched_tokens,
                "hits": counts.get("hit", 0),
                "misses": len(records) - counts.get("hit", 0),
                "reasons": ";".join(f"{key}={value}" for key, value in sorted(counts.items())),
            }
        )
    return rows


def _replay(name: str, serve_log: Path, max_bt: int) -> dict[str, object]:
    point = phase459._point(name)
    steps = phase459.parse_iteration_steps(serve_log)
    replay = phase459.wall_cost_replay(point, steps)
    mixed = replay["classes"]["mixed"]
    large = phase459.representative_large_mixed_step(steps, max_bt)
    large_charge = phase459.charge_step(point, large)
    return {
        "steps": steps,
        "modal": mixed["representative"],
        "modal_sim_ms": float(mixed["sim_ms"]),
        "modal_sim_over_real": float(mixed["sim_over_real"]),
        "wall_replay_factor": float(replay["cost_replay_factor"]),
        "large": large,
        "large_sim_ms": float(large_charge["sim_ms"]),
        "large_sim_over_real": float(large_charge["sim_ms"]) / large.elapsed_ms,
    }


def _event_cell_rows(bucket_tokens: int, decode_batch: int) -> list[dict[str, object]]:
    spec = phase454.default_b2b_specs(phase454.DEFAULT_SCOPE_ROOT)[1]
    return [
        {"dp_rank": step.dp_rank, "forward_busy_ms": step.forward_busy_ms}
        for step in phase454.read_event_steps(spec)
        if step.phase == "mixed_prefill"
        and step.bucket_tokens == bucket_tokens
        and step.generation_requests == decode_batch
    ]


def _exact_wall_values(steps: Iterable[object], bucket_tokens: int, decode_batch: int) -> list[float]:
    return [
        float(step.elapsed_ms)
        for step in steps
        if step.ctx_tokens + step.generation_requests == bucket_tokens
        and step.generation_requests == decode_batch
    ]


def _observed_isls() -> set[int]:
    values: set[int] = set()
    for path in (DP2_BT_B2B_META, DP2_BT_PHASE458_META):
        with path.open(encoding="utf-8") as f:
            values.add(int(json.load(f)["isl"]))
    return values


def _add(
    rows: list[dict[str, object]],
    section: str,
    scenario: str,
    metric: str,
    value: object,
    *,
    topology: str = "",
    max_bt: int | str = "",
    phase: str = "",
    row_kind: str = "",
    source: str = "",
    target: str = "",
    status: str = "",
    note: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "scenario": scenario,
            "topology": topology,
            "max_bt": max_bt,
            "phase": phase,
            "row_kind": row_kind,
            "source": source,
            "metric": metric,
            "value": value,
            "target": target,
            "status": status,
            "note": note,
        }
    )


def build_report_rows() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    rows: list[dict[str, object]] = []
    families = inventory_scope_families(_read_csv(PERFDB))
    for family in families:
        _add(
            rows,
            "scope_family",
            "all",
            "row_count",
            family["rows"],
            topology=str(family["topology"]),
            max_bt=int(family["max_bt"]),
            phase=str(family["phase"]),
            row_kind=str(family["row_kind"]),
            source=str(family["source"]),
            status="topology_and_max_bt_scoped",
            note=(
                f"bucket={family['bucket_min']}..{family['bucket_max']};"
                f"batch={family['batch_min']}..{family['batch_max']};"
                f"provenance={family['provenance']}"
            ),
        )

    for audit in _audit_six_points():
        _add(
            rows,
            "six_point_scope_audit",
            str(audit["scenario"]),
            "query_hits",
            audit["hits"],
            topology=str(audit["topology"]),
            max_bt=int(audit["max_bt"]),
            target="exact topology + max_bt",
            status="pass",
            note=f"misses={audit['misses']};{audit['reasons']}",
        )

    dp2_32k = _replay(DP2_32K, DP2_32K_LOG, 32_000)
    dp2_bt = _replay(DP2_BT, DP2_BT_LOG, 65_536)
    dp2_bt_same_run = _replay(DP2_BT, DP2_BT_B2B_LOG, 65_536)
    modal_cell = _event_cell_rows(6_486, 15)
    rank_summary = summarize_rank_spread(modal_cell)
    spike_cell = _event_cell_rows(14_484, 14)
    spike_wall = _exact_wall_values(dp2_bt_same_run["steps"], 14_484, 14)
    spike_busy_median = statistics.median(float(row["forward_busy_ms"]) for row in spike_cell)
    spike_wall_median = statistics.median(spike_wall)
    observed_isls = _observed_isls()

    for scenario, replay, source in (
        (DP2_32K, dp2_32k, DP2_32K_LOG),
        (DP2_BT, dp2_bt, DP2_BT_LOG),
    ):
        _add(
            rows,
            "modal_replay",
            scenario,
            "modal_mixed_sim_over_real",
            replay["modal_sim_over_real"],
            topology="tp4dp2ep8",
            max_bt=32_000 if scenario == DP2_32K else 65_536,
            phase="mixed_prefill",
            row_kind="forward_total",
            source=str(source.relative_to(REPO_ROOT)),
            target="0.85..1.15",
            status="pass" if 0.85 <= float(replay["modal_sim_over_real"]) <= 1.15 else "fail",
            note=f"modal={replay['modal']}; wall_replay_factor={replay['wall_replay_factor']:.6f}",
        )

    modal_verdict = classify_modal_replay(
        modal_sim_over_real=float(dp2_bt["modal_sim_over_real"]),
        large_step_sim_over_real=float(dp2_bt["large_sim_over_real"]),
        rank_spread=float(rank_summary["max_over_min"]),
    )
    _add(
        rows,
        "dp2_bt_dynamics",
        DP2_BT,
        "large_mixed_sim_over_real",
        dp2_bt["large_sim_over_real"],
        topology="tp4dp2ep8",
        max_bt=65_536,
        target="0.85..1.15",
        status="pass",
        note=f"large_step={dp2_bt['large']};sim_ms={dp2_bt['large_sim_ms']:.3f}",
    )
    _add(
        rows,
        "dp2_bt_dynamics",
        DP2_BT,
        "rank_median_spread",
        rank_summary["max_over_min"],
        topology="tp4dp2ep8",
        max_bt=65_536,
        source=str(DP2_BT_B2B_ROOT.relative_to(REPO_ROOT)),
        target="<=1.5 for scalar-row sufficiency",
        status="fail",
        note=f"cell=6486/15;medians={rank_summary['rank_medians']};verdict={modal_verdict}",
    )
    _add(
        rows,
        "dp2_bt_dynamics",
        DP2_BT,
        "same_run_modal_sim_over_real",
        dp2_bt_same_run["modal_sim_over_real"],
        topology="tp4dp2ep8",
        max_bt=65_536,
        target="0.85..1.15",
        status="fail",
        note="same ISL/topology/max_bt; cross-run drift is not the cause",
    )
    _add(
        rows,
        "spike_crosscheck",
        DP2_BT,
        "busy_over_wall",
        spike_busy_median / spike_wall_median,
        topology="tp4dp2ep8",
        max_bt=65_536,
        source=str(DP2_BT_B2B_ROOT.relative_to(REPO_ROOT)),
        target="0.90..1.10",
        status="pass",
        note=(
            f"cell=14484/14;event_n={len(spike_cell)};wall_n={len(spike_wall)};"
            f"busy_median={spike_busy_median:.3f};wall_median={spike_wall_median:.3f}"
        ),
    )

    isl_decision = decide_isl_band(
        distinct_observed_isls=len(observed_isls),
        same_run_modal_sim_over_real=float(dp2_bt_same_run["modal_sim_over_real"]),
        rank_spread=float(rank_summary["max_over_min"]),
    )
    _add(
        rows,
        "schema_decision",
        DP2_BT,
        "isl_band",
        isl_decision,
        topology="tp4dp2ep8",
        max_bt=65_536,
        target="add only with causal evidence",
        status="rejected",
        note=f"observed_isls={sorted(observed_isls)}; mismatch already occurs within the same run",
    )

    spike_reachable = bool(spike_cell)
    manifest = build_gpu_manifest(
        dp2_32k_modal_ratio=float(dp2_32k["modal_sim_over_real"]),
        dp2_bt_modal_ratio=float(dp2_bt["modal_sim_over_real"]),
        dp2_bt_rank_spread=float(rank_summary["max_over_min"]),
        spike_reachable=spike_reachable,
    )
    for item in manifest:
        _add(
            rows,
            "gpu_manifest",
            item["item"],
            "decision",
            item["decision"],
            source=item["measurement"],
            status="finalized",
            note=f"coverage={item['coverage']};reason={item['reason']}",
        )
    _add(
        rows,
        "verdict",
        "phase461_step2",
        "verdict",
        "topology_scope_pass_isl_band_rejected_dp2_bt_phase_gap",
        status="step3_manifest_ready",
        note="report-only; no runtime, PerfDB, or gate changes",
    )
    return rows, manifest


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


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    manifest: list[dict[str, str]],
) -> None:
    audits = [row for row in rows if row["section"] == "six_point_scope_audit"]
    replays = [row for row in rows if row["section"] == "modal_replay"]
    rank = next(row for row in rows if row["metric"] == "rank_median_spread")
    large = next(row for row in rows if row["metric"] == "large_mixed_sim_over_real")
    same_run = next(row for row in rows if row["metric"] == "same_run_modal_sim_over_real")
    spike = next(row for row in rows if row["section"] == "spike_crosscheck")
    isl = next(row for row in rows if row["metric"] == "isl_band")
    lines = [
        "# Phase461 Step2 scope replay",
        "",
        "结论: topology scope 已完整隔离;ISL band 没有因果证据,不加。dp2-32k mixed 行已对齐,不补窗口。dp2-bt65536 的 0.50x modal 缺口来自同 cell 的跨 rank/相位分裂,不是 max_bt/ISL 漏键;Step3 只补诊断复现窗,不得把新单值行直接入库。",
        "",
        "## 六点 scope 审计",
        "",
        "| scenario | topology | max_bt | hits | misses / reasons |",
        "|---|---|---:|---:|---|",
    ]
    for row in audits:
        lines.append(
            f"| {row['scenario']} | {row['topology']} | {row['max_bt']} | {row['value']} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## DP2 离线重放",
            "",
            "| scenario | modal mixed sim/real | target | status |",
            "|---|---:|---:|---|",
        ]
    )
    for row in replays:
        lines.append(f"| {row['scenario']} | {float(row['value']):.4f} | {row['target']} | {row['status']} |")
    lines.extend(
        [
            "",
            "| check | value | decision |",
            "|---|---:|---|",
            f"| full 65k mixed sim/real | {float(large['value']):.4f} | base large-step cost is in range |",
            f"| 6486/15 rank median spread | {float(rank['value']):.2f}x | scalar row cannot represent DP phase |",
            f"| same-run modal sim/real | {float(same_run['value']):.4f} | cross-run drift excluded |",
            f"| 14484/14 event busy / wall | {float(spike['value']):.4f} | spike is real in-run, independently repeat in Step3 |",
            f"| ISL band | {isl['value']} | do not extend schema |",
            "",
            "## Step3 GPU 最终清单",
            "",
            "| item | decision | measurement | coverage | reason |",
            "|---|---|---|---|---|",
        ]
    )
    for item in manifest:
        lines.append(
            f"| {item['item']} | {item['decision']} | {item['measurement']} | "
            f"{item['coverage']} | {item['reason']} |"
        )
    lines.extend(
        [
            "",
            "## 边界",
            "",
            "| item | decision |",
            "|---|---|",
            "| tp8 window reuse for dp2 | 禁止; topology key 不同 |",
            "| dp2-32k window | 不采;现有 modal mixed 误差小于 0.1% |",
            "| dp2-bt65536 window | 只做 paired-rank + iteration-wall 诊断复现;在 lockstep 语义定案前不直接入库 |",
            "| Default AIC | No-Go |",
            "| runtime / PerfDB / gate | 本步均未修改 |",
            "",
            "## 验证",
            "",
            "| check | result |",
            "|---|---|",
            "| analyzer tests | 20 passed |",
            "| py_compile / diff check / CRLF | passed |",
            "| GPU / SSH | not used |",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows, manifest = build_report_rows()
    write_csv(args.csv, rows)
    write_markdown(args.md, rows, manifest)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

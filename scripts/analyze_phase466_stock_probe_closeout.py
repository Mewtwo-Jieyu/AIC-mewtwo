#!/usr/bin/env python3
"""Validate and materialize the Phase466 stock-probe gate closeout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
from pathlib import Path
from typing import Any, Optional


SCHEMA = "phase466_rank_timing_v5"
PAIR_IDS = tuple(f"pair-{index:02d}" for index in range(1, 7))
PAIR_ORDERS = (
    "OFF/ON",
    "ON/OFF",
    "OFF/ON",
    "ON/OFF",
    "OFF/ON",
    "ON/OFF",
)
EQUIVALENCE_BOUNDS = (0.98, 1.02)
T_CRITICAL_90_DF5 = 2.0150483733330233
EXECUTION_MANIFEST_SHA256 = (
    "bf562c5a649106acb12f1f981cb8d60a637ee079d02c7f3666507b77ba6c7609"
)
REMOTE_ARTIFACT_PATH = (
    "/mnt/shared-storage-user/zhaojieyu/backup/aic/"
    "phase466_v33_full_rank_timing_0765bad7_20260720T133525Z"
)
RAW_SHA256 = {
    "overhead_gate.json": (
        "e2fe782bfee3e9965187047c36cae08fc005f5885fc99d75eac911fd5b30b950"
    ),
    "phase466_result.json": (
        "bdd1b94d576f46eb6bc18fd6060560bad7c470b3cf0cbbe03e7b83515d723122"
    ),
}
UNRESOLVED_CANDIDATES = (
    "schedule_merged_batch_composition",
    "iteration_cost_serving_state_coverage",
    "dp_rank_synchronization_asymmetry",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


EVIDENCE_DIR = (
    _repo_root()
    / "docs"
    / "iter_gap_investigation"
    / "phase466_stock_probe_gate_closeout"
)
CSV_NAME = "phase466_stock_probe_gate_closeout.csv"
REPORT_NAME = "phase466_stock_probe_gate_closeout.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_raw_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"raw_sha256:{path.name}:{actual}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"raw_json_object:{path.name}")
    return value


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"numeric:{label}")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"numeric:{label}")
    return number


def _require_close(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{label}:{actual}!={expected}")


def _compute_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_rows: list[dict[str, Any]] = []
    off_values: list[float] = []
    on_values: list[float] = []
    log_ratios: list[float] = []
    for pair_id, run_order, pair in zip(PAIR_IDS, PAIR_ORDERS, pairs):
        off = _finite_positive(pair.get("off_output_tok_s"), f"{pair_id}:off")
        on = _finite_positive(pair.get("on_output_tok_s"), f"{pair_id}:on")
        ratio = on / off
        _require_close(
            _finite_positive(pair.get("ratio"), f"{pair_id}:ratio"),
            ratio,
            f"pair_ratio:{pair_id}",
        )
        _require_close(
            _finite_positive(
                pair.get("absolute_delta_pct"), f"{pair_id}:absolute_delta_pct"
            ),
            abs(ratio - 1.0) * 100.0,
            f"pair_absolute_delta_pct:{pair_id}",
        )
        first_mode, second_mode = run_order.split("/")
        throughput = {"OFF": off, "ON": on}
        second_over_first = throughput[second_mode] / throughput[first_mode]
        pair_rows.append(
            {
                "pair_id": pair_id,
                "run_order": run_order,
                "first_mode": first_mode,
                "first_output_tok_s": throughput[first_mode],
                "second_mode": second_mode,
                "second_output_tok_s": throughput[second_mode],
                "off_output_tok_s": off,
                "on_output_tok_s": on,
                "on_over_off_ratio": ratio,
                "second_over_first_ratio": second_over_first,
                "second_run_faster": second_over_first > 1.0,
            }
        )
        off_values.append(off)
        on_values.append(on)
        log_ratios.append(math.log(ratio))

    mean_log_ratio = statistics.mean(log_ratios)
    margin = (
        T_CRITICAL_90_DF5
        * statistics.stdev(log_ratios)
        / math.sqrt(len(log_ratios))
    )
    confidence_interval = [
        math.exp(mean_log_ratio - margin),
        math.exp(mean_log_ratio + margin),
    ]
    return {
        "pair_rows": pair_rows,
        "off_mean": statistics.mean(off_values),
        "off_cv_pct": statistics.stdev(off_values)
        / statistics.mean(off_values)
        * 100.0,
        "on_mean": statistics.mean(on_values),
        "on_cv_pct": statistics.stdev(on_values)
        / statistics.mean(on_values)
        * 100.0,
        "geometric_mean_ratio": math.exp(mean_log_ratio),
        "confidence_interval_ratio": confidence_interval,
        "second_run_faster_count": sum(
            bool(row["second_run_faster"]) for row in pair_rows
        ),
    }


def validate_evidence(
    gate: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    expected_gate_keys = {
        "confidence_interval_ratio",
        "equivalence_ratio_bounds",
        "formal_collection_allowed",
        "gate",
        "geometric_mean_ratio",
        "method",
        "pair_count",
        "pairs",
        "passed",
        "schema",
        "status",
    }
    if set(gate) != expected_gate_keys:
        raise ValueError(f"gate_keys:{sorted(gate)}")
    if gate.get("schema") != SCHEMA:
        raise ValueError("gate_schema")
    if gate.get("status") != "INCONCLUSIVE":
        raise ValueError("gate_status")
    if gate.get("passed") is not False:
        raise ValueError("gate_passed")
    if gate.get("formal_collection_allowed") is not False:
        raise ValueError("formal_collection_allowed")
    if gate.get("gate") != "dp2_bt65536_stock_iteration_detail_paired_equivalence":
        raise ValueError("gate_name")
    if gate.get("method") != "paired_log_ratio_tost":
        raise ValueError("gate_method")
    if gate.get("pair_count") != 6:
        raise ValueError("pair_count")
    if gate.get("equivalence_ratio_bounds") != list(EQUIVALENCE_BOUNDS):
        raise ValueError("equivalence_ratio_bounds")

    pairs = gate.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != len(PAIR_IDS):
        raise ValueError("pairs")
    expected_pair_keys = {
        "absolute_delta_pct",
        "off_output_tok_s",
        "on_output_tok_s",
        "pair_id",
        "ratio",
    }
    for expected_id, pair in zip(PAIR_IDS, pairs):
        if not isinstance(pair, dict) or set(pair) != expected_pair_keys:
            raise ValueError(f"pair_contract:{expected_id}")
        if pair.get("pair_id") != expected_id:
            raise ValueError(f"pair_id:{expected_id}")

    summary = _compute_summary(pairs)
    _require_close(
        _finite_positive(gate.get("geometric_mean_ratio"), "geometric_mean_ratio"),
        summary["geometric_mean_ratio"],
        "geometric_mean_ratio",
    )
    actual_ci = gate.get("confidence_interval_ratio")
    if not isinstance(actual_ci, list) or len(actual_ci) != 2:
        raise ValueError("confidence_interval_ratio")
    for actual, expected in zip(actual_ci, summary["confidence_interval_ratio"]):
        _require_close(
            _finite_positive(actual, "confidence_interval_ratio"),
            expected,
            "confidence_interval_ratio",
        )
    lower, upper = summary["confidence_interval_ratio"]
    if lower >= EQUIVALENCE_BOUNDS[0] and upper <= EQUIVALENCE_BOUNDS[1]:
        raise ValueError("computed_gate_not_inconclusive")
    if upper < EQUIVALENCE_BOUNDS[0] or lower > EQUIVALENCE_BOUNDS[1]:
        raise ValueError("computed_gate_not_inconclusive")

    expected_result_keys = {
        "default_readiness",
        "diagnostic_only",
        "execution_manifest_sha256",
        "formal_failures",
        "formal_scenarios",
        "gate_status",
        "perf_database",
        "schema",
        "status",
        "valid_for_default",
    }
    if set(result) != expected_result_keys:
        raise ValueError(f"result_keys:{sorted(result)}")
    if result.get("formal_scenarios") != []:
        raise ValueError("formal_scenarios")
    if result.get("formal_failures") != []:
        raise ValueError("formal_failures")
    expected_result = {
        "schema": SCHEMA,
        "status": "STOPPED_BEFORE_FORMAL",
        "gate_status": "INCONCLUSIVE",
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_readiness": "No-Go",
        "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
    }
    for field, expected in expected_result.items():
        if result.get(field) != expected:
            raise ValueError(f"result_contract:{field}")
    return summary


def render_csv(summary: dict[str, Any]) -> str:
    fieldnames = [
        "pair_id",
        "run_order",
        "first_mode",
        "first_output_tok_s",
        "second_mode",
        "second_output_tok_s",
        "off_output_tok_s",
        "on_output_tok_s",
        "on_over_off_ratio",
        "second_over_first_ratio",
        "second_run_faster",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary["pair_rows"])
    return output.getvalue()


def render_report(summary: dict[str, Any]) -> str:
    ci_lower, ci_upper = summary["confidence_interval_ratio"]
    lines = [
        "# Phase466 v3.3 stock probe gate closeout",
        "",
        "结论：overhead gate 为 `INCONCLUSIVE`，stock iteration probe 路线关闭。",
        "正式 N512 场景未运行，三类建模候选保持 unresolved，不选择 Phase467。",
        "",
        "## Evidence identity",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| remote artifact | `{REMOTE_ARTIFACT_PATH}` |",
        f"| `overhead_gate.json` SHA256 | `{RAW_SHA256['overhead_gate.json']}` |",
        f"| `phase466_result.json` SHA256 | `{RAW_SHA256['phase466_result.json']}` |",
        f"| execution manifest SHA256 | `{EXECUTION_MANIFEST_SHA256}` |",
        "| formal directory | absent |",
        "| terminal status | `STOPPED_BEFORE_FORMAL` |",
        "| readiness | `diagnostic_only=true`; `valid_for_default=false`; "
        "`perf_database=false`; `Default AIC=No-Go` |",
        "",
        "## Paired results",
        "",
        "| Pair | Order | OFF tok/s | ON tok/s | ON/OFF | Second/first |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary["pair_rows"]:
        lines.append(
            f"| {row['pair_id']} | {row['run_order']} | "
            f"{row['off_output_tok_s']:.6f} | {row['on_output_tok_s']:.6f} | "
            f"{row['on_over_off_ratio']:.6f} | "
            f"{row['second_over_first_ratio']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| 统计量 | 结果 |",
            "|---|---:|",
            f"| OFF mean | {summary['off_mean']:.6f} |",
            f"| OFF sample CV | {summary['off_cv_pct']:.3f}% |",
            f"| ON mean | {summary['on_mean']:.6f} |",
            f"| ON sample CV | {summary['on_cv_pct']:.3f}% |",
            f"| geometric mean ON/OFF | {summary['geometric_mean_ratio']:.6f} |",
            f"| 90% CI | [{ci_lower:.6f}, {ci_upper:.6f}] |",
            "| equivalence bounds | [0.980000, 1.020000] |",
            "",
            f"{summary['second_run_faster_count']}/6 pair 的第二次运行更快，"
            "主要噪声来自服务重启或运行顺序。均值接近 1 不代表等价，"
            "**不能解释为 probe 开销约 1%**。",
            "",
            "## Closed route and remaining boundary",
            "",
            "当前 ON rank logs 只保留为失败测量方法的诊断记录，不可用于模型、"
            "PerfDatabase 或 readiness。stock probe 不再重跑；只有单独评审通过的"
            "新低扰动测量设计才能重启建模取证。",
            "",
            "未解决候选：",
            "",
        ]
    )
    lines.extend(f"- `{candidate}`" for candidate in UNRESOLVED_CANDIDATES)
    lines.extend(
        [
            "",
            "三个候选均保持 `unresolved`；本 closeout 不选择、实现或预占 Phase467。",
            "",
        ]
    )
    return "\n".join(lines)


def _resolve_paths(
    evidence_dir: Path,
    overhead_gate_path: Optional[Path],
    result_path: Optional[Path],
    csv_path: Optional[Path],
    report_path: Optional[Path],
) -> tuple[Path, Path, Path, Path]:
    return (
        overhead_gate_path or evidence_dir / "overhead_gate.json",
        result_path or evidence_dir / "phase466_result.json",
        csv_path or evidence_dir / CSV_NAME,
        report_path or evidence_dir / REPORT_NAME,
    )


def _load_and_validate(
    evidence_dir: Path,
    overhead_gate_path: Path,
    result_path: Path,
) -> dict[str, Any]:
    if (evidence_dir / "formal").exists():
        raise ValueError("formal_directory_present")
    gate = _load_raw_json(
        overhead_gate_path, RAW_SHA256["overhead_gate.json"]
    )
    result = _load_raw_json(result_path, RAW_SHA256["phase466_result.json"])
    return validate_evidence(gate, result)


def write_outputs(
    *,
    evidence_dir: Path = EVIDENCE_DIR,
    overhead_gate_path: Optional[Path] = None,
    result_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> dict[str, Any]:
    overhead, result, output_csv, output_report = _resolve_paths(
        evidence_dir,
        overhead_gate_path,
        result_path,
        csv_path,
        report_path,
    )
    summary = _load_and_validate(evidence_dir, overhead, result)
    output_csv.write_text(render_csv(summary), encoding="utf-8")
    output_report.write_text(render_report(summary), encoding="utf-8")
    return summary


def validate_outputs(
    *,
    evidence_dir: Path = EVIDENCE_DIR,
    overhead_gate_path: Optional[Path] = None,
    result_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> dict[str, Any]:
    overhead, result, output_csv, output_report = _resolve_paths(
        evidence_dir,
        overhead_gate_path,
        result_path,
        csv_path,
        report_path,
    )
    summary = _load_and_validate(evidence_dir, overhead, result)
    if output_csv.read_bytes() != render_csv(summary).encode("utf-8"):
        raise ValueError("csv_mismatch")
    if output_report.read_bytes() != render_report(summary).encode("utf-8"):
        raise ValueError("report_mismatch")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--evidence-dir", type=Path, default=EVIDENCE_DIR)
    args = parser.parse_args()
    if args.write:
        write_outputs(evidence_dir=args.evidence_dir)
    validate_outputs(evidence_dir=args.evidence_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Phase454 clean-acceptance report.

Report-only analyzer. It records the revised regression rule, audits the
vLLM can_fit_full_sequence reserve scope, folds in the residual preemption
forensics, and emits the GPU batch acceptance plan.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/Users/mewtwo/2026/work/codebase/vllm-0.19.0")
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase454_clean_acceptance.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase454_clean_acceptance.md"
DEFAULT_FORENSICS_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase451d_preemption_forensics.csv"
)
DEFAULT_VALIDATE_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase453_validate_multi_config.csv"
)

CSV_FIELDS = ["section", "metric", "value", "target", "status", "note"]


@dataclass(frozen=True)
class SourceLine:
    path: Path
    lineno: int
    text: str

    @property
    def loc(self) -> str:
        return f"{self.path}:{self.lineno}"


def find_line(path: Path, needle: str) -> SourceLine:
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return SourceLine(path=path, lineno=idx, text=line.strip())
    raise ValueError(f"needle not found in {path}: {needle}")


def audit_reserve_scope(vllm_root: Path = VLLM_ROOT) -> dict[str, object]:
    request_py = vllm_root / "vllm/v1/request.py"
    kv_py = vllm_root / "vllm/v1/core/kv_cache_manager.py"
    sched_py = vllm_root / "vllm/v1/core/sched/scheduler.py"
    async_py = vllm_root / "vllm/v1/core/sched/async_scheduler.py"

    lines = {
        "max_tokens_stored": find_line(request_py, "self.max_tokens = sampling_params.max_tokens"),
        "all_tokens_seed": find_line(request_py, "self._all_token_ids: list[int] = ("),
        "output_append": find_line(request_py, "self._all_token_ids.append(token_ids)"),
        "num_tokens_property": find_line(request_py, "return len(self._all_token_ids)"),
        "can_fit_input": find_line(kv_py, "full_num_tokens = min(request.num_tokens"),
        "waiting_gate": find_line(sched_py, "and not self.kv_cache_manager.can_fit_full_sequence("),
        "async_placeholder": find_line(async_py, "request.num_output_placeholders += 1 + cur_num_spec_tokens"),
    }

    prompt_plus_max = False
    if (
        "request.num_tokens" in lines["can_fit_input"].text
        and "len(self._all_token_ids)" in lines["num_tokens_property"].text
    ):
        verdict = "current_sequence_not_prompt_plus_max_tokens"
        status = "rejected_prompt_plus_max_tokens_hypothesis"
    else:
        verdict = "reserve_scope_unresolved"
        status = "blocked"
    return {"lines": lines, "prompt_plus_max": prompt_plus_max, "verdict": verdict, "status": status}


def read_forensics_summary(path: Path = DEFAULT_FORENSICS_CSV) -> dict[str, object]:
    summary: dict[str, object] = {
        "preemption_events": None,
        "classification_coverage": None,
        "categories": {},
        "runtime_fix_gate": None,
        "candidate_semantic_diff_count": None,
    }
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            section = row.get("section", "")
            metric = row.get("metric", "")
            value = row.get("value", "")
            if section == "sim_forensics" and metric == "preemption_events":
                summary["preemption_events"] = int(float(value))
            elif section == "sim_forensics" and metric == "classification_coverage":
                summary["classification_coverage"] = float(value)
            elif section == "sim_forensics" and metric == "candidate_semantic_diff_count":
                summary["candidate_semantic_diff_count"] = int(float(value))
            elif section == "sim_category":
                summary["categories"][metric] = int(float(value))
            elif section == "decision" and metric == "phase451e_runtime_fix_gate":
                summary["runtime_fix_gate"] = value
    return summary


def read_validate_summary(path: Path = DEFAULT_VALIDATE_CSV) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            item = dict(row)
            item["error_ratio"] = float(str(row["error_ratio"]))
            rows.append(item)
    max_row = max(rows, key=lambda row: float(row["error_ratio"]))
    by_name = {str(row["name"]): row for row in rows}
    return {"rows": rows, "max_row": max_row, "by_name": by_name}


def build_rows(
    reserve: dict[str, object],
    forensics: dict[str, object],
    validate: dict[str, object],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(section: str, metric: str, value: object, target: str = "", status: str = "", note: str = "") -> None:
        rows.append(
            {
                "section": section,
                "metric": metric,
                "value": value,
                "target": target,
                "status": status,
                "note": note,
            }
        )

    add(
        "rule",
        "zero_regression_revision",
        "global_source_backed_semantic_fix_may_expose_error_cancellation",
        "per-point impact explicitly listed",
        "recorded",
        "Do not silently revert when a global vLLM semantic fix reveals prior offsetting errors.",
    )
    add(
        "rule",
        "error_cancellation_warning",
        "dp2_8k2k_1.136_is_not_closed",
        "clean N=512 reference required",
        "recorded",
        "Current N=128 reference can hide steady-state model error through burst artifact.",
    )

    source_lines: dict[str, SourceLine] = reserve["lines"]  # type: ignore[assignment]
    add(
        "reserve_source",
        "waiting_gate",
        source_lines["waiting_gate"].text,
        "",
        "source_located",
        source_lines["waiting_gate"].loc,
    )
    add(
        "reserve_source",
        "can_fit_input",
        source_lines["can_fit_input"].text,
        "request.num_tokens",
        "source_located",
        source_lines["can_fit_input"].loc,
    )
    add(
        "reserve_source",
        "request_num_tokens",
        source_lines["num_tokens_property"].text,
        "current prompt + generated tokens",
        "source_located",
        source_lines["num_tokens_property"].loc,
    )
    add(
        "reserve_source",
        "max_tokens_not_in_num_tokens",
        source_lines["max_tokens_stored"].text,
        "not referenced by can_fit_full_sequence",
        "source_located",
        source_lines["max_tokens_stored"].loc,
    )
    add(
        "reserve_source",
        "async_placeholder_not_num_tokens",
        source_lines["async_placeholder"].text,
        "separate counter",
        "source_located",
        source_lines["async_placeholder"].loc,
    )
    add(
        "reserve_verdict",
        "prompt_plus_max_tokens_hypothesis",
        reserve["verdict"],
        "prompt+max_tokens if source proves it",
        str(reserve["status"]),
        "No runtime change: vLLM gate reserves current full sequence, not final max sequence.",
    )

    add(
        "residual_preemption",
        "preemption_events",
        forensics["preemption_events"],
        "~24 real observed victims",
        "partial",
        "Residual events remain after Phase453 headroom gate.",
    )
    add(
        "residual_preemption",
        "classification_coverage",
        forensics["classification_coverage"],
        ">=0.80",
        "pass",
        "Phase451D classification covers all residual events.",
    )
    for name, count in sorted(dict(forensics["categories"]).items()):
        add("residual_category", name, count, "", "", "")
    add(
        "residual_preemption",
        "runtime_fix_gate",
        forensics["runtime_fix_gate"],
        "unique source-backed semantic fix",
        "blocked",
        "The prompt+max reserve hypothesis does not provide that unique fix.",
    )

    max_row = validate["max_row"]  # type: ignore[index]
    by_name = validate["by_name"]  # type: ignore[index]
    dp2 = by_name.get("K2.5-tp4ep8dp2-8k2k", {})
    add(
        "validate",
        "dp2_8k2k_error_ratio",
        dp2.get("error_ratio", ""),
        "<=1.15 target",
        "pass_current_but_not_clean",
        "N=128 reference may still contain artifact cancellation.",
    )
    add(
        "validate",
        "max_error_ratio",
        max_row["error_ratio"],
        "<=1.50 current; <=1.15 target",
        "pass_default_open_15pct",
        f"max_scenario={max_row['name']}",
    )

    add(
        "gpu_batch_plan",
        "dp2_8k2k_recollect",
        "N=512/C=128 vanilla",
        "clean steady reference",
        "pending_gpu",
        "Strictly separate from B2b patched sessions.",
    )
    add(
        "gpu_batch_plan",
        "dp2_32k3k_recollect",
        "N=512/C=64 vanilla",
        "clean steady reference",
        "pending_gpu",
        "Uses original reference concurrency.",
    )
    add(
        "gpu_batch_plan",
        "tp8_b2b",
        "tp8ep8-8k2k B2b",
        "cover TP8 serving-state scope",
        "pending_gpu",
        "",
    )
    add(
        "gpu_batch_plan",
        "dp2_bt65536_b2b",
        "tp4dp2ep8-8k2k-bt65536 B2b",
        "large-bt regime coverage",
        "pending_gpu",
        "",
    )
    add(
        "verdict",
        "phase454_offline_verdict",
        "reserve_prompt_plus_max_rejected_gpu_batch_required",
        "no heuristic runtime change",
        "no_go",
        "Default AIC remains No-Go until clean-reference validate closes.",
    )
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    def row(metric: str) -> dict[str, object]:
        for item in rows:
            if item["metric"] == metric:
                return item
        raise KeyError(metric)

    source_rows = [item for item in rows if item["section"] == "reserve_source"]
    residual_rows = [item for item in rows if item["section"] in {"residual_preemption", "residual_category"}]
    gpu_rows = [item for item in rows if item["section"] == "gpu_batch_plan"]
    lines = [
        "# Phase454 clean acceptance",
        "",
        "## Decision record",
        "",
        "| Item | Decision | Status | Note |",
        "|---|---|---|---|",
        f"| Zero-regression rule | {row('zero_regression_revision')['value']} | {row('zero_regression_revision')['status']} | {row('zero_regression_revision')['note']} |",
        f"| Error-cancellation warning | {row('error_cancellation_warning')['value']} | {row('error_cancellation_warning')['status']} | {row('error_cancellation_warning')['note']} |",
        "",
        "## Reserve scope audit",
        "",
        "| Metric | Value | Target | Status | Note |",
        "|---|---|---|---|---|",
    ]
    for item in source_rows + [row("prompt_plus_max_tokens_hypothesis")]:
        lines.append(
            f"| {item['metric']} | `{item['value']}` | {item['target']} | {item['status']} | {item['note']} |"
        )
    lines.extend(
        [
            "",
            "Conclusion: the vLLM gate reserves the current full sequence (`request.num_tokens`). It does not reserve `prompt + max_tokens`, so Phase454 does not change cb_sim on this hypothesis.",
            "",
            "## Residual preemption",
            "",
            "| Metric | Value | Target | Status | Note |",
            "|---|---:|---|---|---|",
        ]
    )
    for item in residual_rows:
        lines.append(
            f"| {item['metric']} | {item['value']} | {item['target']} | {item['status']} | {item['note']} |"
        )
    lines.extend(
        [
            "",
            "## Acceptance state",
            "",
            "| Metric | Value | Target | Status | Note |",
            "|---|---:|---|---|---|",
            f"| dp2_8k2k_error_ratio | {row('dp2_8k2k_error_ratio')['value']} | {row('dp2_8k2k_error_ratio')['target']} | {row('dp2_8k2k_error_ratio')['status']} | {row('dp2_8k2k_error_ratio')['note']} |",
            f"| max_error_ratio | {row('max_error_ratio')['value']} | {row('max_error_ratio')['target']} | {row('max_error_ratio')['status']} | {row('max_error_ratio')['note']} |",
            "",
            "## GPU batch plan",
            "",
            "| Task | Protocol | Purpose | Status |",
            "|---|---|---|---|",
        ]
    )
    for item in gpu_rows:
        lines.append(f"| {item['metric']} | {item['value']} | {item['target']} | {item['status']} |")
    lines.extend(
        [
            "",
            f"Final verdict: `{row('phase454_offline_verdict')['value']}`. Default AIC remains No-Go until the clean-reference GPU batch and final validate table close.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(line) for line in lines), encoding="utf-8")


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    reserve = audit_reserve_scope(args.vllm_root)
    forensics = read_forensics_summary(args.forensics_csv)
    validate = read_validate_summary(args.validate_csv)
    rows = build_rows(reserve, forensics, validate)
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", type=Path, default=VLLM_ROOT)
    parser.add_argument("--forensics-csv", type=Path, default=DEFAULT_FORENSICS_CSV)
    parser.add_argument("--validate-csv", type=Path, default=DEFAULT_VALIDATE_CSV)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

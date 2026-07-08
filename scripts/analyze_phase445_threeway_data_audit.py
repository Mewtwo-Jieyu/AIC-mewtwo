#!/usr/bin/env python3
"""Phase445-A: audit whether kernel/module/serving rows can be compared safely."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVING_STATE = (
    REPO_ROOT
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0/vllm_serving_state_perf.txt"
)
DEFAULT_UPSTREAM_DATA_DIR = (
    Path("/tmp/aiconfigurator")
    / "src/aiconfigurator/systems/data/h200_sxm/vllm/0.19.0"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase445_threeway_data_audit.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase445_threeway_data_audit.md"

SOURCE = "phase445_threeway_data_audit"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "row_type",
    "phase",
    "category",
    "serving_rows",
    "serving_bucket_min",
    "serving_bucket_max",
    "upstream_file",
    "upstream_rows",
    "upstream_filtered_rows",
    "upstream_token_min",
    "upstream_token_max",
    "alignment_status",
    "runtime_action",
    "note",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class AlignmentRow:
    phase: str
    category: str
    serving_rows: int
    serving_bucket_min: int | None
    serving_bucket_max: int | None
    upstream_file: str
    upstream_rows: int
    upstream_filtered_rows: int
    upstream_token_min: int | None
    upstream_token_max: int | None
    alignment_status: str
    runtime_action: str
    note: str


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _base_row() -> dict[str, object]:
    return {
        "source": SOURCE,
        "row_type": "alignment",
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def _range(values: list[int]) -> tuple[int | None, int | None]:
    if not values:
        return None, None
    return min(values), max(values)


def read_serving_summary(path: Path) -> dict[tuple[str, str], dict[str, int | None]]:
    summary: dict[tuple[str, str], dict[str, int | None]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"empty serving-state table: {path}")
        required = {"phase", "category", "bucket_tokens"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"serving-state table missing columns: {sorted(missing)}")
        buckets_by_key: dict[tuple[str, str], list[int]] = {}
        rows_by_key: dict[tuple[str, str], int] = {}
        for row in reader:
            key = (row["phase"], row["category"])
            buckets_by_key.setdefault(key, []).append(int(row["bucket_tokens"]))
            rows_by_key[key] = rows_by_key.get(key, 0) + 1

    for key, buckets in sorted(buckets_by_key.items()):
        bucket_min, bucket_max = _range(buckets)
        summary[key] = {
            "rows": rows_by_key[key],
            "bucket_min": bucket_min,
            "bucket_max": bucket_max,
        }
    if not summary:
        raise ValueError(f"no serving-state rows in {path}")
    return summary


def _parquet_summary(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on local environment
        return {"status": f"pyarrow_missing:{exc}", "rows": 0, "columns": []}

    if not path.exists():
        return {"status": "missing", "rows": 0, "columns": []}
    parquet_file = pq.ParquetFile(path)
    return {
        "status": "available",
        "rows": parquet_file.metadata.num_rows,
        "columns": list(parquet_file.schema_arrow.names),
    }


def inspect_upstream_data(data_dir: Path) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for name in (
        "moe_perf.parquet",
        "mla_context_module_perf.parquet",
        "mla_generation_module_perf.parquet",
        "custom_allreduce_perf.parquet",
        "context_attention_perf.parquet",
        "generation_attention_perf.parquet",
    ):
        summary[name] = _parquet_summary(data_dir / name)

    moe_path = data_dir / "moe_perf.parquet"
    if summary["moe_perf.parquet"].get("status") == "available":
        import pyarrow.parquet as pq

        table = pq.read_table(
            moe_path,
            columns=[
                "moe_dtype",
                "num_tokens",
                "hidden_size",
                "topk",
                "num_experts",
                "moe_tp_size",
                "moe_ep_size",
                "kernel_source",
            ],
        )
        rows = table.to_pylist()
        filtered = [
            row
            for row in rows
            if row["moe_dtype"] == "int4_wo"
            and int(row["hidden_size"]) == 7168
            and int(row["topk"]) == 8
            and int(row["num_experts"]) == 384
            and int(row["moe_tp_size"]) == 1
            and int(row["moe_ep_size"]) == 8
        ]
        tokens = [int(row["num_tokens"]) for row in filtered]
        token_min, token_max = _range(tokens)
        summary["moe_perf.parquet"].update(
            {
                "filtered_rows": len(filtered),
                "token_min": token_min,
                "token_max": token_max,
                "kernel_sources": ",".join(sorted({str(row["kernel_source"]) for row in filtered})),
            }
        )
    else:
        summary["moe_perf.parquet"].update({"filtered_rows": 0, "token_min": None, "token_max": None})

    for name, item in summary.items():
        item.setdefault("filtered_rows", 0)
        item.setdefault("token_min", None)
        item.setdefault("token_max", None)
        item.setdefault("kernel_sources", "")
    return summary


def build_alignment_rows(
    *,
    serving_summary: dict[tuple[str, str], dict[str, int | None]],
    upstream_summary: dict[str, dict[str, Any]],
) -> list[AlignmentRow]:
    rows: list[AlignmentRow] = []
    for (phase, category), serving in sorted(serving_summary.items()):
        upstream_file = ""
        status = "missing_direct_upstream_table"
        note = "No upstream parquet table directly matches this serving-state category."
        upstream = {"rows": 0, "filtered_rows": 0, "token_min": None, "token_max": None}

        if category == "moe_gemm_or_aux":
            upstream_file = "moe_perf.parquet"
            upstream = upstream_summary.get(upstream_file, upstream)
            if int(upstream.get("filtered_rows") or 0) > 0:
                status = "partial_requires_layer_normalization"
                note = (
                    "Upstream MoE rows are module/microbenchmark entries; serving rows are whole-step "
                    "category timings and may include auxiliary kernels. A runtime coefficient needs an "
                    "explicit layer/category normalization first."
                )
            else:
                status = "missing_filtered_moe_rows"
                note = "moe_perf.parquet exists but lacks the K2.5 int4_wo tp1/ep8 filtered scope."
        elif category == "collective_other":
            upstream_file = "custom_allreduce_perf.parquet"
            upstream = upstream_summary.get(upstream_file, upstream)
            if int(upstream.get("rows") or 0) > 0:
                status = "partial_collective_scope_mismatch"
                note = (
                    "custom_allreduce rows are not equivalent to the serving collective_other bucket; "
                    "the serving bucket can contain multiple collective families."
                )
        elif category == "ep_a2a":
            note = "The upstream vLLM 0.19 parquet set has no direct EP8 all-to-all serving/module table."
        elif category == "other_cuda":
            note = "other_cuda is a profiler residual category; no upstream kernel/module table maps to it."

        rows.append(
            AlignmentRow(
                phase=phase,
                category=category,
                serving_rows=int(serving.get("rows") or 0),
                serving_bucket_min=serving.get("bucket_min"),
                serving_bucket_max=serving.get("bucket_max"),
                upstream_file=upstream_file,
                upstream_rows=int(upstream.get("rows") or 0),
                upstream_filtered_rows=int(upstream.get("filtered_rows") or 0),
                upstream_token_min=upstream.get("token_min"),
                upstream_token_max=upstream.get("token_max"),
                alignment_status=status,
                runtime_action="report_only",
                note=note,
            )
        )
    return rows


def decide_verdict(rows: list[AlignmentRow]) -> str:
    if rows and all(row.alignment_status == "direct_aligned" for row in rows):
        return "runtime_coefficient_candidate"
    return "no_runtime_coefficient_schema_alignment_blocked"


def _alignment_to_csv_row(row: AlignmentRow) -> dict[str, object]:
    output = _base_row()
    output.update(
        {
            "phase": row.phase,
            "category": row.category,
            "serving_rows": row.serving_rows,
            "serving_bucket_min": row.serving_bucket_min,
            "serving_bucket_max": row.serving_bucket_max,
            "upstream_file": row.upstream_file,
            "upstream_rows": row.upstream_rows,
            "upstream_filtered_rows": row.upstream_filtered_rows,
            "upstream_token_min": row.upstream_token_min,
            "upstream_token_max": row.upstream_token_max,
            "alignment_status": row.alignment_status,
            "runtime_action": row.runtime_action,
            "note": row.note,
        }
    )
    return output


def write_csv(rows: list[AlignmentRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(value) for key, value in _alignment_to_csv_row(row).items()})


def _md_table(headers: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(value) for value in row) + " |")
    return lines


def write_markdown(
    *,
    rows: list[AlignmentRow],
    upstream_summary: dict[str, dict[str, Any]],
    verdict: str,
    path: Path,
) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row.alignment_status] = status_counts.get(row.alignment_status, 0) + 1

    upstream_rows = [
        [
            name,
            item.get("status", ""),
            item.get("rows", 0),
            item.get("filtered_rows", 0),
            item.get("token_min"),
            item.get("token_max"),
        ]
        for name, item in sorted(upstream_summary.items())
    ]
    alignment_rows = [
        [
            row.phase,
            row.category,
            row.serving_rows,
            row.serving_bucket_min,
            row.serving_bucket_max,
            row.upstream_file or "none",
            row.alignment_status,
        ]
        for row in rows
    ]

    lines = [
        "# Phase445-A Three-Way Data Audit",
        "",
        f"- Verdict: `{verdict}`.",
        "- No runtime coefficient was generated; this is a schema and evidence audit only.",
        "- `moe_gemm_or_aux` is only partially comparable. EP a2a and residual CUDA categories lack a direct upstream table.",
        "- TP8 out-of-sample validation is blocked until a category map and layer normalization make the DP2 serving rows transferable.",
        "",
        "## Alignment Summary",
    ]
    lines.extend(_md_table(["status", "count"], [[key, value] for key, value in sorted(status_counts.items())]))
    lines.extend(["", "## Category Alignment"])
    lines.extend(
        _md_table(
            [
                "phase",
                "category",
                "serving_rows",
                "bucket_min",
                "bucket_max",
                "upstream_file",
                "alignment_status",
            ],
            alignment_rows,
        )
    )
    lines.extend(["", "## Upstream Inventory"])
    lines.extend(
        _md_table(
            ["file", "status", "rows", "filtered_rows", "token_min", "token_max"],
            upstream_rows,
        )
    )
    lines.extend(
        [
            "",
            "## Next Gate",
            "",
            "Before any mechanism coefficient can enter runtime, build an explicit category map that states exactly how a serving-step category maps to module/kernel rows. For MoE, that map must include layer count and prove the serving category is not mixing non-MoE auxiliary work. For EP a2a, a direct EP8 all-to-all table or serving-derived primitive is still required.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(serving_state: Path, upstream_data_dir: Path, csv_path: Path, md_path: Path) -> str:
    serving_summary = read_serving_summary(serving_state)
    upstream_summary = inspect_upstream_data(upstream_data_dir)
    rows = build_alignment_rows(serving_summary=serving_summary, upstream_summary=upstream_summary)
    verdict = decide_verdict(rows)
    write_csv(rows, csv_path)
    write_markdown(rows=rows, upstream_summary=upstream_summary, verdict=verdict, path=md_path)
    return verdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving-state", type=Path, default=DEFAULT_SERVING_STATE)
    parser.add_argument("--upstream-data-dir", type=Path, default=DEFAULT_UPSTREAM_DATA_DIR)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    verdict = run_analysis(args.serving_state, args.upstream_data_dir, args.csv_out, args.md_out)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    print(f"verdict={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

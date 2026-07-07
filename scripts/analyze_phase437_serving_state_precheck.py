#!/usr/bin/env python3
"""Phase437: classify serving-state misses before GPU grid collection."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase437_serving_state_precheck"
DEFAULT_AUDIT = REPO_ROOT / "docs/iter_gap_investigation/phase436_serving_state_hit_audit_after.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase437_serving_state_precheck.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase437_serving_state_precheck.md"
DEFAULT_READINESS = "No-Go"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "phase",
    "category",
    "miss_reason",
    "classification",
    "grid_axis",
    "collection_required",
    "expected",
    "bucket_tokens",
    "decode_batch",
    "bucket_min",
    "bucket_max",
    "decode_batch_min",
    "decode_batch_max",
    "unique_points",
    "count",
    "collection_note",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def classify_audit_row(row: dict[str, str]) -> dict[str, object]:
    hit = row.get("hit") == "true"
    phase = row.get("phase", "")
    reason = "hit" if hit else row.get("miss_reason", "")

    classification = "covered"
    grid_axis = "none"
    collection_required = False
    expected = True
    note = "serving-state query already hits the measured grid"

    if hit:
        pass
    elif reason == "decode_batch_above_range":
        classification = "in_scope_grid_gap"
        grid_axis = "decode_batch_high"
        collection_required = True
        note = "collect higher decode_batch grid points for this phase/category"
    elif reason == "bucket_above_range":
        classification = "in_scope_grid_gap"
        grid_axis = "token_or_decode_bucket_high"
        collection_required = True
        note = "collect higher token bucket points; decode phase uses bucket_tokens=decode_batch"
    elif reason == "interpolation_gap":
        classification = "in_scope_grid_gap"
        grid_axis = "rectangular_grid_density"
        collection_required = True
        note = "existing diagonal/window points do not bracket this 2D query"
    elif reason == "bucket_below_range":
        classification = "expected_ramp_or_tail_fallback"
        grid_axis = "low_ramp_below_grid"
        collection_required = False
        note = "low-batch or partial-prefill ramp/tail query; keep fallback unless it becomes a gate target"
    elif reason == "table_missing" and phase == "prefill":
        classification = "expected_pure_prefill_fallback"
        grid_axis = "pure_prefill_not_in_serving_state_table"
        collection_required = False
        note = "pure prefill remains on the baseline model; Phase437 targets mixed_prefill and decode"
    else:
        classification = "unexpected_table_or_key_gap"
        grid_axis = "unknown"
        collection_required = False
        expected = False
        note = "unexpected serving-state miss; fix table/key structure before GPU collection"

    return {
        "classification": classification,
        "grid_axis": grid_axis,
        "collection_required": collection_required,
        "expected": expected,
        "collection_note": note,
    }


def _base(row_type: str) -> dict[str, object]:
    return {
        "source": SOURCE,
        "row_type": row_type,
        "gpu_allowed": False,
        "ssh_allowed": False,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def build_rows(audit_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    workpoints: list[dict[str, object]] = []
    summary: dict[tuple[str, ...], dict[str, object]] = {}

    for row in audit_rows:
        cls = classify_audit_row(row)
        count = int(row.get("count") or 0)
        item = _base("workpoint_scatter")
        item.update(cls)
        for field in (
            "scenario",
            "phase",
            "category",
            "miss_reason",
            "bucket_tokens",
            "decode_batch",
            "bucket_min",
            "bucket_max",
            "decode_batch_min",
            "decode_batch_max",
        ):
            item[field] = row.get(field, "")
        if row.get("hit") == "true":
            item["miss_reason"] = "hit"
        item["count"] = count
        workpoints.append(item)

        key = (
            str(item.get("scenario", "")),
            str(item.get("phase", "")),
            str(item.get("category", "")),
            str(item.get("miss_reason", "")),
            str(item.get("classification", "")),
            str(item.get("grid_axis", "")),
        )
        if key not in summary:
            summary[key] = _base("miss_class_summary")
            summary[key].update(cls)
            (
                summary[key]["scenario"],
                summary[key]["phase"],
                summary[key]["category"],
                summary[key]["miss_reason"],
                summary[key]["classification"],
                summary[key]["grid_axis"],
            ) = key
            summary[key]["unique_points"] = 0
            summary[key]["count"] = 0
        summary[key]["unique_points"] = int(summary[key]["unique_points"]) + 1
        summary[key]["count"] = int(summary[key]["count"]) + count

    axis_rows = _build_axis_rows(workpoints)
    return sorted(summary.values(), key=_sort_key) + axis_rows + sorted(workpoints, key=_sort_key)


def _sort_key(row: dict[str, object]) -> tuple:
    return (
        str(row.get("row_type", "")),
        str(row.get("scenario", "")),
        str(row.get("phase", "")),
        str(row.get("category", "")),
        str(row.get("miss_reason", "")),
        _int(str(row.get("bucket_tokens", ""))) or -1,
        _int(str(row.get("decode_batch", ""))) or -1,
    )


def _build_axis_rows(workpoints: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in workpoints:
        if row.get("collection_required") is not True:
            continue
        grouped[
            (
                str(row.get("scenario", "")),
                str(row.get("phase", "")),
                str(row.get("category", "")),
                str(row.get("grid_axis", "")),
            )
        ].append(row)

    axis_rows: list[dict[str, object]] = []
    for (scenario, phase, category, grid_axis), rows in sorted(grouped.items()):
        buckets = [_int(str(row.get("bucket_tokens", ""))) for row in rows]
        batches = [_int(str(row.get("decode_batch", ""))) for row in rows]
        buckets_i = [value for value in buckets if value is not None]
        batches_i = [value for value in batches if value is not None]
        axis = _base("collection_axis_requirement")
        axis.update(
            {
                "scenario": scenario,
                "phase": phase,
                "category": category,
                "grid_axis": grid_axis,
                "classification": "phase437_required_grid_coverage",
                "collection_required": True,
                "expected": True,
                "bucket_min": min(buckets_i) if buckets_i else "",
                "bucket_max": max(buckets_i) if buckets_i else "",
                "decode_batch_min": min(batches_i) if batches_i else "",
                "decode_batch_max": max(batches_i) if batches_i else "",
                "unique_points": len(rows),
                "count": sum(int(row.get("count") or 0) for row in rows),
                "collection_note": _axis_note(grid_axis),
            }
        )
        axis_rows.append(axis)
    return axis_rows


def _axis_note(grid_axis: str) -> str:
    if grid_axis == "decode_batch_high":
        return "Phase437 decode-batch sweep must cover the observed high batch endpoint."
    if grid_axis == "token_or_decode_bucket_high":
        return "Phase437 token/decode bucket sweep must cover the observed high token endpoint."
    if grid_axis == "rectangular_grid_density":
        return "Phase437 must collect a real 2D grid; diagonal-only points are insufficient."
    return "Phase437 collection must cover this grid axis."


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def _counter(rows: list[dict[str, object]], *, row_type: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row.get("row_type") != row_type:
            continue
        key = str(row.get("classification", ""))
        counts[key] += int(row.get("count") or 0)
    return counts


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    summary_rows = [row for row in rows if row.get("row_type") == "miss_class_summary"]
    axis_rows = [row for row in rows if row.get("row_type") == "collection_axis_requirement"]
    workpoint_rows = [row for row in rows if row.get("row_type") == "workpoint_scatter"]
    unexpected = [
        row
        for row in summary_rows
        if row.get("expected") is False or str(row.get("classification")) == "unexpected_table_or_key_gap"
    ]
    required_count = sum(int(row.get("count") or 0) for row in workpoint_rows if row.get("collection_required") is True)
    expected_fallback_count = sum(
        int(row.get("count") or 0)
        for row in workpoint_rows
        if row.get("collection_required") is False and row.get("classification") != "covered"
    )
    hit_count = sum(int(row.get("count") or 0) for row in workpoint_rows if row.get("classification") == "covered")
    verdict = "serving_state_grid_collection_can_proceed" if not unexpected else "fix_table_structure_before_gpu"

    class_counts = _counter(rows, row_type="workpoint_scatter")
    miss_reason_counts: Counter[str] = Counter()
    for row in workpoint_rows:
        miss_reason_counts[str(row.get("miss_reason", ""))] += int(row.get("count") or 0)

    lines = [
        "# Phase437 Serving-State Precheck",
        "",
        "## Verdict",
        "",
        f"- verdict: `{verdict}`.",
        f"- hit count: {hit_count}.",
        f"- GPU-grid-required miss count: {required_count}.",
        f"- expected fallback miss count: {expected_fallback_count}.",
        "- No serving-state runtime, PerfDB, or gate logic was changed.",
        "- Default AIC remains No-Go.",
        "",
        "## Miss Classification",
        "",
        "| classification | weighted count | meaning |",
        "|---|---:|---|",
    ]
    meanings = {
        "covered": "already hits current measured grid",
        "in_scope_grid_gap": "Phase437 must collect measured grid points",
        "expected_ramp_or_tail_fallback": "low ramp/tail query remains on baseline fallback",
        "expected_pure_prefill_fallback": "pure prefill remains outside serving-state table",
        "unexpected_table_or_key_gap": "table/key bug; do not collect GPU grid until fixed",
    }
    for key, value in sorted(class_counts.items()):
        lines.append(f"| `{key}` | {value} | {meanings.get(key, '')} |")

    lines.extend(
        [
            "",
            "## Raw Miss Reasons",
            "",
            "| miss reason | weighted count |",
            "|---|---:|",
        ]
    )
    for key, value in sorted(miss_reason_counts.items()):
        lines.append(f"| `{key}` | {value} |")

    lines.extend(
        [
            "",
            "## Collection Requirements",
            "",
            "| scenario | phase | category | axis | bucket range | decode batch range | count |",
            "|---|---|---|---|---:|---:|---:|",
        ]
    )
    for row in axis_rows:
        lines.append(
            "| {scenario} | {phase} | {category} | {axis} | {bmin}-{bmax} | {dmin}-{dmax} | {count} |".format(
                scenario=row.get("scenario", ""),
                phase=row.get("phase", ""),
                category=row.get("category", ""),
                axis=row.get("grid_axis", ""),
                bmin=row.get("bucket_min", ""),
                bmax=row.get("bucket_max", ""),
                dmin=row.get("decode_batch_min", ""),
                dmax=row.get("decode_batch_max", ""),
                count=row.get("count", ""),
            )
        )

    lines.extend(
        [
            "",
            "## Phase437 Acceptance List",
            "",
            "- Mixed prefill token axis must cover real measured points up to the 8k and 32k working points, with decode-batch brackets beyond the observed 8k2k batch 34 and 32k3k batch 9.",
            "- Decode grid must be rectangular enough to bracket the 8k2k steady decode queries through batch 64; diagonal-only measurements are not sufficient under the current inner-only query policy.",
            "- Pure prefill `table_missing` rows are expected fallback rows, not a table-structure blocker for this phase.",
            "- Low-batch ramp/tail rows below batch 8 are recorded but not required for this acceptance gate.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_rows(_read_rows(args.audit_csv))
    write_csv(rows, args.csv)
    write_markdown(rows, args.md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

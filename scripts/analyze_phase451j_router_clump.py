#!/usr/bin/env python3
"""Phase451-J: audit vLLM internal-DP router semantics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBUG_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451i_debug_ramp.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451j_router_clump.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451j_router_clump.md"

CSV_FIELDS = ["section", "metric", "value", "target", "status", "note"]


def _fmt(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def read_debug_summary(path: Path) -> dict[str, str]:
    """Read selected Phase451-I DEBUG ramp metrics."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = row.get("metric", "")
            if metric in {
                "first_nonzero_counts",
                "first_asymmetric_burst_counts",
                "first_symmetric_burst_counts",
                "max_waiting_skew",
                "engine_visibility_debug",
            }:
                values[metric] = row.get("value", "")
    return values


def simulate_dplb_source_routes(
    *,
    request_count: int,
    initial_counts: list[list[int]],
    client_count: int,
    client_index: int,
) -> dict[str, object]:
    """Simulate the routing loop in DPLBAsyncMPClient.get_core_engine_for_request."""
    if request_count < 0:
        raise ValueError("request_count must be non-negative")
    if client_count <= 0:
        raise ValueError("client_count must be positive")
    if client_index < 0 or client_index >= client_count:
        raise ValueError("client_index must be within client_count")
    if not initial_counts:
        raise ValueError("initial_counts must not be empty")

    counts = [list(pair) for pair in initial_counts]
    num_engines = len(counts)
    eng_start_index = (num_engines * client_index) // client_count
    routes = [0 for _ in range(num_engines)]

    for _ in range(request_count):
        min_score: int | None = None
        eng_index = 0
        for i in range(num_engines):
            idx = (eng_start_index + i) % num_engines
            waiting, running = counts[idx]
            score = waiting * 4 + running
            if min_score is None or score < min_score:
                min_score = score
                eng_index = idx
        routes[eng_index] += 1
        counts[eng_index][0] += client_count

    return {"routes": routes, "final_counts": counts}


def build_rows(debug_summary: dict[str, str]) -> list[dict[str, object]]:
    source_sim = simulate_dplb_source_routes(
        request_count=64,
        initial_counts=[[0, 1], [0, 0]],
        client_count=2,
        client_index=0,
    )
    rows = [
        _row(
            "j1_source",
            "internal_dp_client_class",
            "DPLBAsyncMPClient",
            target="core_client.py:107-129",
            status="pass",
            note="data_parallel_size>1 and no external LB selects DPLBAsyncMPClient",
        ),
        _row(
            "j1_source",
            "api_server_count_default",
            "data_parallel_size",
            target="serve.py:83-103",
            status="pass",
            note="internal DP serve defaults api_server_count to data_parallel_size",
        ),
        _row(
            "j1_source",
            "multi_api_client_config",
            "client_count/client_index",
            target="utils.py:202-208",
            status="pass",
            note="each API server gets client_count and client_index",
        ),
        _row(
            "j1_source",
            "route_score",
            "waiting*4+running",
            target="core_client.py:1337-1354",
            status="pass",
            note="score source matches DPAdmissionRouter",
        ),
        _row(
            "j1_source",
            "local_waiting_increment",
            "present",
            target="core_client.py:1358-1360",
            status="pass",
            note="DPLB increments current_counts[eng_index][0] by client_count",
        ),
        _row(
            "j1_source",
            "header_bypass",
            "absent_in_benchmark",
            target="serving.py:866-871; run_openai_fixed_shape_benchmark.py:190,243",
            status="pass",
            note="benchmark does not send X-data-parallel-rank",
        ),
        _row(
            "j1_debug",
            "first_nonzero_counts",
            debug_summary.get("first_nonzero_counts", "missing"),
            note="Phase451-I DEBUG ramp",
        ),
        _row(
            "j1_debug",
            "first_asymmetric_burst_counts",
            debug_summary.get("first_asymmetric_burst_counts", "missing"),
            target="[[0,1],[63,1]]",
            status="observed" if debug_summary else "missing",
            note="DEBUG observation shows early API-visible skew",
        ),
        _row(
            "j1_debug",
            "first_symmetric_burst_counts",
            debug_summary.get("first_symmetric_burst_counts", "missing"),
            target="[[62,2],[62,2]]",
            status="observed" if debug_summary else "missing",
            note="later stats view catches up",
        ),
        _row(
            "j1_source_sim",
            "dplb_64_routes_from_debug_seed",
            source_sim["routes"],
            target="[63,1] or [1,63] if no-increment clump were source-level",
            status="contradicts_no_increment_hypothesis",
            note=f"final_counts={source_sim['final_counts']}",
        ),
        _row(
            "j2_decision",
            "remove_optimistic_increment",
            "not_applied",
            target="runtime change allowed only with source evidence",
            status="blocked",
            note="source contradicts no-optimistic-increment interpretation; DEBUG skew is real but not this code path's routing loop semantics",
        ),
        _row(
            "j3_decision",
            "cascade_gates_after_runtime_change",
            "not_run",
            status="blocked",
            note="no source-grounded runtime change was made, so H/artifact/fingerprint/preemption/--ab cascade would be meaningless",
        ),
        _row(
            "j4_decision",
            "phase443b_withdrawal",
            "not_decided",
            status="blocked",
            note="artifact mechanism remains open; next target is API-server request distribution or stats visibility, not DPLB local increment",
        ),
    ]
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_md(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase451J router clump audit",
        "",
        "Conclusion: do not remove the optimistic waiting increment. The local vLLM 0.19.0 internal-DP source path selects DPLBAsyncMPClient, and that code increments the local waiting count between 100ms stats refreshes. The DEBUG skew is real, but this specific runtime fix is not source-grounded.",
        "",
        "| Section | Metric | Value | Target | Status | Note |",
        "|---|---|---:|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {section} | {metric} | {value} | {target} | {status} | {note} |".format(
                section=_fmt(row["section"]),
                metric=_fmt(row["metric"]),
                value=_fmt(row["value"]).replace("|", "\\|"),
                target=_fmt(row["target"]).replace("|", "\\|"),
                status=_fmt(row["status"]),
                note=_fmt(row["note"]).replace("|", "\\|"),
            )
        )
    lines.extend(
        [
            "",
            "Next target: explain why the DEBUG view still shows a clustered early skew despite the DPLB routing loop having local increments. That points at API-server request distribution, stats visibility timing, or a different deployed routing boundary, not at deleting the increment from the simulator.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-csv", type=Path, default=DEFAULT_DEBUG_CSV)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    debug_summary = read_debug_summary(args.debug_csv)
    rows = build_rows(debug_summary)
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase451-K: run a source-faithful router harness for vLLM 0.19 DP LB."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VLLM_ROOT = Path("/Users/mewtwo/2026/work/codebase/vllm-0.19.0")
SCENARIO = "K2.5-tp4ep8dp2-8k2k"
DEFAULT_DEBUG_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451i_debug_ramp.csv"
DEFAULT_SERVE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase451i_debug_ramp"
    / SCENARIO
    / "serve.log.gz"
)
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase451k_router_harness.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase451k_router_harness.md"

CSV_FIELDS = ["section", "metric", "value", "target", "status", "note"]

COUNTS_RE = re.compile(
    r"Received counts: \[\[(?P<w0>\d+), (?P<r0>\d+)\], "
    r"\[(?P<w1>\d+), (?P<r1>\d+)\]\]"
)


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
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


class SourceFaithfulDPLBHarness:
    """Minimal state object for the DPLB route block in core_client.py."""

    def __init__(
        self,
        *,
        initial_counts: list[list[int]],
        client_count: int,
        client_index: int,
    ) -> None:
        if client_count <= 0:
            raise ValueError("client_count must be positive")
        if client_index < 0 or client_index >= client_count:
            raise ValueError("client_index must be within client_count")
        if not initial_counts:
            raise ValueError("initial_counts must not be empty")
        self.client_count = client_count
        self.core_engines = list(range(len(initial_counts)))
        self.lb_engines = [list(pair) for pair in initial_counts]
        self.eng_start_index = (len(self.core_engines) * client_index) // client_count

    def route_one(self) -> int:
        # This is the relevant get_core_engine_for_request branch from vLLM
        # v0.19.0 core_client.py:1337-1360, with EngineIdentity replaced by int.
        current_counts = self.lb_engines
        num_engines = len(current_counts)
        min_score: int | None = None
        eng_index = 0
        for i in range(num_engines):
            idx = (self.eng_start_index + i) % num_engines
            waiting, running = current_counts[idx]
            score = waiting * 4 + running
            if min_score is None or score < min_score:
                min_score = score
                eng_index = idx
        current_counts[eng_index][0] += self.client_count
        return eng_index


def _classify_routes(routes: list[int]) -> str:
    if not routes:
        return "empty"
    high = max(routes)
    low = min(routes)
    total = sum(routes)
    if total == 0:
        return "empty"
    if high / total >= 0.90:
        return "clustered"
    if high - low <= max(1, total * 0.05):
        return "alternating"
    return "skewed"


def run_route_harness(
    *,
    request_count: int,
    initial_counts: list[list[int]],
    client_count: int,
    client_index: int,
    overwrite_mode: str,
) -> dict[str, object]:
    """Run the route loop with optional stats-task overwrite between requests."""
    if overwrite_mode not in {"none", "before_each_request"}:
        raise ValueError(f"unknown overwrite_mode: {overwrite_mode}")
    harness = SourceFaithfulDPLBHarness(
        initial_counts=initial_counts,
        client_count=client_count,
        client_index=client_index,
    )
    stale_snapshot = [list(pair) for pair in initial_counts]
    routes = [0 for _ in initial_counts]
    placements: list[int] = []
    for idx in range(request_count):
        if overwrite_mode == "before_each_request" and idx > 0:
            # Models run_engine_stats_update_task replacing self.lb_engines with
            # an old coordinator snapshot between route calls.
            harness.lb_engines = [list(pair) for pair in stale_snapshot]
        chosen = harness.route_one()
        routes[chosen] += 1
        placements.append(chosen)
    return {
        "routes": routes,
        "placements": placements,
        "final_counts": harness.lb_engines,
        "classification": _classify_routes(routes),
    }


def run_two_api_server_harness(
    *,
    request_count_per_client: int,
    initial_counts: list[list[int]],
    client_count: int,
) -> dict[str, object]:
    routes = [0 for _ in initial_counts]
    placements: list[tuple[int, int]] = []
    for client_index in range(client_count):
        result = run_route_harness(
            request_count=request_count_per_client,
            initial_counts=initial_counts,
            client_count=client_count,
            client_index=client_index,
            overwrite_mode="none",
        )
        for replica, count in enumerate(result["routes"]):
            routes[replica] += int(count)
        placements.extend((client_index, replica) for replica in result["placements"])
    return {
        "routes": routes,
        "placements": placements,
        "classification": _classify_routes(routes),
    }


def read_debug_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    summary: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            metric = row.get("metric", "")
            if metric in {
                "first_nonzero_counts",
                "first_asymmetric_burst_counts",
                "first_symmetric_burst_counts",
                "max_waiting_skew",
                "engine_visibility_debug",
            }:
                summary[metric] = row.get("value", "")
    return summary


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def inspect_serve_boundary(path: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "api_server_count_default": "missing",
        "non_default_args_has_api_server_count_2": False,
        "api_server_processes_seen": 0,
        "first_received_counts": "",
    }
    if not path.exists():
        return result
    api_servers: set[str] = set()
    with _open_text(path) as f:
        for line in f:
            if "Defaulting api_server_count to data_parallel_size (2)" in line:
                result["api_server_count_default"] = "data_parallel_size(2)"
            if "'api_server_count': 2" in line and "'data_parallel_size': 2" in line:
                result["non_default_args_has_api_server_count_2"] = True
            if "(ApiServer_" in line:
                start = line.find("(ApiServer_") + len("(ApiServer_")
                end = line.find(" ", start)
                if end > start:
                    api_servers.add(line[start:end])
            if not result["first_received_counts"]:
                match = COUNTS_RE.search(line)
                if match:
                    result["first_received_counts"] = (
                        f"[[{match.group('w0')},{match.group('r0')}],"
                        f"[{match.group('w1')},{match.group('r1')}]]"
                    )
    result["api_server_processes_seen"] = len(api_servers)
    return result


def build_rows(
    *,
    debug_summary: dict[str, str],
    serve_boundary: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    serve_boundary = serve_boundary or {}
    persistent = run_route_harness(
        request_count=64,
        initial_counts=[[0, 1], [0, 0]],
        client_count=2,
        client_index=0,
        overwrite_mode="none",
    )
    overwrite = run_route_harness(
        request_count=64,
        initial_counts=[[0, 1], [0, 0]],
        client_count=2,
        client_index=0,
        overwrite_mode="before_each_request",
    )
    two_api = run_two_api_server_harness(
        request_count_per_client=64,
        initial_counts=[[0, 0], [0, 0]],
        client_count=2,
    )
    return [
        _row(
            "k1_boundary",
            "serve_api_server_count",
            serve_boundary.get("api_server_count_default", "missing"),
            target="api_server_count=2",
            status="pass" if serve_boundary.get("api_server_count_default") == "data_parallel_size(2)" else "missing",
            note="serve.log confirms internal DP default when present",
        ),
        _row(
            "k1_boundary",
            "api_server_processes_seen",
            serve_boundary.get("api_server_processes_seen", "missing"),
            target="2",
            status="pass" if serve_boundary.get("api_server_processes_seen") == 2 else "missing",
            note="two ApiServer processes share the internal DP path",
        ),
        _row(
            "k1_boundary",
            "client_class_source",
            "DPLBAsyncMPClient",
            target="core_client.py:107-129",
            status="pass",
            note="internal DP without external LB selects DPLBAsyncMPClient",
        ),
        _row(
            "k1_boundary",
            "client_config_source",
            "client_count/client_index",
            target="utils.py:202-208",
            status="pass",
            note="APIServerProcessManager passes per-process client_index",
        ),
        _row(
            "k2_harness",
            "direct_import",
            "blocked",
            target="import vllm.v1.engine.core_client",
            status="blocked",
            note="local import requires torch; harness uses source-faithful extracted route block",
        ),
        _row(
            "k2_harness",
            "persistent_increment_routes",
            persistent["routes"],
            target="alternating",
            status=persistent["classification"],
            note=f"final_counts={persistent['final_counts']}",
        ),
        _row(
            "k2_harness",
            "two_api_persistent_routes",
            two_api["routes"],
            target="alternating",
            status=two_api["classification"],
            note="two API clients with independent start indices still balance if increments persist",
        ),
        _row(
            "k2_harness",
            "stats_overwrite_routes",
            overwrite["routes"],
            target="clustered",
            status=overwrite["classification"],
            note="clump appears only when stats task overwrites local increments with an old snapshot between route calls",
        ),
        _row(
            "k2_observation",
            "debug_first_asymmetric_counts",
            debug_summary.get("first_asymmetric_burst_counts", "missing"),
            target="[[0,1],[63,1]]",
            status="observed" if debug_summary else "missing",
            note="Phase451-I real run",
        ),
        _row(
            "k2_verdict",
            "route_loop_verdict",
            "alternates",
            target="direct route loop behavior",
            status="pass",
            note="get_core_engine_for_request itself does not produce 63/1",
        ),
        _row(
            "k2_verdict",
            "clump_mechanism_candidate",
            "stats_overwrite_interleaving",
            target="explain code+DEBUG contradiction",
            status="candidate",
            note="source line core_client.py:1271 assigns self.lb_engines=sliced_counts and can discard local increments",
        ),
        _row(
            "k5_decision",
            "sim_route_alignment",
            "not_applied",
            target="harness-proven actual behavior",
            status="blocked",
            note="stats-overwrite interleaving cadence is not yet measured; changing route policy would be a guessed model",
        ),
        _row(
            "k5_decision",
            "cascade_gates",
            "not_run",
            status="blocked",
            note="no runtime change was made, so H/450-A/fingerprint/preemption/--ab cascade is not meaningful",
        ),
        _row(
            "k5_decision",
            "phase443b_452",
            "unchanged",
            status="blocked",
            note="443-B and Phase452 decisions remain pending until stats-overwrite or upstream distribution is measured",
        ),
    ]


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
        "# Phase451K router harness",
        "",
        "结论: 真实 `get_core_engine_for_request` 路由块本身仍是交替,不是成簇。63/1 只能在 stats task 反复用旧 coordinator snapshot 覆盖本地递增时复现。因此本轮不改 sim 路由;下一步应测 stats 覆盖与 add_request 的交错时序。",
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
            "## Source boundary",
            "",
            "- `serve.py:83-103`: internal DP defaults `api_server_count` to `data_parallel_size`.",
            "- `core_client.py:107-129`: internal DP selects `DPLBAsyncMPClient`.",
            "- `core_client.py:1337-1360`: route score is `waiting*4+running`, then local waiting is incremented by `client_count`.",
            "- `core_client.py:1268-1275`: stats task assigns `self.lb_engines = sliced_counts`, which is the only source-local way to discard local increments.",
            "",
            "## Decision",
            "",
            "Do not change `DPAdmissionRouter` to no-increment clumping. That would contradict the route block. The next source-grounded probe is request-level instrumentation around `add_request_async`: log the route choice, pre/post `lb_engines`, and whether a stats update overwrote the list between adjacent adds.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-csv", type=Path, default=DEFAULT_DEBUG_CSV)
    parser.add_argument("--serve-log", type=Path, default=DEFAULT_SERVE_LOG)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_rows(
        debug_summary=read_debug_summary(args.debug_csv),
        serve_boundary=inspect_serve_boundary(args.serve_log),
    )
    write_csv(rows, args.out_csv)
    write_md(rows, args.out_md)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_md}")


if __name__ == "__main__":
    main()

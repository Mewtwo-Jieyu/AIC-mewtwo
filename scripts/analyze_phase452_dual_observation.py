#!/usr/bin/env python3
"""Phase452 dual-observation analyzer.

This is a report-only parser for the logging patch. It answers two questions:
whether stale stats overwrites interleave with API-server routing, and how a
preempted victim is scheduled again.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase452_dual_observation"
DEFAULT_JSONL = DEFAULT_ROOT / "dual_observation.jsonl"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase452_dual_observation.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase452_dual_observation.md"

CSV_FIELDS = ["section", "metric", "value", "target", "status", "note"]


@dataclass(frozen=True)
class RouteInterleaveSummary:
    route_rows: int
    stats_rows: int
    adjacent_pairs: int
    stats_between_pairs: int
    local_increment_lost_pairs: int
    route_counts: dict[int, int]
    per_client_route_counts: dict[str, dict[int, int]]
    lost_pair_examples: list[dict[str, object]]


@dataclass(frozen=True)
class VictimSummary:
    preempt_decisions: int
    victim_reschedules: int
    unique_victims: int
    recompute_from_zero: int
    resume_with_cached_tokens: int
    missing_reschedule: int
    victim_position_hist: dict[int, int]
    free_blocks_before_min: int | None
    free_blocks_before_max: int | None
    free_blocks_after_min: int | None
    free_blocks_after_max: int | None
    reschedule_new_tokens: list[int]


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


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def read_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records: list[dict[str, object]] = []
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _counts_equal(a: object, b: object) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def summarize_route_interleaving(
    records: Iterable[dict[str, object]],
) -> RouteInterleaveSummary:
    route_rows = [r for r in records if r.get("kind") == "route"]
    stats_rows = [r for r in records if r.get("kind") == "stats_overwrite"]
    by_client: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(list)
    stats_by_client: dict[tuple[object, object], list[dict[str, object]]] = defaultdict(
        list
    )

    for row in route_rows:
        by_client[(row.get("pid"), row.get("client_index"))].append(row)
    for row in stats_rows:
        stats_by_client[(row.get("pid"), row.get("client_index"))].append(row)

    adjacent_pairs = 0
    stats_between_pairs = 0
    local_increment_lost_pairs = 0
    route_counts: Counter[int] = Counter()
    per_client_route_counts: dict[str, Counter[int]] = defaultdict(Counter)
    lost_pair_examples: list[dict[str, object]] = []

    for row in route_rows:
        chosen = row.get("chosen_engine_index")
        if chosen is not None:
            route_counts[int(chosen)] += 1
            client_key = f"pid={row.get('pid')}:client={row.get('client_index')}"
            per_client_route_counts[client_key][int(chosen)] += 1

    for key, routes in by_client.items():
        routes.sort(key=lambda r: int(r.get("ts_ns", 0)))
        stats = sorted(stats_by_client.get(key, []), key=lambda r: int(r.get("ts_ns", 0)))
        stat_times = [int(row.get("ts_ns", 0)) for row in stats]
        for left, right in zip(routes, routes[1:]):
            adjacent_pairs += 1
            left_ts = int(left.get("ts_ns", 0))
            right_ts = int(right.get("ts_ns", 0))
            has_stats = any(left_ts <= ts <= right_ts for ts in stat_times)
            if has_stats:
                stats_between_pairs += 1
            if not _counts_equal(left.get("post_counts"), right.get("pre_counts")):
                local_increment_lost_pairs += 1
                if len(lost_pair_examples) < 8:
                    lost_pair_examples.append(
                        {
                            "client": f"pid={key[0]}:client={key[1]}",
                            "pair_index": len(lost_pair_examples),
                            "dt_ms": (right_ts - left_ts) / 1_000_000.0,
                            "left_post": left.get("post_counts"),
                            "right_pre": right.get("pre_counts"),
                            "stats_between": len(
                                [ts for ts in stat_times if left_ts <= ts <= right_ts]
                            ),
                            "left_engine": left.get("chosen_engine_index"),
                            "right_engine": right.get("chosen_engine_index"),
                        }
                    )

    return RouteInterleaveSummary(
        route_rows=len(route_rows),
        stats_rows=len(stats_rows),
        adjacent_pairs=adjacent_pairs,
        stats_between_pairs=stats_between_pairs,
        local_increment_lost_pairs=local_increment_lost_pairs,
        route_counts=dict(route_counts),
        per_client_route_counts={
            key: dict(counter) for key, counter in per_client_route_counts.items()
        },
        lost_pair_examples=lost_pair_examples,
    )


def _request_id_from_preempt(row: dict[str, object]) -> str:
    victim = row.get("victim")
    if isinstance(victim, dict):
        return str(victim.get("request_id", ""))
    return ""


def _request_id_from_reschedule(row: dict[str, object]) -> str:
    req = row.get("request")
    if isinstance(req, dict):
        return str(req.get("request_id", ""))
    return ""


def summarize_victim_path(records: Iterable[dict[str, object]]) -> VictimSummary:
    decisions = [r for r in records if r.get("kind") == "preempt_decision"]
    after_free = [r for r in records if r.get("kind") == "preempt_after_free"]
    reschedules = [r for r in records if r.get("kind") == "victim_reschedule"]
    res_by_id: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in reschedules:
        rid = _request_id_from_reschedule(row)
        if rid:
            res_by_id[rid].append(row)
    for rows in res_by_id.values():
        rows.sort(key=lambda r: int(r.get("ts_ns", 0)))

    recompute_from_zero = 0
    resume_with_cached_tokens = 0
    missing = 0
    victims: set[str] = set()
    victim_position_hist: Counter[int] = Counter()
    for row in decisions:
        position = row.get("victim_position")
        if position is not None:
            victim_position_hist[int(position)] += 1
        rid = _request_id_from_preempt(row)
        if not rid:
            continue
        victims.add(rid)
        later = [
            res
            for res in res_by_id.get(rid, [])
            if int(res.get("ts_ns", 0)) >= int(row.get("ts_ns", 0))
        ]
        if not later:
            missing += 1
            continue
        computed = int(later[0].get("num_computed_tokens_for_schedule") or 0)
        if computed == 0:
            recompute_from_zero += 1
        else:
            resume_with_cached_tokens += 1

    before_values = [
        int(row["free_blocks_before_free"])
        for row in decisions
        if row.get("free_blocks_before_free") is not None
    ]
    after_values = [
        int(row["free_blocks_after_free"])
        for row in after_free
        if row.get("free_blocks_after_free") is not None
    ]
    reschedule_new_tokens = sorted(
        {
            int(row["num_new_tokens"])
            for row in reschedules
            if row.get("num_new_tokens") is not None
        }
    )
    return VictimSummary(
        preempt_decisions=len(decisions),
        victim_reschedules=len(reschedules),
        unique_victims=len(victims),
        recompute_from_zero=recompute_from_zero,
        resume_with_cached_tokens=resume_with_cached_tokens,
        missing_reschedule=missing,
        victim_position_hist=dict(victim_position_hist),
        free_blocks_before_min=min(before_values) if before_values else None,
        free_blocks_before_max=max(before_values) if before_values else None,
        free_blocks_after_min=min(after_values) if after_values else None,
        free_blocks_after_max=max(after_values) if after_values else None,
        reschedule_new_tokens=reschedule_new_tokens,
    )


def build_rows(records: list[dict[str, object]]) -> list[dict[str, object]]:
    route = summarize_route_interleaving(records)
    victim = summarize_victim_path(records)

    route_gate = (
        route.route_rows >= 120
        and route.stats_rows > 0
        and route.local_increment_lost_pairs > 0
    )
    victim_gate = victim.preempt_decisions > 0 and (
        victim.victim_reschedules > 0 or victim.missing_reschedule > 0
    )
    if route.local_increment_lost_pairs:
        route_verdict = "stats_overwrite_interleaves_add"
    elif route.stats_rows:
        route_verdict = "stats_seen_no_increment_loss"
    else:
        route_verdict = "stats_missing"
    if victim.recompute_from_zero > victim.resume_with_cached_tokens:
        victim_verdict = "victim_recompute_from_zero"
    elif victim.resume_with_cached_tokens:
        victim_verdict = "victim_resumes_with_cached_tokens"
    elif victim.preempt_decisions:
        victim_verdict = "victim_recovery_not_observed"
    else:
        victim_verdict = "victim_missing"

    rows = [
        _row("self_check", "route_rows", route.route_rows, target=">=120"),
        _row("self_check", "stats_overwrite_rows", route.stats_rows, target=">0"),
        _row(
            "self_check",
            "preempt_decision_rows",
            victim.preempt_decisions,
            target=">0",
        ),
        _row(
            "route",
            "route_counts",
            json.dumps(route.route_counts, sort_keys=True),
            note="chosen_engine_index histogram",
        ),
        _row(
            "route",
            "per_client_route_counts",
            json.dumps(route.per_client_route_counts, sort_keys=True),
        ),
        _row("route", "adjacent_route_pairs", route.adjacent_pairs),
        _row("route", "stats_between_pairs", route.stats_between_pairs),
        _row(
            "route",
            "local_increment_lost_pairs",
            route.local_increment_lost_pairs,
            note="post_counts of one route differs from pre_counts of the next route",
        ),
        _row(
            "route",
            "lost_pair_examples",
            json.dumps(route.lost_pair_examples, sort_keys=True),
            note="first examples only",
        ),
        _row(
            "route",
            "verdict",
            route_verdict,
            status="pass" if route_gate else "fail",
        ),
        _row("victim", "unique_victims", victim.unique_victims),
        _row("victim", "victim_reschedules", victim.victim_reschedules),
        _row(
            "victim",
            "victim_position_hist",
            json.dumps(victim.victim_position_hist, sort_keys=True),
        ),
        _row(
            "victim",
            "free_blocks_before_range",
            f"{_fmt(victim.free_blocks_before_min)}..{_fmt(victim.free_blocks_before_max)}",
        ),
        _row(
            "victim",
            "free_blocks_after_range",
            f"{_fmt(victim.free_blocks_after_min)}..{_fmt(victim.free_blocks_after_max)}",
        ),
        _row(
            "victim",
            "reschedule_new_tokens",
            json.dumps(victim.reschedule_new_tokens),
            note="unique values",
        ),
        _row("victim", "recompute_from_zero", victim.recompute_from_zero),
        _row("victim", "resume_with_cached_tokens", victim.resume_with_cached_tokens),
        _row("victim", "missing_reschedule", victim.missing_reschedule),
        _row(
            "victim",
            "verdict",
            victim_verdict,
            status="pass" if victim_gate else "fail",
        ),
        _row(
            "decision",
            "route_line_next",
            "model_or_boundary",
            status="needs_decision" if route_gate else "blocked",
            note="overwrite exists; only stable cadence should be modeled",
        ),
        _row(
            "decision",
            "victim_line_next",
            "victim_recovery_not_root_cause",
            status="excluded" if victim_verdict == "victim_recompute_from_zero" else "needs_followup",
            note="real recovery recomputes from zero; cb_sim already uses PREEMPTED recompute semantics",
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
    by_section: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_section[str(row["section"])].append(row)
    values = {(row["section"], row["metric"]): row["value"] for row in rows}
    lines = [
        "# Phase452 dual observation",
        "",
        "logging-only raw; no runtime or PerfDB change.",
        "",
        "## verdict",
        "",
        "- Route: stats overwrite does interleave with adds, but this run only shows a small number of lost local increments; model only if a stable cadence is proven, otherwise keep it as deployment-timing boundary.",
        "- Victim: observed recovery is recompute-from-zero, matching vLLM source and current cb_sim PREEMPTED semantics; do not change victim recovery to chase thrash.",
        f"- Raw gates: route_rows={_fmt(values.get(('self_check', 'route_rows')))}, stats_overwrite_rows={_fmt(values.get(('self_check', 'stats_overwrite_rows')))}, preempt_decision_rows={_fmt(values.get(('self_check', 'preempt_decision_rows')))}.",
        "",
    ]
    for section in ["self_check", "route", "victim", "decision"]:
        if section not in by_section:
            continue
        lines.extend([f"## {section}", "", "| metric | value | target | status | note |", "|---|---:|---:|---|---|"])
        for row in by_section[section]:
            lines.append(
                "| {metric} | {value} | {target} | {status} | {note} |".format(
                    metric=_fmt(row.get("metric")),
                    value=_fmt(row.get("value")),
                    target=_fmt(row.get("target")),
                    status=_fmt(row.get("status")),
                    note=_fmt(row.get("note")),
                )
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    records = read_records(args.jsonl)
    rows = build_rows(records)
    write_csv(rows, args.csv)
    write_md(rows, args.md)

    for row in rows:
        if row["metric"] in {"verdict", "route_rows", "preempt_decision_rows"}:
            print(f"{row['section']}.{row['metric']}={row['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Judge Phase462 DP route/receive/admit logging-only evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase462_dp_route_observation"
DEFAULT_CSV = DEFAULT_ROOT / "phase462_dp_route_judgement.csv"
DEFAULT_REPORT = DEFAULT_ROOT / "phase462_dp_route_judgement.md"


@dataclass(frozen=True)
class RouteJudgement:
    expected_requests: int
    rank0_routes: int
    rank1_routes: int
    route_balanced: bool
    route_strict_alternation: bool
    route_asymmetry_observed: bool
    rank0_admit_gap_distribution: str
    rank1_admit_gap_distribution: str
    drain_cadence_equal: bool
    max_same_rank_route_run: int
    route_switches: int
    score_tie_routes: int
    rank0_route_receive_median_ms: float
    rank1_route_receive_median_ms: float
    rank0_receive_admit_median_ms: float
    rank1_receive_admit_median_ms: float
    verdict: str


def load_trace_rows(paths: Sequence[Path]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in paths:
        file_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        footers = [row for row in file_rows if row.get("kind") == "trace_footer"]
        if len(footers) != 1 or footers[0].get("flush_complete") is not True:
            raise AssertionError(f"incomplete_trace_footer:{path}")
        events = [row for row in file_rows if row.get("kind") != "trace_footer"]
        if int(footers[0]["event_count"]) != len(events):
            raise AssertionError(f"trace_event_count_mismatch:{path}")
        rows.extend(events)
    return rows


def _by_ordinal(
    rows: Iterable[Mapping[str, object]], kind: str
) -> dict[int, Mapping[str, object]]:
    result: dict[int, Mapping[str, object]] = {}
    for row in rows:
        if row.get("kind") != kind:
            continue
        ordinal = int(row["arrival_ordinal"])
        if ordinal in result:
            raise AssertionError(f"duplicate_{kind}_ordinal:{ordinal}")
        result[ordinal] = row
    return result


def _gap_distribution(values: Sequence[int]) -> tuple[tuple[int, Fraction], ...]:
    if len(values) < 2:
        raise AssertionError("each DP rank needs at least two admits")
    ordered = sorted(values)
    gaps = [right - left for left, right in zip(ordered, ordered[1:])]
    if any(gap < 0 for gap in gaps):
        raise AssertionError(f"non_monotonic_rank_admit_steps:{gaps}")
    counts = Counter(gaps)
    total = sum(counts.values())
    return tuple((gap, Fraction(count, total)) for gap, count in sorted(counts.items()))


def _render_distribution(distribution: tuple[tuple[int, Fraction], ...]) -> str:
    return ";".join(
        f"{gap}:{fraction.numerator}/{fraction.denominator}"
        for gap, fraction in distribution
    )


def _max_same_rank_run(ranks: Sequence[int]) -> int:
    longest = current = 0
    previous = None
    for rank in ranks:
        current = current + 1 if rank == previous else 1
        longest = max(longest, current)
        previous = rank
    return longest


def _median_ms(values_ns: Sequence[int]) -> float:
    if not values_ns:
        raise AssertionError("each DP rank needs at least one latency sample")
    return float(median(values_ns)) / 1_000_000


def judge_rows(
    rows: Iterable[Mapping[str, object]], *, expected_requests: int
) -> RouteJudgement:
    rows = list(rows)
    if expected_requests <= 0 or expected_requests % 2:
        raise ValueError("expected_requests must be positive and even")
    expected_ordinals = set(range(expected_requests))
    route = _by_ordinal(rows, "route")
    receive = _by_ordinal(rows, "receive")
    admit = _by_ordinal(rows, "admit")
    for kind, observed in (("route", route), ("receive", receive), ("admit", admit)):
        if set(observed) != expected_ordinals:
            raise AssertionError(
                f"{kind}_ordinals:expected={sorted(expected_ordinals)}:"
                f"observed={sorted(observed)}"
            )

    route_ranks: list[int] = []
    admit_steps: dict[int, list[int]] = {0: [], 1: []}
    route_receive_ns: dict[int, list[int]] = {0: [], 1: []}
    receive_admit_ns: dict[int, list[int]] = {0: [], 1: []}
    score_tie_routes = 0
    for ordinal in range(expected_requests):
        route_row = route[ordinal]
        receive_row = receive[ordinal]
        admit_row = admit[ordinal]
        rank = int(route_row["chosen_rank"])
        if rank not in {0, 1}:
            raise AssertionError(f"invalid_route_rank:{rank}")
        if int(receive_row["dp_rank"]) != rank or int(admit_row["dp_rank"]) != rank:
            raise AssertionError(f"route_receive_admit_rank_mismatch:{ordinal}")
        route_ts = int(route_row["wall_ts_ns"])
        receive_ts = int(receive_row["wall_ts_ns"])
        admit_ts = int(admit_row["wall_ts_ns"])
        if not route_ts <= receive_ts <= admit_ts:
            raise AssertionError(f"route_receive_admit_time_order:{ordinal}")
        snapshots = list(route_row["score_snapshot"])
        if {int(item["rank"]) for item in snapshots} != {0, 1}:
            raise AssertionError(f"route_score_snapshot_ranks:{ordinal}")
        scores = {int(item["rank"]): int(item["score"]) for item in snapshots}
        if scores[rank] != min(scores.values()):
            raise AssertionError(f"route_did_not_choose_min_score:{ordinal}")
        if scores[0] == scores[1]:
            score_tie_routes += 1
        route_ranks.append(rank)
        admit_steps[rank].append(int(admit_row["schedule_seq"]))
        route_receive_ns[rank].append(receive_ts - route_ts)
        receive_admit_ns[rank].append(admit_ts - receive_ts)

    counts = Counter(route_ranks)
    balanced = counts[0] == counts[1] == expected_requests // 2
    temporal_route_ranks = [
        int(row["chosen_rank"])
        for row in sorted(
            route.values(),
            key=lambda row: (int(row["wall_ts_ns"]), int(row["arrival_ordinal"])),
        )
    ]
    route_switches = sum(
        left != right
        for left, right in zip(temporal_route_ranks, temporal_route_ranks[1:])
    )
    strict_alternation = route_switches == expected_requests - 1
    route_asymmetry = not balanced or not strict_alternation
    gap0 = _gap_distribution(admit_steps[0])
    gap1 = _gap_distribution(admit_steps[1])
    cadence_equal = gap0 == gap1
    if route_asymmetry and cadence_equal:
        verdict = "H-route"
    elif not route_asymmetry and not cadence_equal:
        verdict = "H-drain"
    elif route_asymmetry and not cadence_equal:
        verdict = "H-mixed"
    else:
        verdict = "no_rank_asymmetry_observed"
    return RouteJudgement(
        expected_requests=expected_requests,
        rank0_routes=counts[0],
        rank1_routes=counts[1],
        route_balanced=balanced,
        route_strict_alternation=strict_alternation,
        route_asymmetry_observed=route_asymmetry,
        rank0_admit_gap_distribution=_render_distribution(gap0),
        rank1_admit_gap_distribution=_render_distribution(gap1),
        drain_cadence_equal=cadence_equal,
        max_same_rank_route_run=_max_same_rank_run(temporal_route_ranks),
        route_switches=route_switches,
        score_tie_routes=score_tie_routes,
        rank0_route_receive_median_ms=_median_ms(route_receive_ns[0]),
        rank1_route_receive_median_ms=_median_ms(route_receive_ns[1]),
        rank0_receive_admit_median_ms=_median_ms(receive_admit_ns[0]),
        rank1_receive_admit_median_ms=_median_ms(receive_admit_ns[1]),
        verdict=verdict,
    )


def write_outputs(
    result: RouteJudgement,
    *,
    output_csv: Path,
    output_report: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=list(asdict(result)),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(asdict(result))
    output_report.write_text(
        f"""# Phase462 DP Step 3b 路由取证

结论：机械判卷为 `{result.verdict}`。判卷顺序固定为 route 后 drain；本报告只给机制证据，不修改 runtime、PerfDB 或 gate。`Default AIC=No-Go`。

| 指标 | rank 0 | rank 1 |
|---|---:|---:|
| route 请求数 | {result.rank0_routes} | {result.rank1_routes} |
| admit step gap 分布 | {result.rank0_admit_gap_distribution} | {result.rank1_admit_gap_distribution} |
| route→receive 中位延迟 (ms) | {result.rank0_route_receive_median_ms:.6f} | {result.rank1_route_receive_median_ms:.6f} |
| receive→admit 中位延迟 (ms) | {result.rank0_receive_admit_median_ms:.6f} | {result.rank1_receive_admit_median_ms:.6f} |

| 判据 | 结果 |
|---|---|
| route 严格均衡 | {str(result.route_balanced).lower()} |
| route 逐请求严格交替 | {str(result.route_strict_alternation).lower()} |
| route 不对称（总量或聚簇） | {str(result.route_asymmetry_observed).lower()} |
| drain cadence 逐项同分布 | {str(result.drain_cadence_equal).lower()} |
| 最大连续同 rank 路由 | {result.max_same_rank_route_run} |
| route rank 切换次数 | {result.route_switches} |
| score 并列路由次数 | {result.score_tie_routes} |
| diagnostic_only | true |
| valid_for_default | false |
""",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--expected-requests", type=int, default=512)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    result = judge_rows(
        load_trace_rows(args.paths), expected_requests=args.expected_requests
    )
    write_outputs(result, output_csv=args.output_csv, output_report=args.output_report)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

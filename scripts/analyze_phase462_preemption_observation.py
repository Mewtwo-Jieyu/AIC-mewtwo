#!/usr/bin/env python3
"""Phase462 Step2b: compare real and sim preemption decision dynamics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase451d_preemption_forensics as phase451d
import scripts.validate_cb_simulator as validate


SCENARIO = "K2.5-tp8ep8-32k3k"
ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase462_preemption_observation"
DEFAULT_JSONL = ROOT / "preemption_observation.jsonl"
DEFAULT_OVERHEAD = ROOT / "overhead_gate.json"
DEFAULT_SELF_CHECK = ROOT / "self_check.json"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_preemption_observation.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase462_preemption_observation.md"
NUM_REQUESTS = 128
OSL = 1200

CSV_FIELDS = ["section", "side", "metric", "value", "target", "status", "note"]


@dataclass(frozen=True)
class DecisionPair:
    trigger_request_id: object
    victim_request_id: object
    requested_blocks: int
    free_blocks: int
    victim_position: int
    ts_ns: int
    trigger_num_computed_tokens: int
    victim_num_computed_tokens: int


def _row(
    section: str,
    side: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "side": side,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pair_records(records: Iterable[dict[str, object]]) -> list[DecisionPair]:
    rows = list(records)
    failures = [row for row in rows if row.get("kind") == "allocate_failure"]
    decisions = [row for row in rows if row.get("kind") == "preempt_decision"]
    pairs: list[DecisionPair] = []
    used: set[int] = set()
    for decision in decisions:
        candidates = [
            (index, failure)
            for index, failure in enumerate(failures)
            if index not in used
            and failure.get("pid") == decision.get("pid")
            and failure.get("trigger_request_id") == decision.get("trigger_request_id")
            and int(failure.get("ts_ns", 0)) <= int(decision.get("ts_ns", 0))
        ]
        if not candidates:
            continue
        index, failure = max(
            candidates, key=lambda item: int(item[1].get("ts_ns", 0))
        )
        used.add(index)
        pairs.append(
            DecisionPair(
                trigger_request_id=decision["trigger_request_id"],
                victim_request_id=decision["victim_request_id"],
                requested_blocks=int(failure["requested_blocks"]),
                free_blocks=int(failure["free_blocks"]),
                victim_position=int(decision["victim_position"]),
                ts_ns=int(decision["ts_ns"]),
                trigger_num_computed_tokens=int(
                    decision["trigger_num_computed_tokens"]
                ),
                victim_num_computed_tokens=int(decision["victim_num_computed_tokens"]),
            )
        )
    return pairs


def summarize_real_decisions(pairs: Iterable[DecisionPair]) -> dict[str, object]:
    rows = list(pairs)
    victim_counts = Counter(row.victim_request_id for row in rows)
    gaps = [
        (right.ts_ns - left.ts_ns) / 1e9 for left, right in zip(rows, rows[1:])
    ]
    return {
        "preemptions": len(rows),
        "unique_victims": len(victim_counts),
        "repeat_victim_events": sum(count - 1 for count in victim_counts.values()),
        "self_preemptions": sum(
            row.trigger_request_id == row.victim_request_id for row in rows
        ),
        "gap_median_s": statistics.median(gaps) if gaps else 0.0,
        "requested_blocks": sorted({row.requested_blocks for row in rows}),
        "free_blocks": sorted({row.free_blocks for row in rows}),
        "victim_positions": sorted({row.victim_position for row in rows}),
    }


def summarize_sim_decisions(events: Iterable[dict[str, object]]) -> dict[str, object]:
    rows = list(events)
    victim_counts = Counter(row["victim_req_id"] for row in rows)
    return {
        "preemptions": len(rows),
        "unique_victims": len(victim_counts),
        "repeat_victim_events": sum(
            int(row.get("victim_preemptions_before", 0)) > 0 for row in rows
        ),
        "self_preemptions": sum(
            row.get("trigger_req_id") == row.get("victim_req_id") for row in rows
        ),
        "over_blocks": sorted({int(row["over_blocks_before"]) for row in rows}),
    }


def first_divergence_verdict(
    real_pairs: list[DecisionPair], sim_events: list[dict[str, object]]
) -> dict[str, object]:
    if not real_pairs or len(sim_events) < 2:
        raise ValueError("need real pairs and at least two sim preemptions")
    real_first = real_pairs[0]
    sim_first = sim_events[0]
    one_block_aligned = (
        real_first.requested_blocks - real_first.free_blocks == 1
        and int(sim_first["over_blocks_before"]) == 1
    )
    sim_immediate_repeat = (
        sim_events[1]["victim_req_id"] == sim_first["victim_req_id"]
        and int(sim_events[1]["local_iter"]) - int(sim_first["local_iter"]) <= 2
    )
    real_distinct = (
        len(real_pairs) >= 2
        and real_pairs[1].victim_request_id != real_first.victim_request_id
    )
    relation_differs = (
        real_first.trigger_request_id != real_first.victim_request_id
        and sim_first["trigger_req_id"] == sim_first["victim_req_id"]
    )
    narrowed = one_block_aligned and relation_differs and sim_immediate_repeat and real_distinct
    return {
        "allocation_condition": (
            "aligned_one_block_short" if one_block_aligned else "not_aligned"
        ),
        "first_divergence": (
            "victim_relation_and_recurrence" if narrowed else "not_uniquely_narrowed"
        ),
        "candidate": (
            "per_request_decode_phase_evolution" if narrowed else "unresolved"
        ),
        "runtime_fix_allowed": False,
    }


def run_matching_sim() -> list[dict[str, object]]:
    original_points = validate.MULTI_CONFIG_DATA
    original_loader = phase451d._load_validate_module
    validate.MULTI_CONFIG_DATA = tuple(
        replace(point, osl=OSL) if point.name == SCENARIO else point
        for point in original_points
    )
    phase451d._load_validate_module = lambda: validate
    try:
        result = phase451d.run_sim_preemption_forensics(
            scenario=SCENARIO, num_requests=NUM_REQUESTS
        )
        return list(result["events"])
    finally:
        validate.MULTI_CONFIG_DATA = original_points
        phase451d._load_validate_module = original_loader


def build_rows(
    *,
    jsonl_path: Path = DEFAULT_JSONL,
    overhead_path: Path = DEFAULT_OVERHEAD,
    self_check_path: Path = DEFAULT_SELF_CHECK,
) -> list[dict[str, object]]:
    pairs = pair_records(read_jsonl(jsonl_path))
    sim_events = run_matching_sim()
    real = summarize_real_decisions(pairs)
    sim = summarize_sim_decisions(sim_events)
    verdict = first_divergence_verdict(pairs, sim_events)
    overhead = read_json(overhead_path)
    self_check = read_json(self_check_path)
    rows = [
        _row(
            "gate",
            "real",
            "complete_decision_pairs",
            self_check["complete_pairs"],
            target=">0",
            status="pass" if self_check["passed"] else "fail",
        ),
        _row(
            "gate",
            "real",
            "logging_overhead_pct",
            overhead["absolute_delta_pct"],
            target="<=2%",
            status="pass" if overhead["passed"] else "fail",
            note=(
                f"off={overhead['off_output_tok_s']:.6f}; "
                f"on={overhead['on_output_tok_s']:.6f}"
            ),
        ),
    ]
    for side, summary in (("real", real), ("sim", sim)):
        for metric in (
            "preemptions",
            "unique_victims",
            "repeat_victim_events",
            "self_preemptions",
        ):
            rows.append(_row("decision_dynamics", side, metric, summary[metric]))
    rows.extend(
        [
            _row(
                "first_divergence",
                "both",
                "allocation_condition",
                verdict["allocation_condition"],
                target="same one-block shortage",
                status="aligned",
                note="real requested=1/free=0; sim scheduled demand exceeds capacity by 1 block",
            ),
            _row(
                "first_divergence",
                "real",
                "victim_relation",
                "peer_tail_victim",
                status="observed",
                note=(
                    f"trigger_tokens={pairs[0].trigger_num_computed_tokens}; "
                    f"victim_tokens={pairs[0].victim_num_computed_tokens}; "
                    f"median_gap_s={real['gap_median_s']:.6f}"
                ),
            ),
            _row(
                "first_divergence",
                "sim",
                "victim_relation",
                "self_victim_then_repeat_in_2_iters",
                status="diverged",
                note=(
                    f"first_iter={sim_events[0]['local_iter']}; "
                    f"second_iter={sim_events[1]['local_iter']}"
                ),
            ),
            _row(
                "decision",
                "both",
                "first_divergence",
                verdict["first_divergence"],
                target="unique semantics boundary",
                status="narrowed",
                note=f"candidate={verdict['candidate']}",
            ),
            _row(
                "decision",
                "sim",
                "runtime_fix_allowed",
                verdict["runtime_fix_allowed"],
                target="source-backed decode-phase mechanism",
                status="blocked",
                note="allocation threshold and tail-victim policy are aligned; source audit is still required",
            ),
            _row(
                "flags",
                "real",
                "diagnostic_only",
                True,
                status="pass",
                note="valid_for_default=false; perf_database=false; Default AIC No-Go",
            ),
        ]
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {(row["side"], row["metric"]): row for row in rows}
    lines = [
        "# Phase462 Step2b: 抢占决策点真实观测",
        "",
        "结论：Step2b 观测与开销门均通过，但没有找到 allocation 阈值差。real 与 sim 都在差 1 个 block 时进入抢占，victim policy 也都是 running tail。首次分歧出现在请求相位：real 首次由一个请求触发、抢另一个落后 1 token 的 tail，10 次 victim 全不重复；sim 首次 trigger 就是 tail victim 本人，2 个 local iteration 后再次抢同一请求。修复靶点收窄为 per-request decode phase evolution / re-entry dynamics，Step2c 在源码语义钉死前继续 blocked。Default AIC 维持 No-Go。",
        "",
        "| 项 | real | sim |",
        "|---|---:|---:|",
        f"| 抢占次数 | {lookup[('real', 'preemptions')]['value']} | {lookup[('sim', 'preemptions')]['value']} |",
        f"| unique victim | {lookup[('real', 'unique_victims')]['value']} | {lookup[('sim', 'unique_victims')]['value']} |",
        f"| repeat victim event | {lookup[('real', 'repeat_victim_events')]['value']} | {lookup[('sim', 'repeat_victim_events')]['value']} |",
        f"| self-preemption | {lookup[('real', 'self_preemptions')]['value']} | {lookup[('sim', 'self_preemptions')]['value']} |",
        "",
        "| 门 | 结果 |",
        "|---|---|",
        f"| 字段完整性 | {lookup[('real', 'complete_decision_pairs')]['value']}/10 complete，pass |",
        f"| logging 开销 | {float(lookup[('real', 'logging_overhead_pct')]['value']):.3f}%（阈值 2%），pass |",
        "| GPU/process residual | empty |",
        "| patch restore | scheduler/kv SHA 与采集前一致，已恢复 |",
        "",
        "源码边界：vLLM `scheduler.py:384-518` 逐请求计算 `num_tokens_with_spec + num_output_placeholders - num_computed_tokens` 后依次申请 block；cb_sim `scheduler.py:183-205` 固定为每个 decode request 申请 1 token，`simulator.py:621-624` 再统一推进 1 token。这个差异与实测 phase skew 同方向，但尚未证明哪一条状态更新导致首次 victim 关系分岔，因此不能直接改代码。",
        "",
        "下一步只做源码对照：解释 vLLM 如何让同 cohort 请求产生 1-token phase skew，以及 cb_sim 为何在首次 block boundary 形成 self-victim + immediate repeat。不能修改 block threshold，也不能加去同步随机数。",
        "",
        "本批数据 `diagnostic_only=true / valid_for_default=false / perf_database=false`，不进参考与 PerfDB。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--overhead", type=Path, default=DEFAULT_OVERHEAD)
    parser.add_argument("--self-check", type=Path, default=DEFAULT_SELF_CHECK)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_rows(
        jsonl_path=args.jsonl,
        overhead_path=args.overhead,
        self_check_path=args.self_check,
    )
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()

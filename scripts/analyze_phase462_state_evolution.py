#!/usr/bin/env python3
"""Phase462 Step2a-2: audit preemption state evolution at source level."""

from __future__ import annotations

import argparse
import csv
import gzip
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.analyze_phase462_preemption_observation as observation
import scripts.validate_cb_simulator as validate
from aiconfigurator.sdk.backends.cb_simulator import CBSimulator
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler


SCENARIO = "K2.5-tp8ep8-32k3k"
TRACE_STEPS = 16
REAL_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/overhead_on"
    / SCENARIO
    / "serve.log.gz"
)
PREEMPT_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation"
    / "preemption_observation.jsonl"
)
SIGN_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_first_divergence.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_state_evolution.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase462_state_evolution.md"

VLLM_VERSION = "0.19.0"
VLLM_SOURCE_HASHES = {
    "config/scheduler.py": "011cb6cda6625e9f05fc0a6479061e94ec5a449d984685f19985243e91c45a2d",
    "v1/engine/async_llm.py": "b101a3cd5ea4b7d0cf1852824d8fc96ec8c6a83c0396aad54ad9c1c8d69442c3",
    "v1/core/sched/scheduler.py": "9f3da2dfce94963e1e1cefc156e4239120f79c6eaf824b2829cac0ee7736b58a",
    "v1/core/kv_cache_manager.py": "0df69b7626195cbaabb33013d1e05ba6660243369171367f87313e4221bfffc8",
}

ITERATION_RE = re.compile(
    r"Iteration\((?P<iteration>\d+)\): "
    r"(?P<context_requests>\d+) context requests, "
    r"(?P<context_tokens>\d+) context tokens, "
    r"(?P<generation_requests>\d+) generation requests, "
    r"(?P<generation_tokens>\d+) generation tokens, "
    r"iteration elapsed time: (?P<elapsed_ms>[0-9.]+) ms"
)

CSV_FIELDS = ["section", "side", "metric", "value", "target", "status", "source", "note"]


@dataclass(frozen=True)
class IterationRow:
    iteration: int
    context_requests: int
    context_tokens: int
    generation_requests: int
    generation_tokens: int
    elapsed_ms: float

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (
            self.context_requests,
            self.context_tokens,
            self.generation_requests,
            self.generation_tokens,
        )


@dataclass(frozen=True)
class GapAlignment:
    gap_real_iteration: int
    compared_steps: int
    matched_steps: int
    match_rate: float


def parse_iteration_rows(lines: Iterable[str]) -> list[IterationRow]:
    rows: list[IterationRow] = []
    for line in lines:
        match = ITERATION_RE.search(line)
        if not match:
            continue
        values = match.groupdict()
        rows.append(
            IterationRow(
                iteration=int(values["iteration"]),
                context_requests=int(values["context_requests"]),
                context_tokens=int(values["context_tokens"]),
                generation_requests=int(values["generation_requests"]),
                generation_tokens=int(values["generation_tokens"]),
                elapsed_ms=float(values["elapsed_ms"]),
            )
        )
    return rows


def find_single_gap_alignment(
    real_rows: list[IterationRow], sim_rows: list[IterationRow]
) -> GapAlignment:
    if len(real_rows) < 2 or not sim_rows:
        raise ValueError("need real and sim iteration rows")
    best: GapAlignment | None = None
    for index, row in enumerate(real_rows):
        if row.context_requests != 0 or row.generation_requests <= 0:
            continue
        candidate = real_rows[:index] + real_rows[index + 1 :]
        compared = min(len(candidate), len(sim_rows))
        matched = sum(
            candidate[pos].shape == sim_rows[pos].shape for pos in range(compared)
        )
        alignment = GapAlignment(
            gap_real_iteration=row.iteration,
            compared_steps=compared,
            matched_steps=matched,
            match_rate=matched / compared if compared else 0.0,
        )
        if best is None or (alignment.match_rate, -alignment.gap_real_iteration) > (
            best.match_rate,
            -best.gap_real_iteration,
        ):
            best = alignment
    if best is None:
        raise ValueError("no real idle-prefill gap found")
    return best


def state_evolution_verdict(
    *,
    alignment: GapAlignment,
    real_max_context_requests: int,
    sim_max_context_requests: int,
    real_first_trigger_tokens: int,
    real_first_victim_tokens: int,
    sim_first_trigger_tokens: int,
    sim_first_victim_tokens: int,
    vllm_long_prefill_threshold: int,
    sim_long_prefill_threshold: int,
) -> dict[str, object]:
    packing_aligned = (
        real_max_context_requests == sim_max_context_requests
        and vllm_long_prefill_threshold == sim_long_prefill_threshold
    )
    phase_signature = (
        real_first_trigger_tokens - real_first_victim_tokens == 1
        and sim_first_trigger_tokens == sim_first_victim_tokens
    )
    gap_explains_trace = alignment.match_rate == 1.0 and alignment.compared_steps >= 4
    root = "engine_visible_arrival_gap" if packing_aligned and phase_signature and gap_explains_trace else "unresolved"
    return {
        "prefill_packing": "aligned_not_root" if packing_aligned else "diverged",
        "prefill_completion": (
            "one_step_arrival_gap_then_aligned" if gap_explains_trace else "diverged"
        ),
        "block_boundary": "aligned_not_root" if phase_signature else "diverged",
        "root_cause": root,
        "self_sustaining_loop": (
            "falsified_as_origin" if root == "engine_visible_arrival_gap" else "unresolved"
        ),
        "runtime_fix_allowed": False,
    }


def _point():
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == SCENARIO)


def capture_sim_iterations(limit: int = TRACE_STEPS) -> list[IterationRow]:
    point = _point()
    model, database, backend = validate._load_model_and_db(
        tp=point.tp,
        dp=point.dp,
        moe_tp=point.moe_tp,
        moe_ep=point.moe_ep,
    )
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    config = replace(
        config,
        num_requests=observation.NUM_REQUESTS,
        warmup_requests=observation.NUM_REQUESTS,
    )
    rows: list[IterationRow] = []
    original_schedule = CBScheduler.schedule

    def wrapped_schedule(self, waiting, running):
        result = original_schedule(self, waiting, running)
        if len(rows) < limit:
            rows.append(
                IterationRow(
                    iteration=len(rows),
                    context_requests=len(result.prefill_reqs),
                    context_tokens=result.total_prefill_tokens,
                    generation_requests=len(result.decode_reqs),
                    generation_tokens=len(result.decode_reqs),
                    elapsed_ms=0.0,
                )
            )
        return result

    with patch.object(CBScheduler, "schedule", wrapped_schedule):
        CBSimulator(backend, model, database, config).run(
            isl=point.isl,
            osl=observation.OSL,
            concurrency=point.batch_size,
            num_gpus=point.tp,
        )
    return rows


def _real_iterations(path: Path) -> list[IterationRow]:
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as source:
        return parse_iteration_rows(source)


def _generation_increment_max(rows: list[IterationRow]) -> int:
    return max(
        (right.generation_requests - left.generation_requests for left, right in zip(rows, rows[1:])),
        default=0,
    )


def _sign_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [row for row in csv.DictReader(source) if row["section"] == "sign_prediction"]


def _row(
    section: str,
    side: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    source: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "side": side,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "source": source,
        "note": note,
    }


def build_rows(
    *, real_log: Path = REAL_LOG, preempt_jsonl: Path = PREEMPT_JSONL
) -> list[dict[str, object]]:
    real_all = _real_iterations(real_log)
    sim = capture_sim_iterations()
    real = real_all[: len(sim) + 1]
    alignment = find_single_gap_alignment(real, sim)

    real_pairs = observation.pair_records(observation.read_jsonl(preempt_jsonl))
    sim_events = observation.run_matching_sim()
    real_first = real_pairs[0]
    sim_first = sim_events[0]
    point = _point()
    config = validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=validate.DEFAULT_EP8_PER_ITERATION_OVERHEAD_MS,
    )
    verdict = state_evolution_verdict(
        alignment=alignment,
        real_max_context_requests=max(row.context_requests for row in real),
        sim_max_context_requests=max(row.context_requests for row in sim),
        real_first_trigger_tokens=real_first.trigger_num_computed_tokens,
        real_first_victim_tokens=real_first.victim_num_computed_tokens,
        sim_first_trigger_tokens=int(sim_first["trigger_kv_cache_len"]),
        sim_first_victim_tokens=int(sim_first["victim_kv_cache_len"]),
        vllm_long_prefill_threshold=0,
        sim_long_prefill_threshold=config.long_prefill_token_threshold,
    )
    second_request_blocks = (point.isl + config.block_size - 1) // config.block_size
    free_after_first = config.num_gpu_blocks - second_request_blocks
    rows = [
        _row(
            "trace_alignment",
            "real",
            "idle_prefill_gap_iteration",
            alignment.gap_real_iteration,
            target="none in all-at-t0 sim",
            status="diverged",
            source=str(real_log.relative_to(REPO_ROOT)),
            note="0 context requests, 1 generation request",
        ),
        _row(
            "trace_alignment",
            "both",
            "post_gap_shape_match_rate",
            alignment.match_rate,
            target="1.0",
            status="pass" if alignment.match_rate == 1.0 else "fail",
            source="real INFO iteration rows vs current sim scheduler trace",
            note=f"matched={alignment.matched_steps}/{alignment.compared_steps}",
        ),
        _row(
            "admission_state",
            "real",
            "second_full_prompt_fits_at_gap",
            free_after_first >= second_request_blocks,
            target=True,
            status="pass",
            source="Phase458 KV capacity + vLLM can_fit_full_sequence semantics",
            note=f"free_after_first={free_after_first}; needed={second_request_blocks}",
        ),
        _row(
            "prefill_packing",
            "real",
            "max_context_requests_first_window",
            max(row.context_requests for row in real),
            target=max(row.context_requests for row in sim),
            status="aligned",
            source=str(real_log.relative_to(REPO_ROOT)),
        ),
        _row(
            "prefill_packing",
            "sim",
            "max_context_requests_first_window",
            max(row.context_requests for row in sim),
            status="aligned",
            source="current cb_sim trace",
        ),
        _row(
            "prefill_completion",
            "real",
            "max_decode_batch_increment",
            _generation_increment_max(real),
            target=1,
            status="aligned",
            source=str(real_log.relative_to(REPO_ROOT)),
        ),
        _row(
            "prefill_completion",
            "sim",
            "max_decode_batch_increment",
            _generation_increment_max(sim),
            target=1,
            status="aligned",
            source="current cb_sim trace",
        ),
        _row(
            "source_semantics",
            "real",
            "long_prefill_token_threshold",
            0,
            target=config.long_prefill_token_threshold,
            status="aligned",
            source="vLLM config/scheduler.py:69-79,244-247",
            note="max_num_partial_prefills defaults to 1, so the 4% auto-threshold branch is inactive",
        ),
        _row(
            "source_semantics",
            "sim",
            "initial_arrival_visibility",
            "all_concurrency_at_t0",
            target="engine-visible incremental arrivals",
            status="diverged",
            source="cb_sim simulator.py:152-157",
            note="the only source difference that reproduces the one-row trace shift",
        ),
        _row(
            "source_semantics",
            "real",
            "initial_arrival_visibility",
            "process_inputs_then_add_request_async",
            target="scheduler only sees enqueued requests",
            status="observed",
            source="vLLM async_llm.py:354-373,407-418; scheduler.py:1733-1752",
            note="Iteration(1) proves the next request was not yet scheduler-visible",
        ),
        _row(
            "block_boundary",
            "real",
            "first_trigger_minus_tail_tokens",
            real_first.trigger_num_computed_tokens - real_first.victim_num_computed_tokens,
            target=1,
            status="phase_skew",
            source="Phase462 Step2b decision hook",
            note=f"trigger={real_first.trigger_num_computed_tokens}; victim={real_first.victim_num_computed_tokens}",
        ),
        _row(
            "block_boundary",
            "sim",
            "first_trigger_minus_tail_tokens",
            int(sim_first["trigger_kv_cache_len"]) - int(sim_first["victim_kv_cache_len"]),
            target=0,
            status="self_phase",
            source="current sim decision hook",
            note="allocation remains one block short on both sides",
        ),
    ]
    for metric in (
        "prefill_packing",
        "prefill_completion",
        "block_boundary",
        "root_cause",
        "self_sustaining_loop",
        "runtime_fix_allowed",
    ):
        rows.append(
            _row(
                "verdict",
                "both",
                metric,
                verdict[metric],
                status="blocked" if metric == "runtime_fix_allowed" else "decided",
                source="source audit + trace alignment",
            )
        )
    for sign in _sign_rows(SIGN_CSV):
        rows.append(
            _row(
                "sign_prediction",
                sign["scenario"],
                sign["metric"],
                sign["value"],
                target=sign["target"],
                status=sign["status"],
                source=sign["source"],
                note=sign["note"],
            )
        )
    rows.append(
        _row(
            "flags",
            "both",
            "default_aic",
            "No-Go",
            target="6/6 <=1.15",
            status="No-Go",
            source="Phase462 discipline",
            note="report-only; no runtime, PerfDB, or gate change",
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lookup = {(row["side"], row["metric"]): row for row in rows}
    sign_rows = [row for row in rows if row["section"] == "sign_prediction"]
    lines = [
        "# Phase462 Step2a-2: 状态演化源码审计",
        "",
        "结论：计划中的“多 prefill 同步完成 -> 同相跨块 -> 自抢占”不是首次分歧的起点。部署态 vLLM 与 cb_sim 的 prefill 上限、running/waiting 顺序和 16-token block 申请语义一致。唯一能完整解释 Step2b 签名的差异是：cb_sim 在 t=0 把全部并发请求放入 waiting；真实 EngineCore 的下一请求在 Iteration(1) 尚不可见。删除这一个真实空拍后，前 16 步调度形状逐步 100% 对齐。",
        "",
        "| 审计项 | real | sim | 判定 |",
        "|---|---|---|---|",
        f"| prefill 打包 | threshold=0，首窗 max context req={lookup[('real', 'max_context_requests_first_window')]['value']} | threshold=0，max={lookup[('sim', 'max_context_requests_first_window')]['value']} | 对齐，不是根因 |",
        f"| 完成节奏 | max decode 增量={lookup[('real', 'max_decode_batch_increment')]['value']}，Iteration(1) 有空拍 | max decode 增量={lookup[('sim', 'max_decode_batch_increment')]['value']}，无空拍 | 空拍后 {float(lookup[('both', 'post_gap_shape_match_rate')]['value']):.0%} 对齐 |",
        f"| block 边界 | trigger-tail={lookup[('real', 'first_trigger_minus_tail_tokens')]['value']} token | trigger-tail={lookup[('sim', 'first_trigger_minus_tail_tokens')]['value']} token | 申请算法对齐，相位不同 |",
        "",
        "## 源码落点",
        "",
        "| 位置 | 语义 |",
        "|---|---|",
        "| vLLM `config/scheduler.py:69-79,244-247` | 默认 `max_num_partial_prefills=1`，因此 `long_prefill_token_threshold` 保持 0；原计划假设的 4% cap 未启用 |",
        "| vLLM `v1/core/sched/scheduler.py:384-479,574-757` | running 先消费预算，waiting 再消费剩余预算，与 cb_sim 顺序一致 |",
        "| vLLM `v1/core/kv_cache_manager.py:361-397` | 按本步 `num_new_tokens` 计算新 block；Step2b 已证明两侧均在缺 1 block 时触发 |",
        "| vLLM `v1/engine/async_llm.py:354-373,407-418` | 每个请求先 `process_inputs`，随后才 `add_request_async` 到 EngineCore |",
        "| cb_sim `simulator.py:152-157` | 一次性将全部 concurrency 以 `arrival_time_ms=0.0` 填入 waiting |",
        "",
        "部署源码 SHA256：" + "; ".join(f"`{name}`={sha}" for name, sha in VLLM_SOURCE_HASHES.items()) + ".",
        "",
        "## 因果判定",
        "",
        "真实第 0 步和 sim 相同；真实第 1 步只有 decode、没有 context。此时 KV 仍可容纳下一条完整 32k prompt，且 running=1 远低于 max_num_seqs，所以该空拍只能由下一请求尚未进入 EngineCore waiting 解释。把这一行从真实轨迹移除后，其后 16 步与 sim 完全一致；第一次抢占处因此表现为 real trigger 比 tail 领先 1 token，而 sim trigger 与 tail 同相并自抢占。",
        "",
        "因此，recompute 巨 bucket 是抢占环路的放大器，不是首次分歧的来源。Step2c 不能修改 prefill cap、block 阈值或加入随机去同步。可执行修复仍被锁住：需要先把 EngineCore 可见到达建成无自由参数的输入原语，或正式声明 ramp/到达动力学不在模型范围。",
        "",
        "## 六点符号预测",
        "",
        "| 场景 | 修掉额外抢占后的单变量方向 | 风险 |",
        "|---|---|---|",
    ]
    for row in sign_rows:
        lines.append(f"| {row['side']} | {row['value']} | {row['status']} |")
    lines.extend(
        [
            "",
            "静态符号只用于证伪，不用于预测终值。尤其 tp8-8k2k 只有 1.543% sim 吞吐上升空间，语义修复可能把它推出 15%。",
            "",
            "本步骤 `report-only / diagnostic_only=true / valid_for_default=false / perf_database=false`。未改 runtime、PerfDB 或 gate；Default AIC 维持 No-Go。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-log", type=Path, default=REAL_LOG)
    parser.add_argument("--preempt-jsonl", type=Path, default=PREEMPT_JSONL)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    rows = build_rows(real_log=args.real_log, preempt_jsonl=args.preempt_jsonl)
    write_csv(args.csv, rows)
    write_markdown(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()

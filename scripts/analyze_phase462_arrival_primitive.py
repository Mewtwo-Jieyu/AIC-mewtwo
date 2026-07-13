#!/usr/bin/env python3
"""Phase462 Step2a-3: audit the EngineCore-visible arrival primitive."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_arrival_primitive.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase462_arrival_primitive.md"

VLLM_VERSION = "0.19.0"
VLLM_SOURCE_HASHES = {
    "utils/async_utils.py": "82953898a30494e4da43236d140153c2c9d05a876b8267efbea27170ac074d7b",
    "renderers/base.py": "74d716ff06a0e5f480dbab51fd6c36246fc22f51178051810d5d3b0fe54653cb",
    "v1/engine/async_llm.py": "b101a3cd5ea4b7d0cf1852824d8fc96ec8c6a83c0396aad54ad9c1c8d69442c3",
    "v1/engine/core.py": "896730e749cbcabb487ce50c703974594d197fc31a1d3b26fe096197d142d2d5",
}

OBSERVATION_LOGS = [
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_8k2k"
    / "K2.5-tp8ep8-8k2k/serve.log",
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_32k3k"
    / "K2.5-tp8ep8-32k3k/serve.log",
    REPO_ROOT
    / "docs/iter_gap_investigation/phase458_n512_unify/recollect_tp8_8k2k_bt65536"
    / "K2.5-tp8ep8-8k2k-bt65536/serve.log",
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/overhead_on"
    / "K2.5-tp8ep8-32k3k/serve.log.gz",
]

BENCH_RECORDS = [path.parent / "bench_records.jsonl" for path in OBSERVATION_LOGS[:3]]

PIPELINE_START_RE = re.compile(
    r"request_pipeline_start.*request_id=(?P<request_id>[^\s]+)"
)
TOKENIZER_BATCH_RE = re.compile(r"tokenizer_batch_(?:start|end)")
POST_ADD_RE = re.compile(r"Added request (?P<request_id>[^.\s]+)\.")
ENGINE_RECEIVE_RE = re.compile(
    r"engine_core_receive.*request_id=(?P<request_id>[^\s]+)"
)

CSV_FIELDS = ["section", "metric", "value", "status", "source", "note"]


@dataclass(frozen=True)
class RuntimeSourceSpec:
    max_batch_size: int
    batch_wait_timeout_ms: float
    executor_workers: int
    engine_drains_before_step: bool

    @property
    def structure(self) -> str:
        return "microbatch_queue" if self.max_batch_size > 1 else "serial_request_queue"

    @property
    def per_request_serial_formula_allowed(self) -> bool:
        return self.max_batch_size == 1


@dataclass(frozen=True)
class ObservationInventory:
    pipeline_start_rows: int
    tokenizer_batch_rows: int
    post_add_rows: int
    engine_receive_rows: int
    complete_request_pairs: int

    @property
    def measurement_ready(self) -> bool:
        return (
            self.pipeline_start_rows > 0
            and self.tokenizer_batch_rows > 0
            and self.complete_request_pairs == self.pipeline_start_rows
            and self.complete_request_pairs == self.post_add_rows
            and self.complete_request_pairs == self.engine_receive_rows
        )


def scan_observation_lines(lines: Iterable[str]) -> ObservationInventory:
    starts: set[str] = set()
    post_add: set[str] = set()
    engine_receive: set[str] = set()
    tokenizer_rows = 0
    for line in lines:
        if match := PIPELINE_START_RE.search(line):
            starts.add(match.group("request_id"))
        if TOKENIZER_BATCH_RE.search(line):
            tokenizer_rows += 1
        if match := POST_ADD_RE.search(line):
            post_add.add(match.group("request_id"))
        if match := ENGINE_RECEIVE_RE.search(line):
            engine_receive.add(match.group("request_id"))
    return ObservationInventory(
        pipeline_start_rows=len(starts),
        tokenizer_batch_rows=tokenizer_rows,
        post_add_rows=len(post_add),
        engine_receive_rows=len(engine_receive),
        complete_request_pairs=len(starts & post_add & engine_receive),
    )


def arrival_primitive_verdict(
    source: RuntimeSourceSpec, observations: ObservationInventory
) -> dict[str, object]:
    structure_passed = (
        source.max_batch_size == 32
        and source.batch_wait_timeout_ms == 2.0
        and source.executor_workers == 1
        and source.engine_drains_before_step
    )
    measurement_passed = observations.measurement_ready
    prototype_allowed = structure_passed and measurement_passed
    signature_status = "pending" if prototype_allowed else "not_run"
    return {
        "structure_gate": "pass" if structure_passed else "fail",
        "measurement_gate": "pass" if measurement_passed else "fail",
        "prototype_gate": "pending" if prototype_allowed else "blocked",
        "runtime_fix_allowed": False,
        "visible_step_signature": signature_status,
        "first_16_steps_signature": signature_status,
        "preemption_signature": signature_status,
    }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _combine_inventories(items: Iterable[ObservationInventory]) -> ObservationInventory:
    values = list(items)
    return ObservationInventory(
        pipeline_start_rows=sum(item.pipeline_start_rows for item in values),
        tokenizer_batch_rows=sum(item.tokenizer_batch_rows for item in values),
        post_add_rows=sum(item.post_add_rows for item in values),
        engine_receive_rows=sum(item.engine_receive_rows for item in values),
        complete_request_pairs=sum(item.complete_request_pairs for item in values),
    )


def _bench_schema(path: Path) -> tuple[int, set[str]]:
    count = 0
    fields: set[str] = set()
    with path.open(encoding="utf-8") as source:
        for line in source:
            count += 1
            if count == 1:
                fields = set(json.loads(line))
    return count, fields


def _row(
    section: str,
    metric: str,
    value: object,
    *,
    status: str,
    source: str,
    note: str,
) -> dict[str, object]:
    return {
        "section": section,
        "metric": metric,
        "value": value,
        "status": status,
        "source": source,
        "note": note,
    }


def build_rows() -> tuple[list[dict[str, object]], dict[str, object]]:
    source = RuntimeSourceSpec(32, 2.0, 1, True)
    rows = [
        _row(
            "source_structure",
            "arrival_queue",
            source.structure,
            status="pass",
            source="vllm/utils/async_utils.py:24-52,90-132",
            note="up to 32 prompts wait 2ms and share one tokenizer batch call",
        ),
        _row(
            "source_structure",
            "completion_path",
            "render_cmpl_async->tokenize_prompts_async->process_inputs->add_request_async",
            status="pass",
            source="vllm/renderers/base.py:843-862; v1/engine/async_llm.py:340-420",
            note="tokenization precedes process_inputs; process_inputs is not the serial tokenizer",
        ),
        _row(
            "source_structure",
            "engine_queue_boundary",
            "drain_before_each_engine_step",
            status="pass",
            source="vllm/v1/engine/core.py:1136-1175",
            note="all queued client requests are consumed before the next step",
        ),
        _row(
            "source_structure",
            "per_request_serial_formula_allowed",
            source.per_request_serial_formula_allowed,
            status="fail",
            source="vllm/utils/async_utils.py:32-52,90-132",
            note="single executor serializes batches, not individual requests",
        ),
    ]

    inventories = []
    for path in OBSERVATION_LOGS:
        with _open_text(path) as source_file:
            inventory = scan_observation_lines(source_file)
        inventories.append(inventory)
        rows.append(
            _row(
                "observation_inventory",
                path.parent.name,
                inventory.complete_request_pairs,
                status="fail" if not inventory.measurement_ready else "pass",
                source=str(path.relative_to(REPO_ROOT)),
                note=(
                    f"start={inventory.pipeline_start_rows}; "
                    f"tokenizer_batch={inventory.tokenizer_batch_rows}; "
                    f"post_add={inventory.post_add_rows}; "
                    f"engine_receive={inventory.engine_receive_rows}"
                ),
            )
        )

    bench_start_fields = {"started_at", "start_time", "arrival_time", "ts_ns"}
    for path in BENCH_RECORDS:
        count, fields = _bench_schema(path)
        found = sorted(fields & bench_start_fields)
        rows.append(
            _row(
                "observation_inventory",
                f"{path.parent.name}_bench_records",
                count,
                status="fail" if not found else "pass",
                source=str(path.relative_to(REPO_ROOT)),
                note=f"request_start_fields={','.join(found) if found else 'none'}",
            )
        )

    combined = _combine_inventories(inventories)
    verdict = arrival_primitive_verdict(source, combined)
    for metric, value in verdict.items():
        rows.append(
            _row(
                "gate",
                metric,
                value,
                status=(
                    "pass"
                    if value == "pass"
                    else "blocked"
                    if value in {"blocked", "not_run"}
                    else "fail"
                ),
                source="phase462_step2a3",
                note="no runtime, PerfDB, or gate change",
            )
        )
    return rows, verdict


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, verdict: dict[str, object]) -> None:
    hashes = "; ".join(f"`{name}`={digest}" for name, digest in VLLM_SOURCE_HASHES.items())
    text = f"""# Phase462 Step2a-3: EngineCore 可见到达原语审计

结论：**Step2a-3 测量门失败，Step2c 继续锁住。** Step2a-2 的 16/16 对齐仍证明首次分歧位于调度器外、EngineCore 可见到达边界；但“`process_inputs` 逐请求串行，因此相邻到达差一个处理耗时”的结构不成立。部署态 vLLM {VLLM_VERSION} 在 `process_inputs` 之前使用异步 tokenizer 微批：最多 32 条请求、2ms 聚合窗、单线程 executor 每次执行一个批次。现有 raw 又没有 pipeline 起点、tokenizer batch 和 EngineCore 入队的同 request-id 时间戳，无法测得数值原语，也不能运行三签名原型。

| 门 | 结果 | 含义 |
|---|---|---|
| 源码结构门 | {verdict['structure_gate']} | 可定义为微批排队，不是逐请求串行排队 |
| 数值测量门 | {verdict['measurement_gate']} | 现有 4 份 serve log 的完整边界配对为 0 |
| 离线原型门 | {verdict['prototype_gate']} | 禁止把 Iteration(1) 的“一拍”反推成处理耗时 |
| runtime | locked | 不改 runtime、PerfDB 或 gate |

## 源码规格

| 位置 | 已确认语义 |
|---|---|
| `vllm/utils/async_utils.py:24-52,90-132` | `AsyncMicrobatchTokenizer(max_batch_size=32, batch_wait_timeout_s=0.002)`；单 executor 串行的是微批调用，不是请求 |
| `vllm/renderers/base.py:141-147,379-390,843-862` | completion 请求先进入共享 async tokenizer，再转成 EngineInput |
| `vllm/v1/engine/async_llm.py:340-420` | tokenized EngineInput 经同步 `process_inputs` 后 `add_request_async`；`Added request` 仅在 `log_requests` 开启时出现 |
| `vllm/v1/engine/core.py:1136-1175` | 每个 engine step 前 drain input queue；空拍只证明下一请求当时尚未进入该 queue |
| cb_sim `simulator.py:152-157` | 当前把初始并发全部设为 `arrival_time_ms=0.0` |

部署源码 SHA256：{hashes}。

## 原语边界

正确的候选结构是**微批排队原语**：HTTP 请求进入 tokenizer 队列，按源码固定的 32 条上限和 2ms 聚合窗形成批次；单 executor 依次处理批次；批内请求完成 tokenization 后，再经 `process_inputs` 和 IPC 到 EngineCore。需要实测的数值至少有两段：`tokenizer batch service time(total_prompt_tokens, batch_size)` 与 `tokenize_done -> EngineCore received` 延迟。只按单请求 prompt 长度存一个常数，会漏掉 batch size，仍是隐藏经验项。

现有 Phase458 `bench_records.jsonl` 只有 request index、总 latency 和 token 数，没有提交时刻或 EngineCore 可见时刻；serve log 未开启 `Added request`，Step2b 也只记录抢占决策。因此以下三项均为 `not_run`：

| 原型签名 | 状态 |
|---|---|
| 逐请求可见步对齐 | {verdict['visible_step_signature']} |
| 前 16 步继续 16/16 对齐 | {verdict['first_16_steps_signature']} |
| 自抢占归零、重复 victim 收敛、26 -> 约 10 | {verdict['preemption_signature']} |

## 最小补观测规格

下一步只能是 logging-only 短跑，且先经确认：用同一个 request id 记录 `HTTP/render start`、`tokenizer batch id/start/end + batch size + total prompt tokens`、`add_request_async call/end`、`EngineCore receive + visible iteration`，统一 monotonic ns。patch 只加日志，`diagnostic_only=true / perf_database=false`，开销门仍为 2%。拿到这组边界时间后，才能测量原语并运行三签名原型。

本步骤 `report-only / diagnostic_only=true / valid_for_default=false / perf_database=false`。Default AIC 维持 No-Go。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, verdict = build_rows()
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, verdict)
    print(
        f"structure={verdict['structure_gate']} "
        f"measurement={verdict['measurement_gate']} "
        f"prototype={verdict['prototype_gate']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

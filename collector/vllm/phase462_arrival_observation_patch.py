#!/usr/bin/env python3
"""Temporarily add logging-only arrival hooks to vLLM 0.19."""

from __future__ import annotations

import argparse
from pathlib import Path


ASYNC_BEGIN = "# AIC PHASE462 ARRIVAL ASYNC BEGIN\n"
ASYNC_END = "# AIC PHASE462 ARRIVAL ASYNC END\n"
COMPLETION_BEGIN = "# AIC PHASE462 ARRIVAL COMPLETION BEGIN\n"
COMPLETION_END = "# AIC PHASE462 ARRIVAL COMPLETION END\n"
ENGINE_BEGIN = "    # AIC PHASE462 ARRIVAL ENGINE BEGIN\n"
ENGINE_END = "    # AIC PHASE462 ARRIVAL ENGINE END\n"
SCHED_BEGIN = "    # AIC PHASE462 ARRIVAL SCHEDULER BEGIN\n"
SCHED_END = "    # AIC PHASE462 ARRIVAL SCHEDULER END\n"

ASYNC_IMPORT_ANCHOR = "import contextlib\n"
ASYNC_INIT_ANCHOR = "        self._executor = executor or ThreadPoolExecutor(max_workers=1)\n"
ASYNC_CALL_ANCHOR = """        result_future: Future = self._loop.create_future()
        key = self._queue_key("encode", kwargs)
        queue = self._get_queue(self._loop, key)
        await queue.put((prompt, kwargs, result_future))
        return await result_future
"""
ASYNC_BATCH_ANCHOR = """    async def _batch_encode_loop(self, queue: asyncio.Queue, can_batch: bool):
        \"\"\"Batch incoming encode requests for efficiency.\"\"\"
        while True:
            prompt, kwargs, result_future = await queue.get()
            prompts = [prompt]
            kwargs_list = [kwargs]
            result_futures = [result_future]
            deadline = self._loop.time() + self.batch_wait_timeout_s

            while len(prompts) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0:
                    break
                try:
                    prompt, kwargs, result_future = await asyncio.wait_for(
                        queue.get(), timeout
                    )
                    prompts.append(prompt)
                    result_futures.append(result_future)
                    if not can_batch:
                        kwargs_list.append(kwargs)
                except asyncio.TimeoutError:
                    break

            try:
                # If every request uses identical kwargs we can run a single
                # batched tokenizer call for a big speed-up.
                if can_batch and len(prompts) > 1:
                    batch_encode_fn = partial(self.tokenizer, prompts, **kwargs)
                    results = await self._loop.run_in_executor(
                        self._executor, batch_encode_fn
                    )

                    for i, fut in enumerate(result_futures):
                        if not fut.done():
                            data = {k: v[i] for k, v in results.items()}
                            fut.set_result(BatchEncoding(data))
                else:
                    encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [
                        self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs)
                    ]
                    results = await self._loop.run_in_executor(
                        self._executor, encode_fn
                    )

                    for fut, res in zip(result_futures, results):
                        if not fut.done():
                            fut.set_result(res)
            except Exception as e:
                for fut in result_futures:
                    if not fut.done():
                        fut.set_exception(e)

"""

COMPLETION_IMPORT_ANCHOR = "from vllm.utils.async_utils import merge_async_iterators\n"
COMPLETION_RENDER_ANCHOR = """        result = await self.render_completion_request(request)
        if isinstance(result, ErrorResponse):
            return result

        engine_inputs = result
"""

ENGINE_HELPER_ANCHOR = "    def add_request(self, request: Request, request_wave: int = 0):\n"
ENGINE_RECEIVE_ANCHOR = "        self.scheduler.add_request(request)\n"

SCHED_HELPER_ANCHOR = "    def schedule(self) -> SchedulerOutput:\n"
SCHED_START_ANCHOR = """        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []
"""
SCHED_RETURN_ANCHOR = "        return scheduler_output\n"


ASYNC_HELPERS = ASYNC_BEGIN + """import contextvars as _aic_phase462_contextvars
import json as _aic_phase462_json
import os as _aic_phase462_os
import time as _aic_phase462_time

aic_phase462_arrival_trace_id = _aic_phase462_contextvars.ContextVar(
    "aic_phase462_arrival_trace_id", default=None
)
aic_phase462_arrival_expected_prompt_tokens = _aic_phase462_contextvars.ContextVar(
    "aic_phase462_arrival_expected_prompt_tokens", default=None
)


def _aic_phase462_arrival_emit(payload):
    path = _aic_phase462_os.environ.get("AIC_PHASE462_ARRIVAL_JSONL")
    if not path:
        return
    row = dict(payload)
    row.setdefault("schema", "phase462_arrival_observation_v1")
    row.setdefault("scenario", _aic_phase462_os.environ.get("AIC_PHASE462_SCENARIO"))
    row.setdefault("ts_ns", _aic_phase462_time.monotonic_ns())
    row.setdefault("wall_ts_ns", _aic_phase462_time.time_ns())
    row.setdefault("pid", _aic_phase462_os.getpid())
    data = (_aic_phase462_json.dumps(row, sort_keys=True) + "\\n").encode("utf-8")
    fd = _aic_phase462_os.open(
        path,
        _aic_phase462_os.O_APPEND | _aic_phase462_os.O_CREAT | _aic_phase462_os.O_WRONLY,
        0o644,
    )
    try:
        _aic_phase462_os.write(fd, data)
    finally:
        _aic_phase462_os.close(fd)


""" + ASYNC_END

ASYNC_INIT_REPLACEMENT = ASYNC_INIT_ANCHOR + """        self._aic_phase462_batch_seq = 0
        _aic_phase462_arrival_emit(
            {
                "kind": "runtime_config",
                "max_batch_size": int(self.max_batch_size),
                "batch_wait_timeout_s": float(self.batch_wait_timeout_s),
                "executor_max_workers": 1,
            }
        )
"""

ASYNC_CALL_REPLACEMENT = """        result_future: Future = self._loop.create_future()
        key = self._queue_key("encode", kwargs)
        queue = self._get_queue(self._loop, key)
        await queue.put(
            (
                prompt,
                kwargs,
                result_future,
                aic_phase462_arrival_trace_id.get(),
                aic_phase462_arrival_expected_prompt_tokens.get(),
                _aic_phase462_time.monotonic_ns(),
            )
        )
        return await result_future
"""

ASYNC_BATCH_REPLACEMENT = """    async def _batch_encode_loop(self, queue: asyncio.Queue, can_batch: bool):
        \"\"\"Batch incoming encode requests for efficiency.\"\"\"
        while True:
            prompt, kwargs, result_future, trace_id, expected_tokens, queue_ts_ns = (
                await queue.get()
            )
            prompts = [prompt]
            kwargs_list = [kwargs]
            result_futures = [result_future]
            trace_ids = [trace_id]
            expected_prompt_tokens = [expected_tokens]
            queue_enter_ns = [queue_ts_ns]
            deadline = self._loop.time() + self.batch_wait_timeout_s

            while len(prompts) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0:
                    break
                try:
                    (
                        prompt,
                        kwargs,
                        result_future,
                        trace_id,
                        expected_tokens,
                        queue_ts_ns,
                    ) = await asyncio.wait_for(queue.get(), timeout)
                    prompts.append(prompt)
                    result_futures.append(result_future)
                    trace_ids.append(trace_id)
                    expected_prompt_tokens.append(expected_tokens)
                    queue_enter_ns.append(queue_ts_ns)
                    if not can_batch:
                        kwargs_list.append(kwargs)
                except asyncio.TimeoutError:
                    break

            self._aic_phase462_batch_seq += 1
            batch_id = f"{_aic_phase462_os.getpid()}-{self._aic_phase462_batch_seq}"
            batch_start_ns = _aic_phase462_time.monotonic_ns()
            _aic_phase462_arrival_emit(
                {
                    "kind": "tokenizer_batch_enter",
                    "batch_id": batch_id,
                    "batch_size": len(prompts),
                    "trace_ids": trace_ids,
                    "expected_prompt_tokens": expected_prompt_tokens,
                    "queue_enter_ns": queue_enter_ns,
                    "batch_start_ns": batch_start_ns,
                    "max_batch_size": int(self.max_batch_size),
                    "batch_wait_timeout_s": float(self.batch_wait_timeout_s),
                }
            )

            try:
                # If every request uses identical kwargs we can run a single
                # batched tokenizer call for a big speed-up.
                if can_batch and len(prompts) > 1:
                    batch_encode_fn = partial(self.tokenizer, prompts, **kwargs)
                    results = await self._loop.run_in_executor(
                        self._executor, batch_encode_fn
                    )
                    prompt_token_lengths = [len(ids) for ids in results["input_ids"]]

                    for i, fut in enumerate(result_futures):
                        if not fut.done():
                            data = {k: v[i] for k, v in results.items()}
                            fut.set_result(BatchEncoding(data))
                else:
                    encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [
                        self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs)
                    ]
                    results = await self._loop.run_in_executor(
                        self._executor, encode_fn
                    )
                    prompt_token_lengths = [len(res.input_ids) for res in results]

                    for fut, res in zip(result_futures, results):
                        if not fut.done():
                            fut.set_result(res)
                _aic_phase462_arrival_emit(
                    {
                        "kind": "tokenizer_batch_complete",
                        "batch_id": batch_id,
                        "batch_size": len(prompts),
                        "trace_ids": trace_ids,
                        "prompt_token_lengths": prompt_token_lengths,
                        "batch_start_ns": batch_start_ns,
                        "batch_complete_ns": _aic_phase462_time.monotonic_ns(),
                    }
                )
            except Exception as e:
                for fut in result_futures:
                    if not fut.done():
                        fut.set_exception(e)

"""

COMPLETION_IMPORT_REPLACEMENT = """from vllm.utils.async_utils import (
    aic_phase462_arrival_expected_prompt_tokens,
    aic_phase462_arrival_trace_id,
    merge_async_iterators,
)
"""
COMPLETION_RENDER_REPLACEMENT = COMPLETION_BEGIN + """        aic_phase462_trace_id = (
            raw_request.headers.get("X-Request-Id")
            if raw_request is not None
            else None
        )
        aic_phase462_trace_token = aic_phase462_arrival_trace_id.set(
            aic_phase462_trace_id
        )
        aic_phase462_expected_tokens = (
            raw_request.headers.get("X-AIC-Prompt-Tokens")
            if raw_request is not None
            else None
        )
        aic_phase462_prompt_token = (
            aic_phase462_arrival_expected_prompt_tokens.set(
                int(aic_phase462_expected_tokens)
                if aic_phase462_expected_tokens is not None
                else None
            )
        )
        try:
            result = await self.render_completion_request(request)
        finally:
            aic_phase462_arrival_expected_prompt_tokens.reset(
                aic_phase462_prompt_token
            )
            aic_phase462_arrival_trace_id.reset(aic_phase462_trace_token)
""" + COMPLETION_END + """        if isinstance(result, ErrorResponse):
            return result

        engine_inputs = result
"""


def _class_writer(begin: str, end: str, method_name: str) -> str:
    return begin + f"""    def {method_name}(self, payload):
        import json
        import os
        import time
        path = os.environ.get("AIC_PHASE462_ARRIVAL_JSONL")
        if not path:
            return
        row = dict(payload)
        row.setdefault("schema", "phase462_arrival_observation_v1")
        row.setdefault("scenario", os.environ.get("AIC_PHASE462_SCENARIO"))
        row.setdefault("ts_ns", time.monotonic_ns())
        row.setdefault("wall_ts_ns", time.time_ns())
        row.setdefault("pid", os.getpid())
        data = (json.dumps(row, sort_keys=True) + "\\n").encode("utf-8")
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)

    @staticmethod
    def _aic_phase462_trace_id(request):
        external = getattr(request, "external_req_id", None)
        value = external or request.request_id
        if external is None:
            prefix, separator, random_suffix = value.rpartition("-")
            if (
                separator
                and len(random_suffix) == 8
                and all(char in "0123456789abcdef" for char in random_suffix)
            ):
                value = prefix
        if value.startswith("cmpl-") and value.endswith("-0"):
            return value[5:-2]
        return value

""" + end


ENGINE_HELPERS = _class_writer(
    ENGINE_BEGIN, ENGINE_END, "_aic_phase462_write_arrival_observation"
)
ENGINE_RECEIVE_REPLACEMENT = """        self._aic_phase462_write_arrival_observation(
            {
                "kind": "engine_receive",
                "trace_id": self._aic_phase462_trace_id(request),
                "request_id": request.request_id,
                "external_req_id": getattr(request, "external_req_id", None),
                "prompt_tokens": len(request.prompt_token_ids),
            }
        )
        self.scheduler.add_request(request)
"""

SCHED_HELPERS = _class_writer(
    SCHED_BEGIN, SCHED_END, "_aic_phase462_write_arrival_observation"
)
SCHED_START_REPLACEMENT = SCHED_START_ANCHOR + """        aic_phase462_waiting_before = len(self.waiting) + len(self.skipped_waiting)
        aic_phase462_step = getattr(self, "_aic_phase462_arrival_step", 0) + 1
        self._aic_phase462_arrival_step = aic_phase462_step
"""
SCHED_RETURN_REPLACEMENT = """        self._aic_phase462_write_arrival_observation(
            {
                "kind": "scheduler_step",
                "step": aic_phase462_step,
                "waiting_before": aic_phase462_waiting_before,
                "waiting_after": len(self.waiting) + len(self.skipped_waiting),
                "running_after": len(self.running),
                "new_context_count": len(scheduled_new_reqs),
                "new_context_trace_ids": [
                    self._aic_phase462_trace_id(request)
                    for request in scheduled_new_reqs
                ],
                "resumed_context_count": len(scheduled_resumed_reqs),
            }
        )
        return scheduler_output
"""


def _remove_block(text: str, begin: str, end: str) -> str:
    while begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + text[stop:]
    return text


def restore_async_source(text: str) -> str:
    text = _remove_block(text, ASYNC_BEGIN, ASYNC_END)
    text = text.replace(ASYNC_INIT_REPLACEMENT, ASYNC_INIT_ANCHOR)
    text = text.replace(ASYNC_CALL_REPLACEMENT, ASYNC_CALL_ANCHOR)
    return text.replace(ASYNC_BATCH_REPLACEMENT, ASYNC_BATCH_ANCHOR)


def patch_async_source(text: str) -> str:
    text = restore_async_source(text)
    for anchor in (ASYNC_IMPORT_ANCHOR, ASYNC_INIT_ANCHOR, ASYNC_CALL_ANCHOR, ASYNC_BATCH_ANCHOR):
        if anchor not in text:
            raise ValueError(f"async_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(ASYNC_IMPORT_ANCHOR, ASYNC_IMPORT_ANCHOR + ASYNC_HELPERS, 1)
    text = text.replace(ASYNC_INIT_ANCHOR, ASYNC_INIT_REPLACEMENT, 1)
    text = text.replace(ASYNC_CALL_ANCHOR, ASYNC_CALL_REPLACEMENT, 1)
    return text.replace(ASYNC_BATCH_ANCHOR, ASYNC_BATCH_REPLACEMENT, 1)


def restore_completion_source(text: str) -> str:
    text = text.replace(COMPLETION_IMPORT_REPLACEMENT, COMPLETION_IMPORT_ANCHOR)
    return text.replace(COMPLETION_RENDER_REPLACEMENT, COMPLETION_RENDER_ANCHOR)


def patch_completion_source(text: str) -> str:
    text = restore_completion_source(text)
    for anchor in (COMPLETION_IMPORT_ANCHOR, COMPLETION_RENDER_ANCHOR):
        if anchor not in text:
            raise ValueError(f"completion_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(COMPLETION_IMPORT_ANCHOR, COMPLETION_IMPORT_REPLACEMENT, 1)
    return text.replace(COMPLETION_RENDER_ANCHOR, COMPLETION_RENDER_REPLACEMENT, 1)


def restore_engine_source(text: str) -> str:
    text = _remove_block(text, ENGINE_BEGIN, ENGINE_END)
    return text.replace(ENGINE_RECEIVE_REPLACEMENT, ENGINE_RECEIVE_ANCHOR)


def patch_engine_source(text: str) -> str:
    text = restore_engine_source(text)
    for anchor in (ENGINE_HELPER_ANCHOR, ENGINE_RECEIVE_ANCHOR):
        if anchor not in text:
            raise ValueError(f"engine_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(ENGINE_HELPER_ANCHOR, ENGINE_HELPERS + ENGINE_HELPER_ANCHOR, 1)
    return text.replace(ENGINE_RECEIVE_ANCHOR, ENGINE_RECEIVE_REPLACEMENT, 1)


def restore_scheduler_source(text: str) -> str:
    text = _remove_block(text, SCHED_BEGIN, SCHED_END)
    text = text.replace(SCHED_START_REPLACEMENT, SCHED_START_ANCHOR)
    return text.replace(SCHED_RETURN_REPLACEMENT, SCHED_RETURN_ANCHOR)


def patch_scheduler_source(text: str) -> str:
    text = restore_scheduler_source(text)
    for anchor in (SCHED_HELPER_ANCHOR, SCHED_START_ANCHOR, SCHED_RETURN_ANCHOR):
        if anchor not in text:
            raise ValueError(f"scheduler_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(SCHED_HELPER_ANCHOR, SCHED_HELPERS + SCHED_HELPER_ANCHOR, 1)
    text = text.replace(SCHED_START_ANCHOR, SCHED_START_REPLACEMENT, 1)
    return text.replace(SCHED_RETURN_ANCHOR, SCHED_RETURN_REPLACEMENT, 1)


def _backup_path(backup_dir: Path, target: Path) -> Path:
    return backup_dir / f"{target.name}.phase462.arrival_observation.bak"


def _apply(target: Path, backup_dir: Path, patcher) -> None:
    original = target.read_text(encoding="utf-8")
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = _backup_path(backup_dir, target)
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patcher(original), encoding="utf-8")


def _restore(target: Path, backup_dir: Path, restorer) -> None:
    backup = _backup_path(backup_dir, target)
    if backup.exists():
        target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        target.write_text(restorer(target.read_text(encoding="utf-8")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--async-target", required=True, type=Path)
    parser.add_argument("--completion-target", required=True, type=Path)
    parser.add_argument("--engine-target", required=True, type=Path)
    parser.add_argument("--scheduler-target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true")
    action.add_argument("--restore", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    targets = (
        (args.async_target, patch_async_source, restore_async_source, ASYNC_BEGIN),
        (args.completion_target, patch_completion_source, restore_completion_source, COMPLETION_BEGIN),
        (args.engine_target, patch_engine_source, restore_engine_source, ENGINE_BEGIN),
        (args.scheduler_target, patch_scheduler_source, restore_scheduler_source, SCHED_BEGIN),
    )
    if args.apply:
        for target, patcher, _, _ in targets:
            _apply(target, args.backup_dir, patcher)
        return 0
    if args.restore:
        for target, _, restorer, _ in targets:
            _restore(target, args.backup_dir, restorer)
        return 0
    ok = all(marker in target.read_text(encoding="utf-8") for target, _, _, marker in targets)
    print(f"phase462_arrival_observation_patch={'applied' if ok else 'missing'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

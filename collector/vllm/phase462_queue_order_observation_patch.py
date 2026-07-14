#!/usr/bin/env python3
"""Temporarily add buffered queue-order observation hooks to vLLM 0.19."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from collector.vllm import phase462_arrival_observation_patch as arrival


ASYNC_BEGIN = "# AIC PHASE462 QUEUE ASYNC BEGIN\n"
ASYNC_END = "# AIC PHASE462 QUEUE ASYNC END\n"
COMPLETION_BEGIN = "# AIC PHASE462 QUEUE COMPLETION BEGIN\n"
COMPLETION_END = "# AIC PHASE462 QUEUE COMPLETION END\n"
SCHED_MODULE_BEGIN = "# AIC PHASE462 QUEUE SCHED MODULE BEGIN\n"
SCHED_MODULE_END = "# AIC PHASE462 QUEUE SCHED MODULE END\n"
SCHED_CLASS_BEGIN = "    # AIC PHASE462 QUEUE SCHED CLASS BEGIN\n"
SCHED_CLASS_END = "    # AIC PHASE462 QUEUE SCHED CLASS END\n"
ENGINE_BEGIN = "        # AIC PHASE462 QUEUE ENGINE BEGIN\n"
ENGINE_END = "        # AIC PHASE462 QUEUE ENGINE END\n"

ASYNC_IMPORT_ANCHOR = arrival.ASYNC_IMPORT_ANCHOR
ASYNC_INIT_ANCHOR = arrival.ASYNC_INIT_ANCHOR
ASYNC_CALL_ANCHOR = arrival.ASYNC_CALL_ANCHOR
ASYNC_BATCH_ANCHOR = arrival.ASYNC_BATCH_ANCHOR
COMPLETION_IMPORT_ANCHOR = arrival.COMPLETION_IMPORT_ANCHOR
COMPLETION_RENDER_ANCHOR = arrival.COMPLETION_RENDER_ANCHOR

SCHED_MODULE_ANCHOR = "import time\n"
SCHED_HELPER_ANCHOR = "    def schedule(self) -> SchedulerOutput:\n"
SCHED_START_ANCHOR = arrival.SCHED_START_ANCHOR
SCHED_CAPTURE_ANCHOR = "        # Check if the scheduling constraints are satisfied.\n"
SCHED_RETURN_ANCHOR = """        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output
"""
PRIORITY_ANCHOR = """                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
"""
TAIL_ANCHOR = """                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
"""
PREEMPT_WAITING_ANCHOR = "        self.waiting.prepend_request(request)\n"
WAITING_POP_ANCHOR = "                request = request_queue.pop_request()\n"
RUNNING_APPEND_ANCHOR = "                self.running.append(request)\n"
ENQUEUE_ANCHOR = """            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
"""
FINISH_REMOVE_ANCHOR = """        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)
"""

KV_FAILURE_ANCHOR = """        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None
"""

ENGINE_FUTURE_ANCHOR = """        with (
            self.log_error_detail(scheduler_output),
            self.log_iteration_details(scheduler_output),
        ):
            model_output = future.result()
            if model_output is None:
                # None from sample_tokens() implies that the original execute_model()
                # call failed - raise that exception.
                exec_model_fut.result()
                raise RuntimeError("unexpected error")
"""
ENGINE_SHUTDOWN_ANCHOR = """    def shutdown(self):
        self.structured_output_manager.clear_backend()
"""


def _buffer_helpers(*, component: str, capacity: int) -> str:
    return f'''import atexit as _aic_phase462_atexit
import json as _aic_phase462_json
import os as _aic_phase462_os
import time as _aic_phase462_time

_aic_phase462_queue_trace_dir = _aic_phase462_os.environ.get(
    "AIC_PHASE462_QUEUE_TRACE_DIR"
)
_aic_phase462_queue_capacity = int(
    _aic_phase462_os.environ.get("AIC_PHASE462_QUEUE_EVENT_CAPACITY", "{capacity}")
)
_aic_phase462_queue_buffer = [None] * _aic_phase462_queue_capacity
_aic_phase462_queue_count = 0
_aic_phase462_queue_flushed = False


def aic_phase462_queue_emit(payload):
    if not _aic_phase462_queue_trace_dir:
        return
    global _aic_phase462_queue_count
    if _aic_phase462_queue_count >= _aic_phase462_queue_capacity:
        raise RuntimeError("phase462_queue_trace_buffer_exhausted")
    row = dict(payload)
    row.setdefault("schema", "phase462_queue_order_observation_v1")
    row.setdefault("component", "{component}")
    row.setdefault("event_seq", _aic_phase462_queue_count + 1)
    row.setdefault("ts_ns", _aic_phase462_time.monotonic_ns())
    row.setdefault("wall_ts_ns", _aic_phase462_time.time_ns())
    row.setdefault("pid", _aic_phase462_os.getpid())
    _aic_phase462_queue_buffer[_aic_phase462_queue_count] = (
        _aic_phase462_json.dumps(row, sort_keys=True) + "\\n"
    )
    _aic_phase462_queue_count += 1


def aic_phase462_queue_flush():
    global _aic_phase462_queue_flushed
    if not _aic_phase462_queue_trace_dir or _aic_phase462_queue_flushed:
        return
    _aic_phase462_os.makedirs(_aic_phase462_queue_trace_dir, exist_ok=True)
    pid = _aic_phase462_os.getpid()
    final_path = _aic_phase462_os.path.join(
        _aic_phase462_queue_trace_dir, f"{component}-{{pid}}.jsonl"
    )
    temp_path = final_path + ".tmp"
    footer = {{
        "schema": "phase462_queue_order_observation_v1",
        "component": "{component}",
        "kind": "trace_footer",
        "event_count": _aic_phase462_queue_count,
        "buffer_capacity": _aic_phase462_queue_capacity,
        "flush_complete": True,
        "event_seq": _aic_phase462_queue_count + 1,
        "ts_ns": _aic_phase462_time.monotonic_ns(),
        "wall_ts_ns": _aic_phase462_time.time_ns(),
        "pid": pid,
    }}
    with open(temp_path, "w", encoding="utf-8") as output:
        output.writelines(
            line for line in _aic_phase462_queue_buffer[:_aic_phase462_queue_count]
            if line is not None
        )
        output.write(_aic_phase462_json.dumps(footer, sort_keys=True) + "\\n")
        output.flush()
        _aic_phase462_os.fsync(output.fileno())
    _aic_phase462_os.replace(temp_path, final_path)
    _aic_phase462_queue_flushed = True


_aic_phase462_atexit.register(aic_phase462_queue_flush)

'''


ASYNC_HELPERS = (
    ASYNC_BEGIN
    + "import contextvars as _aic_phase462_contextvars\n"
    + _buffer_helpers(component="tokenizer", capacity=4096)
    + '''aic_phase462_queue_trace_id = _aic_phase462_contextvars.ContextVar(
    "aic_phase462_queue_trace_id", default=None
)
aic_phase462_queue_expected_prompt_tokens = _aic_phase462_contextvars.ContextVar(
    "aic_phase462_queue_expected_prompt_tokens", default=None
)

'''
    + ASYNC_END
)

ASYNC_INIT_REPLACEMENT = ASYNC_INIT_ANCHOR + """        self._aic_phase462_batch_seq = 0
        self._aic_phase462_arrival_ordinal = 0
"""
ASYNC_CALL_REPLACEMENT = (
    arrival.ASYNC_CALL_REPLACEMENT
    .replace("aic_phase462_arrival_trace_id", "aic_phase462_queue_trace_id")
    .replace(
        "aic_phase462_arrival_expected_prompt_tokens",
        "aic_phase462_queue_expected_prompt_tokens",
    )
)
ASYNC_BATCH_REPLACEMENT = (
    arrival.ASYNC_BATCH_REPLACEMENT
    .replace("_aic_phase462_arrival_emit", "aic_phase462_queue_emit")
    .replace('"kind": "tokenizer_batch_enter"', '"kind": "arrival_map"')
    .replace(
        "            batch_start_ns = _aic_phase462_time.monotonic_ns()\n",
        "            batch_start_ns = _aic_phase462_time.monotonic_ns()\n"
        "            first_ordinal = self._aic_phase462_arrival_ordinal\n"
        "            arrival_ordinals = list(\n"
        "                range(first_ordinal, first_ordinal + len(prompts))\n"
        "            )\n"
        "            self._aic_phase462_arrival_ordinal += len(prompts)\n",
    )
    .replace(
        '                    "trace_ids": trace_ids,\n',
        '                    "trace_ids": trace_ids,\n'
        '                    "arrival_ordinals": arrival_ordinals,\n'
        '                    "batch_positions": list(range(len(prompts))),\n',
    )
    .replace("aic_phase462_arrival_trace_id", "aic_phase462_queue_trace_id")
    .replace(
        "aic_phase462_arrival_expected_prompt_tokens",
        "aic_phase462_queue_expected_prompt_tokens",
    )
)

COMPLETION_IMPORT_REPLACEMENT = """from vllm.utils.async_utils import (
    aic_phase462_queue_expected_prompt_tokens,
    aic_phase462_queue_trace_id,
    merge_async_iterators,
)
"""
COMPLETION_RENDER_REPLACEMENT = COMPLETION_BEGIN + """        aic_phase462_trace_id = (
            raw_request.headers.get("X-Request-Id")
            if raw_request is not None
            else None
        )
        aic_phase462_trace_token = aic_phase462_queue_trace_id.set(
            aic_phase462_trace_id
        )
        aic_phase462_expected_tokens = (
            raw_request.headers.get("X-AIC-Prompt-Tokens")
            if raw_request is not None
            else None
        )
        aic_phase462_prompt_token = aic_phase462_queue_expected_prompt_tokens.set(
            int(aic_phase462_expected_tokens)
            if aic_phase462_expected_tokens is not None
            else None
        )
        try:
            result = await self.render_completion_request(request)
        finally:
            aic_phase462_queue_expected_prompt_tokens.reset(
                aic_phase462_prompt_token
            )
            aic_phase462_queue_trace_id.reset(aic_phase462_trace_token)
""" + COMPLETION_END + """        if isinstance(result, ErrorResponse):
            return result

        engine_inputs = result
"""

SCHED_MODULE_HELPERS = (
    SCHED_MODULE_BEGIN
    + _buffer_helpers(component="engine", capacity=100000)
    + SCHED_MODULE_END
)

SCHED_CLASS_HELPERS = SCHED_CLASS_BEGIN + '''    @staticmethod
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

    def _aic_phase462_request_phase(self, request):
        blocks = self.kv_cache_manager.get_blocks(request.request_id).blocks
        return {
            "trace_id": self._aic_phase462_trace_id(request),
            "request_id": request.request_id,
            "status": request.status.name,
            "num_computed_tokens": int(request.num_computed_tokens),
            "num_output_placeholders": int(request.num_output_placeholders),
            "num_tokens": int(request.num_tokens),
            "num_prompt_tokens": int(request.num_prompt_tokens),
            "block_counts": [len(group) for group in blocks],
            "block_phase": int(request.num_computed_tokens % self.block_size),
            "num_preemptions": int(request.num_preemptions),
        }

    def _aic_phase462_queue_state(self):
        running = list(self.running)
        waiting = list(self.waiting)
        skipped = list(self.skipped_waiting)
        visible = [*running, *waiting, *skipped]
        return {
            "visible_request_ids": [request.request_id for request in visible],
            "visible_trace_ids": [
                self._aic_phase462_trace_id(request) for request in visible
            ],
            "running_order": [request.request_id for request in running],
            "running_trace_order": [
                self._aic_phase462_trace_id(request) for request in running
            ],
            "waiting_order": [request.request_id for request in waiting],
            "waiting_trace_order": [
                self._aic_phase462_trace_id(request) for request in waiting
            ],
            "skipped_waiting_order": [request.request_id for request in skipped],
            "skipped_waiting_trace_order": [
                self._aic_phase462_trace_id(request) for request in skipped
            ],
            "request_phase": [
                self._aic_phase462_request_phase(request) for request in visible
            ],
            "free_blocks": int(
                self.kv_cache_manager.block_pool.get_num_free_blocks()
            ),
        }

    def _aic_phase462_emit_queue(self, kind, **payload):
        aic_phase462_queue_emit(
            {
                "kind": kind,
                "schedule_seq": int(
                    getattr(self, "_aic_phase462_schedule_seq", 0)
                ),
                **payload,
            }
        )

    def _aic_phase462_log_future_complete(self, scheduler_output):
        sequence_by_output = getattr(
            self, "_aic_phase462_sequence_by_output", {}
        )
        schedule_seq = sequence_by_output.pop(id(scheduler_output), None)
        if schedule_seq is None:
            raise RuntimeError("phase462_future_without_schedule_seq")
        aic_phase462_queue_emit(
            {
                "kind": "future_complete",
                "schedule_seq": int(schedule_seq),
                "boundary": "before_update_from_output",
                **self._aic_phase462_queue_state(),
            }
        )

    @staticmethod
    def _aic_phase462_flush_queue_trace():
        aic_phase462_queue_flush()

''' + SCHED_CLASS_END

SCHED_START_REPLACEMENT = SCHED_START_ANCHOR + '''        aic_phase462_schedule_seq = (
            getattr(self, "_aic_phase462_schedule_seq", 0) + 1
        )
        self._aic_phase462_schedule_seq = aic_phase462_schedule_seq
        aic_phase462_queue_emit(
            {
                "kind": "scheduler_input",
                "schedule_seq": aic_phase462_schedule_seq,
                "boundary": "before_schedule",
                **self._aic_phase462_queue_state(),
            }
        )
'''

SCHED_CAPTURE_REPLACEMENT = '''        aic_phase462_scheduled_new_ids = [
            request.request_id for request in scheduled_new_reqs
        ]
        aic_phase462_scheduled_resumed_ids = [
            request.request_id for request in scheduled_resumed_reqs
        ]
        aic_phase462_scheduled_running_ids = [
            request.request_id for request in scheduled_running_reqs
        ]
        aic_phase462_preempted_ids = [
            request.request_id for request in preempted_reqs
        ]

''' + SCHED_CAPTURE_ANCHOR

SCHED_RETURN_REPLACEMENT = '''        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        sequence_by_output = getattr(
            self, "_aic_phase462_sequence_by_output", None
        )
        if sequence_by_output is None:
            sequence_by_output = self._aic_phase462_sequence_by_output = {}
        sequence_by_output[id(scheduler_output)] = aic_phase462_schedule_seq
        aic_phase462_queue_emit(
            {
                "kind": "scheduler_output",
                "schedule_seq": aic_phase462_schedule_seq,
                "boundary": "after_update_after_schedule",
                "scheduled_new_request_ids": aic_phase462_scheduled_new_ids,
                "scheduled_resumed_request_ids": aic_phase462_scheduled_resumed_ids,
                "scheduled_running_request_ids": aic_phase462_scheduled_running_ids,
                "preempted_request_ids": aic_phase462_preempted_ids,
                "num_scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
                **self._aic_phase462_queue_state(),
            }
        )
        return scheduler_output
'''

PRIORITY_REPLACEMENT = '''                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        aic_phase462_victim_position = self.running.index(preempted_req)
                        aic_phase462_victim_policy = "priority"
                        aic_phase462_trigger_position = self.running.index(request)
                        self.running.remove(preempted_req)
'''
TAIL_REPLACEMENT = '''                    else:
                        aic_phase462_victim_position = len(self.running) - 1
                        aic_phase462_victim_policy = "tail"
                        aic_phase462_trigger_position = self.running.index(request)
                        preempted_req = self.running.pop()

                    aic_phase462_failure = getattr(
                        self.kv_cache_manager, "_aic_phase462_last_failure", {}
                    )
                    self._aic_phase462_emit_queue(
                        "preempt_decision",
                        trigger_request_id=request.request_id,
                        trigger_trace_id=self._aic_phase462_trace_id(request),
                        trigger_position=aic_phase462_trigger_position,
                        victim_request_id=preempted_req.request_id,
                        victim_trace_id=self._aic_phase462_trace_id(preempted_req),
                        victim_position=aic_phase462_victim_position,
                        victim_policy=aic_phase462_victim_policy,
                        requested_blocks=aic_phase462_failure.get("requested_blocks"),
                        free_blocks=aic_phase462_failure.get("free_blocks"),
                        block_shortage=(
                            aic_phase462_failure.get("requested_blocks", 0)
                            - aic_phase462_failure.get("free_blocks", 0)
                        ),
                        running_order_after_pop=[
                            item.request_id for item in self.running
                        ],
                    )
                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
'''
PREEMPT_WAITING_REPLACEMENT = PREEMPT_WAITING_ANCHOR + '''        self._aic_phase462_emit_queue(
            "queue_transition",
            operation="preempt_to_waiting_front",
            request_id=request.request_id,
            trace_id=self._aic_phase462_trace_id(request),
            running_order=[item.request_id for item in self.running],
            waiting_order=[item.request_id for item in self.waiting],
        )
'''
WAITING_POP_REPLACEMENT = WAITING_POP_ANCHOR + '''                aic_phase462_waiting_status = request.status
                self._aic_phase462_emit_queue(
                    "queue_transition",
                    operation="waiting_remove",
                    request_id=request.request_id,
                    trace_id=self._aic_phase462_trace_id(request),
                    prior_status=aic_phase462_waiting_status.name,
                )
'''
RUNNING_APPEND_REPLACEMENT = RUNNING_APPEND_ANCHOR + '''                self._aic_phase462_emit_queue(
                    "queue_transition",
                    operation=(
                        "resume"
                        if aic_phase462_waiting_status == RequestStatus.PREEMPTED
                        else "admit"
                    ),
                    request_id=request.request_id,
                    trace_id=self._aic_phase462_trace_id(request),
                    running_order=[item.request_id for item in self.running],
                )
'''
ENQUEUE_REPLACEMENT = '''            self._enqueue_waiting_request(request)
            self._aic_phase462_emit_queue(
                "queue_transition",
                operation="waiting_insert_new",
                request_id=request.request_id,
                trace_id=self._aic_phase462_trace_id(request),
                waiting_order=[item.request_id for item in self.waiting],
            )
            self.requests[request.request_id] = request
'''
FINISH_REMOVE_REPLACEMENT = '''        if stopped_running_reqs:
            aic_phase462_stopped_running_ids = [
                request.request_id for request in stopped_running_reqs
            ]
            self.running = remove_all(self.running, stopped_running_reqs)
            self._aic_phase462_emit_queue(
                "queue_transition",
                operation="finish_remove_running",
                request_ids=aic_phase462_stopped_running_ids,
                running_order=[item.request_id for item in self.running],
            )
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            aic_phase462_stopped_waiting_ids = [
                request.request_id for request in stopped_preempted_reqs
            ]
            self.waiting.remove_requests(stopped_preempted_reqs)
            self._aic_phase462_emit_queue(
                "queue_transition",
                operation="finish_remove_waiting",
                request_ids=aic_phase462_stopped_waiting_ids,
                waiting_order=[item.request_id for item in self.waiting],
            )
'''

KV_FAILURE_REPLACEMENT = '''        aic_phase462_free_blocks = self.block_pool.get_num_free_blocks()
        if num_blocks_to_allocate > aic_phase462_free_blocks:
            self._aic_phase462_last_failure = {
                "request_id": request.request_id,
                "requested_blocks": int(num_blocks_to_allocate),
                "free_blocks": int(aic_phase462_free_blocks),
                "num_new_tokens": int(num_new_tokens),
                "num_tokens_need_slot": int(num_tokens_need_slot),
                "total_computed_tokens": int(total_computed_tokens),
            }
            # Cannot allocate new blocks
            return None
'''

ENGINE_FUTURE_REPLACEMENT = (
    ENGINE_FUTURE_ANCHOR
    + '''        self.scheduler._aic_phase462_log_future_complete(scheduler_output)
'''
)
ENGINE_SHUTDOWN_REPLACEMENT = '''    def shutdown(self):
        if self.scheduler:
            self.scheduler._aic_phase462_flush_queue_trace()
''' + ENGINE_BEGIN + ENGINE_END + '''        self.structured_output_manager.clear_backend()
'''


def _remove_block(text: str, begin: str, end: str) -> str:
    while begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + text[stop:]
    return text


def patch_async_source(text: str) -> str:
    text = restore_async_source(text)
    for anchor in (
        ASYNC_IMPORT_ANCHOR,
        ASYNC_INIT_ANCHOR,
        ASYNC_CALL_ANCHOR,
        ASYNC_BATCH_ANCHOR,
    ):
        if anchor not in text:
            raise ValueError(f"async_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(ASYNC_IMPORT_ANCHOR, ASYNC_IMPORT_ANCHOR + ASYNC_HELPERS, 1)
    text = text.replace(ASYNC_INIT_ANCHOR, ASYNC_INIT_REPLACEMENT, 1)
    text = text.replace(ASYNC_CALL_ANCHOR, ASYNC_CALL_REPLACEMENT, 1)
    return text.replace(ASYNC_BATCH_ANCHOR, ASYNC_BATCH_REPLACEMENT, 1)


def restore_async_source(text: str) -> str:
    text = _remove_block(text, ASYNC_BEGIN, ASYNC_END)
    text = text.replace(ASYNC_INIT_REPLACEMENT, ASYNC_INIT_ANCHOR)
    text = text.replace(ASYNC_CALL_REPLACEMENT, ASYNC_CALL_ANCHOR)
    return text.replace(ASYNC_BATCH_REPLACEMENT, ASYNC_BATCH_ANCHOR)


def patch_completion_source(text: str) -> str:
    text = restore_completion_source(text)
    for anchor in (COMPLETION_IMPORT_ANCHOR, COMPLETION_RENDER_ANCHOR):
        if anchor not in text:
            raise ValueError(f"completion_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(COMPLETION_IMPORT_ANCHOR, COMPLETION_IMPORT_REPLACEMENT, 1)
    return text.replace(COMPLETION_RENDER_ANCHOR, COMPLETION_RENDER_REPLACEMENT, 1)


def restore_completion_source(text: str) -> str:
    text = text.replace(COMPLETION_IMPORT_REPLACEMENT, COMPLETION_IMPORT_ANCHOR)
    return text.replace(COMPLETION_RENDER_REPLACEMENT, COMPLETION_RENDER_ANCHOR)


def patch_scheduler_source(text: str) -> str:
    text = restore_scheduler_source(text)
    anchors = (
        SCHED_MODULE_ANCHOR,
        SCHED_HELPER_ANCHOR,
        SCHED_START_ANCHOR,
        SCHED_CAPTURE_ANCHOR,
        SCHED_RETURN_ANCHOR,
        PRIORITY_ANCHOR,
        TAIL_ANCHOR,
        PREEMPT_WAITING_ANCHOR,
        WAITING_POP_ANCHOR,
        RUNNING_APPEND_ANCHOR,
        ENQUEUE_ANCHOR,
        FINISH_REMOVE_ANCHOR,
    )
    for anchor in anchors:
        if anchor not in text:
            raise ValueError(f"scheduler_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(SCHED_MODULE_ANCHOR, SCHED_MODULE_ANCHOR + SCHED_MODULE_HELPERS, 1)
    text = text.replace(SCHED_HELPER_ANCHOR, SCHED_CLASS_HELPERS + SCHED_HELPER_ANCHOR, 1)
    text = text.replace(SCHED_START_ANCHOR, SCHED_START_REPLACEMENT, 1)
    text = text.replace(SCHED_CAPTURE_ANCHOR, SCHED_CAPTURE_REPLACEMENT, 1)
    text = text.replace(SCHED_RETURN_ANCHOR, SCHED_RETURN_REPLACEMENT, 1)
    text = text.replace(PRIORITY_ANCHOR, PRIORITY_REPLACEMENT, 1)
    text = text.replace(TAIL_ANCHOR, TAIL_REPLACEMENT, 1)
    text = text.replace(PREEMPT_WAITING_ANCHOR, PREEMPT_WAITING_REPLACEMENT, 1)
    text = text.replace(WAITING_POP_ANCHOR, WAITING_POP_REPLACEMENT, 1)
    text = text.replace(RUNNING_APPEND_ANCHOR, RUNNING_APPEND_REPLACEMENT, 1)
    text = text.replace(ENQUEUE_ANCHOR, ENQUEUE_REPLACEMENT, 1)
    return text.replace(FINISH_REMOVE_ANCHOR, FINISH_REMOVE_REPLACEMENT, 1)


def restore_scheduler_source(text: str) -> str:
    text = _remove_block(text, SCHED_MODULE_BEGIN, SCHED_MODULE_END)
    text = _remove_block(text, SCHED_CLASS_BEGIN, SCHED_CLASS_END)
    replacements = (
        (SCHED_START_REPLACEMENT, SCHED_START_ANCHOR),
        (SCHED_CAPTURE_REPLACEMENT, SCHED_CAPTURE_ANCHOR),
        (SCHED_RETURN_REPLACEMENT, SCHED_RETURN_ANCHOR),
        (PRIORITY_REPLACEMENT, PRIORITY_ANCHOR),
        (TAIL_REPLACEMENT, TAIL_ANCHOR),
        (PREEMPT_WAITING_REPLACEMENT, PREEMPT_WAITING_ANCHOR),
        (WAITING_POP_REPLACEMENT, WAITING_POP_ANCHOR),
        (RUNNING_APPEND_REPLACEMENT, RUNNING_APPEND_ANCHOR),
        (ENQUEUE_REPLACEMENT, ENQUEUE_ANCHOR),
        (FINISH_REMOVE_REPLACEMENT, FINISH_REMOVE_ANCHOR),
    )
    for replacement, anchor in replacements:
        text = text.replace(replacement, anchor)
    return text


def patch_kv_source(text: str) -> str:
    text = restore_kv_source(text)
    if KV_FAILURE_ANCHOR not in text:
        raise ValueError("kv_anchor_not_found:allocation_failure")
    return text.replace(KV_FAILURE_ANCHOR, KV_FAILURE_REPLACEMENT, 1)


def restore_kv_source(text: str) -> str:
    return text.replace(KV_FAILURE_REPLACEMENT, KV_FAILURE_ANCHOR)


def patch_engine_source(text: str) -> str:
    text = restore_engine_source(text)
    for anchor in (ENGINE_FUTURE_ANCHOR, ENGINE_SHUTDOWN_ANCHOR):
        if anchor not in text:
            raise ValueError(f"engine_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(ENGINE_FUTURE_ANCHOR, ENGINE_FUTURE_REPLACEMENT, 1)
    return text.replace(ENGINE_SHUTDOWN_ANCHOR, ENGINE_SHUTDOWN_REPLACEMENT, 1)


def restore_engine_source(text: str) -> str:
    text = text.replace(ENGINE_FUTURE_REPLACEMENT, ENGINE_FUTURE_ANCHOR)
    return text.replace(ENGINE_SHUTDOWN_REPLACEMENT, ENGINE_SHUTDOWN_ANCHOR)


def scheduler_fixture_source() -> str:
    return (
        SCHED_MODULE_ANCHOR
        + "class Scheduler:\n"
        + SCHED_HELPER_ANCHOR
        + SCHED_START_ANCHOR
        + PRIORITY_ANCHOR
        + TAIL_ANCHOR
        + WAITING_POP_ANCHOR
        + RUNNING_APPEND_ANCHOR
        + SCHED_CAPTURE_ANCHOR
        + SCHED_RETURN_ANCHOR
        + "    def _preempt_request(self, request, timestamp):\n"
        + PREEMPT_WAITING_ANCHOR
        + "    def add_request(self, request):\n"
        + ENQUEUE_ANCHOR
        + "    def update_from_output(self):\n"
        + FINISH_REMOVE_ANCHOR
    )


def engine_fixture_source() -> str:
    return "class EngineCore:\n" + ENGINE_FUTURE_ANCHOR + ENGINE_SHUTDOWN_ANCHOR


def _backup_path(backup_dir: Path, target: Path) -> Path:
    return backup_dir / f"{target.name}.phase462.queue_order_observation.bak"


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
    parser.add_argument("--kv-target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true")
    action.add_argument("--restore", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    targets = (
        (args.async_target, patch_async_source, restore_async_source),
        (args.completion_target, patch_completion_source, restore_completion_source),
        (args.engine_target, patch_engine_source, restore_engine_source),
        (args.scheduler_target, patch_scheduler_source, restore_scheduler_source),
        (args.kv_target, patch_kv_source, restore_kv_source),
    )
    if args.apply:
        for target, patcher, _restorer in targets:
            _apply(target, args.backup_dir, patcher)
        return 0
    if args.restore:
        for target, _patcher, restorer in targets:
            _restore(target, args.backup_dir, restorer)
        return 0
    checks = (
        ASYNC_BEGIN in args.async_target.read_text(encoding="utf-8"),
        COMPLETION_BEGIN in args.completion_target.read_text(encoding="utf-8"),
        ENGINE_BEGIN in args.engine_target.read_text(encoding="utf-8"),
        SCHED_MODULE_BEGIN in args.scheduler_target.read_text(encoding="utf-8"),
        KV_FAILURE_REPLACEMENT in args.kv_target.read_text(encoding="utf-8"),
    )
    print(f"phase462_queue_order_patch={'applied' if all(checks) else 'missing'}")
    return 0 if all(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())

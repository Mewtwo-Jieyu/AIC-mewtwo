#!/usr/bin/env python3
"""Temporarily add buffered DP route/receive/admit hooks to vLLM 0.19."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

CLIENT_MODULE_ANCHOR = "import asyncio\n"
ENGINE_MODULE_ANCHOR = "import os\n"
CLIENT_HELPER_BEGIN = "# AIC PHASE462 DP ROUTE CLIENT HELPERS BEGIN\n"
CLIENT_HELPER_END = "# AIC PHASE462 DP ROUTE CLIENT HELPERS END\n"
ENGINE_HELPER_BEGIN = "# AIC PHASE462 DP ROUTE ENGINE HELPERS BEGIN\n"
ENGINE_HELPER_END = "# AIC PHASE462 DP ROUTE ENGINE HELPERS END\n"
ROUTE_BEGIN = "    # AIC PHASE462 DP ROUTE DECISION BEGIN\n"
ROUTE_END = "    # AIC PHASE462 DP ROUTE DECISION END\n"
CLIENT_FLUSH_BEGIN = "        # AIC PHASE462 DP CLIENT FLUSH BEGIN\n"
CLIENT_FLUSH_END = "        # AIC PHASE462 DP CLIENT FLUSH END\n"
RECEIVE_BEGIN = "        # AIC PHASE462 DP RECEIVE BEGIN\n"
RECEIVE_END = "        # AIC PHASE462 DP RECEIVE END\n"
ASYNC_ADMIT_BEGIN = "        # AIC PHASE462 DP ASYNC ADMIT BEGIN\n"
ASYNC_ADMIT_END = "        # AIC PHASE462 DP ASYNC ADMIT END\n"
SYNC_ADMIT_BEGIN = "            # AIC PHASE462 DP SYNC ADMIT BEGIN\n"
SYNC_ADMIT_END = "            # AIC PHASE462 DP SYNC ADMIT END\n"
FLUSH_BEGIN = "        # AIC PHASE462 DP FLUSH BEGIN\n"
FLUSH_END = "        # AIC PHASE462 DP FLUSH END\n"

ENGINE_ADD_ANCHOR = "        self.scheduler.add_request(request)\n"
ASYNC_SCHEDULE_ANCHOR = "        scheduler_output = self.scheduler.schedule()\n"
ASYNC_SCHEDULE_FOLLOW = (
    "        future = self.model_executor.execute_model(scheduler_output, non_block=True)\n"
)
SYNC_SCHEDULE_ANCHOR = "            scheduler_output = self.scheduler.schedule()\n"
SYNC_SCHEDULE_FOLLOW = "            with self.log_error_detail(scheduler_output):\n"
ENGINE_SHUTDOWN_ANCHOR = "        self.structured_output_manager.clear_backend()\n"
CLIENT_SHUTDOWN_ANCHOR = '''    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
'''

GET_CORE_ANCHOR = '''    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine
'''


def _buffer_helpers(*, component: str, marker_begin: str, marker_end: str) -> str:
    return marker_begin + f'''import atexit as _aic_phase462_dp_atexit
import json as _aic_phase462_dp_json
import os as _aic_phase462_dp_os
import time as _aic_phase462_dp_time

_aic_phase462_dp_trace_dir = _aic_phase462_dp_os.environ.get(
    "AIC_PHASE462_DP_ROUTE_TRACE_DIR"
)
_aic_phase462_dp_request_prefix = _aic_phase462_dp_os.environ.get(
    "AIC_PHASE462_DP_REQUEST_PREFIX", "phase462-dp-route"
)
_aic_phase462_dp_capacity = int(
    _aic_phase462_dp_os.environ.get("AIC_PHASE462_DP_EVENT_CAPACITY", "8192")
)
_aic_phase462_dp_buffer = [None] * _aic_phase462_dp_capacity
_aic_phase462_dp_count = 0
_aic_phase462_dp_flushed = False


def aic_phase462_dp_identity(request):
    value = None
    for field in ("external_req_id", "req_id", "request_id"):
        candidate = getattr(request, field, None)
        if candidate:
            value = str(candidate)
            break
    if value is None and isinstance(request, str):
        value = request
    if value is None:
        return None
    prefix, separator, random_suffix = value.rpartition("-")
    if (
        separator
        and len(random_suffix) == 8
        and all(char in "0123456789abcdef" for char in random_suffix)
    ):
        value = prefix
    if value.startswith("cmpl-") and value.endswith("-0"):
        value = value[5:-2]
    expected = _aic_phase462_dp_request_prefix + "-"
    if not value.startswith(expected):
        return None
    ordinal_text = value[len(expected):]
    if len(ordinal_text) != 6 or not ordinal_text.isdigit():
        raise RuntimeError(f"phase462_dp_bad_arrival_ordinal:{{value}}")
    return value, int(ordinal_text)


def aic_phase462_dp_emit(payload):
    if not _aic_phase462_dp_trace_dir:
        return
    global _aic_phase462_dp_count
    if _aic_phase462_dp_count >= _aic_phase462_dp_capacity:
        raise RuntimeError("phase462_dp_trace_buffer_exhausted")
    row = dict(payload)
    row.setdefault("schema", "phase462_dp_route_observation_v1")
    row.setdefault("component", "{component}")
    row.setdefault("event_seq", _aic_phase462_dp_count + 1)
    row.setdefault("ts_ns", _aic_phase462_dp_time.monotonic_ns())
    row.setdefault("wall_ts_ns", _aic_phase462_dp_time.time_ns())
    row.setdefault("pid", _aic_phase462_dp_os.getpid())
    _aic_phase462_dp_buffer[_aic_phase462_dp_count] = (
        _aic_phase462_dp_json.dumps(row, sort_keys=True) + "\\n"
    )
    _aic_phase462_dp_count += 1


def aic_phase462_dp_emit_request(payload, request):
    identity = aic_phase462_dp_identity(request)
    if identity is None:
        return
    trace_id, arrival_ordinal = identity
    aic_phase462_dp_emit(
        {{"trace_id": trace_id, "arrival_ordinal": arrival_ordinal, **payload}}
    )


def aic_phase462_dp_flush():
    global _aic_phase462_dp_flushed
    if not _aic_phase462_dp_trace_dir or _aic_phase462_dp_flushed:
        return
    _aic_phase462_dp_os.makedirs(_aic_phase462_dp_trace_dir, exist_ok=True)
    pid = _aic_phase462_dp_os.getpid()
    final_path = _aic_phase462_dp_os.path.join(
        _aic_phase462_dp_trace_dir, f"{component}-{{pid}}.jsonl"
    )
    temp_path = final_path + ".tmp"
    footer = {{
        "schema": "phase462_dp_route_observation_v1",
        "component": "{component}",
        "kind": "trace_footer",
        "event_count": _aic_phase462_dp_count,
        "buffer_capacity": _aic_phase462_dp_capacity,
        "flush_complete": True,
        "event_seq": _aic_phase462_dp_count + 1,
        "ts_ns": _aic_phase462_dp_time.monotonic_ns(),
        "wall_ts_ns": _aic_phase462_dp_time.time_ns(),
        "pid": pid,
    }}
    with open(temp_path, "w", encoding="utf-8") as output:
        output.writelines(
            line
            for line in _aic_phase462_dp_buffer[:_aic_phase462_dp_count]
            if line is not None
        )
        output.write(_aic_phase462_dp_json.dumps(footer, sort_keys=True) + "\\n")
        output.flush()
        _aic_phase462_dp_os.fsync(output.fileno())
    _aic_phase462_dp_os.replace(temp_path, final_path)
    _aic_phase462_dp_flushed = True


_aic_phase462_dp_atexit.register(aic_phase462_dp_flush)

''' + marker_end


CLIENT_HELPERS = _buffer_helpers(
    component="route",
    marker_begin=CLIENT_HELPER_BEGIN,
    marker_end=CLIENT_HELPER_END,
)

ENGINE_HELPERS = _buffer_helpers(
    component="engine",
    marker_begin=ENGINE_HELPER_BEGIN,
    marker_end=ENGINE_HELPER_END,
).removesuffix(ENGINE_HELPER_END) + '''_aic_phase462_dp_admitted_ordinals = set()


def aic_phase462_dp_rank(engine_core):
    return int(
        getattr(
            engine_core,
            "dp_rank",
            getattr(
                engine_core.vllm_config.parallel_config,
                "data_parallel_index",
                0,
            )
            or 0,
        )
    )


def aic_phase462_dp_log_admits(engine_core, scheduler_output):
    schedule_seq = int(getattr(engine_core, "_aic_phase462_dp_schedule_seq", 0)) + 1
    engine_core._aic_phase462_dp_schedule_seq = schedule_seq
    dp_rank = aic_phase462_dp_rank(engine_core)
    for request in scheduler_output.scheduled_new_reqs:
        identity = aic_phase462_dp_identity(request)
        if identity is None:
            continue
        trace_id, arrival_ordinal = identity
        if arrival_ordinal in _aic_phase462_dp_admitted_ordinals:
            raise RuntimeError(f"phase462_dp_duplicate_admit:{{arrival_ordinal}}")
        _aic_phase462_dp_admitted_ordinals.add(arrival_ordinal)
        aic_phase462_dp_emit(
            {
                "kind": "admit",
                "trace_id": trace_id,
                "arrival_ordinal": arrival_ordinal,
                "dp_rank": dp_rank,
                "schedule_seq": schedule_seq,
            }
        )


''' + ENGINE_HELPER_END

ROUTE_REPLACEMENT = ROUTE_BEGIN + '''    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        aic_phase462_dp_pre_counts = [
            [int(waiting), int(running)] for waiting, running in self.lb_engines
        ]
        aic_phase462_dp_scores = [
            {
                "rank": rank,
                "waiting": waiting,
                "running": running,
                "score": waiting * 4 + running,
            }
            for rank, (waiting, running) in enumerate(aic_phase462_dp_pre_counts)
        ]
        aic_phase462_dp_reason = "fixed"
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            aic_phase462_dp_reason = "load_balance"
            current_counts = self.lb_engines
            # TODO use P2C alg for larger DP sizes
            num_engines = len(current_counts)
            min_score = sys.maxsize
            eng_index = 0
            for i in range(num_engines):
                # Start from client_index to help with balancing when engines
                # are empty.
                idx = (self.eng_start_index + i) % num_engines
                waiting, running = current_counts[idx]
                score = waiting * 4 + running
                if score < min_score:
                    min_score = score
                    eng_index = idx
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count

        aic_phase462_dp_emit_request(
            {
                "kind": "route",
                "route_reason": aic_phase462_dp_reason,
                "chosen_rank": int(eng_index),
                "score_snapshot": aic_phase462_dp_scores,
                "eng_start_index": int(self.eng_start_index),
                "client_count": int(self.client_count),
            },
            request,
        )
        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine
''' + ROUTE_END


def _remove_block(text: str, begin: str, end: str) -> str:
    while begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + text[stop:]
    return text


def restore_core_client_source(text: str) -> str:
    text = _remove_block(text, CLIENT_HELPER_BEGIN, CLIENT_HELPER_END)
    text = _remove_block(text, CLIENT_FLUSH_BEGIN, CLIENT_FLUSH_END)
    if ROUTE_BEGIN in text:
        start = text.index(ROUTE_BEGIN)
        stop = text.index(ROUTE_END, start) + len(ROUTE_END)
        text = text[:start] + GET_CORE_ANCHOR + text[stop:]
    return text


def patch_core_client_source(text: str) -> str:
    clean = restore_core_client_source(text)
    if CLIENT_MODULE_ANCHOR not in clean:
        raise ValueError("core_client_module_anchor_not_found")
    if GET_CORE_ANCHOR not in clean:
        raise ValueError("core_client_route_anchor_not_found")
    if clean.count(CLIENT_SHUTDOWN_ANCHOR) != 1:
        raise ValueError(
            f"core_client_shutdown_anchor_count:{clean.count(CLIENT_SHUTDOWN_ANCHOR)}"
        )
    return clean.replace(
        CLIENT_MODULE_ANCHOR, CLIENT_MODULE_ANCHOR + CLIENT_HELPERS, 1
    ).replace(GET_CORE_ANCHOR, ROUTE_REPLACEMENT, 1).replace(
        CLIENT_SHUTDOWN_ANCHOR,
        CLIENT_SHUTDOWN_ANCHOR
        + CLIENT_FLUSH_BEGIN
        + "        aic_phase462_dp_flush()\n"
        + CLIENT_FLUSH_END,
        1,
    )


def restore_engine_source(text: str) -> str:
    text = _remove_block(text, ENGINE_HELPER_BEGIN, ENGINE_HELPER_END)
    for begin, end in (
        (RECEIVE_BEGIN, RECEIVE_END),
        (ASYNC_ADMIT_BEGIN, ASYNC_ADMIT_END),
        (SYNC_ADMIT_BEGIN, SYNC_ADMIT_END),
        (FLUSH_BEGIN, FLUSH_END),
    ):
        text = _remove_block(text, begin, end)
    return text


def patch_engine_source(text: str) -> str:
    clean = restore_engine_source(text)
    async_pair = ASYNC_SCHEDULE_ANCHOR + ASYNC_SCHEDULE_FOLLOW
    sync_pair = SYNC_SCHEDULE_ANCHOR + SYNC_SCHEDULE_FOLLOW
    for name, anchor in (
        ("module", ENGINE_MODULE_ANCHOR),
        ("receive", ENGINE_ADD_ANCHOR),
        ("async_schedule", async_pair),
        ("sync_schedule", sync_pair),
        ("shutdown", ENGINE_SHUTDOWN_ANCHOR),
    ):
        if clean.count(anchor) != 1:
            raise ValueError(f"engine_{name}_anchor_count:{clean.count(anchor)}")
    text = clean.replace(ENGINE_MODULE_ANCHOR, ENGINE_MODULE_ANCHOR + ENGINE_HELPERS, 1)
    text = text.replace(
        ENGINE_ADD_ANCHOR,
        RECEIVE_BEGIN
        + '''        aic_phase462_dp_emit_request(
            {"kind": "receive", "dp_rank": aic_phase462_dp_rank(self)},
            request,
        )
'''
        + RECEIVE_END
        + ENGINE_ADD_ANCHOR,
        1,
    )
    text = text.replace(
        async_pair,
        ASYNC_SCHEDULE_ANCHOR
        + ASYNC_ADMIT_BEGIN
        + "        aic_phase462_dp_log_admits(self, scheduler_output)\n"
        + ASYNC_ADMIT_END
        + ASYNC_SCHEDULE_FOLLOW,
        1,
    )
    text = text.replace(
        sync_pair,
        SYNC_SCHEDULE_ANCHOR
        + SYNC_ADMIT_BEGIN
        + "            aic_phase462_dp_log_admits(self, scheduler_output)\n"
        + SYNC_ADMIT_END
        + SYNC_SCHEDULE_FOLLOW,
        1,
    )
    return text.replace(
        ENGINE_SHUTDOWN_ANCHOR,
        FLUSH_BEGIN
        + "        aic_phase462_dp_flush()\n"
        + FLUSH_END
        + ENGINE_SHUTDOWN_ANCHOR,
        1,
    )


def core_client_fixture_source() -> str:
    return CLIENT_MODULE_ANCHOR + CLIENT_SHUTDOWN_ANCHOR + "        pass\n\n" + GET_CORE_ANCHOR


def engine_fixture_source() -> str:
    return (
        ENGINE_MODULE_ANCHOR
        + "class EngineCore:\n"
        + "    def add_request(self, request):\n"
        + ENGINE_ADD_ANCHOR
        + "\n    def step_with_batch_queue(self):\n"
        + ASYNC_SCHEDULE_ANCHOR
        + ASYNC_SCHEDULE_FOLLOW
        + "\n    def step(self):\n"
        + SYNC_SCHEDULE_ANCHOR
        + SYNC_SCHEDULE_FOLLOW
        + "\n    def shutdown(self):\n"
        + ENGINE_SHUTDOWN_ANCHOR
    )


def _backup_path(backup_dir: Path, target: Path) -> Path:
    return backup_dir / target.name


def _apply(target: Path, backup_dir: Path, transform) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = _backup_path(backup_dir, target)
    if not backup.exists():
        shutil.copy2(target, backup)
    target.write_text(transform(target.read_text(encoding="utf-8")), encoding="utf-8")


def _restore(target: Path, backup_dir: Path, transform) -> None:
    backup = _backup_path(backup_dir, target)
    if backup.exists():
        shutil.copy2(backup, target)
    else:
        target.write_text(transform(target.read_text(encoding="utf-8")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--core-client-target", type=Path, required=True)
    parser.add_argument("--engine-target", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true")
    action.add_argument("--restore", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.apply:
        _apply(args.core_client_target, args.backup_dir, patch_core_client_source)
        _apply(args.engine_target, args.backup_dir, patch_engine_source)
    elif args.restore:
        _restore(args.core_client_target, args.backup_dir, restore_core_client_source)
        _restore(args.engine_target, args.backup_dir, restore_engine_source)
    else:
        client = args.core_client_target.read_text(encoding="utf-8")
        engine = args.engine_target.read_text(encoding="utf-8")
        if CLIENT_HELPER_BEGIN not in client or ROUTE_BEGIN not in client:
            raise SystemExit("phase462_dp_core_client_patch_missing")
        if ENGINE_HELPER_BEGIN not in engine or RECEIVE_BEGIN not in engine:
            raise SystemExit("phase462_dp_engine_patch_missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

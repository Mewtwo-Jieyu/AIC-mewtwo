#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


CORE_HELPER_BEGIN = "    # AIC PHASE452 DUAL OBSERVATION CORE HELPERS BEGIN\n"
CORE_HELPER_END = "    # AIC PHASE452 DUAL OBSERVATION CORE HELPERS END\n"
SCHED_HELPER_BEGIN = "    # AIC PHASE452 DUAL OBSERVATION SCHED HELPERS BEGIN\n"
SCHED_HELPER_END = "    # AIC PHASE452 DUAL OBSERVATION SCHED HELPERS END\n"

ADD_REQUEST_ANCHOR = (
    "    async def add_request_async(self, request: EngineCoreRequest) -> None:\n"
)

STATS_ANCHOR = """                        sliced_counts = counts[count_slice]
                        self.lb_engines = sliced_counts
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )
"""

STATS_REPLACEMENT = """                        sliced_counts = counts[count_slice]
                        aic_phase452_before_counts = self._aic_phase452_counts_snapshot(
                            getattr(self, "lb_engines", None)
                        )
                        self.lb_engines = sliced_counts
                        self._aic_phase452_write_observation(
                            {
                                "kind": "stats_overwrite",
                                "before_counts": aic_phase452_before_counts,
                                "after_counts": self._aic_phase452_counts_snapshot(
                                    self.lb_engines
                                ),
                                "count_slice": [count_slice.start, count_slice.stop],
                                "wave": self.current_wave,
                                "engines_running": self.engines_running,
                            }
                        )
                        logger.debug(
                            "Received counts: %s (%s)", sliced_counts, count_slice
                        )
"""

GET_CORE_ANCHOR = """    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
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
"""

GET_CORE_REPLACEMENT = """    def get_core_engine_for_request(self, request: EngineCoreRequest) -> EngineIdentity:
        # Engines are in rank order.
        aic_phase452_pre_counts = self._aic_phase452_counts_snapshot(self.lb_engines)
        aic_phase452_route_reason = "fixed"
        aic_phase452_min_score = None
        if (eng_index := request.data_parallel_rank) is None and (
            eng_index := get_late_interaction_engine_index(
                request.pooling_params, len(self.core_engines)
            )
        ) is None:
            aic_phase452_route_reason = "load_balance"
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
            aic_phase452_min_score = min_score
            # Increment local waiting count for better balancing between stats
            # updates from the coordinator (which happen every 100ms).
            current_counts[eng_index][0] += self.client_count

        aic_phase452_post_counts = self._aic_phase452_counts_snapshot(self.lb_engines)
        self._aic_phase452_write_observation(
            {
                "kind": "route",
                "request_id": getattr(request, "request_id", None),
                "route_reason": aic_phase452_route_reason,
                "chosen_engine_index": int(eng_index),
                "pre_counts": aic_phase452_pre_counts,
                "post_counts": aic_phase452_post_counts,
                "min_score": aic_phase452_min_score,
                "current_wave": getattr(request, "current_wave", None),
                "request_client_index": getattr(request, "client_index", None),
            }
        )
        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen for this request, to handle aborts.
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine
"""

PRIORITY_VICTIM_ANCHOR = """                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
"""

PRIORITY_VICTIM_REPLACEMENT = """                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        aic_phase452_victim_pos = self.running.index(preempted_req)
                        aic_phase452_victim_policy = "priority"
                        self.running.remove(preempted_req)
"""

TAIL_VICTIM_ANCHOR = """                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
"""

TAIL_VICTIM_REPLACEMENT = """                    else:
                        aic_phase452_victim_pos = len(self.running) - 1
                        aic_phase452_victim_policy = "tail"
                        preempted_req = self.running.pop()

                    self._aic_phase452_log_preempt_decision(
                        preempted_req,
                        scheduled_timestamp,
                        trigger_request=request,
                        victim_position=aic_phase452_victim_pos,
                        victim_policy=aic_phase452_victim_policy,
                    )
                    self._preempt_request(preempted_req, scheduled_timestamp)
                    self._aic_phase452_log_preempt_after_free(
                        preempted_req,
                        scheduled_timestamp,
                        victim_position=aic_phase452_victim_pos,
                        victim_policy=aic_phase452_victim_policy,
                    )
                    preempted_reqs.append(preempted_req)
"""

RUNNING_APPEND_ANCHOR = """                self.running.append(request)
                if self.log_stats:
"""

RUNNING_APPEND_REPLACEMENT = """                self.running.append(request)
                if request.status == RequestStatus.PREEMPTED:
                    self._aic_phase452_log_victim_reschedule(
                        request,
                        scheduled_timestamp,
                        num_new_tokens=num_new_tokens,
                        num_computed_tokens=num_computed_tokens,
                    )
                if self.log_stats:
"""

PREEMPT_HELPER_ANCHOR = """    def _preempt_request(self, request: Request, timestamp: float) -> None:
"""


def _core_helper_block() -> str:
    return (
        CORE_HELPER_BEGIN
        + "    def _aic_phase452_observation_path(self):\n"
        + "        import os\n"
        + "        return os.environ.get(\"AIC_PHASE452_OBS_JSONL\")\n"
        + "\n"
        + "    def _aic_phase452_counts_snapshot(self, counts):\n"
        + "        if counts is None:\n"
        + "            return None\n"
        + "        snapshot = []\n"
        + "        try:\n"
        + "            iterable = list(counts)\n"
        + "        except Exception:\n"
        + "            return str(counts)\n"
        + "        for item in iterable:\n"
        + "            try:\n"
        + "                row = list(item)\n"
        + "            except Exception:\n"
        + "                snapshot.append(str(item))\n"
        + "                continue\n"
        + "            if len(row) >= 2:\n"
        + "                snapshot.append([int(row[0]), int(row[1])])\n"
        + "            else:\n"
        + "                snapshot.append([int(value) for value in row])\n"
        + "        return snapshot\n"
        + "\n"
        + "    def _aic_phase452_write_observation(self, payload):\n"
        + "        path = self._aic_phase452_observation_path()\n"
        + "        if not path:\n"
        + "            return\n"
        + "        import json\n"
        + "        import os\n"
        + "        import time\n"
        + "        from pathlib import Path\n"
        + "        record = dict(payload)\n"
        + "        record.setdefault(\"schema\", \"phase452_dual_observation_v1\")\n"
        + "        record.setdefault(\"ts_ns\", time.time_ns())\n"
        + "        record.setdefault(\"pid\", os.getpid())\n"
        + "        record.setdefault(\"client_index\", getattr(self, \"client_index\", None))\n"
        + "        record.setdefault(\"client_count\", getattr(self, \"client_count\", None))\n"
        + "        path_obj = Path(path)\n"
        + "        path_obj.parent.mkdir(parents=True, exist_ok=True)\n"
        + "        data = (json.dumps(record, sort_keys=True) + \"\\n\").encode(\"utf-8\")\n"
        + "        fd = os.open(str(path_obj), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)\n"
        + "        try:\n"
        + "            os.write(fd, data)\n"
        + "        finally:\n"
        + "            os.close(fd)\n"
        + "\n"
        + CORE_HELPER_END
    )


def _scheduler_helper_block() -> str:
    return (
        SCHED_HELPER_BEGIN
        + "    def _aic_phase452_observation_path(self):\n"
        + "        import os\n"
        + "        return os.environ.get(\"AIC_PHASE452_OBS_JSONL\")\n"
        + "\n"
        + "    def _aic_phase452_write_observation(self, payload):\n"
        + "        path = self._aic_phase452_observation_path()\n"
        + "        if not path:\n"
        + "            return\n"
        + "        import json\n"
        + "        import os\n"
        + "        import time\n"
        + "        from pathlib import Path\n"
        + "        record = dict(payload)\n"
        + "        record.setdefault(\"schema\", \"phase452_dual_observation_v1\")\n"
        + "        record.setdefault(\"ts_ns\", time.time_ns())\n"
        + "        record.setdefault(\"pid\", os.getpid())\n"
        + "        record.setdefault(\"scheduler_id\", hex(id(self)))\n"
        + "        path_obj = Path(path)\n"
        + "        path_obj.parent.mkdir(parents=True, exist_ok=True)\n"
        + "        data = (json.dumps(record, sort_keys=True) + \"\\n\").encode(\"utf-8\")\n"
        + "        fd = os.open(str(path_obj), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)\n"
        + "        try:\n"
        + "            os.write(fd, data)\n"
        + "        finally:\n"
        + "            os.close(fd)\n"
        + "\n"
        + "    def _aic_phase452_free_blocks(self):\n"
        + "        try:\n"
        + "            return int(self.kv_cache_manager.block_pool.get_num_free_blocks())\n"
        + "        except Exception:\n"
        + "            return None\n"
        + "\n"
        + "    def _aic_phase452_queue_len(self, queue):\n"
        + "        try:\n"
        + "            return int(len(queue))\n"
        + "        except Exception:\n"
        + "            return None\n"
        + "\n"
        + "    def _aic_phase452_request_snapshot(self, request):\n"
        + "        def maybe_int(value):\n"
        + "            try:\n"
        + "                return int(value)\n"
        + "            except Exception:\n"
        + "                return None\n"
        + "        status = getattr(request, \"status\", None)\n"
        + "        return {\n"
        + "            \"request_id\": getattr(request, \"request_id\", None),\n"
        + "            \"status\": str(status) if status is not None else None,\n"
        + "            \"num_computed_tokens\": maybe_int(getattr(request, \"num_computed_tokens\", None)),\n"
        + "            \"num_tokens\": maybe_int(getattr(request, \"num_tokens\", None)),\n"
        + "            \"num_prompt_tokens\": maybe_int(getattr(request, \"num_prompt_tokens\", None)),\n"
        + "            \"num_preemptions\": maybe_int(getattr(request, \"num_preemptions\", None)),\n"
        + "        }\n"
        + "\n"
        + "    def _aic_phase452_log_preempt_decision(\n"
        + "        self,\n"
        + "        victim,\n"
        + "        timestamp,\n"
        + "        *,\n"
        + "        trigger_request,\n"
        + "        victim_position,\n"
        + "        victim_policy,\n"
        + "    ):\n"
        + "        self._aic_phase452_write_observation(\n"
        + "            {\n"
        + "                \"kind\": \"preempt_decision\",\n"
        + "                \"scheduler_timestamp\": timestamp,\n"
        + "                \"victim\": self._aic_phase452_request_snapshot(victim),\n"
        + "                \"trigger_request\": self._aic_phase452_request_snapshot(trigger_request),\n"
        + "                \"victim_position\": int(victim_position),\n"
        + "                \"victim_policy\": victim_policy,\n"
        + "                \"running_count_after_pop\": len(self.running),\n"
        + "                \"waiting_count\": self._aic_phase452_queue_len(self.waiting),\n"
        + "                \"free_blocks_before_free\": self._aic_phase452_free_blocks(),\n"
        + "            }\n"
        + "        )\n"
        + "\n"
        + "    def _aic_phase452_log_preempt_after_free(\n"
        + "        self, victim, timestamp, *, victim_position, victim_policy\n"
        + "    ):\n"
        + "        self._aic_phase452_write_observation(\n"
        + "            {\n"
        + "                \"kind\": \"preempt_after_free\",\n"
        + "                \"scheduler_timestamp\": timestamp,\n"
        + "                \"victim\": self._aic_phase452_request_snapshot(victim),\n"
        + "                \"victim_position\": int(victim_position),\n"
        + "                \"victim_policy\": victim_policy,\n"
        + "                \"running_count\": len(self.running),\n"
        + "                \"waiting_count\": self._aic_phase452_queue_len(self.waiting),\n"
        + "                \"free_blocks_after_free\": self._aic_phase452_free_blocks(),\n"
        + "            }\n"
        + "        )\n"
        + "\n"
        + "    def _aic_phase452_log_victim_reschedule(\n"
        + "        self, request, timestamp, *, num_new_tokens, num_computed_tokens\n"
        + "    ):\n"
        + "        self._aic_phase452_write_observation(\n"
        + "            {\n"
        + "                \"kind\": \"victim_reschedule\",\n"
        + "                \"scheduler_timestamp\": timestamp,\n"
        + "                \"request\": self._aic_phase452_request_snapshot(request),\n"
        + "                \"num_new_tokens\": int(num_new_tokens),\n"
        + "                \"num_computed_tokens_for_schedule\": int(num_computed_tokens),\n"
        + "                \"running_count\": len(self.running),\n"
        + "                \"waiting_count\": self._aic_phase452_queue_len(self.waiting),\n"
        + "                \"free_blocks_after_alloc\": self._aic_phase452_free_blocks(),\n"
        + "            }\n"
        + "        )\n"
        + "\n"
        + SCHED_HELPER_END
    )


def _remove_marked_block(text: str, begin: str, end: str) -> str:
    while begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + text[stop:]
    return text


def restore_core_source(text: str) -> str:
    text = _remove_marked_block(text, CORE_HELPER_BEGIN, CORE_HELPER_END)
    text = text.replace(STATS_REPLACEMENT, STATS_ANCHOR)
    text = text.replace(GET_CORE_REPLACEMENT, GET_CORE_ANCHOR)
    return text


def patch_core_source(text: str) -> str:
    if CORE_HELPER_BEGIN in text and STATS_REPLACEMENT in text:
        return text
    text = restore_core_source(text)
    for anchor in (ADD_REQUEST_ANCHOR, STATS_ANCHOR, GET_CORE_ANCHOR):
        if anchor not in text:
            raise ValueError(f"core_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(ADD_REQUEST_ANCHOR, _core_helper_block() + ADD_REQUEST_ANCHOR, 1)
    text = text.replace(STATS_ANCHOR, STATS_REPLACEMENT, 1)
    text = text.replace(GET_CORE_ANCHOR, GET_CORE_REPLACEMENT, 1)
    return text


def restore_scheduler_source(text: str) -> str:
    text = _remove_marked_block(text, SCHED_HELPER_BEGIN, SCHED_HELPER_END)
    text = text.replace(PRIORITY_VICTIM_REPLACEMENT, PRIORITY_VICTIM_ANCHOR)
    text = text.replace(TAIL_VICTIM_REPLACEMENT, TAIL_VICTIM_ANCHOR)
    text = text.replace(RUNNING_APPEND_REPLACEMENT, RUNNING_APPEND_ANCHOR)
    return text


def patch_scheduler_source(text: str) -> str:
    if SCHED_HELPER_BEGIN in text and TAIL_VICTIM_REPLACEMENT in text:
        return text
    text = restore_scheduler_source(text)
    for anchor in (
        PREEMPT_HELPER_ANCHOR,
        PRIORITY_VICTIM_ANCHOR,
        TAIL_VICTIM_ANCHOR,
        RUNNING_APPEND_ANCHOR,
    ):
        if anchor not in text:
            raise ValueError(f"scheduler_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(
        PREEMPT_HELPER_ANCHOR, _scheduler_helper_block() + PREEMPT_HELPER_ANCHOR, 1
    )
    text = text.replace(PRIORITY_VICTIM_ANCHOR, PRIORITY_VICTIM_REPLACEMENT, 1)
    text = text.replace(TAIL_VICTIM_ANCHOR, TAIL_VICTIM_REPLACEMENT, 1)
    text = text.replace(RUNNING_APPEND_ANCHOR, RUNNING_APPEND_REPLACEMENT, 1)
    return text


def _backup_path(backup_dir: Path, target: Path) -> Path:
    return backup_dir / f"{target.name}.phase452.dual_observation.bak"


def _apply_one(target: Path, backup_dir: Path, patcher) -> None:
    original = target.read_text(encoding="utf-8")
    patched = patcher(original)
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup = _backup_path(backup_dir, target)
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    target.write_text(patched, encoding="utf-8")


def _restore_one(target: Path, backup_dir: Path, restorer) -> None:
    backup = _backup_path(backup_dir, target)
    if backup.exists():
        target.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
        return
    target.write_text(restorer(target.read_text(encoding="utf-8")), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch vLLM core_client and scheduler for Phase452 dual observation."
    )
    parser.add_argument("--core-client-target", required=True, type=Path)
    parser.add_argument("--scheduler-target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true")
    group.add_argument("--restore", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.apply:
        _apply_one(args.core_client_target, args.backup_dir, patch_core_source)
        _apply_one(args.scheduler_target, args.backup_dir, patch_scheduler_source)
        return 0
    if args.restore:
        _restore_one(args.core_client_target, args.backup_dir, restore_core_source)
        _restore_one(args.scheduler_target, args.backup_dir, restore_scheduler_source)
        return 0

    core = args.core_client_target.read_text(encoding="utf-8")
    scheduler = args.scheduler_target.read_text(encoding="utf-8")
    ok = (
        CORE_HELPER_BEGIN in core
        and STATS_REPLACEMENT in core
        and SCHED_HELPER_BEGIN in scheduler
        and TAIL_VICTIM_REPLACEMENT in scheduler
    )
    print(f"phase452_dual_observation_patch={'applied' if ok else 'missing'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

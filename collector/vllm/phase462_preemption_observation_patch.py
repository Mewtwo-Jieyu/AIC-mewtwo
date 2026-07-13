#!/usr/bin/env python3
"""Temporarily add logging-only preemption decision hooks to vLLM 0.19."""

from __future__ import annotations

import argparse
from pathlib import Path


SCHED_BEGIN = "    # AIC PHASE462 PREEMPTION OBSERVATION BEGIN\n"
SCHED_END = "    # AIC PHASE462 PREEMPTION OBSERVATION END\n"
KV_BEGIN = "    # AIC PHASE462 ALLOCATION OBSERVATION BEGIN\n"
KV_END = "    # AIC PHASE462 ALLOCATION OBSERVATION END\n"

SCHED_HELPER_ANCHOR = "    def _preempt_request(self, request: Request, timestamp: float) -> None:\n"
PRIORITY_ANCHOR = """                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
"""
PRIORITY_REPLACEMENT = """                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        aic_phase462_victim_position = self.running.index(preempted_req)
                        aic_phase462_victim_policy = "priority"
                        self.running.remove(preempted_req)
"""
TAIL_ANCHOR = """                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
"""
TAIL_REPLACEMENT = """                    else:
                        aic_phase462_victim_position = len(self.running) - 1
                        aic_phase462_victim_policy = "tail"
                        preempted_req = self.running.pop()

                    self._aic_phase462_log_preempt_decision(
                        trigger_request=request,
                        victim=preempted_req,
                        victim_position=aic_phase462_victim_position,
                        victim_policy=aic_phase462_victim_policy,
                        scheduler_timestamp=scheduled_timestamp,
                    )
                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
"""

KV_HELPER_ANCHOR = "    def allocate_slots(\n"
KV_FAILURE_ANCHOR = """        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None
"""
KV_FAILURE_REPLACEMENT = """        aic_phase462_free_blocks = self.block_pool.get_num_free_blocks()
        if num_blocks_to_allocate > aic_phase462_free_blocks:
            self._aic_phase462_log_allocate_failure(
                request=request,
                requested_blocks=num_blocks_to_allocate,
                free_blocks=aic_phase462_free_blocks,
                num_new_tokens=num_new_tokens,
                num_tokens_need_slot=num_tokens_need_slot,
                total_computed_tokens=total_computed_tokens,
            )
            # Cannot allocate new blocks
            return None
"""


def _writer_helpers(begin: str, end: str, method_name: str) -> str:
    return (
        begin
        + f"    def {method_name}(self, payload):\n"
        + "        import json\n"
        + "        import os\n"
        + "        import time\n"
        + "        path = os.environ.get(\"AIC_PHASE462_PREEMPT_JSONL\")\n"
        + "        if not path:\n"
        + "            return\n"
        + "        record = dict(payload)\n"
        + "        record.setdefault(\"schema\", \"phase462_preemption_observation_v1\")\n"
        + "        record.setdefault(\"ts_ns\", time.time_ns())\n"
        + "        record.setdefault(\"pid\", os.getpid())\n"
        + "        data = (json.dumps(record, sort_keys=True) + \"\\n\").encode(\"utf-8\")\n"
        + "        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)\n"
        + "        try:\n"
        + "            os.write(fd, data)\n"
        + "        finally:\n"
        + "            os.close(fd)\n"
        + "\n"
        + end
    )


def _scheduler_helpers() -> str:
    writer = _writer_helpers(SCHED_BEGIN, "", "_aic_phase462_write_observation")
    return (
        writer
        + "    def _aic_phase462_log_preempt_decision(\n"
        + "        self, *, trigger_request, victim, victim_position, victim_policy, scheduler_timestamp\n"
        + "    ):\n"
        + "        self._aic_phase462_write_observation(\n"
        + "            {\n"
        + "                \"kind\": \"preempt_decision\",\n"
        + "                \"scheduler_timestamp\": scheduler_timestamp,\n"
        + "                \"trigger_request_id\": trigger_request.request_id,\n"
        + "                \"trigger_num_computed_tokens\": int(trigger_request.num_computed_tokens),\n"
        + "                \"trigger_num_tokens\": int(trigger_request.num_tokens),\n"
        + "                \"victim_request_id\": victim.request_id,\n"
        + "                \"victim_num_computed_tokens\": int(victim.num_computed_tokens),\n"
        + "                \"victim_num_tokens\": int(victim.num_tokens),\n"
        + "                \"victim_position\": int(victim_position),\n"
        + "                \"victim_policy\": victim_policy,\n"
        + "                \"running_count_after_pop\": len(self.running),\n"
        + "                \"waiting_count\": len(self.waiting),\n"
        + "                \"free_blocks_before_victim_free\": int(\n"
        + "                    self.kv_cache_manager.block_pool.get_num_free_blocks()\n"
        + "                ),\n"
        + "            }\n"
        + "        )\n"
        + "\n"
        + SCHED_END
    )


def _kv_helpers() -> str:
    writer = _writer_helpers(KV_BEGIN, "", "_aic_phase462_write_observation")
    return (
        writer
        + "    def _aic_phase462_log_allocate_failure(\n"
        + "        self, *, request, requested_blocks, free_blocks, num_new_tokens,\n"
        + "        num_tokens_need_slot, total_computed_tokens\n"
        + "    ):\n"
        + "        self._aic_phase462_write_observation(\n"
        + "            {\n"
        + "                \"kind\": \"allocate_failure\",\n"
        + "                \"trigger_request_id\": request.request_id,\n"
        + "                \"trigger_num_computed_tokens\": int(request.num_computed_tokens),\n"
        + "                \"trigger_num_tokens\": int(request.num_tokens),\n"
        + "                \"requested_blocks\": int(requested_blocks),\n"
        + "                \"free_blocks\": int(free_blocks),\n"
        + "                \"num_new_tokens\": int(num_new_tokens),\n"
        + "                \"num_tokens_need_slot\": int(num_tokens_need_slot),\n"
        + "                \"total_computed_tokens\": int(total_computed_tokens),\n"
        + "            }\n"
        + "        )\n"
        + "\n"
        + KV_END
    )


def _remove_block(text: str, begin: str, end: str) -> str:
    while begin in text:
        start = text.index(begin)
        stop = text.index(end, start) + len(end)
        text = text[:start] + text[stop:]
    return text


def restore_scheduler_source(text: str) -> str:
    text = _remove_block(text, SCHED_BEGIN, SCHED_END)
    text = text.replace(PRIORITY_REPLACEMENT, PRIORITY_ANCHOR)
    return text.replace(TAIL_REPLACEMENT, TAIL_ANCHOR)


def patch_scheduler_source(text: str) -> str:
    text = restore_scheduler_source(text)
    for anchor in (SCHED_HELPER_ANCHOR, PRIORITY_ANCHOR, TAIL_ANCHOR):
        if anchor not in text:
            raise ValueError(f"scheduler_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(SCHED_HELPER_ANCHOR, _scheduler_helpers() + SCHED_HELPER_ANCHOR, 1)
    text = text.replace(PRIORITY_ANCHOR, PRIORITY_REPLACEMENT, 1)
    return text.replace(TAIL_ANCHOR, TAIL_REPLACEMENT, 1)


def restore_kv_source(text: str) -> str:
    text = _remove_block(text, KV_BEGIN, KV_END)
    return text.replace(KV_FAILURE_REPLACEMENT, KV_FAILURE_ANCHOR)


def patch_kv_source(text: str) -> str:
    text = restore_kv_source(text)
    for anchor in (KV_HELPER_ANCHOR, KV_FAILURE_ANCHOR):
        if anchor not in text:
            raise ValueError(f"kv_anchor_not_found:{anchor.splitlines()[0]}")
    text = text.replace(KV_HELPER_ANCHOR, _kv_helpers() + KV_HELPER_ANCHOR, 1)
    return text.replace(KV_FAILURE_ANCHOR, KV_FAILURE_REPLACEMENT, 1)


def _backup_path(backup_dir: Path, target: Path) -> Path:
    return backup_dir / f"{target.name}.phase462.preemption_observation.bak"


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
    parser.add_argument("--scheduler-target", required=True, type=Path)
    parser.add_argument("--kv-target", required=True, type=Path)
    parser.add_argument("--backup-dir", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--apply", action="store_true")
    action.add_argument("--restore", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.apply:
        _apply(args.scheduler_target, args.backup_dir, patch_scheduler_source)
        _apply(args.kv_target, args.backup_dir, patch_kv_source)
        return 0
    if args.restore:
        _restore(args.scheduler_target, args.backup_dir, restore_scheduler_source)
        _restore(args.kv_target, args.backup_dir, restore_kv_source)
        return 0
    scheduler = args.scheduler_target.read_text(encoding="utf-8")
    kv = args.kv_target.read_text(encoding="utf-8")
    ok = SCHED_BEGIN in scheduler and KV_BEGIN in kv
    print(f"phase462_preemption_observation_patch={'applied' if ok else 'missing'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

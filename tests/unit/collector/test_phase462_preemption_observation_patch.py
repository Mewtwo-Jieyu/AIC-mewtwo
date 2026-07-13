#!/usr/bin/env python3
"""Tests for the Phase462 logging-only vLLM patch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "collector/vllm/phase462_preemption_observation_patch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_patch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scheduler_source(mod) -> str:
    return (
        "class Scheduler:\n"
        + mod.PRIORITY_ANCHOR
        + mod.TAIL_ANCHOR
        + mod.SCHED_HELPER_ANCHOR
        + "        pass\n"
    )


def _kv_source(mod) -> str:
    return (
        "class KVCacheManager:\n"
        + mod.KV_HELPER_ANCHOR
        + "        pass\n"
        + mod.KV_FAILURE_ANCHOR
    )


def test_scheduler_patch_is_reversible_and_logs_decision_fields() -> None:
    mod = _load_module()
    source = _scheduler_source(mod)

    patched = mod.patch_scheduler_source(source)

    assert "trigger_request_id" in patched
    assert "victim_request_id" in patched
    assert "victim_position" in patched
    assert "free_blocks_before_victim_free" in patched
    assert mod.restore_scheduler_source(patched) == source


def test_kv_patch_is_reversible_and_logs_exact_block_demand() -> None:
    mod = _load_module()
    source = _kv_source(mod)

    patched = mod.patch_kv_source(source)

    assert '"requested_blocks"' in patched
    assert '"free_blocks"' in patched
    assert "num_blocks_to_allocate" in patched
    assert mod.restore_kv_source(patched) == source


def test_patch_is_idempotent() -> None:
    mod = _load_module()
    scheduler = mod.patch_scheduler_source(_scheduler_source(mod))
    kv = mod.patch_kv_source(_kv_source(mod))

    assert mod.patch_scheduler_source(scheduler) == scheduler
    assert mod.patch_kv_source(kv) == kv

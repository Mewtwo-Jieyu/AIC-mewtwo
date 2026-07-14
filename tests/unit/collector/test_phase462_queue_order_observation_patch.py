#!/usr/bin/env python3
"""Tests for the Phase462 queue-order logging-only vLLM patch."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "collector/vllm/phase462_queue_order_observation_patch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_queue_patch", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tokenizer_patch_assigns_ordinals_at_microbatch_boundary() -> None:
    mod = _load_module()
    source = (
        mod.ASYNC_IMPORT_ANCHOR
        + mod.ASYNC_INIT_ANCHOR
        + mod.ASYNC_CALL_ANCHOR
        + mod.ASYNC_BATCH_ANCHOR
    )

    patched = mod.patch_async_source(source)

    assert '"kind": "arrival_map"' in patched
    assert '"arrival_ordinals": arrival_ordinals' in patched
    assert '"batch_positions": list(range(len(prompts)))' in patched
    assert "self._aic_phase462_arrival_ordinal" in patched
    assert "[None] * _aic_phase462_queue_capacity" in patched
    assert mod.restore_async_source(patched) == source


def test_scheduler_patch_records_boundaries_queue_events_and_request_phase() -> None:
    mod = _load_module()
    source = mod.scheduler_fixture_source()

    patched = mod.patch_scheduler_source(source)

    for field in (
        '"kind": "scheduler_input"',
        '"kind": "scheduler_output"',
        '"queue_transition"',
        '"running_order"',
        '"waiting_order"',
        '"num_computed_tokens"',
        '"num_output_placeholders"',
        '"block_counts"',
        '"preempt_decision"',
        "trigger_position=",
    ):
        assert field in patched
    assert "[None] * _aic_phase462_queue_capacity" in patched
    assert mod.restore_scheduler_source(patched) == source


def test_engine_patch_records_future_completion_and_flushes_on_shutdown() -> None:
    mod = _load_module()
    source = mod.engine_fixture_source()

    patched = mod.patch_engine_source(source)

    assert "_aic_phase462_log_future_complete(scheduler_output)" in patched
    assert "_aic_phase462_flush_queue_trace()" in patched
    assert mod.restore_engine_source(patched) == source


def test_completion_patch_carries_external_request_id_into_tokenizer_context() -> None:
    mod = _load_module()
    source = mod.COMPLETION_IMPORT_ANCHOR + mod.COMPLETION_RENDER_ANCHOR

    patched = mod.patch_completion_source(source)

    assert 'headers.get("X-Request-Id")' in patched
    assert "aic_phase462_queue_trace_id.set" in patched
    assert mod.restore_completion_source(patched) == source


def test_all_queue_order_patches_are_idempotent() -> None:
    mod = _load_module()
    async_source = (
        mod.ASYNC_IMPORT_ANCHOR
        + mod.ASYNC_INIT_ANCHOR
        + mod.ASYNC_CALL_ANCHOR
        + mod.ASYNC_BATCH_ANCHOR
    )
    completion_source = mod.COMPLETION_IMPORT_ANCHOR + mod.COMPLETION_RENDER_ANCHOR
    scheduler_source = mod.scheduler_fixture_source()
    engine_source = mod.engine_fixture_source()

    assert mod.patch_async_source(mod.patch_async_source(async_source)) == mod.patch_async_source(async_source)
    assert mod.patch_completion_source(
        mod.patch_completion_source(completion_source)
    ) == mod.patch_completion_source(completion_source)
    assert mod.patch_scheduler_source(
        mod.patch_scheduler_source(scheduler_source)
    ) == mod.patch_scheduler_source(scheduler_source)
    assert mod.patch_engine_source(mod.patch_engine_source(engine_source)) == mod.patch_engine_source(engine_source)


def test_patcher_cli_is_runnable_from_its_file_path() -> None:
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr

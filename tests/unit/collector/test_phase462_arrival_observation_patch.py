#!/usr/bin/env python3
"""Tests for the Phase462 arrival-observation vLLM patch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "collector/vllm/phase462_arrival_observation_patch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_arrival_patch", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_async_tokenizer_patch_logs_runtime_batch_boundaries_and_prompt_lengths() -> None:
    mod = _load_module()
    source = mod.ASYNC_IMPORT_ANCHOR + mod.ASYNC_INIT_ANCHOR + mod.ASYNC_CALL_ANCHOR + mod.ASYNC_BATCH_ANCHOR

    patched = mod.patch_async_source(source)

    assert '"runtime_config"' in patched
    assert '"tokenizer_batch_enter"' in patched
    assert '"tokenizer_batch_complete"' in patched
    assert '"prompt_token_lengths"' in patched
    assert "max_batch_size" in patched
    assert "batch_wait_timeout_s" in patched
    assert mod.restore_async_source(patched) == source


def test_completion_patch_scopes_trace_id_around_render_only() -> None:
    mod = _load_module()
    source = mod.COMPLETION_IMPORT_ANCHOR + mod.COMPLETION_RENDER_ANCHOR

    patched = mod.patch_completion_source(source)

    assert 'headers.get("X-Request-Id")' in patched
    assert "aic_phase462_arrival_trace_id.set" in patched
    assert "finally:" in patched
    assert "aic_phase462_arrival_trace_id.reset" in patched
    assert mod.restore_completion_source(patched) == source


def test_engine_and_scheduler_patch_keep_external_id_and_step_admission_fields() -> None:
    mod = _load_module()
    engine_source = mod.ENGINE_HELPER_ANCHOR + mod.ENGINE_RECEIVE_ANCHOR
    scheduler_source = (
        mod.SCHED_HELPER_ANCHOR
        + mod.SCHED_START_ANCHOR
        + mod.SCHED_RETURN_ANCHOR
    )

    patched_engine = mod.patch_engine_source(engine_source)
    patched_scheduler = mod.patch_scheduler_source(scheduler_source)

    assert "external_req_id" in patched_engine
    assert 'getattr(request, "external_req_id", None)' in patched_engine
    assert '"external_req_id": request.external_req_id' not in patched_engine
    assert "len(random_suffix) == 8" in patched_engine
    assert "all(char in \"0123456789abcdef\"" in patched_engine
    assert '"engine_receive"' in patched_engine
    assert '"scheduler_step"' in patched_scheduler
    assert '"waiting_before"' in patched_scheduler
    assert '"waiting_after"' in patched_scheduler
    assert '"new_context_count"' in patched_scheduler
    assert "scheduled_new_reqs" in patched_scheduler
    assert mod.restore_engine_source(patched_engine) == engine_source
    assert mod.restore_scheduler_source(patched_scheduler) == scheduler_source


def test_all_patches_are_idempotent() -> None:
    mod = _load_module()
    async_source = mod.ASYNC_IMPORT_ANCHOR + mod.ASYNC_INIT_ANCHOR + mod.ASYNC_CALL_ANCHOR + mod.ASYNC_BATCH_ANCHOR
    completion_source = mod.COMPLETION_IMPORT_ANCHOR + mod.COMPLETION_RENDER_ANCHOR
    engine_source = mod.ENGINE_HELPER_ANCHOR + mod.ENGINE_RECEIVE_ANCHOR
    scheduler_source = mod.SCHED_HELPER_ANCHOR + mod.SCHED_START_ANCHOR + mod.SCHED_RETURN_ANCHOR

    assert mod.patch_async_source(mod.patch_async_source(async_source)) == mod.patch_async_source(async_source)
    assert mod.patch_completion_source(mod.patch_completion_source(completion_source)) == mod.patch_completion_source(completion_source)
    assert mod.patch_engine_source(mod.patch_engine_source(engine_source)) == mod.patch_engine_source(engine_source)
    assert mod.patch_scheduler_source(mod.patch_scheduler_source(scheduler_source)) == mod.patch_scheduler_source(scheduler_source)

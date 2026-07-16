#!/usr/bin/env python3
"""Tests for the Phase462 DP route logging-only patch."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "collector/vllm/phase462_dp_route_observation_patch.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase462_dp_route_patch", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_core_client_patch_records_route_snapshot_and_restores_exactly() -> None:
    mod = _load_module()
    source = mod.core_client_fixture_source()

    patched = mod.patch_core_client_source(source)

    for field in (
        '"kind": "route"',
        '"chosen_rank"',
        '"score_snapshot"',
        '"waiting"',
        '"running"',
        '"score"',
        "[None] * _aic_phase462_dp_capacity",
        "aic_phase462_dp_flush()",
    ):
        assert field in patched
    assert mod.restore_core_client_source(patched) == source


def test_engine_patch_records_receive_and_first_admit_on_both_paths() -> None:
    mod = _load_module()
    source = mod.engine_fixture_source()

    patched = mod.patch_engine_source(source)

    assert '"kind": "receive"' in patched
    assert patched.count('"kind": "admit"') == 1
    assert patched.count("aic_phase462_dp_log_admits(self, scheduler_output)") == 2
    assert "scheduler_output.scheduled_new_reqs" in patched
    assert "aic_phase462_dp_rank(self)" in patched
    assert "aic_phase462_dp_flush()" in patched
    assert mod.restore_engine_source(patched) == source


def test_patches_are_idempotent_and_do_not_write_per_event() -> None:
    mod = _load_module()
    core_client = mod.core_client_fixture_source()
    engine = mod.engine_fixture_source()

    patched_client = mod.patch_core_client_source(core_client)
    patched_engine = mod.patch_engine_source(engine)

    assert mod.patch_core_client_source(patched_client) == patched_client
    assert mod.patch_engine_source(patched_engine) == patched_engine
    for patched in (patched_client, patched_engine):
        emit_body = patched.split("def aic_phase462_dp_emit", 1)[1].split(
            "def aic_phase462_dp_flush", 1
        )[0]
        assert "open(" not in emit_body
        assert ".write(" not in emit_body


def test_engine_patch_names_the_missing_anchor() -> None:
    mod = _load_module()
    source = mod.engine_fixture_source().replace(mod.ENGINE_ADD_ANCHOR, "")

    with pytest.raises(ValueError, match="engine_receive_anchor_count:0"):
        mod.patch_engine_source(source)


def test_patcher_cli_is_runnable() -> None:
    completed = subprocess.run(
        [sys.executable, str(MODULE_PATH), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr

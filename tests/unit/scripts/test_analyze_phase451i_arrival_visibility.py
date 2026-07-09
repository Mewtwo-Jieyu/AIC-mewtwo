#!/usr/bin/env python3
"""Tests for Phase451-I arrival visibility analyzer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase451i_arrival_visibility.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase451i", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_discriminator_marks_waiting_jump_as_engine_burst() -> None:
    mod = _load_module()
    samples = [
        {"t_s": 0.0, "engine": "0", "running": 0.0, "waiting": 0.0},
        {"t_s": 2.0, "engine": "0", "running": 1.0, "waiting": 61.0},
        {"t_s": 4.0, "engine": "0", "running": 2.0, "waiting": 60.0},
    ]

    verdict = mod.classify_engine_visibility(samples)

    assert verdict["visibility"] == "engine_burst"
    assert verdict["route"] == "debug_observation_required"


def test_discriminator_marks_low_waiting_ramp_as_engine_drizzle() -> None:
    mod = _load_module()
    samples = [
        {"t_s": 0.0, "engine": "0", "running": 0.0, "waiting": 0.0},
        {"t_s": 2.0, "engine": "0", "running": 2.0, "waiting": 1.0},
        {"t_s": 4.0, "engine": "0", "running": 5.0, "waiting": 2.0},
        {"t_s": 6.0, "engine": "0", "running": 9.0, "waiting": 3.0},
    ]

    verdict = mod.classify_engine_visibility(samples)

    assert verdict["visibility"] == "engine_drizzle"
    assert verdict["route"] == "engine_visible_arrival_replay"


def test_find_added_request_lines_returns_matching_lines() -> None:
    mod = _load_module()
    lines = [
        "INFO other line\n",
        "DEBUG Added request chatcmpl-123 to engine 0\n",
        "INFO scheduled something else\n",
    ]

    matches = mod.find_added_request_lines(lines)

    assert len(matches) == 1
    assert "chatcmpl-123" in matches[0]


if __name__ == "__main__":
    test_discriminator_marks_waiting_jump_as_engine_burst()
    test_discriminator_marks_low_waiting_ramp_as_engine_drizzle()
    test_find_added_request_lines_returns_matching_lines()

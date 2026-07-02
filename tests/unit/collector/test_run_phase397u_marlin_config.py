from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _load_runner(monkeypatch, calls):
    def fake_run_moe_torch(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setitem(
        sys.modules,
        "collect_moe",
        types.SimpleNamespace(run_moe_torch=fake_run_moe_torch),
    )
    from collector.vllm import run_phase397u_marlin_config

    return importlib.reload(run_phase397u_marlin_config)


def test_phase397u_defaults_probe_two_structural_levers(tmp_path: Path, monkeypatch) -> None:
    calls = []
    runner = _load_runner(monkeypatch, calls)
    output_dir = tmp_path / "phase397u"

    assert runner.main(["--out-dir", str(output_dir)]) == 0

    assert [call["kwargs"]["distributed"] for call in calls] == [
        "power_law_local48_direct",
        "power_law_force_block64",
    ]
    assert all(call["args"][1] == [128] for call in calls)
    assert all(call["args"][6] == 1 for call in calls)
    assert all(call["args"][7] == 8 for call in calls)
    assert all(call["kwargs"]["power_law_alpha"] == 1.01 for call in calls)


def test_phase397u_can_run_one_selected_shot(tmp_path: Path, monkeypatch) -> None:
    calls = []
    runner = _load_runner(monkeypatch, calls)

    assert (
        runner.main(
            [
                "--out-dir",
                str(tmp_path),
                "--shot",
                "local48_direct",
            ]
        )
        == 0
    )

    assert len(calls) == 1
    assert calls[0]["kwargs"]["distributed"] == "power_law_local48_direct"

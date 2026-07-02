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
    from collector.vllm import run_phase397m_moe4bit

    return importlib.reload(run_phase397m_moe4bit)


def test_phase397m_defaults_preserve_ep8_power_law_scope(tmp_path: Path, monkeypatch) -> None:
    calls = []
    runner = _load_runner(monkeypatch, calls)
    output = tmp_path / "moe4bit.txt"

    assert runner.main(["--out", str(output), "--max-num-tokens", "1"]) == 0

    assert len(calls) == 2
    first_args = calls[0]["args"]
    assert first_args[1] == [1]
    assert first_args[6] == 1
    assert first_args[7] == 8
    assert calls[0]["kwargs"]["distributed"] == "power_law"
    assert calls[0]["kwargs"]["power_law_alpha"] == 1.01
    assert calls[1]["kwargs"]["power_law_alpha"] == 1.2


def test_phase397s_can_collect_single_tp16_ep1_balanced_route(tmp_path: Path, monkeypatch) -> None:
    calls = []
    runner = _load_runner(monkeypatch, calls)
    output = tmp_path / "moe4bit_tp16_ep1.txt"

    assert (
        runner.main(
            [
                "--out",
                str(output),
                "--moe-tp",
                "16",
                "--moe-ep",
                "1",
                "--num-tokens",
                "128",
                "--alphas",
                "1.01",
                "--distributed",
                "balanced",
            ]
        )
        == 0
    )

    assert len(calls) == 1
    args = calls[0]["args"]
    assert args[1] == [128]
    assert args[6] == 16
    assert args[7] == 1
    assert calls[0]["kwargs"]["distributed"] == "balanced"
    assert calls[0]["kwargs"]["power_law_alpha"] == 1.01


def test_phase397s_can_request_power_law_eplb_route(tmp_path: Path, monkeypatch) -> None:
    calls = []
    runner = _load_runner(monkeypatch, calls)
    output = tmp_path / "moe4bit_eplb.txt"

    assert (
        runner.main(
            [
                "--out",
                str(output),
                "--moe-tp",
                "1",
                "--moe-ep",
                "8",
                "--num-tokens",
                "128",
                "--alphas",
                "1.01",
                "--distributed",
                "power_law_eplb",
            ]
        )
        == 0
    )

    assert len(calls) == 1
    assert calls[0]["args"][1] == [128]
    assert calls[0]["kwargs"]["distributed"] == "power_law_eplb"
    assert calls[0]["kwargs"]["power_law_alpha"] == 1.01

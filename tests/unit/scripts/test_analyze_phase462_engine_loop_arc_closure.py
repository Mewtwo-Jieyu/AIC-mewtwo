from __future__ import annotations

import csv
from pathlib import Path

import pytest

import scripts.analyze_phase462_engine_loop_arc_closure as analysis


def _write_scoreboard(path: Path, *, passing: set[str]) -> None:
    scenarios = [
        "K2.5-tp8ep8-8k2k",
        "K2.5-tp4ep8dp2-8k2k",
        "K2.5-tp8ep8-32k3k",
        "K2.5-tp4ep8dp2-32k3k",
        "K2.5-tp8ep8-8k2k-bt65536",
        "K2.5-tp4ep8dp2-8k2k-bt65536",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "scenario",
                "real_output_tok_s_gpu",
                "sim_output_tok_s_gpu",
                "error_ratio",
                "gate_pass",
            ],
        )
        writer.writeheader()
        for scenario in scenarios:
            gate_pass = scenario in passing
            writer.writerow(
                {
                    "scenario": scenario,
                    "real_output_tok_s_gpu": 100.0,
                    "sim_output_tok_s_gpu": 110.0 if gate_pass else 120.0,
                    "error_ratio": 1.1 if gate_pass else 1.2,
                    "gate_pass": gate_pass,
                }
            )


def test_closure_accepts_only_the_locked_null_only_scoreboard(tmp_path: Path) -> None:
    scoreboard = tmp_path / "scoreboard.csv"
    report = tmp_path / "report.md"
    _write_scoreboard(scoreboard, passing=set(analysis.PASSING_SCENARIOS))

    result = analysis.run_closure(scoreboard=scoreboard, report=report)

    assert result["scoreboard"] == "3/6"
    assert result["official_baseline"] == "null_only"
    assert result["engine_loop_default"] == "off"
    text = report.read_text(encoding="utf-8")
    assert "null block 修复" in text
    assert "diagnostic" in text
    assert "Default AIC=No-Go" in text


def test_closure_rejects_a_different_passing_set(tmp_path: Path) -> None:
    scoreboard = tmp_path / "scoreboard.csv"
    _write_scoreboard(
        scoreboard,
        passing={
            "K2.5-tp8ep8-8k2k",
            "K2.5-tp4ep8dp2-8k2k",
            "K2.5-tp4ep8dp2-32k3k",
        },
    )

    with pytest.raises(AssertionError, match="unexpected null-only passing set"):
        analysis.load_scoreboard(scoreboard)

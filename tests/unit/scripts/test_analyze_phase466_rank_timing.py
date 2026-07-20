from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase466_rank_timing.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_phase466_rank_timing", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_scenario(root: Path, scenario: str, rank_count: int) -> None:
    run_dir = root / "formal" / scenario
    run_dir.mkdir(parents=True)
    (run_dir / "probe_summary.json").write_text(
        json.dumps({"status": "ARTIFACT_VALID"}), encoding="utf-8"
    )
    iteration_rows = []
    rank_rows = []
    for rank in range(rank_count):
        iteration_rows.append(
            {
                "rank_id": rank,
                "iteration_seq": 10 + rank,
                "progress_window_id": 0,
                "progress_start_tokens": 0,
                "progress_end_tokens": 100 + rank,
                "iteration_elapsed_ms": 10.0 + rank,
                "scheduled_prefill_tokens": 80,
                "scheduled_decode_tokens": 20,
                "prefill_request_count": 2,
                "decode_request_count": 8,
            }
        )
        rank_rows.append({"rank_id": rank, "preemptions": rank})
    _write_csv(run_dir / "iteration_rows.csv", iteration_rows)
    _write_csv(run_dir / "rank_summary.csv", rank_rows)


def test_rank_timing_report_is_diagnostic_only_and_never_fakes_sim_rows(
    tmp_path: Path,
) -> None:
    analysis = _load_module()
    for scenario, rank_count in analysis.FORMAL_SCENARIOS.items():
        _write_scenario(tmp_path, scenario, rank_count)

    result = analysis.analyze(tmp_path)

    assert result["status"] == "DIAGNOSTIC_COMPLETE"
    assert result["route_selection_executed"] is False
    assert result["simulator_rank_rows_generated"] is False
    assert result["diagnostic_only"] is True
    assert result["valid_for_default"] is False
    assert result["perf_database"] is False
    assert result["default_readiness"] == "No-Go"
    by_scenario = {item["scenario"]: item for item in result["scenarios"]}
    assert by_scenario["K2.5-tp8ep8-8k2k-bt65536"]["topology_role"] == (
        "single_rank_control"
    )
    assert by_scenario["K2.5-tp4ep8dp2-8k2k-bt65536"]["rank_count"] == 2
    assert by_scenario["K2.5-tp4ep8dp2-8k2k-bt65536"][
        "rank_comparison"
    ] is not None

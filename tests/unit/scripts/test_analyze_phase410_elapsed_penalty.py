import csv
import json

import pytest

from scripts import analyze_phase410_elapsed_penalty as phase410


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_elapsed_penalty_uses_wallclock_overlap_not_iteration_index(tmp_path):
    serve_log = tmp_path / "serve.log"
    serve_log.write_text(
        "\n".join(
            [
                "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP1 Iteration(0): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP1 Iteration(1): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP0 Iteration(2): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 10.00 ms",
                "INFO EngineCore_DP1 Iteration(2): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 10.00 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    steps = phase410.phase409.parse_iteration_steps(serve_log)
    timelines = phase410.build_wallclock_timelines(steps)
    summary = phase410.summarize_elapsed_penalty(steps, needed_penalty=1.75)

    assert timelines["0"][1].start_ms == pytest.approx(100.0)
    assert timelines["1"][1].start_ms == pytest.approx(100.0)
    assert summary["intrinsic_decode_ms"] == pytest.approx(10.0)
    assert summary["stalled_decode_steps"] == 2
    assert summary["dp_lockstep_extra_ms"] == pytest.approx(180.0)
    assert summary["real_wall_ms"] == pytest.approx(210.0)
    assert summary["counterfactual_wall_ms"] == pytest.approx(120.0)
    assert summary["elapsed_penalty"] == pytest.approx(1.75)
    assert summary["penalty_gate"] == "passed"


def test_phase410_rows_are_report_only_and_do_not_read_phase405(tmp_path):
    raw_root = tmp_path / "phase409_iter_trace"
    scenario = phase410.DEFAULT_SCENARIO
    out_dir = raw_root / scenario
    out_dir.mkdir(parents=True)
    (out_dir / "serve.log").write_text(
        "\n".join(
            [
                "INFO EngineCore_DP0 Iteration(0): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP1 Iteration(0): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP0 Iteration(1): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP1 Iteration(1): 1 context requests, 100 context tokens, 0 generation requests, 0 generation tokens, iteration elapsed time: 100.00 ms",
                "INFO EngineCore_DP0 Iteration(2): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 10.00 ms",
                "INFO EngineCore_DP1 Iteration(2): 0 context requests, 0 context tokens, 1 generation requests, 1 generation tokens, iteration elapsed time: 10.00 ms",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    phase407_csv = tmp_path / "phase407.csv"
    _write_csv(
        phase407_csv,
        [
            {
                "scenario": scenario,
                "uncoupled_joint_output_tok_s_gpu": "1.75",
                "real_output_tok_s_gpu": "1.0",
            }
        ],
    )

    row = phase410.build_phase410_rows(
        raw_root=raw_root,
        phase407_csv=phase407_csv,
        scenario=scenario,
    )[0]

    assert row["phase405_penalty_read"] == "false"
    assert row["runtime_modified"] == "false"
    assert row["perf_database"] == "false"
    assert row["valid_for_default"] == "false"
    assert row["default_readiness"] == "No-Go"
    assert row["elapsed_penalty"] == "1.750000"
    assert row["penalty_gate"] == "passed"
    assert row["mechanism_verdict"] == "elapsed_lockstep_reproduces_needed"


def test_phase410_writer_rejects_default_claim(tmp_path):
    rows = [{field: "" for field in phase410.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase410.SOURCE,
            "scenario": phase410.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "true",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    with pytest.raises(ValueError, match="valid_for_default"):
        phase410.write_phase410_csv(tmp_path / "bad.csv", rows)

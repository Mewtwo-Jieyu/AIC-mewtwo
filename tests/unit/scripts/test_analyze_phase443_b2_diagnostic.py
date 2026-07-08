import gzip
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase443_b2_diagnostic.py"
    spec = importlib.util.spec_from_file_location("phase443_b2_diagnostic", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase443_parses_gz_event_rows_and_wall_steps(tmp_path):
    phase443 = _load_module()
    event_path = tmp_path / "event_timing.jsonl.gz"
    with gzip.open(event_path, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": "phase442_graph_outer_event_v1",
                    "ctx_requests": 1,
                    "ctx_tokens": 8000,
                    "generation_requests": 60,
                    "generation_tokens": 60,
                    "dp_rank": 0,
                    "forward_busy_ms": 100.0,
                    "cudagraph_mode": "NONE",
                }
            )
            + "\n"
        )
    wall_path = tmp_path / "serve.log.gz"
    with gzip.open(wall_path, "wt", encoding="utf-8") as handle:
        handle.write(
            "(EngineCore_DP0 pid=1) INFO Iteration(7): 1 context requests, "
            "8000 context tokens, 60 generation requests, 60 generation tokens, "
            "iteration elapsed time: 150.00 ms\n"
        )

    events = phase443.parse_event_rows(event_path)
    wall_steps = phase443.parse_wall_steps(wall_path)

    assert events[0].key == (8000, 60)
    assert events[0].forward_busy_ms == 100.0
    assert wall_steps[0].key == (8000, 60)
    assert wall_steps[0].elapsed_ms == 150.0


def test_phase443_classifies_wall_gap_as_b2b_not_useful():
    phase443 = _load_module()
    verdict = phase443.classify_bucket(
        event_count=50,
        wall_count=50,
        busy_ms=101.0,
        sim_charge_ms=100.0,
        wall_ms=150.0,
    )

    assert verdict == "wall_gap_dominates_b2b_not_useful"


def test_phase443_classifies_busy_gap_as_b2b_candidate():
    phase443 = _load_module()
    verdict = phase443.classify_bucket(
        event_count=50,
        wall_count=50,
        busy_ms=150.0,
        sim_charge_ms=100.0,
        wall_ms=154.0,
    )

    assert verdict == "busy_gap_dominates_b2b_candidate"


def test_phase443_overall_prefers_wall_gap_when_weighted_gap_is_wall_side():
    phase443 = _load_module()
    rows = [
        {
            "row_type": "bucket",
            "classification": "wall_gap_dominates_b2b_not_useful",
            "wall_count": 10,
            "sim_charge_ms": 100.0,
            "event_forward_busy_median_ms": 100.0,
            "wall_elapsed_median_ms": 150.0,
        },
        {
            "row_type": "bucket",
            "classification": "busy_gap_dominates_b2b_candidate",
            "wall_count": 1,
            "sim_charge_ms": 100.0,
            "event_forward_busy_median_ms": 140.0,
            "wall_elapsed_median_ms": 142.0,
        },
    ]

    summary = phase443.summarize_b2b_gate(rows)

    assert summary["b2b_gate"] == "not_recommended"
    assert summary["primary_gap"] == "wall_over_busy"

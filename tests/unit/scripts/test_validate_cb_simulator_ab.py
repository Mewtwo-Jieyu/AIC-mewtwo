import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "validate_cb_simulator.py"
    spec = importlib.util.spec_from_file_location("validate_cb_simulator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase444_ab_classification_uses_error_ratio_delta():
    validate_cb = _load_module()

    assert validate_cb._classify_ab_delta(1.10, 1.00, threshold=0.005) == "improved"
    assert validate_cb._classify_ab_delta(1.00, 1.10, threshold=0.005) == "regressed"
    assert validate_cb._classify_ab_delta(1.001, 1.004, threshold=0.005) == "unchanged"


def test_phase444_ab_rows_join_by_scenario_and_flag_regression():
    validate_cb = _load_module()
    baseline = {
        "a": {
            "name": "a",
            "sim_output_tok_s_gpu": 100.0,
            "real_output_tok_s_gpu": 100.0,
            "error_ratio": 1.0,
        },
        "b": {
            "name": "b",
            "sim_output_tok_s_gpu": 100.0,
            "real_output_tok_s_gpu": 100.0,
            "error_ratio": 1.2,
        },
    }
    current = [
        validate_cb.MultiConfigTopKRow(
            name="a",
            tp=4,
            dp=2,
            ep=8,
            max_bt=8000,
            real_output_tok_s_gpu=100.0,
            sim_output_tok_s_gpu=120.0,
            error_ratio=1.2,
            rank=1,
            real_rank=1,
            rank_delta=0,
            diagnostic_only=True,
            valid_for_default=False,
            perf_database=False,
        ),
        validate_cb.MultiConfigTopKRow(
            name="b",
            tp=4,
            dp=2,
            ep=8,
            max_bt=8000,
            real_output_tok_s_gpu=100.0,
            sim_output_tok_s_gpu=110.0,
            error_ratio=1.1,
            rank=2,
            real_rank=2,
            rank_delta=0,
            diagnostic_only=True,
            valid_for_default=False,
            perf_database=False,
        ),
    ]

    rows = validate_cb.build_multi_config_ab_rows(
        baseline,
        current,
        delta_threshold=0.005,
    )
    by_name = {row.name: row for row in rows}

    assert by_name["a"].classification == "regressed"
    assert by_name["b"].classification == "improved"
    assert validate_cb.multi_config_ab_has_regression(rows)

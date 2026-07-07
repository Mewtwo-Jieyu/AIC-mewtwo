import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase440_nonattn_total.py"
    spec = importlib.util.spec_from_file_location("phase440_nonattn_total", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase440_derives_non_attn_total_from_iteration_elapsed():
    phase440 = _load_module()
    step = phase440.StepRecord(
        source="unit",
        bucket_tokens=8000,
        decode_batch=64,
        elapsed_ms=1000.0,
        context_attention_ms=70.0,
        generation_attention_ms=30.0,
    )

    rows = phase440.derive_non_attn_total_rows([step])

    assert rows[0]["row_type"] == "non_attn_total_curve"
    assert rows[0]["bucket_tokens"] == 8000
    assert rows[0]["decode_batch"] == 64
    assert rows[0]["latency_ms"] == 900.0


def test_phase440_overlap_calibration_gate_compares_category_sum():
    phase440 = _load_module()
    non_attn_rows = [
        {
            "row_type": "non_attn_total_curve",
            "phase": "mixed_prefill",
            "bucket_tokens": 8000,
            "decode_batch": 34,
            "latency_ms": 1000.0,
        }
    ]
    category_rows = [
        phase440.ServingRow("base", "mixed_prefill", "ep_a2a", 8000, 34, 400.0, "a"),
        phase440.ServingRow("base", "mixed_prefill", "moe_gemm_or_aux", 8000, 34, 500.0, "b"),
        phase440.ServingRow("base", "mixed_prefill", "other_cuda", 8000, 34, 50.0, "c"),
        phase440.ServingRow("base", "mixed_prefill", "collective_other", 8000, 34, 40.0, "d"),
    ]

    gate = phase440.calibration_gate_rows(non_attn_rows, category_rows, tolerance_pct=10.0)

    assert gate[0]["row_type"] == "calibration_gate"
    assert gate[0]["reconstruction_gate"] == "passed"
    assert round(gate[0]["reconstruction_error_pct"], 4) == 1.0101

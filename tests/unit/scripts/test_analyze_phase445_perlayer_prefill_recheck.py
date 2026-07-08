import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase445_perlayer_prefill_recheck.py"
    spec = importlib.util.spec_from_file_location("phase445_perlayer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase445b_classifies_busy_gap_before_wall_gap():
    phase445 = _load_module()
    row = phase445.BucketRow(
        scenario="unit",
        ctx_tokens=8000,
        decode_batch=60,
        event_count=48,
        confidence="high",
        sim_charge_ms=400.0,
        event_forward_busy_median_ms=1000.0,
        wall_elapsed_median_ms=1040.0,
    )

    classified = phase445.classify_bucket(row)

    assert classified.mechanism_candidate == "busy_over_sim"
    assert classified.busy_gap_share > 0.9
    assert classified.wall_gap_share < 0.1


def test_phase445b_verdict_prefers_busy_charge_when_mixed_prefill_gap_is_busy_dominated():
    phase445 = _load_module()
    classified = [
        phase445.ClassifiedBucket(
            scenario="unit",
            ctx_tokens=8000,
            decode_batch=60,
            event_count=48,
            confidence="high",
            sim_charge_ms=400.0,
            event_forward_busy_median_ms=1000.0,
            wall_elapsed_median_ms=1040.0,
            busy_minus_sim_ms=600.0,
            wall_minus_busy_ms=40.0,
            busy_gap_share=0.9375,
            wall_gap_share=0.0625,
            mechanism_candidate="busy_over_sim",
            note="unit",
        ),
        phase445.ClassifiedBucket(
            scenario="unit",
            ctx_tokens=8000,
            decode_batch=61,
            event_count=44,
            confidence="high",
            sim_charge_ms=410.0,
            event_forward_busy_median_ms=1000.0,
            wall_elapsed_median_ms=1045.0,
            busy_minus_sim_ms=590.0,
            wall_minus_busy_ms=45.0,
            busy_gap_share=0.9291,
            wall_gap_share=0.0709,
            mechanism_candidate="busy_over_sim",
            note="unit",
        ),
    ]

    verdict = phase445.decide_verdict(classified)

    assert verdict == "prefill_gap_is_busy_charge_not_wall_only"


if __name__ == "__main__":
    test_phase445b_classifies_busy_gap_before_wall_gap()
    test_phase445b_verdict_prefers_busy_charge_when_mixed_prefill_gap_is_busy_dominated()

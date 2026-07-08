import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase445_threeway_data_audit.py"
    spec = importlib.util.spec_from_file_location("phase445_threeway", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase445_alignment_requires_direct_upstream_category_match():
    phase445 = _load_module()

    rows = phase445.build_alignment_rows(
        serving_summary={
            ("mixed_prefill", "moe_gemm_or_aux"): {"rows": 4, "bucket_min": 8000, "bucket_max": 32000},
            ("mixed_prefill", "ep_a2a"): {"rows": 4, "bucket_min": 8000, "bucket_max": 32000},
            ("decode", "other_cuda"): {"rows": 2, "bucket_min": 8, "bucket_max": 64},
        },
        upstream_summary={
            "moe_perf.parquet": {
                "status": "available",
                "rows": 48195,
                "filtered_rows": 162,
                "token_min": 1,
                "token_max": 16384,
            },
            "custom_allreduce_perf.parquet": {
                "status": "available",
                "rows": 138,
                "filtered_rows": 0,
                "token_min": None,
                "token_max": None,
            },
        },
    )

    by_category = {(row.phase, row.category): row for row in rows}

    assert by_category[("mixed_prefill", "moe_gemm_or_aux")].alignment_status == "partial_requires_layer_normalization"
    assert by_category[("mixed_prefill", "ep_a2a")].alignment_status == "missing_direct_upstream_table"
    assert by_category[("decode", "other_cuda")].alignment_status == "missing_direct_upstream_table"


def test_phase445_verdict_blocks_runtime_coefficient_without_full_alignment():
    phase445 = _load_module()
    rows = [
        phase445.AlignmentRow(
            phase="mixed_prefill",
            category="moe_gemm_or_aux",
            serving_rows=4,
            serving_bucket_min=8000,
            serving_bucket_max=32000,
            upstream_file="moe_perf.parquet",
            upstream_rows=48195,
            upstream_filtered_rows=162,
            upstream_token_min=1,
            upstream_token_max=16384,
            alignment_status="partial_requires_layer_normalization",
            runtime_action="report_only",
            note="unit",
        ),
        phase445.AlignmentRow(
            phase="mixed_prefill",
            category="ep_a2a",
            serving_rows=4,
            serving_bucket_min=8000,
            serving_bucket_max=32000,
            upstream_file="",
            upstream_rows=0,
            upstream_filtered_rows=0,
            upstream_token_min=None,
            upstream_token_max=None,
            alignment_status="missing_direct_upstream_table",
            runtime_action="report_only",
            note="unit",
        ),
    ]

    verdict = phase445.decide_verdict(rows)

    assert verdict == "no_runtime_coefficient_schema_alignment_blocked"


if __name__ == "__main__":
    test_phase445_alignment_requires_direct_upstream_category_match()
    test_phase445_verdict_blocks_runtime_coefficient_without_full_alignment()

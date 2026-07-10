import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase460_mla_decode_lookup.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase460_mla_decode_lookup", MODULE_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyzer)


def test_mla_lookup_audit_identifies_the_32k_batch8_outlier():
    rows = [
        analyzer.MlaRow(8, 8, 16_384, 0.07758933305740356, "vllm_flash_attn_mla"),
        analyzer.MlaRow(8, 8, 32_768, 0.9586453437805176, "vllm_flash_attn_mla"),
        analyzer.MlaRow(8, 8, 65_536, 0.2446026603380839, "vllm_flash_attn_mla"),
        analyzer.MlaRow(8, 16, 32_768, 0.2449386715888977, "vllm_flash_attn_mla"),
        analyzer.MlaRow(8, 16, 65_536, 0.47361600399017334, "vllm_flash_attn_mla"),
    ]

    audit = analyzer.audit_lookup(
        rows,
        batch=13,
        kv_len=33_793,
        local_heads=8,
        layers=61,
        current_decode_total_ms=39.07168780417416,
        real_decode_total_ms=21.0,
    )

    assert audit.batch_bracket == (8, 16)
    assert audit.kv_bracket == (32_768, 65_536)
    assert abs(audit.current_per_layer_ms - 0.5086735302165835) < 1e-12
    assert audit.outlier_batch == 8
    assert audit.outlier_kv_len == 32_768
    assert audit.outlier_actual_over_context_trend > 7.0
    assert abs(audit.counterfactual_decode_total_ms - 20.78160561028787) < 1e-9
    assert abs(audit.counterfactual_decode_sim_over_real - 1.0) <= 0.10
    assert audit.verdict == "perfdb_outlier_row_requires_recollect"


def test_context_semantics_cannot_explain_a_stable_30ms_attention_band():
    contribution = analyzer.classify_context_contribution(
        minimum_attention_ms=30.125752694963012,
        maximum_attention_ms=31.267299093306065,
        observed_excess_ms=18.07168780417416,
    )

    assert contribution["attention_span_ms"] < 1.2
    assert contribution["share_of_observed_excess"] < 0.07
    assert contribution["verdict"] == "context_semantics_not_primary"

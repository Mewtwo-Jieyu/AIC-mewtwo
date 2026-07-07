import csv
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _load_module():
    script = Path(__file__).resolve().parents[3] / "scripts/analyze_phase437_serving_state_collect.py"
    spec = importlib.util.spec_from_file_location("analyze_phase437_serving_state_collect", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _step(window: str, ctx: int, gen: int, category_ms: dict[str, float]):
    return SimpleNamespace(window=window, ctx_tokens=ctx, gen_tokens=gen, category_ms=category_ms)


def test_phase437_mixed_key_uses_ctx_plus_decode_tokens() -> None:
    mod = _load_module()
    rows = mod.build_rows_from_rank_steps(
        [
            _step("mixed_isl8k_c128", 7991, 9, {"ep_a2a": 100.0, "moe_gemm_or_aux": 300.0}),
            _step("mixed_isl8k_c128", 7991, 9, {"ep_a2a": 120.0}),
        ],
        scenario="synthetic",
    )
    by_key = {
        (row["phase"], row["category"], row["bucket_tokens"], row["decode_batch"]): row
        for row in rows
    }

    ep = by_key[("mixed_prefill", "ep_a2a", 8000, 9)]
    moe = by_key[("mixed_prefill", "moe_gemm_or_aux", 8000, 9)]

    assert ep["latency_ms"] == 110.0
    assert ep["sample_count"] == 2
    assert moe["latency_ms"] == 300.0
    assert ("mixed_prefill", "ep_a2a", 7991, 9) not in by_key


def test_phase437_decode_key_and_perfdb_provenance(tmp_path: Path) -> None:
    mod = _load_module()
    rows = mod.build_rows_from_rank_steps(
        [_step("decode_isl1k_c128", 0, 64, {"tp_or_dp_allreduce": 7.0})],
        scenario="synthetic",
    )
    perfdb = tmp_path / "vllm_serving_state_perf.txt"
    mod.write_perfdb(rows, perfdb)
    written = list(csv.DictReader(perfdb.open()))

    assert rows[0]["phase"] == "decode"
    assert rows[0]["category"] == "collective_other"
    assert rows[0]["bucket_tokens"] == 64
    assert rows[0]["decode_batch"] == 64
    assert written[0]["kernel_source"] == "phase437_serving_state_collect"
    assert written[0]["provenance"] == "phase437_serving_state_collect_step_bucket"
    assert written[0]["latency"] == "7.000000"


def test_phase437_execute_context_fallback_without_profiler_step() -> None:
    mod = _load_module()
    phase428 = mod.phase428
    events = [
        phase428.TraceEvent(
            name="execute_context_2000(2000)_generation_9(9)",
            cat="user_annotation",
            ts_us=1000.0,
            dur_us=10_000.0,
        ),
        phase428.TraceEvent(
            name="ncclDevKernel_AllGather_RING_LL",
            cat="kernel",
            ts_us=2000.0,
            dur_us=2000.0,
        ),
        phase428.TraceEvent(
            name="void marlin_moe_wna16::Marlin<synthetic>",
            cat="kernel",
            ts_us=5000.0,
            dur_us=3000.0,
        ),
    ]

    steps = mod._steps_from_execute_context_events(
        events,
        trace_name="synthetic.pt.trace.json.gz",
        window="mixed_isl2k_c128",
    )
    rows = mod.build_rows_from_rank_steps(steps, scenario="synthetic")
    by_key = {
        (row["phase"], row["category"], row["bucket_tokens"], row["decode_batch"]): row
        for row in rows
    }

    assert len(steps) == 1
    assert steps[0].ctx_tokens == 2000
    assert steps[0].gen_tokens == 9
    assert by_key[("mixed_prefill", "ep_a2a", 2009, 9)]["latency_ms"] == 2.0
    assert by_key[("mixed_prefill", "moe_gemm_or_aux", 2009, 9)]["latency_ms"] == 3.0

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase444_loo_serving_state.py"
    spec = importlib.util.spec_from_file_location("phase444_loo", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _row(bucket: int, batch: int, latency: float):
    phase444 = _load_module()
    return phase444.ServingStateRow(
        model="kimi-k2.5",
        topology="tp4dp2ep8",
        phase="decode",
        row_kind="category",
        category="ep_a2a",
        bucket_tokens=bucket,
        decode_batch=batch,
        hidden_size=7168,
        topk=8,
        moe_ep_size=8,
        quant_runtime="CompressedTensorsWNA16MarlinMoEMethod",
        latency_ms=latency,
        kernel_source="unit",
        provenance="unit",
    )


def test_phase444_loo_predicts_interior_row_from_remaining_grid():
    phase444 = _load_module()
    rows = [_row(b, d, float(b + d)) for b in (0, 10, 20) for d in (0, 10, 20)]

    results = phase444.run_loo(rows)
    center = next(row for row in results if row.bucket_tokens == 10 and row.decode_batch == 10)

    assert center.fold == "interior"
    assert center.predicted_ms == 20.0
    assert center.error_pct == 0.0


def test_phase444_gate_ignores_frontier_rows():
    phase444 = _load_module()
    summary = phase444.summarize_loo(
        [
            phase444.LOORow(
                phase="decode",
                row_kind="category",
                category="ep_a2a",
                bucket_tokens=10,
                decode_batch=10,
                actual_ms=20.0,
                predicted_ms=20.0,
                error_pct=0.0,
                fold="interior",
                gate="passed",
                note="",
            ),
            phase444.LOORow(
                phase="decode",
                row_kind="category",
                category="ep_a2a",
                bucket_tokens=0,
                decode_batch=0,
                actual_ms=0.0,
                predicted_ms=None,
                error_pct=None,
                fold="frontier",
                gate="not_counted",
                note="",
            ),
        ]
    )

    by_fold = {row["fold"]: row for row in summary}
    assert by_fold["interior"]["loo_gate"] == "passed"
    assert by_fold["frontier"]["loo_gate"] == "not_counted"

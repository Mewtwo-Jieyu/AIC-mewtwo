from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase448_dp_admission.py"
    spec = importlib.util.spec_from_file_location("analyze_phase448_dp_admission", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase448_summarizes_mixed_distribution_and_phase_pairs() -> None:
    phase448 = _load_module()
    tolerances = {
        ("unit", "mixed_decode_batch_p10_p90"): (34.0, 52.0),
        ("unit", "mixed_bucket_tokens_p10_p90"): (8000.0, 8000.0),
    }
    trace = [
        {
            "replica_id": 0,
            "local_iter": 1,
            "prefill_tokens": 7958,
            "decode_reqs": 42,
            "total_tokens": 8000,
        },
        {
            "replica_id": 1,
            "local_iter": 1,
            "prefill_tokens": 0,
            "decode_reqs": 42,
            "total_tokens": 42,
        },
        {
            "replica_id": 0,
            "local_iter": 2,
            "prefill_tokens": 0,
            "decode_reqs": 43,
            "total_tokens": 43,
        },
        {
            "replica_id": 1,
            "local_iter": 2,
            "prefill_tokens": 7957,
            "decode_reqs": 43,
            "total_tokens": 8000,
        },
    ]

    rows, gates = phase448.summarize_trace("unit", trace, tolerances)

    assert all(gate.passed for gate in gates)
    assert any(row["metric"] == "mixed_prefill+decode" for row in rows)
    assert any(row["metric"] == "decode+mixed_prefill" for row in rows)


if __name__ == "__main__":
    test_phase448_summarizes_mixed_distribution_and_phase_pairs()

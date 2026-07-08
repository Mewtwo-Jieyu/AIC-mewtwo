from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aiconfigurator.sdk.backends.cb_simulator.iteration_latency import (  # noqa: E402
    IterationLatencyCalculator,
)
from aiconfigurator.sdk.performance_result import PerformanceResult  # noqa: E402


class _BackendMustNotRun:
    def run_static(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("forward_total must short-circuit before backend.run_static")


class _ForwardTotalDB:
    system = "h200_sxm"
    backend = "vllm"
    version = "0.19.0"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def query_vllm_serving_state(self, **kwargs):  # noqa: ANN003
        self.calls.append(kwargs)
        if kwargs.get("row_kind") != "forward_total":
            raise AssertionError(f"unexpected non-forward_total query: {kwargs}")
        if kwargs["phase"] == "mixed_prefill":
            return PerformanceResult(1100.0, energy=0.0)
        if kwargs["phase"] == "decode":
            return PerformanceResult(40.0, energy=0.0)
        return None


def _model():
    return SimpleNamespace(
        config=SimpleNamespace(
            tp_size=4,
            attention_dp_size=2,
            moe_tp_size=1,
            moe_ep_size=8,
        )
    )


def test_forward_total_replaces_mixed_step_before_backend() -> None:
    db = _ForwardTotalDB()
    calc = IterationLatencyCalculator(_BackendMustNotRun(), _model(), db)

    total = calc.compute(
        prefill_tokens=8000,
        prefill_batch_size=1,
        prefill_seq_len=8000,
        decode_batch_size=64,
        decode_avg_kv_len=8000,
    )

    assert math.isclose(total, 1100.0)
    assert [call["row_kind"] for call in db.calls] == ["forward_total"]


def test_forward_total_replaces_decode_step_before_backend() -> None:
    db = _ForwardTotalDB()
    calc = IterationLatencyCalculator(_BackendMustNotRun(), _model(), db)

    total = calc.compute(
        prefill_tokens=0,
        prefill_batch_size=0,
        prefill_seq_len=1,
        decode_batch_size=64,
        decode_avg_kv_len=8000,
    )

    assert math.isclose(total, 40.0)
    assert [call["row_kind"] for call in db.calls] == ["forward_total"]


if __name__ == "__main__":
    test_forward_total_replaces_mixed_step_before_backend()
    test_forward_total_replaces_decode_step_before_backend()

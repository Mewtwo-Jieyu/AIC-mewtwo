from aiconfigurator.sdk.backends.cb_simulator.datatypes import (
    CBSimConfig,
    Request,
    RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler

from scripts import validate_cb_simulator as validate_cb


def _multi_config_point(name: str):
    return next(pt for pt in validate_cb.MULTI_CONFIG_DATA if pt.name == name)


def test_phase398_kv_capacity_table_uses_per_engine_blocks() -> None:
    point = _multi_config_point("K2.5-tp4ep8dp2-32k3k")

    capacity = validate_cb._multi_config_kv_capacity(point)

    assert capacity.kv_cache_tokens == 320_800
    assert capacity.num_gpu_blocks == 20_050
    assert capacity.serve_log_line_numbers == (204, 212)
    assert capacity.override_num_gpu_blocks == 512
    assert capacity.capacity_scope == "per_engine"


def test_phase398_scheduler_capacity_limits_tp4_32k_decode_batch() -> None:
    running = []
    for request_id in range(64):
        request = Request(request_id=request_id, isl=32_000, osl=3_000, arrival_time_ms=0.0)
        request.state = RequestState.DECODING
        request.prefill_tokens_remaining = 0
        running.append(request)

    result = CBScheduler(
        CBSimConfig(
            max_num_batched_tokens=65_536,
            num_gpu_blocks=20_050,
            block_size=16,
        )
    ).schedule(waiting=[], running=running)

    assert len(result.decode_reqs) <= 10


def test_phase398_multi_config_diagnostic_passes_kv_capacity(monkeypatch) -> None:
    point = _multi_config_point("K2.5-tp4ep8dp2-32k3k")
    monkeypatch.setattr(validate_cb, "MULTI_CONFIG_DATA", [point])

    class FakeSummary:
        def get_result_dict(self) -> dict[str, float]:
            return {"tokens/s/gpu": 1.0}

        def get_per_ops_data(self) -> dict[str, dict[str, float]]:
            return {
                "cb_sim_scheduling": {
                    "avg_prefill_reqs_per_iter": 0.0,
                    "avg_decode_reqs_per_iter": 10.0,
                    "avg_tokens_per_iter": 10.0,
                    "peak_prefill_reqs_per_iter": 0.0,
                    "peak_decode_reqs_per_iter": 10.0,
                    "peak_tokens_per_iter": 10.0,
                    "steady_state_iterations": 1.0,
                    "steady_state_time_ms": 1.0,
                }
            }

    class FakeBackend:
        def run_agg(self, *args, **kwargs) -> FakeSummary:
            cb_config = kwargs["cb_config"]
            assert cb_config.num_gpu_blocks == 20_050
            assert cb_config.block_size == 16
            return FakeSummary()

    monkeypatch.setattr(validate_cb, "VLLMBackend", FakeBackend)
    monkeypatch.setattr(validate_cb, "_load_model_and_db", lambda **kwargs: (object(), object(), object()))

    rows = validate_cb._run_multi_config_diagnostic_raw(
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=0.0,
    )

    assert rows[0][2] == 1.0

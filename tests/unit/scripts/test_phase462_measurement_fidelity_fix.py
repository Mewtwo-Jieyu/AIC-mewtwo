from types import SimpleNamespace

import pytest

from scripts import validate_cb_simulator as validate


def _point(name: str):
    return next(point for point in validate.MULTI_CONFIG_DATA if point.name == name)


def _base_config(point):
    return validate._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=0.0,
    )


def test_official_validation_keeps_tp8_config_unchanged() -> None:
    point = _point("K2.5-tp8ep8-8k2k-bt65536")

    assert validate._make_official_validation_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=0.0,
    ) == _base_config(point)


@pytest.mark.parametrize(
    ("name", "expected_requests"),
    [
        ("K2.5-tp4ep8dp2-8k2k", 512),
        ("K2.5-tp4ep8dp2-8k2k-bt65536", 256),
    ],
)
def test_official_validation_matches_dp2_global_n512_full_window(
    name: str,
    expected_requests: int,
) -> None:
    config = validate._make_official_validation_cb_config(
        _point(name),
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=0.0,
    )

    assert config.num_requests == expected_requests
    assert config.warmup_requests == 0


def test_full_closed_loop_metric_counts_full_osl_until_last_completion() -> None:
    completed = [
        SimpleNamespace(osl=3, arrival_time_ms=0.0, finish_ms=100.0),
        SimpleNamespace(osl=3, arrival_time_ms=0.0, finish_ms=200.0),
    ]

    metric = validate._full_closed_loop_metric(
        completed,
        num_gpus=2,
        expected_requests=2,
    )

    assert metric.output_tokens == 6
    assert metric.wall_ms == pytest.approx(200.0)
    assert metric.throughput_tok_s == pytest.approx(30.0)
    assert metric.throughput_tok_s_gpu == pytest.approx(15.0)


def test_full_closed_loop_metric_rejects_incomplete_validation_run() -> None:
    with pytest.raises(AssertionError, match="completed_request_count"):
        validate._full_closed_loop_metric(
            [SimpleNamespace(osl=3, finish_ms=200.0)],
            num_gpus=2,
            expected_requests=2,
        )


def test_official_dp2_validation_uses_full_closed_loop_metric(monkeypatch) -> None:
    point = _point("K2.5-tp4ep8dp2-8k2k-bt65536")
    monkeypatch.setattr(validate, "MULTI_CONFIG_DATA", [point])

    def fake_collect(self, completed, num_gpus, total_iters, *args, **kwargs):
        return SimpleNamespace(throughput_tok_s=999.0, throughput_tok_s_gpu=999.0)

    monkeypatch.setattr(validate.CBSimulator, "_collect_metrics", fake_collect)
    monkeypatch.setattr(
        validate,
        "_load_model_and_db",
        lambda **kwargs: (object(), object(), object()),
    )

    class FakeSummary:
        def __init__(self, result) -> None:
            self.result = result

        def get_result_dict(self):
            return {"tokens/s/gpu": self.result.throughput_tok_s_gpu}

        def get_per_ops_data(self):
            return {"cb_sim_boundary": {"throughput_source": "unit"}}

    class FakeBackend:
        def run_agg(self, *args, **kwargs):
            config = kwargs["cb_config"]
            completed = [
                SimpleNamespace(
                    osl=point.osl,
                    arrival_time_ms=0.0,
                    finish_ms=1000.0,
                )
                for _ in range(config.num_requests)
            ]
            result = validate.CBSimulator._collect_metrics(
                SimpleNamespace(_config=config),
                completed,
                point.tp,
                1,
            )
            return FakeSummary(result)

    monkeypatch.setattr(validate, "VLLMBackend", FakeBackend)

    errors = validate._run_multi_config_validation(
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
        ep8_per_iteration_overhead_ms=0.0,
        verbose=False,
    )

    expected = point.osl * 256 / point.tp
    assert errors == [pytest.approx(expected / point.real_output_tok_s_gpu)]

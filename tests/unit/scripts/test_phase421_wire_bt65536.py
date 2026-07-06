from scripts import validate_cb_simulator as validate_cb


def _multi_config_point(name: str):
    return next(pt for pt in validate_cb.MULTI_CONFIG_DATA if pt.name == name)


def test_phase421_multi_config_validation_wires_bt65536_config(monkeypatch) -> None:
    point = _multi_config_point("K2.5-tp8ep8-8k2k-bt65536")
    monkeypatch.setattr(validate_cb, "MULTI_CONFIG_DATA", [point])

    class FakeSummary:
        def get_result_dict(self) -> dict[str, float]:
            return {"tokens/s/gpu": point.real_output_tok_s_gpu}

        def get_per_ops_data(self) -> dict[str, dict[str, str]]:
            return {"cb_sim_boundary": {"throughput_source": "unit"}}

    class FakeBackend:
        def run_agg(self, *args, **kwargs) -> FakeSummary:
            cb_config = kwargs["cb_config"]
            assert kwargs["ctx_tokens"] == point.max_num_batched_tokens
            assert cb_config.max_num_batched_tokens == point.max_num_batched_tokens
            assert cb_config.num_gpu_blocks == validate_cb._multi_config_num_gpu_blocks(point)
            assert cb_config.block_size == validate_cb.KV_CACHE_BLOCK_SIZE
            assert cb_config.max_num_seqs == 256
            return FakeSummary()

    monkeypatch.setattr(validate_cb, "VLLMBackend", FakeBackend)
    monkeypatch.setattr(validate_cb, "_load_model_and_db", lambda **kwargs: (object(), object(), object()))

    errs = validate_cb._run_multi_config_validation(
        overlap_factor=0.0,
        per_iteration_overhead_ms=0.0,
        ep8_per_iteration_overhead_ms=0.0,
        verbose=False,
    )

    assert errs == [1.0]

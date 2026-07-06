import scripts.validate_cb_simulator as validate_cb


def test_phase422_dp2_bt65536_uses_real_retry5e_capacity_config() -> None:
    point = next(
        pt for pt in validate_cb.MULTI_CONFIG_DATA
        if pt.name == "K2.5-tp4ep8dp2-8k2k-bt65536"
    )

    capacity = validate_cb._multi_config_kv_capacity(point)
    config = validate_cb._make_multi_config_cb_config(
        point,
        overlap_factor=0.0,
        ep8_per_iteration_overhead_ms=0.0,
    )

    assert capacity.kv_cache_tokens == 25_744
    assert capacity.num_gpu_blocks == 1_609
    assert capacity.max_num_seqs == 128
    assert config.num_gpu_blocks == 1_609
    assert config.max_num_batched_tokens == 65_536
    assert config.max_num_seqs == 128

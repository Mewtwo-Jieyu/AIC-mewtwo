from pathlib import Path


def test_phase423_runner_uses_clean_dp2_bt65536_protocol() -> None:
    runner = Path("collector/vllm/run_phase423_bt65536_recollect.sh")
    text = runner.read_text()

    assert 'SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k-bt65536}"' in text
    assert 'TP="${TP:-4}"' in text
    assert 'DP="${DP:-2}"' in text
    assert 'ISL="${ISL:-8000}"' in text
    assert 'OSL="${OSL:-2000}"' in text
    assert 'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-65536}"' in text
    assert "--no-enable-prefix-caching" in text
    assert "--enable-logging-iteration-details" in text
    assert "--cudagraph-metrics" in text
    assert "--prompt-variant-mode rotating" in text
    assert 'PHASE_NAME="${PHASE_NAME:-phase423}"' in text
    assert '"phase": "${PHASE_NAME}"' in text
    assert "snapshot_process_residuals" in text
    assert "snapshot_gpu_apps \"${OUT_ROOT}/gpu_compute_apps_after.txt\"" in text
    assert "snapshot_process_residuals \"${OUT_ROOT}/process_residual_after.txt\"" in text

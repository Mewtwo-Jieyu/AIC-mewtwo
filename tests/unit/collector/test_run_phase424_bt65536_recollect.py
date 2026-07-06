from pathlib import Path


def test_phase424_runner_only_changes_protocol_version_and_max_model_len() -> None:
    phase423 = Path("collector/vllm/run_phase423_bt65536_recollect.sh").read_text()
    runner = Path("collector/vllm/run_phase424_bt65536_recollect.sh")
    text = runner.read_text()

    assert 'PHASE_NAME="${PHASE_NAME:-phase423}"' in phase423
    assert 'OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase423_bt65536_recollect}"' in phase423
    assert 'MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"' in phase423
    assert '"phase": "${PHASE_NAME}"' in phase423

    assert 'export PHASE_NAME="${PHASE_NAME:-phase424}"' in text
    assert 'export OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase424_bt65536_recollect}"' in text
    assert 'export TMP_ROOT="${TMP_ROOT:-/tmp/phase424_bt65536_recollect_$$}"' in text
    assert 'export MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"' in text
    assert 'exec "$(dirname "$0")/run_phase423_bt65536_recollect.sh" "$@"' in text

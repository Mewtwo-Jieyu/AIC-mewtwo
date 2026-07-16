from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector/vllm/run_phase462_dp_route_observation.sh"


def test_runner_is_valid_shell_and_pins_the_approved_protocol() -> None:
    completed = subprocess.run(
        ["bash", "-n", str(RUNNER)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr

    text = RUNNER.read_text(encoding="utf-8")
    for value in (
        "CAPTURE_PROMPTS=512",
        "CONCURRENCY=128",
        "ISL=8000",
        "OSL=2000",
        "MAX_BT=65536",
        "TP=4 DP=2 EP=8",
        "MAX_MODEL_LEN=131072",
        'MAX_OVERHEAD_PCT=2.0',
        '"diagnostic_only": true',
        '"valid_for_default": false',
        '"perf_database": false',
    ):
        assert value in text


def test_runner_restores_both_vllm_sources() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert 'sha256sum "${CORE_CLIENT_PATH}" "${ENGINE_PATH}"' in text
    assert "source_restore_hash_mismatch" in text
    assert "trap cleanup EXIT" in text

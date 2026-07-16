from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector/vllm/run_phase462_dp_route_observation.sh"
POSTPROCESS_HASHES = REPO_ROOT / (
    "docs/iter_gap_investigation/phase462_dp_route_observation/"
    "tools_postprocess.sha256"
)


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


def test_runner_does_not_hide_residual_probe_failures() -> None:
    text = RUNNER.read_text(encoding="utf-8")

    assert '2>/dev/null || true' not in text
    assert 'record_residuals || status=1' in text
    assert 'pgrep_status' in text


def test_postprocess_hashes_match_committed_tools() -> None:
    tools = {
        "phase462_dp_route_observation_patch.py": REPO_ROOT
        / "collector/vllm/phase462_dp_route_observation_patch.py",
        "analyze_phase462_dp_route_observation.py": REPO_ROOT
        / "scripts/analyze_phase462_dp_route_observation.py",
        "run_phase462_dp_route_observation.sh": RUNNER,
    }

    for line in POSTPROCESS_HASHES.read_text(encoding="utf-8").splitlines():
        expected, name = line.split(maxsplit=1)
        actual = hashlib.sha256(tools[name].read_bytes()).hexdigest()
        assert actual == expected

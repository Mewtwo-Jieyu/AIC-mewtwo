from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_phase425_runner_wraps_phase412_with_8k2k_defaults():
    runner = REPO_ROOT / "collector/vllm/run_phase425_8k2k_sweep.sh"
    text = runner.read_text(encoding="utf-8")

    assert 'export PHASE_NAME="${PHASE_NAME:-phase425}"' in text
    assert 'export OUT_ROOT="${OUT_ROOT:-docs/iter_gap_investigation/phase425_8k2k_sweep}"' in text
    assert 'export SCENARIO="${SCENARIO:-K2.5-tp4ep8dp2-8k2k}"' in text
    assert 'export ISL="${ISL:-8000}"' in text
    assert 'export OSL="${OSL:-2000}"' in text
    assert 'export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8000}"' in text
    assert 'export BENCH_NUM_PROMPTS="${BENCH_NUM_PROMPTS:-512}"' in text
    assert 'export BENCH_MAX_CONCURRENCY="${BENCH_MAX_CONCURRENCY:-128}"' in text
    assert 'run_phase412_arrival_sweep.sh' in text


def test_phase412_runner_allows_phase_name_override_for_wrappers():
    runner = REPO_ROOT / "collector/vllm/run_phase412_arrival_sweep.sh"
    text = runner.read_text(encoding="utf-8")

    assert 'PHASE_NAME="${PHASE_NAME:-phase412}"' in text
    assert '"phase": "${PHASE_NAME}"' in text

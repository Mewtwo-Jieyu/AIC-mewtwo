from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SUPERVISOR = REPO_ROOT / "collector/vllm/run_phase463_background_supervisor.sh"


def test_supervisor_waits_for_existing_runner_and_checks_all_remaining_points() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")

    assert 'WAIT_PID="${WAIT_PID:-}"' in text
    assert 'START_INDEX="${START_INDEX:-2}"' in text
    assert 'kill -0 "${WAIT_PID}"' in text
    assert "K2.5-tp8ep8-32k3k" in text
    assert "K2.5-tp4ep8dp2-32k3k-c64-diagnostic" in text
    assert "scenario_analysis.json" in text


def test_supervisor_archives_failed_points_and_continues_without_overwrite() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")

    assert "archive_partial" in text
    assert 'mv "${out_dir}" "${archived}"' in text
    assert "POINT_SKIPPED_AFTER_FAILURE" in text
    assert "continue" in text
    assert "FORCE=1" not in text
    assert 'ONLY_POINTS="${idx}"' in text


def test_supervisor_is_suitable_for_nohup() -> None:
    text = SUPERVISOR.read_text(encoding="utf-8")

    assert "trap" not in text
    assert "git commit" not in text
    assert "git push" not in text
    assert "supervisor_results.tsv" in text

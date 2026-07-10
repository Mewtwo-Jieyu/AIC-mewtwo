from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUNNER = REPO_ROOT / "collector/vllm/run_phase461_cost_recollect.sh"
COLLECTOR = REPO_ROOT / "collector/vllm/collect_phase461_mla_grid.py"


def test_runner_pins_hardware_environment_and_new_shared_path():
    text = RUNNER.read_text(encoding="utf-8")

    assert "/mnt/shared-storage-user/zhaojieyu/backup/aic" in text
    assert "VLLM_ENABLE_CUDA_COMPATIBILITY" in text
    assert "/usr/local/cuda-12.9/compat" in text


def test_runner_separates_four_measurement_tasks():
    text = RUNNER.read_text(encoding="utf-8")

    assert 'TASKS="${TASKS:-A B C D}"' in text
    assert "run_mla_grid" in text
    assert "run_tp8_bt_b2b" in text
    assert "run_tp8_32k_b2b" in text
    assert "run_dp2_bt_diagnostic" in text
    assert "paired_rank_event_plus_iteration_wall" in text


def test_runner_uses_committed_collector_snapshot():
    text = RUNNER.read_text(encoding="utf-8")

    assert "COMMITTED_COLLECTOR_ROOT" in text
    assert "collect_phase461_mla_grid.py" in text
    assert "collector_commit.txt" in text


def test_mla_collector_has_exact_grid_and_three_repeats():
    text = COLLECTOR.read_text(encoding="utf-8")

    assert "BATCHES = (8, 16)" in text
    assert "TARGET_SEQ_LENS = (16_384, 32_768, 65_536)" in text
    assert "REPEATS = 3" in text
    assert "GLOBAL_NUM_HEADS = 64" in text
    assert "TP_SIZE = 8" in text
    assert "MAX_REPEAT_SPREAD = 1.15" in text

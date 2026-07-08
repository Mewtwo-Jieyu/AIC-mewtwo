import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase446_lockstep_replay.py"
    spec = importlib.util.spec_from_file_location("phase446_replay", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_phase446_replay_uses_max_busy_per_coupled_cycle():
    phase446 = _load_module()
    cycles = [
        phase446.CoupledCycle(
            index=0,
            engine0_ctx_tokens=8000,
            engine0_decode_batch=40,
            engine0_busy_ms=1000.0,
            engine0_wall_ms=1010.0,
            engine1_ctx_tokens=0,
            engine1_decode_batch=40,
            engine1_busy_ms=35.0,
            engine1_wall_ms=36.0,
        ),
        phase446.CoupledCycle(
            index=1,
            engine0_ctx_tokens=0,
            engine0_decode_batch=40,
            engine0_busy_ms=36.0,
            engine0_wall_ms=37.0,
            engine1_ctx_tokens=8000,
            engine1_decode_batch=40,
            engine1_busy_ms=990.0,
            engine1_wall_ms=1000.0,
        ),
    ]

    result = phase446.replay_window("unit", cycles)

    assert result.cycle_count == 2
    assert result.replay_wall_ms == 1990.0
    assert result.real_wall_ms == 1047.0
    assert result.sum_max_wall_ms == 2010.0
    assert abs(result.replay_vs_sum_max_wall_error_pct - -0.9950248756218906) < 1e-9


def test_phase446_grouping_reports_key_mismatch_inside_tp_group():
    phase446 = _load_module()
    records = [
        phase446.EventRecord("0", 8000, 0, 10.0),
        phase446.EventRecord("0", 0, 1, 11.0),
    ]

    grouped = phase446.group_event_steps(records, tp_width=2)

    assert grouped.mismatch_count == 1
    assert grouped.steps == []


if __name__ == "__main__":
    test_phase446_replay_uses_max_busy_per_coupled_cycle()
    test_phase446_grouping_reports_key_mismatch_inside_tp_group()

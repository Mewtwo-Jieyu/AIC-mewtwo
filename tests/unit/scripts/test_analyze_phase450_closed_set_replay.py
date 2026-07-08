import importlib.util
import sys
from pathlib import Path


def _load_module():
    path = Path(__file__).resolve().parents[3] / "scripts" / "analyze_phase450_closed_set_replay.py"
    spec = importlib.util.spec_from_file_location("analyze_phase450_closed_set_replay", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_total_wall_throughput_uses_closed_set_wall_time() -> None:
    phase450 = _load_module()
    rows = [
        {"start_ms": 10.0, "end_ms": 60.0, "replica_id": 0, "decode_reqs": 2},
        {"start_ms": 60.0, "end_ms": 110.0, "replica_id": 1, "decode_reqs": 2},
    ]

    metric = phase450.total_wall_metric(rows, num_requests=4, output_tokens_per_request=2, num_gpus=2)

    assert metric.wall_ms == 100.0
    assert metric.output_tok_s_gpu == 40.0


def test_lockstep_padding_summary_counts_peer_extra() -> None:
    phase450 = _load_module()
    rows = [
        {"cycle": 0, "replica_id": 0, "charge_ms": 100.0, "phase": "mixed_prefill"},
        {"cycle": 0, "replica_id": 1, "charge_ms": 25.0, "phase": "decode"},
        {"cycle": 1, "replica_id": 0, "charge_ms": 30.0, "phase": "decode"},
        {"cycle": 1, "replica_id": 1, "charge_ms": 30.0, "phase": "decode"},
    ]

    summary = phase450.lockstep_padding_summary(rows)

    assert summary["cycles"] == 2
    assert summary["padded_extra_ms"] == 75.0
    assert summary["padded_extra_share"] == 75.0 / 130.0
    assert summary["mixed_decode_share"] == 0.5


def test_closed_set_gate_checks_direction_and_split() -> None:
    phase450 = _load_module()

    verdict = phase450.closed_set_gate(
        sim_n128_tput=100.0,
        sim_n512_tput=130.0,
        sim_n128_split=1.70,
        sim_n512_split=1.02,
        real_artifact=1.30,
        real_n128_split=1.723,
        real_n512_split=1.008,
        tolerance=0.10,
    )

    assert verdict["passed"] is True
    assert verdict["sim_artifact"] == 1.3


if __name__ == "__main__":
    test_total_wall_throughput_uses_closed_set_wall_time()
    test_lockstep_padding_summary_counts_peer_extra()
    test_closed_set_gate_checks_direction_and_split()

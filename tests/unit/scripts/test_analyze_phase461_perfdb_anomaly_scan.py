import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase461_perfdb_anomaly_scan.py"
SPEC = importlib.util.spec_from_file_location("analyze_phase461_perfdb_anomaly_scan", MODULE_PATH)
analyzer = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyzer)


def test_scan_axis_flags_strict_spike_and_valley():
    rows = [
        {"kernel": "a", "batch_size": "1", "latency": "1.0", "__line__": 2},
        {"kernel": "a", "batch_size": "2", "latency": "10.0", "__line__": 3},
        {"kernel": "a", "batch_size": "3", "latency": "1.0", "__line__": 4},
        {"kernel": "b", "batch_size": "1", "latency": "10.0", "__line__": 5},
        {"kernel": "b", "batch_size": "2", "latency": "1.0", "__line__": 6},
        {"kernel": "b", "batch_size": "3", "latency": "10.0", "__line__": 7},
    ]

    anomalies = analyzer.scan_axis(rows, axis="batch_size", threshold=3.0)

    assert [(row.line, row.direction) for row in anomalies] == [(3, "spike"), (6, "valley")]
    assert all(row.deviation_ratio == 10.0 for row in anomalies)


def test_scan_axis_never_mixes_different_semantic_keys():
    rows = [
        {"kernel": "a", "batch_size": "1", "latency": "1.0", "__line__": 2},
        {"kernel": "b", "batch_size": "2", "latency": "20.0", "__line__": 3},
        {"kernel": "a", "batch_size": "3", "latency": "1.0", "__line__": 4},
    ]

    assert analyzer.scan_axis(rows, axis="batch_size", threshold=3.0) == []


def test_generation_mla_effective_sequence_is_isl_plus_step():
    rows = [
        {"batch_size": "8", "isl": "1", "step": "16383", "latency": "1.0"},
        {"batch_size": "8", "isl": "1", "step": "32767", "latency": "10.0"},
        {"batch_size": "8", "isl": "1", "step": "65535", "latency": "1.0"},
    ]

    analyzer.add_effective_sequence(rows)

    assert [row["effective_seq"] for row in rows] == [16_384, 32_768, 65_536]

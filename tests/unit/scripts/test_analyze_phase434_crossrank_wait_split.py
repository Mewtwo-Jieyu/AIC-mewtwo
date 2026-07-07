import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase434_crossrank_wait_split as phase434


def _write_trace(path: Path, *, rank: int, clock_offset_ms: float) -> None:
    offset_us = clock_offset_ms * 1000.0
    events = [
        {
            "name": "ProfilerStep#0",
            "cat": "user_annotation",
            "ph": "X",
            "ts": offset_us,
            "dur": 300_000.0,
        },
        {
            "name": "execute_context_32000(32000)_generation_6(6)",
            "cat": "user_annotation",
            "ph": "X",
            "ts": offset_us + 10.0,
            "dur": 299_000.0,
        },
    ]
    if rank == 0:
        kernels = [
            ("void marlin_moe_wna16::Marlin<test>", 60_000.0, 30_000.0),
            ("ncclDevKernel_AllGather_RING_LL(test)", 100_000.0, 40_000.0),
        ]
    else:
        kernels = [
            ("void marlin_moe_wna16::Marlin<test>", 50_000.0, 50_000.0),
            ("ncclDevKernel_AllGather_RING_LL(test)", 120_000.0, 20_000.0),
        ]
    for name, start_us, dur_us in kernels:
        events.append(
            {
                "name": name,
                "cat": "kernel",
                "ph": "X",
                "ts": offset_us + start_us,
                "dur": dur_us,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)


def _fixture_root(tmp_path: Path) -> Path:
    artifact = tmp_path / "phase433_nsys_prefill" / phase434.DEFAULT_SCENARIO / "prof_prefill"
    _write_trace(artifact / "dp0_pp0_tp0_dcp0_ep0_rank0.pt.trace.json.gz", rank=0, clock_offset_ms=0.0)
    _write_trace(artifact / "dp0_pp0_tp1_dcp0_ep1_rank1.pt.trace.json.gz", rank=1, clock_offset_ms=10.0)
    return tmp_path / "phase433_nsys_prefill"


def test_phase434_splits_collective_wait_and_keeps_no_go() -> None:
    with tempfile.TemporaryDirectory() as raw:
        rows = phase434.build_phase434_rows(artifact_root=_fixture_root(Path(raw)))

    summary = [row for row in rows if row["row_type"] == "summary"][0]
    collective = [row for row in rows if row["row_type"] == "collective_split"][0]
    imbalance = [row for row in rows if row["row_type"] == "imbalance_summary"][0]

    assert summary["source"] == phase434.SOURCE
    assert summary["clock_alignment_gate"] == "passed"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"
    assert float(collective["wait_ms_per_step"]) == 10.0
    assert float(collective["transfer_ms_per_step"]) == 20.0
    assert float(imbalance["moe_hot_mean_ratio"]) > 1.20
    assert summary["runtime_modified"] == "false"
    assert summary["perf_database"] == "false"


def test_phase434_writer_rejects_default_claim() -> None:
    rows = [{field: "" for field in phase434.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase434.SOURCE,
            "row_type": "summary",
            "scenario": phase434.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "true",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    try:
        phase434.write_phase434_csv(Path("/tmp/phase434_bad.csv"), rows)
    except ValueError as exc:
        assert "valid_for_default" in str(exc)
    else:
        raise AssertionError("writer accepted valid_for_default=true")


if __name__ == "__main__":
    test_phase434_splits_collective_wait_and_keeps_no_go()
    test_phase434_writer_rejects_default_claim()

import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase433_nsys_prefill as phase433


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path: Path, *, rank: int, ep_ms: float) -> None:
    wall_ms = 4718.954849
    events: list[dict[str, object]] = [
        {
            "name": "ProfilerStep#0",
            "cat": "user_annotation",
            "ph": "X",
            "ts": 0.0,
            "dur": wall_ms * 1000.0,
        },
        {
            "name": "execute_context_32000(32000)_generation_6(6)",
            "cat": "user_annotation",
            "ph": "X",
            "ts": 10.0,
            "dur": wall_ms * 1000.0 - 20.0,
        },
    ]
    kernels = [
        ("ncclDevKernel_AllGather_RING_LL(test)", ep_ms),
        ("void marlin_moe_wna16::Marlin<test>", 352.717448),
        ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 402.019077),
        ("cublasLtMatmulKernel", 255.654474),
        ("custom_ar_all_reduce", 242.290304),
        ("phase433_other_kernel", 61.549135),
    ]
    cursor = 100.0
    for name, dur_ms in kernels:
        events.append(
            {
                "name": name,
                "cat": "kernel",
                "ph": "X",
                "ts": cursor,
                "dur": dur_ms * 1000.0,
            }
        )
        cursor += dur_ms * 1000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)


def _fixture_root(tmp_path: Path) -> tuple[Path, Path]:
    scenario = phase433.DEFAULT_SCENARIO
    artifact_root = tmp_path / "phase433_nsys_prefill"
    artifact_dir = artifact_root / scenario
    _write_trace(
        artifact_dir / "prof_prefill/dp0_pp0_tp0_dcp0_ep0_rank0.pt.trace.json.gz",
        rank=0,
        ep_ms=2700.0,
    )
    _write_trace(
        artifact_dir / "prof_prefill/dp0_pp0_tp1_dcp0_ep0_rank1.pt.trace.json.gz",
        rank=1,
        ep_ms=3300.0,
    )
    (artifact_dir / "meta.json").write_text(
        json.dumps(
            {
                "phase": "phase433",
                "profiler_mode": "torch_profiler_cpu_fallback",
                "nsys_available": False,
            }
        ),
        encoding="utf-8",
    )
    phase415_csv = tmp_path / "phase415.csv"
    _write_csv(
        phase415_csv,
        [
            {"row_type": "mixed_component", "scenario": scenario, "component": "prefill_mla_attention", "sim_mean_ms": "402.019077"},
            {"row_type": "mixed_component", "scenario": scenario, "component": "moe_compute", "sim_mean_ms": "352.717448"},
            {"row_type": "mixed_component", "scenario": scenario, "component": "ep_dispatch_combine", "sim_mean_ms": "497.491058"},
            {"row_type": "mixed_component", "scenario": scenario, "component": "gemm_other", "sim_mean_ms": "255.654474"},
            {"row_type": "mixed_component", "scenario": scenario, "component": "tp_comm", "sim_mean_ms": "242.290304"},
            {"row_type": "mixed_component", "scenario": scenario, "component": "other", "sim_mean_ms": "61.549135"},
            {"row_type": "summary", "scenario": scenario, "component": "summary", "sim_mean_ms": "1818.637996"},
        ],
    )
    return artifact_root, phase415_csv


def test_phase433_decomposes_fallback_trace_and_keeps_default_no_go() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        artifact_root, phase415_csv = _fixture_root(tmp_path)
        rows = phase433.build_phase433_rows(
            artifact_root=artifact_root,
            phase415_csv=phase415_csv,
            scenario=phase433.DEFAULT_SCENARIO,
        )
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    gpu_rows = [row for row in rows if row["row_type"] == "per_gpu_busy"]

    assert summary["trace_precision"] == "torch_profiler_fallback"
    assert summary["reconstruction_gate"] == "passed"
    assert "collective_or_transfer" in summary["dominant_mechanism"]
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"
    assert float(summary["max_mean_busy_ratio"]) > 1.0
    assert len(gpu_rows) == 2


def test_phase433_window_check_requires_32k_prefill_step() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        artifact_root, phase415_csv = _fixture_root(tmp_path)
        check = phase433.build_window_check(
            artifact_root=artifact_root,
            scenario=phase433.DEFAULT_SCENARIO,
            window="prefill",
            phase415_csv=phase415_csv,
        )
    assert check["passed"] is True
    assert check["prefill_step_count"] == 2
    assert check["max_ctx_tokens"] == 32000


def test_phase433_writer_rejects_default_claim() -> None:
    rows = [{field: "" for field in phase433.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase433.SOURCE,
            "row_type": "summary",
            "scenario": phase433.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "true",
            "ssh_allowed": "true",
            "runtime_modified": "false",
            "perf_database": "false",
            "valid_for_default": "true",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    try:
        phase433.write_phase433_csv(Path("/tmp/phase433_bad.csv"), rows)
    except ValueError as exc:
        assert "valid_for_default" in str(exc)
    else:
        raise AssertionError("writer accepted valid_for_default=true")


if __name__ == "__main__":
    test_phase433_decomposes_fallback_trace_and_keeps_default_no_go()
    test_phase433_window_check_requires_32k_prefill_step()
    test_phase433_writer_rejects_default_claim()

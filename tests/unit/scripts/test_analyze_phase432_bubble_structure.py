import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase432_bubble_structure as phase432


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path: Path, steps: list[dict[str, object]]) -> None:
    events: list[dict[str, object]] = []
    for idx, step in enumerate(steps):
        start_us = idx * 200_000.0
        wall_us = float(step["wall_ms"]) * 1000.0
        events.append(
            {
                "name": f"ProfilerStep#{idx}",
                "cat": "user_annotation",
                "ph": "X",
                "ts": start_us,
                "dur": wall_us,
            }
        )
        events.append(
            {
                "name": (
                    f"execute_context_{step['ctx_tokens']}({step['ctx_tokens']})_"
                    f"generation_{step['gen_tokens']}({step['gen_tokens']})"
                ),
                "cat": "user_annotation",
                "ph": "X",
                "ts": start_us + 5.0,
                "dur": wall_us - 10.0,
            }
        )
        cursor_us = start_us + 1000.0
        for kernel in step["kernels"]:
            events.append(
                {
                    "name": kernel["name"],
                    "cat": "kernel",
                    "ph": "X",
                    "ts": cursor_us,
                    "dur": float(kernel["dur_ms"]) * 1000.0,
                }
            )
            cursor_us += float(kernel["dur_ms"]) * 1000.0 + float(kernel.get("gap_after_ms", 0.0)) * 1000.0
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)


def _fixture_root(tmp_path: Path) -> tuple[Path, Path]:
    scenario = phase432.DEFAULT_SCENARIO
    artifact_root = tmp_path / "phase429_reprofile"
    artifact_dir = artifact_root / scenario
    windows = {
        "w0_prefill": [
            {
                "ctx_tokens": 8000,
                "gen_tokens": 8,
                "wall_ms": 80.0,
                "kernels": [
                    {"name": "ncclDevKernel_AllGather_RING_LL(test)", "dur_ms": 10.0, "gap_after_ms": 2.0},
                    {"name": "void marlin_moe_wna16::Marlin<test>", "dur_ms": 20.0, "gap_after_ms": 2.0},
                    {"name": "ncclDevKernel_AllGather_RING_LL(test)", "dur_ms": 10.0},
                ],
            }
        ],
        "w1_decode_c16": [
            {
                "ctx_tokens": 0,
                "gen_tokens": 8,
                "wall_ms": 40.0,
                "kernels": [
                    {"name": "ncclDevKernel_AllGather_RING_LL(test)", "dur_ms": 5.0, "gap_after_ms": 1.0},
                    {"name": "void marlin_moe_wna16::Marlin<test>", "dur_ms": 10.0},
                ],
            }
        ],
        "w2_decode_c64": [
            {
                "ctx_tokens": 0,
                "gen_tokens": 32,
                "wall_ms": 70.0,
                "kernels": [
                    {"name": "ncclDevKernel_AllGather_RING_LL(test)", "dur_ms": 7.0, "gap_after_ms": 2.0},
                    {"name": "void marlin_moe_wna16::Marlin<test>", "dur_ms": 20.0},
                ],
            }
        ],
        "w3_decode_c128": [
            {
                "ctx_tokens": 0,
                "gen_tokens": 64,
                "wall_ms": 120.0,
                "kernels": [
                    {"name": "ncclDevKernel_AllGather_RING_LL(test)", "dur_ms": 9.0, "gap_after_ms": 3.0},
                    {"name": "void marlin_moe_wna16::Marlin<test>", "dur_ms": 30.0},
                ],
            }
        ],
    }
    for window, steps in windows.items():
        _write_trace(
            artifact_dir / f"prof_{window}/dp0_pp0_tp0_dcp0_ep0_rank0.test.pt.trace.json.gz",
            steps,
        )
    phase429_csv = tmp_path / "phase429.csv"
    _write_csv(
        phase429_csv,
        [
            {
                "source": "phase429_reprofile",
                "row_type": "summary",
                "scenario": scenario,
                "prefill_step_count": "1",
                "decode_step_count": "3",
                "prefill_verdict": "prefill_serialization_bubble_unmodeled",
                "decode_verdict": "decode_mla_attention_slope_dominates",
                "phase426_steady_metric_penalty": "1.516977",
            }
        ],
    )
    return artifact_root, phase429_csv


def test_phase432_extracts_gap_timeline_and_rejects_inconsistent_structure() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        artifact_root, phase429_csv = _fixture_root(tmp_path)
        rows = phase432.build_phase432_rows(
            artifact_root=artifact_root,
            phase429_csv=phase429_csv,
            scenario=phase432.DEFAULT_SCENARIO,
        )
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    gap_rows = [row for row in rows if row["row_type"] == "gap_summary"]

    assert summary["phase432_verdict"] == "execution_overhead_structure_unresolved"
    assert summary["selected_hypothesis"] == "none"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"
    assert len(gap_rows) == 4
    assert float(gap_rows[0]["gap_ms_per_step_mean"]) > 0.0


def test_phase432_writer_rejects_runtime_claim() -> None:
    rows = [{field: "" for field in phase432.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase432.SOURCE,
            "row_type": "summary",
            "scenario": phase432.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "false",
            "ssh_allowed": "false",
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    try:
        phase432.write_phase432_csv(Path("/tmp/phase432_bad.csv"), rows)
    except ValueError as exc:
        assert "runtime_modified" in str(exc)
    else:
        raise AssertionError("writer accepted runtime_modified=true")


if __name__ == "__main__":
    test_phase432_extracts_gap_timeline_and_rejects_inconsistent_structure()
    test_phase432_writer_rejects_runtime_claim()

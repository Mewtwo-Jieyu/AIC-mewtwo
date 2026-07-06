import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase429_reprofile as phase429


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path, steps):
    events = []
    for idx, step in enumerate(steps):
        start = idx * 100_000.0
        wall = step["wall_ms"] * 1000.0
        events.append({"name": f"ProfilerStep#{idx}", "cat": "user_annotation", "ph": "X", "ts": start, "dur": wall})
        events.append(
            {
                "name": f"execute_context_{step['ctx_tokens']}({step['ctx_tokens']})_generation_{step['gen_tokens']}({step['gen_tokens']})",
                "cat": "user_annotation",
                "ph": "X",
                "ts": start + 10.0,
                "dur": wall - 20.0,
            }
        )
        offset = 100.0
        for name, dur_ms in step["kernels"]:
            events.append({"name": name, "cat": "kernel", "ph": "X", "ts": start + offset, "dur": dur_ms * 1000.0})
            offset += max(10.0, dur_ms * 1000.0 / 10.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)


def _fixture_root(tmp_path):
    scenario = phase429.DEFAULT_SCENARIO
    artifact_root = tmp_path / "phase429_reprofile"
    artifact_dir = artifact_root / scenario
    windows = {
        "w0_prefill": [
            {
                "ctx_tokens": 8000,
                "gen_tokens": 8,
                "wall_ms": 120.0,
                "kernels": [
                    ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 40.0),
                    ("void marlin_moe_wna16::Marlin<test>", 25.0),
                    ("ncclDevKernel_AllGather_RING_LL(test)", 20.0),
                ],
            }
        ],
        "w1_decode_c16": [
            {
                "ctx_tokens": 0,
                "gen_tokens": 8,
                "wall_ms": 20.0,
                "kernels": [
                    ("void marlin_moe_wna16::Marlin<test>", 2.0),
                    ("ncclDevKernel_AllGather_RING_LL(test)", 3.0),
                    ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 6.0),
                ],
            }
        ],
        "w2_decode_c64": [
            {
                "ctx_tokens": 0,
                "gen_tokens": 32,
                "wall_ms": 35.0,
                "kernels": [
                    ("void marlin_moe_wna16::Marlin<test>", 14.0),
                    ("ncclDevKernel_AllGather_RING_LL(test)", 4.0),
                    ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 7.0),
                ],
            }
        ],
        "w3_decode_c128": [
            {
                "ctx_tokens": 0,
                "gen_tokens": 64,
                "wall_ms": 58.0,
                "kernels": [
                    ("void marlin_moe_wna16::Marlin<test>", 34.0),
                    ("ncclDevKernel_AllGather_RING_LL(test)", 5.0),
                    ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 9.0),
                ],
            }
        ],
    }
    for window, steps in windows.items():
        _write_trace(artifact_dir / f"prof_{window}/dp0_pp0_tp0_dcp0_ep0_rank0.test.pt.trace.json.gz", steps)
    phase426_csv = tmp_path / "phase426.csv"
    _write_csv(
        phase426_csv,
        [
            {
                "source": "phase426_8k2k_steady_decompose",
                "row_type": "summary",
                "scenario": scenario,
                "steady_metric_penalty": "1.516977",
                "prefill_attribution_share": "0.595460",
                "decode_gap_attribution_share": "0.357492",
                "peer_stall_attribution_share": "0.047048",
            }
        ],
    )
    return artifact_root, phase426_csv


def test_phase429_prefill_hit_and_decode_slope_verdicts():
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        artifact_root, phase426_csv = _fixture_root(tmp_path)
        rows = phase429.build_phase429_rows(
            artifact_root=artifact_root,
            phase426_csv=phase426_csv,
            scenario=phase429.DEFAULT_SCENARIO,
        )
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    slope = [row for row in rows if row["row_type"] == "decode_slope" and row["category"] == "moe_gemm_or_aux"][0]

    assert summary["prefill_hit_gate"] == "passed"
    assert summary["decode_batch_span_gate"] == "passed"
    assert summary["decode_verdict"] == "decode_moe_gemm_or_aux_slope_dominates"
    assert summary["mechanism_verdict"] == "phase429_reprofile_window_hits_attributed"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"
    assert float(slope["real_slope_ms_per_request"]) > 0.0


def test_phase429_writer_rejects_default_claim():
    rows = [{field: "" for field in phase429.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase429.SOURCE,
            "row_type": "summary",
            "scenario": phase429.DEFAULT_SCENARIO,
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
        phase429.write_phase429_csv(Path("/tmp/phase429_bad.csv"), rows)
    except ValueError as exc:
        assert "valid_for_default" in str(exc)
    else:
        raise AssertionError("writer accepted valid_for_default=true")


def test_phase429_fast_window_check_uses_iteration_logs():
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        scenario = phase429.DEFAULT_SCENARIO
        artifact_root = tmp_path / "phase429_reprofile"
        artifact_dir = artifact_root / scenario
        prof_dir = artifact_dir / "prof_w0_prefill"
        prof_dir.mkdir(parents=True)
        (prof_dir / "dp0_pp0_tp0_dcp0_ep0_rank0.test.pt.trace.json.gz").write_bytes(b"not parsed by fast check")
        (artifact_dir / "serve_w0_prefill_attempt1.log").write_text(
            "(EngineCore_DP0 pid=1) INFO Iteration(0): "
            "1 context requests, 8000 context tokens, "
            "0 generation requests, 0 generation tokens, iteration elapsed time: 1.0 ms\n",
            encoding="utf-8",
        )
        check = phase429.build_window_check_from_logs(
            artifact_root=artifact_root,
            scenario=scenario,
            window="w0_prefill",
        )
    assert check.passed is True
    assert check.prefill_step_count == 1
    assert check.reason == "prefill_step_seen"


if __name__ == "__main__":
    test_phase429_prefill_hit_and_decode_slope_verdicts()
    test_phase429_writer_rejects_default_claim()
    test_phase429_fast_window_check_uses_iteration_logs()

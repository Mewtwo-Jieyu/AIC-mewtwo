import csv
import gzip
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase428_perstep_kernel_attrib as phase428


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
        events.append(
            {
                "name": f"ProfilerStep#{idx}",
                "cat": "user_annotation",
                "ph": "X",
                "ts": start,
                "dur": wall,
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
                "ts": start + 10.0,
                "dur": wall - 20.0,
            }
        )
        for name, dur_ms in step["kernels"]:
            events.append(
                {
                    "name": name,
                    "cat": "kernel",
                    "ph": "X",
                    "ts": start + 100.0,
                    "dur": dur_ms * 1000.0,
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)


def _fixture_root(tmp_path):
    scenario = phase428.DEFAULT_SCENARIO
    artifact_root = tmp_path / "phase427_kernel_profile"
    artifact_dir = artifact_root / scenario
    _write_trace(
        artifact_dir / "prof_mixed/dp0_pp0_tp0_dcp0_ep0_rank0.test.pt.trace.json.gz",
        [
            {
                "ctx_tokens": 0,
                "gen_tokens": 56,
                "wall_ms": 40.0,
                "kernels": [
                    ("void marlin_moe_wna16::Marlin<test>", 12.0),
                    ("ncclDevKernel_AllGather_RING_LL(test)", 8.0),
                    ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 9.0),
                ],
            }
        ],
    )
    _write_trace(
        artifact_dir / "prof_decode/dp0_pp0_tp0_dcp0_ep0_rank0.test.pt.trace.json.gz",
        [
            {
                "ctx_tokens": 0,
                "gen_tokens": 40,
                "wall_ms": 30.0,
                "kernels": [
                    ("void marlin_moe_wna16::Marlin<test>", 8.0),
                    ("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(test)", 4.0),
                    ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 6.0),
                ],
            },
            {
                "ctx_tokens": 0,
                "gen_tokens": 80,
                "wall_ms": 50.0,
                "kernels": [
                    ("void marlin_moe_wna16::Marlin<test>", 22.0),
                    ("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(test)", 5.0),
                    ("void cutlass::device_kernel<flash::enable_sm90_or_later>", 7.0),
                ],
            },
        ],
    )
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
                "clean_decode_gap_ms": "146761.520489",
            }
        ],
    )
    return artifact_root, phase426_csv


def test_phase428_detects_missed_prefill_and_decode_moe_slope():
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        artifact_root, phase426_csv = _fixture_root(tmp_path)
        rows = phase428.build_phase428_rows(
            artifact_root=artifact_root,
            phase426_csv=phase426_csv,
            scenario=phase428.DEFAULT_SCENARIO,
        )
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    decode_slope = [
        row
        for row in rows
        if row["row_type"] == "decode_slope" and row["category"] == "moe_gemm_or_aux"
    ][0]

    assert summary["prefill_verdict"] == "prefill_steps_not_captured"
    assert summary["decode_verdict"] == "decode_moe_gemm_or_aux_slope_dominates"
    assert summary["mechanism_verdict"] == "phase427_trace_decode_only_prefill_recollect_required"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"
    assert float(decode_slope["real_slope_ms_per_request"]) > 0.0


def test_phase428_writer_rejects_default_claim():
    rows = [{field: "" for field in phase428.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase428.SOURCE,
            "row_type": "summary",
            "scenario": phase428.DEFAULT_SCENARIO,
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
        phase428.write_phase428_csv(Path("/tmp/phase428_bad.csv"), rows)
    except ValueError as exc:
        assert "valid_for_default" in str(exc)
    else:
        raise AssertionError("writer accepted valid_for_default=true")


if __name__ == "__main__":
    test_phase428_detects_missed_prefill_and_decode_moe_slope()
    test_phase428_writer_rejects_default_claim()

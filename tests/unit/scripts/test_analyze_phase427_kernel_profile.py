import csv
import json

import pytest

from scripts import analyze_phase427_kernel_profile as phase427


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profiler_text(rows):
    lines = [
        "Name  Self CPU %  Self CPU  CPU total %  CPU total  CPU time avg  Self CUDA  Self CUDA %  CUDA total  CUDA time avg  # of Calls",
        "----  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------",
    ]
    for name, self_cuda, cuda_total, avg, calls in rows:
        lines.append(
            f"{name}         0.00%       0.000us         0.00%       0.000us       "
            f"0.000us     {self_cuda}        1.00%     {cuda_total}     {avg}          {calls}"
        )
    lines.extend(
        [
            "----  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------",
            "Self CPU time total: 100.000ms",
            "Self CUDA time total: 85.000ms",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_phase427_artifact(root, scenario):
    artifact_dir = root / scenario
    mixed = artifact_dir / "prof_mixed"
    decode = artifact_dir / "prof_decode"
    mixed.mkdir(parents=True)
    decode.mkdir(parents=True)
    (artifact_dir / "meta.json").write_text(
        json.dumps({"scenario": scenario, "profile_windows": ["mixed", "decode"]}),
        encoding="utf-8",
    )
    mixed_rows = [
        ("ProfilerStep*", "100.000ms", "85.000ms", "100.000ms", 1),
        ("void marlin_moe_wna16::Marlin<test>", "30.000ms", "30.000ms", "5.000ms", 6),
        ("ncclDevKernel_AllGather_RING_LL(test)", "20.000ms", "20.000ms", "2.000ms", 10),
        ("void cutlass::device_kernel<flash::enable_sm90_or_later>", "10.000ms", "10.000ms", "1.000ms", 10),
        ("void vllm::cross_device_reduce_2stage<bf16>", "5.000ms", "5.000ms", "1.000ms", 5),
    ]
    decode_rows = [
        ("ProfilerStep*", "40.000ms", "30.000ms", "40.000ms", 1),
        ("void marlin_moe_wna16::Marlin<test>", "10.000ms", "10.000ms", "2.000ms", 5),
        ("ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(test)", "8.000ms", "8.000ms", "1.600ms", 5),
        ("void cutlass::device_kernel<flash::enable_sm90_or_later>", "7.000ms", "7.000ms", "1.400ms", 5),
    ]
    (mixed / "profiler_out_0.txt").write_text(_profiler_text(mixed_rows), encoding="utf-8")
    (decode / "profiler_out_0.txt").write_text(_profiler_text(decode_rows), encoding="utf-8")
    return artifact_dir


def test_phase427_aggregates_kernel_windows_and_keeps_report_only_flags(tmp_path):
    scenario = phase427.DEFAULT_SCENARIO
    artifact_root = tmp_path / "phase427_kernel_profile"
    artifact_dir = _write_phase427_artifact(artifact_root, scenario)

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
                "peer_stall_attribution_share": "0.047048",
                "decode_gap_attribution_share": "0.357492",
                "clean_decode_gap_ms": "146761.520489",
            }
        ],
    )
    phase415_csv = tmp_path / "phase415.csv"
    _write_csv(
        phase415_csv,
        [
            {
                "source": "phase415_prefill_charge_components",
                "row_type": "decode_component",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "component": "decode_ep_dispatch_combine",
                "sim_ms": "0.031093",
            }
        ],
    )

    rows = phase427.build_phase427_rows(
        artifact_root=artifact_root,
        phase426_csv=phase426_csv,
        phase415_csv=phase415_csv,
        scenario=scenario,
    )
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    mixed_ep = [
        row
        for row in rows
        if row["row_type"] == "kernel_category"
        and row["window"] == "mixed"
        and row["category"] == "ep_a2a"
    ][0]

    assert summary["source"] == phase427.SOURCE
    assert summary["mechanism_verdict"] == "kernel_profile_collected_prefill_and_decode"
    assert summary["reconstruction_gate"] == "passed"
    assert mixed_ep["category_cuda_ms"] == "20.000000"
    assert summary["phase405_penalty_read"] == "false"
    assert summary["runtime_modified"] == "false"
    assert summary["perf_database"] == "false"
    assert summary["valid_for_default"] == "false"
    assert summary["default_readiness"] == "No-Go"


def test_phase427_writer_rejects_default_or_runtime_claim(tmp_path):
    rows = [{field: "" for field in phase427.CSV_FIELDS}]
    rows[0].update(
        {
            "source": phase427.SOURCE,
            "row_type": "summary",
            "scenario": phase427.DEFAULT_SCENARIO,
            "phase405_penalty_read": "false",
            "gpu_allowed": "true",
            "ssh_allowed": "true",
            "runtime_modified": "true",
            "perf_database": "false",
            "valid_for_default": "false",
            "diagnostic_only": "true",
            "default_readiness": "No-Go",
        }
    )
    with pytest.raises(ValueError, match="runtime_modified"):
        phase427.write_phase427_csv(tmp_path / "bad.csv", rows)

    rows[0]["runtime_modified"] = "false"
    rows[0]["valid_for_default"] = "true"
    with pytest.raises(ValueError, match="valid_for_default"):
        phase427.write_phase427_csv(tmp_path / "bad.csv", rows)


def test_phase427_rejects_reused_profiler_window(tmp_path):
    scenario = phase427.DEFAULT_SCENARIO
    artifact_root = tmp_path / "phase427_kernel_profile"
    artifact_dir = _write_phase427_artifact(artifact_root, scenario)
    mixed_text = (artifact_dir / "prof_mixed/profiler_out_0.txt").read_text(encoding="utf-8")
    (artifact_dir / "prof_decode/profiler_out_0.txt").write_text(mixed_text, encoding="utf-8")

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
                "peer_stall_attribution_share": "0.047048",
                "decode_gap_attribution_share": "0.357492",
                "clean_decode_gap_ms": "146761.520489",
            }
        ],
    )
    phase415_csv = tmp_path / "phase415.csv"
    _write_csv(
        phase415_csv,
        [
            {
                "source": "phase415_prefill_charge_components",
                "row_type": "decode_component",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "component": "decode_ep_dispatch_combine",
                "sim_ms": "0.031093",
            }
        ],
    )

    with pytest.raises(ValueError, match="profiler windows are byte-identical"):
        phase427.build_phase427_rows(
            artifact_root=artifact_root,
            phase426_csv=phase426_csv,
            phase415_csv=phase415_csv,
            scenario=scenario,
        )

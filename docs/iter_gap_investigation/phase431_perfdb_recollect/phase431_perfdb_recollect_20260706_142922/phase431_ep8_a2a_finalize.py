import csv
import json
import os
from pathlib import Path

OUT = Path(os.environ["PHASE431_OUT"])
BUCKETS = [8, 16, 32, 64, 128]
EXPECTED_BACKEND = "allgather_reducescatter"
EXPECTED_MANAGER = "AgRsAll2AllManager"
torchrun_exit = int(os.environ.get("PHASE431_TORCHRUN_EXIT", "999"))


def read_text(name: str) -> str:
    path = OUT / name
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


gpu_after = read_text("gpu_compute_apps_after.txt")
residual = read_text("process_residual_after.txt")
cleanup_ok = (not gpu_after.strip()) and (not residual.strip())
error_paths = sorted(OUT.glob("phase431_ep_rank_*_error.json"))
rank_paths = sorted(OUT.glob("phase431_ep_rank_[0-7].json"))
rank_error = len(error_paths)
base_failure = []
if torchrun_exit != 0:
    base_failure.append(f"torchrun_exit_{torchrun_exit}")
if rank_error:
    first = json.loads(error_paths[0].read_text(encoding="utf-8"))
    base_failure.append(first.get("error", first.get("error_type", "rank_error")))
    (OUT / "phase431_ep_error.txt").write_text(
        "\n\n".join(p.read_text(encoding="utf-8", errors="replace") for p in error_paths),
        encoding="utf-8",
    )
if not cleanup_ok:
    base_failure.append("cleanup_residual")

rank_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in rank_paths]
if not base_failure and len(rank_payloads) != 8:
    base_failure.append(f"rank_result_count_{len(rank_payloads)}")

rows = []
for bucket in BUCKETS:
    failure = list(base_failure)
    bucket_rank_rows = []
    for payload in rank_payloads:
        for item in payload.get("bucket_results", []):
            if int(item.get("bucket_tokens", -1)) == bucket:
                bucket_rank_rows.append((payload, item))
    if not failure and len(bucket_rank_rows) != 8:
        failure.append(f"bucket_rank_count_{len(bucket_rank_rows)}")
    backends = {payload.get("backend", "") for payload, _ in bucket_rank_rows}
    managers = {payload.get("manager_class", "") for payload, _ in bucket_rank_rows}
    if not failure and backends != {EXPECTED_BACKEND}:
        failure.append("backend_drift:" + ";".join(sorted(backends)))
    if not failure and managers != {EXPECTED_MANAGER}:
        failure.append("manager_drift:" + ";".join(sorted(managers)))
    finite_values = [item.get("output_all_finite") is True for _, item in bucket_rank_rows]
    if not failure and (not finite_values or not all(finite_values)):
        failure.append("non_finite_output")
    latencies = [float(item.get("latency_ms_median", 0.0)) for _, item in bucket_rank_rows]
    rank0_payload = bucket_rank_rows[0][0] if bucket_rank_rows else {}
    rank0_item = bucket_rank_rows[0][1] if bucket_rank_rows else {}
    row = {
        "source": "phase431_perfdb_recollect",
        "row_type": "ep_a2a_raw",
        "ok": "false" if failure else "true",
        "worker": str(rank0_payload.get("worker", "")),
        "vllm_version": str(rank0_payload.get("vllm_version", "")),
        "source_root": str(rank0_payload.get("source_root", "")),
        "measurement_boundary": "vllm_ep_group_dispatch_router_logits_plus_combine",
        "bucket_tokens": str(bucket),
        "backend": next(iter(backends)) if len(backends) == 1 else ";".join(sorted(backends)),
        "manager": next(iter(managers)) if len(managers) == 1 else ";".join(sorted(managers)),
        "configured_backend": str(rank0_payload.get("configured_backend", "")),
        "shape": str(rank0_item.get("shape", f"num_tokens={bucket},hidden_size=7168,num_experts=384,top_k=8,dtype=bfloat16")),
        "latency_ms": f"{max(latencies):.6f}" if latencies else "",
        "rank_error": str(rank_error),
        "cleanup": "true" if cleanup_ok else "false",
        "gpu_process_residue": "false" if cleanup_ok else "true",
        "failure_reason": ";".join(failure),
        "runtime_dispatch_scope": "fixed_model_config_hw_vllm_version_tuple",
        "default_readiness": "No-Go",
        "diagnostic_only": "false",
        "valid_for_default": "false",
        "perf_database": "true",
    }
    rows.append(row)

with (OUT / "phase431_ep_a2a_raw.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)

passed = all(row["ok"] == "true" for row in rows)
(OUT / "phase431_ep_a2a_summary.json").write_text(
    json.dumps({"ok": passed, "rows": rows}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
if not passed:
    raise SystemExit("phase431 EP a2a self-check failed")

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import analyze_phase431_perfdb_recollect as phase431


def _write_artifact(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "phase431_generation_mla_raw.csv").write_text(
        "framework,version,device,op_name,kernel_source,mla_dtype,kv_cache_dtype,num_heads,batch_size,isl,tp_size,step,latency\n"
        "VLLM,0.19.0,NVIDIA H200,generation_mla,vllm_flash_attn_mla,float16,float16,16,8,1,8,8191,0.1\n"
        "VLLM,0.19.0,NVIDIA H200,generation_mla,vllm_flash_attn_mla,float16,float16,16,16,1,8,8191,0.2\n"
        "VLLM,0.19.0,NVIDIA H200,generation_mla,vllm_flash_attn_mla,float16,float16,16,32,1,8,8191,0.3\n"
        "VLLM,0.19.0,NVIDIA H200,generation_mla,vllm_flash_attn_mla,float16,float16,16,64,1,8,8191,0.4\n"
        "VLLM,0.19.0,NVIDIA H200,generation_mla,vllm_flash_attn_mla,float16,float16,16,128,1,8,8191,0.5\n"
    )
    (root / "phase431_moe_int4_wo_raw.csv").write_text(
        "framework,version,device,op_name,kernel_source,moe_dtype,num_tokens,hidden_size,inter_size,topk,num_experts,moe_tp_size,moe_ep_size,distribution,latency\n"
        "VLLM,0.19.0,NVIDIA H200,moe,vllm_marlin_moe_wna16,int4_wo,8,7168,2048,8,384,1,8,power_law_1.01,0.2\n"
        "VLLM,0.19.0,NVIDIA H200,moe,vllm_marlin_moe_wna16,int4_wo,16,7168,2048,8,384,1,8,power_law_1.01,0.3\n"
        "VLLM,0.19.0,NVIDIA H200,moe,vllm_marlin_moe_wna16,int4_wo,32,7168,2048,8,384,1,8,power_law_1.01,0.4\n"
        "VLLM,0.19.0,NVIDIA H200,moe,vllm_marlin_moe_wna16,int4_wo,64,7168,2048,8,384,1,8,power_law_1.01,0.5\n"
        "VLLM,0.19.0,NVIDIA H200,moe,vllm_marlin_moe_wna16,int4_wo,128,7168,2048,8,384,1,8,power_law_1.01,0.6\n"
    )
    (root / "phase431_ep_a2a_raw.csv").write_text(
        "source,row_type,ok,worker,vllm_version,source_root,measurement_boundary,bucket_tokens,backend,manager,configured_backend,shape,latency_ms,rank_error,cleanup,gpu_process_residue,failure_reason,runtime_dispatch_scope,default_readiness,diagnostic_only,valid_for_default,perf_database\n"
        "phase431_perfdb_recollect,ep_a2a_raw,true,worker,0.19.0,/vllm,boundary,8,allgather_reducescatter,AgRsAll2AllManager,allgather_reducescatter,shape,0.07,0,true,false,,scope,No-Go,false,false,true\n"
        "phase431_perfdb_recollect,ep_a2a_raw,true,worker,0.19.0,/vllm,boundary,16,allgather_reducescatter,AgRsAll2AllManager,allgather_reducescatter,shape,0.08,0,true,false,,scope,No-Go,false,false,true\n"
        "phase431_perfdb_recollect,ep_a2a_raw,true,worker,0.19.0,/vllm,boundary,32,allgather_reducescatter,AgRsAll2AllManager,allgather_reducescatter,shape,0.09,0,true,false,,scope,No-Go,false,false,true\n"
        "phase431_perfdb_recollect,ep_a2a_raw,true,worker,0.19.0,/vllm,boundary,64,allgather_reducescatter,AgRsAll2AllManager,allgather_reducescatter,shape,0.10,0,true,false,,scope,No-Go,false,false,true\n"
        "phase431_perfdb_recollect,ep_a2a_raw,true,worker,0.19.0,/vllm,boundary,128,allgather_reducescatter,AgRsAll2AllManager,allgather_reducescatter,shape,0.11,0,true,false,,scope,No-Go,false,false,true\n"
    )
    (root / "gpu_compute_apps_after.txt").write_text("")
    (root / "process_residual_after.txt").write_text("")


def test_phase431_artifact_and_slope_rows():
    with tempfile.TemporaryDirectory() as raw:
        artifact = Path(raw) / "artifact"
        _write_artifact(artifact)

        rows = phase431.build_phase431_rows(artifact, include_database=False)
        summary = next(row for row in rows if row["row_type"] == "summary")
        moe_slope = next(
            row for row in rows if row["row_type"] == "slope_recheck" and row["category"] == "moe_gemm_or_aux"
        )
        cleanup = next(row for row in rows if row["row_type"] == "cleanup")

        assert summary["verdict"] == "phase431_perfdb_recollect_ingested_default_no_go"
        assert summary["perf_database"] == "true"
        assert summary["valid_for_default"] == "false"
        assert summary["default_readiness"] == "No-Go"
        assert float(moe_slope["phase431_slope_ms_per_request"]) > 0.0
        assert cleanup["verdict"] == "remote_gpu_and_process_residue_empty"


def test_phase431_writer_rejects_default_claim():
    row = {field: "" for field in phase431.CSV_FIELDS}
    row.update(
        {
            "source": phase431.SOURCE,
            "row_type": "summary",
            "perf_database": "true",
            "runtime_modified": "true",
            "valid_for_default": "true",
            "diagnostic_only": "false",
            "default_readiness": "No-Go",
        }
    )
    with tempfile.TemporaryDirectory() as raw:
        try:
            phase431.write_phase431_csv(Path(raw) / "bad.csv", [row])
        except ValueError as exc:
            assert "valid_for_default" in str(exc)
        else:
            raise AssertionError("writer accepted valid_for_default=true")


if __name__ == "__main__":
    test_phase431_artifact_and_slope_rows()
    test_phase431_writer_rejects_default_claim()

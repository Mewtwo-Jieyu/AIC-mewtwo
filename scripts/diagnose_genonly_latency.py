#!/usr/bin/env python3
"""
诊断脚本：分析 genonly_step_latency 随 batch_size 的变化。
输出每个 batch_size 下的 per-op 延迟分解，定位延迟膨胀来源。
"""

import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aiconfigurator.sdk.perf_database import PerfDatabase
from aiconfigurator.sdk.models import get_model
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig
from aiconfigurator.sdk import common
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend

logging.basicConfig(level=logging.WARNING, format="%(name)s - %(levelname)s - %(message)s")
# 对 perf_database 的 DEBUG 日志只在需要时启用
pdb_logger = logging.getLogger("aiconfigurator.sdk.perf_database")


def diagnose(isl: int = 3000, osl: int = 3000, tp: int = 16):
    print(f"\n{'='*80}")
    print(f"诊断：ISL={isl}, OSL={osl}, TP={tp}")
    print(f"{'='*80}")

    model_config = ModelConfig(tp_size=tp, pp_size=1, moe_tp_size=tp, moe_ep_size=1)
    model = get_model("moonshotai/Kimi-K2.5", model_config, backend_name="vllm")

    systems_root = str(Path(__file__).resolve().parent.parent / "src" / "aiconfigurator" / "systems")
    db = PerfDatabase(system="h200_sxm", backend="vllm", version="0.12.0", systems_root=systems_root)
    backend = VLLMBackend()

    print(f"KV Cache Quant Mode: {model.config.kvcache_quant_mode}")
    num_heads = model._num_heads // tp
    print(f"num_heads total: {model._num_heads}, per GPU: {num_heads}")

    # Part 1: 直接查询 generation_mla
    print(f"\n--- Part 1: generation_mla 查询测试 ---")
    kvcache_mode = model.config.kvcache_quant_mode
    pdb_logger.setLevel(logging.DEBUG)
    for b_test in [1, 16]:
        for s_test in [33, isl + osl // 2]:
            try:
                result = db.query_generation_mla(
                    b=b_test, s=s_test, num_heads=num_heads,
                    kvcache_quant_mode=kvcache_mode,
                    database_mode=common.DatabaseMode.HYBRID,
                )
                lat = float(result)  # PerformanceResult extends float
                print(f"  b={b_test:>3}, s={s_test:>6}, heads={num_heads} → {lat:.4f} ms")
            except Exception as e:
                print(f"  b={b_test:>3}, s={s_test:>6}, heads={num_heads} → ERROR: {e}")
    pdb_logger.setLevel(logging.WARNING)

    # Part 2: per-op 分解
    print(f"\n--- Part 2: per-op genonly_step_latency 分解 ---")
    all_ops = {}
    for b in [1, 2, 4, 8, 16, 32, 64]:
        # ctx_tokens = isl means processing 1 request's full context per step
        # steps_to_finish_ctx = ceil(isl * b / isl) = b
        # For b < osl: there will be (osl - b) genonly steps
        ctx_tokens = isl
        try:
            summary = backend.run_agg(
                model, db,
                RuntimeConfig(isl=isl, osl=osl, batch_size=b),
                database_mode=common.DatabaseMode.HYBRID,
                ctx_tokens=ctx_tokens,
            )
            per_ops = summary.get_per_ops_data()
            if per_ops and "genonly_step" in per_ops:
                genonly = per_ops["genonly_step"]
                sched = per_ops.get("scheduling", {})
                total = sum(genonly.values())
                all_ops[b] = genonly
                df = summary.get_summary_df()
                tpot_val = df.iloc[0].get("tpot", "N/A") if df is not None and not df.empty else "N/A"
                tps_gpu = df.iloc[0].get("tokens_s_gpu", "N/A") if df is not None and not df.empty else "N/A"
                print(f"\n  b={b:>3}: genonly_lat={total:.3f}ms, "
                      f"steps={sched.get('num_genonly_steps', '?')}, "
                      f"TPOT={tpot_val}, tok/s/GPU={tps_gpu}")
                sorted_ops = sorted(genonly.items(), key=lambda x: x[1], reverse=True)
                for op_name, lat in sorted_ops[:8]:
                    pct = lat / total * 100 if total > 0 else 0
                    print(f"      {op_name:45s} {lat:>8.4f} ms ({pct:5.1f}%)")
        except Exception as e:
            print(f"\n  b={b:>3}: ERROR: {e}")

    # Part 3: 膨胀分析
    if 1 in all_ops and 16 in all_ops:
        print(f"\n--- Part 3: b=1 → b=16 膨胀分析 ---")
        ops_b1, ops_b16 = all_ops[1], all_ops[16]
        total_b1, total_b16 = sum(ops_b1.values()), sum(ops_b16.values())
        print(f"  总延迟: {total_b1:.3f}ms → {total_b16:.3f}ms ({total_b16/total_b1:.2f}x)")
        all_op_names = set(ops_b1.keys()) | set(ops_b16.keys())
        growth = []
        for op in all_op_names:
            v1, v16 = ops_b1.get(op, 0), ops_b16.get(op, 0)
            if v1 > 0.001:
                growth.append((op, v1, v16, v16/v1, v16-v1))
        growth.sort(key=lambda x: x[4], reverse=True)
        print(f"  {'Op':45s} {'b=1':>8s} {'b=16':>8s} {'Ratio':>7s} {'Delta':>8s}")
        for op, v1, v16, r, d in growth[:15]:
            print(f"  {op:45s} {v1:>8.3f} {v16:>8.3f} {r:>6.2f}x {d:>+8.3f}")


if __name__ == "__main__":
    diagnose(isl=3000, osl=3000, tp=16)
    print("\n\n")
    diagnose(isl=32000, osl=1000, tp=16)

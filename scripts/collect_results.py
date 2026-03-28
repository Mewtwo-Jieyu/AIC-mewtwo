#!/usr/bin/env python3
"""
收集 A100 vs L40、agg vs disagg 对比实验的结果数据
用于生成报告
"""

import os
import sys
import math
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

# 成本计算常量
GPUS_PER_NODE = 8.0  # 单机8卡
MONTHLY_NODE_RENT = {
    "a100": 35000.0,  # 元 / 月 / 8 卡节点
    "l40": 15000.0,   # 元 / 月 / 8 卡节点
}
TEN_K_CURRENCY = 10_000.0


def infer_gpu_type(model_dir_name: str) -> str:
    """
    根据模型目录名推断主要 GPU 类型：
    - 包含 "a100" => a100
    - 包含 "l40" => l40
    其它情况返回 "unknown"。
    """
    lower = model_dir_name.lower()
    if "a100" in lower:
        return "a100"
    if "l40" in lower:
        return "l40"
    return "unknown"


def has_disagg_columns(df: pd.DataFrame) -> bool:
    """判断 DataFrame 是否包含 disagg 成本计算所需的关键列。"""
    cols = set(df.columns)
    if "(p)workers" not in cols or "(d)workers" not in cols:
        return False
    # p_gpus_worker/d_gpus_worker 或者 (p)tp/(p)pp/(p)dp 等其中一组即可
    has_p_gpu_info = "p_gpus_worker" in cols or "(p)tp" in cols
    has_d_gpu_info = "d_gpus_worker" in cols or "(d)tp" in cols
    return has_p_gpu_info and has_d_gpu_info


def calculate_tokens_per_10k_rmb(
    tokens_per_gpu: float, num_total_gpus: int, gpu_type: str
) -> float:
    """
    把 tokens/s/gpu 换算成 tokens/s/10k(元)。

    成本计算规则：8 张 GPU 视为一个租赁单位（一机），向上取整。
    - <= 8 卡 -> 1 机
    - 9~16 卡 -> 2 机
    - 以此类推

    计算公式：
    - 租赁单元数 = ceil(num_total_gpus / GPUS_PER_NODE)
    - 总月成本 = 单元数 * 单机月租金
    - 总吞吐 = tokens/s/gpu * num_total_gpus
    - tokens/s/10k RMB = (总吞吐 / 总月成本) * 10000

    参数：
      tokens_per_gpu: tokens/s/gpu
      num_total_gpus: 使用的 GPU 总数
      gpu_type: "a100" 或 "l40"

    返回：
      tokens/s/10k RMB（浮点数），如果 GPU 类型未知返回 0
    """
    if gpu_type not in MONTHLY_NODE_RENT:
        return 0.0

    # 计算单机数（向上取整）
    num_nodes = int(math.ceil(num_total_gpus / GPUS_PER_NODE))

    # 计算总月成本（元/月）
    total_monthly_cost = num_nodes * MONTHLY_NODE_RENT[gpu_type]

    # 计算总吞吐（tokens/s）
    total_tokens_per_sec = tokens_per_gpu * num_total_gpus

    # 换算成 tokens/s/10k RMB
    if total_monthly_cost == 0:
        return 0.0

    tokens_per_10k = (total_tokens_per_sec / total_monthly_cost) * TEN_K_CURRENCY

    return tokens_per_10k


def normalize_gpu_type_for_cost(raw_type: str) -> str:
    """
    将更细粒度的 GPU 标识（如 a100_sxm, l40s 等）映射到成本表使用的粗粒度类型。
    """
    if not raw_type:
        return "unknown"
    lower = str(raw_type).lower()
    if "a100" in lower:
        return "a100"
    if "l40" in lower:
        return "l40"
    return "unknown"


def calculate_tokens_per_10k_rmb_disagg(
    tokens_per_gpu: float,
    num_total_gpus: int,
    p_gpu_type: str,
    d_gpu_type: str,
    p_workers: int,
    d_workers: int,
    p_gpus_worker: int,
    d_gpus_worker: int,
    replicas: Union[int, float],
) -> float:
    """
    disagg 场景下按“先算单套、先按单套算节点，再乘 replica”的逻辑计算节点数与成本：

    Step 1: 先算单套服务（replica=1）的 GPU 需求
      P_single_replica_gpus = p_workers × p_gpus_per_worker
      D_single_replica_gpus = d_workers × d_gpus_per_worker

    Step 2: 计算单套的节点数（单机 8 卡）
      P_single_replica_nodes = ceil(P_single_replica_gpus / 8)
      D_single_replica_nodes = ceil(D_single_replica_gpus / 8)

    Step 3: 乘以 replica 数量得到总节点数
      P_total_nodes = P_single_replica_nodes × replicas
      D_total_nodes = D_single_replica_nodes × replicas
      total_nodes = P_total_nodes + D_total_nodes

    成本:
      cost = P_total_nodes × MONTHLY_NODE_RENT[p_type] + D_total_nodes × MONTHLY_NODE_RENT[d_type]
    """
    p_type = normalize_gpu_type_for_cost(p_gpu_type)
    d_type = normalize_gpu_type_for_cost(d_gpu_type)

    try:
        tokens_per_gpu_f = float(tokens_per_gpu)
        num_total_gpus_f = float(num_total_gpus)
    except (TypeError, ValueError):
        return 0.0

    if tokens_per_gpu_f <= 0 or num_total_gpus_f <= 0:
        return 0.0

    # 处理 replicas，允许为 float，按四舍五入取整，且至少为 1
    try:
        r = float(replicas)
    except (TypeError, ValueError):
        r = 1.0
    replicas_int = max(1, int(round(r)))

    # Step 1: 单套服务（replica=1）的 GPU 需求
    try:
        p_single_gpus = max(0, int(p_workers) * int(p_gpus_worker))
        d_single_gpus = max(0, int(d_workers) * int(d_gpus_worker))
    except (TypeError, ValueError):
        return 0.0

    # Step 2: 计算单套的节点数（单机 8 卡）
    def nodes_for_single_replica(single_gpus: int) -> int:
        if single_gpus <= 0:
            return 0
        return int(math.ceil(single_gpus / GPUS_PER_NODE))

    p_single_nodes = nodes_for_single_replica(p_single_gpus)
    d_single_nodes = nodes_for_single_replica(d_single_gpus)

    if p_single_nodes <= 0 and d_single_nodes <= 0:
        return 0.0

    # Step 3: 乘以 replica 数量得到总节点数
    p_total_nodes = p_single_nodes * replicas_int
    d_total_nodes = d_single_nodes * replicas_int

    if p_total_nodes <= 0 and d_total_nodes <= 0:
        return 0.0

    # 成本 = 各自节点数 × 对应 GPU 类型的单机月租金
    total_monthly_cost = 0.0
    if p_total_nodes > 0 and p_type in MONTHLY_NODE_RENT:
        total_monthly_cost += p_total_nodes * MONTHLY_NODE_RENT[p_type]
    if d_total_nodes > 0 and d_type in MONTHLY_NODE_RENT:
        total_monthly_cost += d_total_nodes * MONTHLY_NODE_RENT[d_type]

    if total_monthly_cost <= 0:
        return 0.0

    total_tokens_per_sec = tokens_per_gpu_f * num_total_gpus_f
    return (total_tokens_per_sec / total_monthly_cost) * TEN_K_CURRENCY


def find_result_dirs(base_dir: str) -> Dict[str, List[str]]:
    """找到所有包含 best_config_topn.csv 的实验目录，按实验名称分组。

    目录结构大致为：
    <base>/Qwen/Qwen3-32B-FP8_.../<exp_name>/best_config_topn.csv
    """
    base_path = Path(base_dir)
    result_dirs: Dict[str, List[str]] = {}

    # 直接以 best_config_topn.csv 为锚点，更稳健
    for csv_path in base_path.rglob("best_config_topn.csv"):
        exp_dir = csv_path.parent              # .../<exp_name>
        model_dir = exp_dir.parent            # .../Qwen3-32B...
        if not model_dir.name.startswith("Qwen3-32B"):
            continue

        dir_name = model_dir.name
        parts = dir_name.split("_")
        
        # 解析 system, isl, osl, ttft, tpot（从 model_dir 名里）
        system = None
        isl = None
        osl = None
        ttft = None
        tpot = None
        
        for i, part in enumerate(parts):
            if part in ["a100", "l40s"]:
                system = part
            elif part.startswith("isl"):
                isl = int(part.replace("isl", ""))
            elif part.startswith("osl"):
                osl = int(part.replace("osl", ""))
            elif part.startswith("ttft"):
                ttft = float(part.replace("ttft", ""))
            elif part.startswith("tpot"):
                tpot = float(part.replace("tpot", ""))
        
        # 根据 CSV 列判断是 agg 还是 disagg，而不依赖目录名
        try:
            df_temp = pd.read_csv(csv_path)
            is_disagg = has_disagg_columns(df_temp)
        except Exception:
            is_disagg = False
        
        if is_disagg:
            # 如果有 p/d GPU 类型信息，从第一行取值构成更详细的 exp_key
            p_gpu = None
            d_gpu = None
            try:
                first = df_temp.iloc[0]
                if "(p)system" in first:
                    p_gpu = str(first.get("(p)system", "unknown"))
                if "(d)system" in first:
                    d_gpu = str(first.get("(d)system", "unknown"))
            except Exception:
                pass
            if p_gpu and d_gpu:
                # 格式：<system>_disagg_<d_gpu>_d_<p_gpu>_p
                exp_key_base = f"{system}_disagg_{d_gpu}_d_{p_gpu}_p"
            else:
                exp_key_base = f"{system}_disagg"
        else:
            exp_key_base = f"{system}_agg"
        
        # 添加场景标识
        if isl == 512 and osl == 128:
            scene = "short"
        elif isl == 2048 and osl == 512:
            scene = "medium"
        elif isl == 4000 and osl == 1000:
            scene = "long"
        else:
            scene = "unknown"
        
        full_key = f"{exp_key_base}_{scene}"
        
        result_dirs.setdefault(full_key, []).append(str(exp_dir))

    return result_dirs

def extract_metrics(result_dir: str) -> Optional[Dict]:
    """从单个结果目录提取关键指标"""
    result_path = Path(result_dir)
    
    # 优先尝试带 replica 信息的文件（disagg）
    best_with_replica = result_path / "best_config_topn_with_replica.csv"
    best_config_file = None
    if best_with_replica.exists():
        best_config_file = best_with_replica
    else:
        # 退回到普通文件（agg）
        candidate = result_path / "best_config_topn.csv"
        if candidate.exists():
            best_config_file = candidate
    
    if best_config_file is None:
        return None
    
    try:
        df = pd.read_csv(best_config_file)
        if df.empty:
            return None
        
        # 取 top-1 配置
        top1 = df.iloc[0].to_dict()
        
        # 判断是否为 disagg（根据 CSV 列判断，而非目录名）
        is_disagg = has_disagg_columns(df)
        
        # 针对 agg/disagg 确定 GPU 类型
        model_dir = result_path.parent
        gpu_type = infer_gpu_type(model_dir.name)

        # 如果是 disagg，top1 包含 (p)system 和 (d)system 两列，用各自类型替代
        p_gpu_type = None
        d_gpu_type = None
        if is_disagg and ("(p)system" in top1 or "(d)system" in top1):
            p_gpu_type = top1.get("(p)system", "unknown")
            d_gpu_type = top1.get("(d)system", "unknown")
            # 总体 gpu_type 标记为混合，防止误用
            gpu_type = f"p:{p_gpu_type},d:{d_gpu_type}"
        
        # 提取关键指标
        tokens_per_gpu = top1.get("tokens/s/gpu", 0)
        tokens_per_user = top1.get("tokens/s/user", 0)
        num_total_gpus = int(top1.get("num_total_gpus", 0))
        
        metrics = {
            "experiment": result_path.name,
            "tokens_per_sec": top1.get("tokens/s", 0),
            "tokens_per_sec_per_gpu": tokens_per_gpu,
            "tokens_per_sec_per_user": tokens_per_user,
            "seq_per_sec": top1.get("seq/s", 0),
            "seq_per_sec_per_gpu": top1.get("seq/s/gpu", 0),
            "num_total_gpus": num_total_gpus,
            "ttft_ms": top1.get("ttft", 0),
            "tpot_ms": top1.get("tpot", 0),
            "request_latency_ms": top1.get("request_latency", 0),
            "isl": top1.get("isl", 0),
            "osl": top1.get("osl", 0),
            "gpu_type": gpu_type,
        }
        if p_gpu_type is not None:
            metrics["p_gpu_type"] = p_gpu_type
        if d_gpu_type is not None:
            metrics["d_gpu_type"] = d_gpu_type
        
        # 计算 tokens/s/10k rmb
        if tokens_per_gpu > 0 and num_total_gpus > 0:
            # disagg：分别对 p/d 两侧按更保守的节点数算法计费
            if is_disagg:
                p_workers = top1.get("(p)workers", 0) or 0
                d_workers = top1.get("(d)workers", 0) or 0
                replicas = top1.get("replicas", 1) or 1
                p_gpus_worker = top1.get("p_gpus_worker", 0) or 0
                d_gpus_worker = top1.get("d_gpus_worker", 0) or 0

                metrics["tokens_per_10k_rmb"] = calculate_tokens_per_10k_rmb_disagg(
                    tokens_per_gpu,
                    num_total_gpus,
                    p_gpu_type,
                    d_gpu_type,
                    int(p_workers),
                    int(d_workers),
                    int(p_gpus_worker),
                    int(d_gpus_worker),
                    replicas,
                )
            else:
                # agg：沿用原有的总 GPU 数 / 节点数计费逻辑
                metrics["tokens_per_10k_rmb"] = calculate_tokens_per_10k_rmb(
                    tokens_per_gpu, num_total_gpus, gpu_type
                )
        else:
            metrics["tokens_per_10k_rmb"] = 0.0
        
        # 不同模式的额外字段
        if "(p)" in str(df.columns):
            metrics["prefill_tp"] = top1.get("(p)tp", 0)
            metrics["prefill_pp"] = top1.get("(p)pp", 0)
            metrics["prefill_workers"] = top1.get("(p)workers", 0)
            metrics["prefill_batch_size"] = top1.get("(p)bs", 0)
            metrics["decode_tp"] = top1.get("(d)tp", 0)
            metrics["decode_pp"] = top1.get("(d)pp", 0)
            metrics["decode_workers"] = top1.get("(d)workers", 0)
            metrics["decode_batch_size"] = top1.get("(d)bs", 0)
        else:
            metrics["tp"] = top1.get("tp", 0)
            metrics["pp"] = top1.get("pp", 0)
            metrics["batch_size"] = top1.get("bs", 0)
        
        # replica 相关列（如果有的话）
        if "replicas" in top1:
            metrics["replicas"] = top1.get("replicas", 0)
        if "p_gpus_worker" in top1:
            metrics["p_gpus_worker"] = top1.get("p_gpus_worker", 0)
        if "d_gpus_worker" in top1:
            metrics["d_gpus_worker"] = top1.get("d_gpus_worker", 0)
        
        return metrics
    except Exception as e:
        print(f"Error reading {best_config_file}: {e}", file=sys.stderr)
        return None

def collect_all_results(base_dir: str, output_file: str = "results_collect.csv"):
    """收集所有实验结果并汇总"""
    result_dirs = find_result_dirs(base_dir)
    
    all_metrics = []
    
    for exp_key, dirs in sorted(result_dirs.items()):
        for result_dir in dirs:
            metrics = extract_metrics(result_dir)
            if metrics:
                # 添加实验标识
                metrics["exp_key"] = exp_key
                all_metrics.append(metrics)
    
    if not all_metrics:
        print(f"No results found in {base_dir}", file=sys.stderr)
        return
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_metrics)
    
    # 保存为 CSV
    output_path = Path(base_dir) / output_file
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"\nTotal experiments: {len(df)}")
    print(f"\nSummary by experiment type:")
    cols = ["tokens_per_sec_per_gpu", "tokens_per_sec_per_user", "num_total_gpus", "tokens_per_10k_rmb"]
    print(df.groupby("exp_key")[cols].agg({
        "tokens_per_sec_per_gpu": ["mean", "max"],
        "tokens_per_sec_per_user": ["mean", "max"],
        "num_total_gpus": "mean",
        "tokens_per_10k_rmb": ["mean", "max"]
    }))
    
    return df

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python collect_results.py <output_dir> [output_csv_name]")
        print("Example: python collect_results.py output/a100_l40_full_comparison")
        sys.exit(1)
    
    base_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "results_summary.csv"
    
    df = collect_all_results(base_dir, output_file)
    
    if df is not None:
        print("\n=== Key Metrics Summary ===")
        print("\nBy System and Mode:")
        summary = df.groupby(["exp_key"])[[
            "tokens_per_sec_per_gpu",
            "tokens_per_sec_per_user",
            "num_total_gpus",
            "ttft_ms",
            "tpot_ms",
            "tokens_per_10k_rmb"
        ]].mean()
        print(summary)

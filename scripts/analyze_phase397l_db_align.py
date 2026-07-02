#!/usr/bin/env python3
"""Phase397l: align cb_sim per-op DECODE latencies against the profiled ground truth.

For each 8k2k topology this dumps cb_sim's ``generation_latency_dict`` (the exact
per-op decode decomposition ``iteration_latency.py`` serial-sums for a pure decode
iteration), maps every op to the same structural categories the torch-profiler
analyzer uses, and prints measured-vs-DB per-category latency plus the serial-sum
overcount. This isolates WHERE cb_sim's decode over-prediction comes from
(attention / moe / proj / comm) and quantifies how much of the ``5x`` gap is the
fictional ``ep8_per_iteration_overhead_ms=90`` vs a real DB/summation overcount.

Report only. Does NOT modify the perf DB, the simulator, or any gate constant.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend  # noqa: E402
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig  # noqa: E402
from aiconfigurator.sdk.models import get_model  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402

from validate_cb_simulator import BACKEND, MODEL_PATH, SYSTEM  # noqa: E402

SYSTEMS_ROOT = str(REPO_ROOT / "src" / "aiconfigurator" / "systems")


def _classify(name: str) -> str:
    low = name.lower()
    if "dispatch" in low or "combine" in low or "all2all" in low or "alltoall" in low:
        return "comm_all2all"
    if "allreduce" in low or "all_reduce" in low or "reduce" in low or "comm" in low \
            or "nccl" in low or "allgather" in low or "all_gather" in low \
            or re.search(r"(^|_)ar(_|$)", low):  # generation_ar_1/2, context_ar_1/2 (CustomAllReduce)
        return "comm_reduce"
    if "attention" in low or "mla" in low or "attn" in low:
        return "attention"
    if "moe" in low or "expert" in low:
        return "moe"
    return "proj_gemm"


def dump_decode(tp: int, dp: int, moe_tp: int, moe_ep: int, version: str,
                decode_bs: int, kv_len: int) -> dict:
    model_config = ModelConfig(
        tp_size=tp, pp_size=1, moe_tp_size=moe_tp, moe_ep_size=moe_ep,
        attention_dp_size=dp,
    )
    model = get_model(MODEL_PATH, model_config, backend_name=BACKEND)
    # Force the standard kernel-table path (bypass the experimental 0.19 Kimi
    # module-runtime route that only triggers for tp4dp2ep8), matching phase397k.
    model.model_name = "phase397l-stdkernel"
    db = PerfDatabase(system=SYSTEM, backend=BACKEND, version=version,
                      systems_root=SYSTEMS_ROOT)
    backend = VLLMBackend()
    summary = backend.run_static(
        model, db,
        RuntimeConfig(batch_size=decode_bs, beam_width=1, isl=kv_len, osl=2),
        mode="static_gen",
    )
    return summary.get_generation_latency_dict()


def analyze(label: str, tp: int, dp: int, moe_tp: int, moe_ep: int, version: str,
            decode_bs: int, kv_len: int, measured: dict[str, float],
            out_csv: Path | None) -> None:
    gen = dump_decode(tp, dp, moe_tp, moe_ep, version, decode_bs, kv_len)
    print(f"\n=== cb_sim decode per-op ({label}, DB {version}, bs={decode_bs}, kv={kv_len}) ===")
    cats: dict[str, float] = {}
    for name, lat in sorted(gen.items(), key=lambda kv: -kv[1]):
        c = _classify(name)
        cats[c] = cats.get(c, 0.0) + lat
        print(f"  {lat:8.3f} ms  [{c:12s}] {name}")
    attn = cats.get("attention", 0.0)
    moe = cats.get("moe", 0.0)
    proj = cats.get("proj_gemm", 0.0)
    comm = cats.get("comm_reduce", 0.0) + cats.get("comm_all2all", 0.0)
    sim_serial = sum(gen.values())

    # Measured category rollups (profiler): attention, moe(=expert+aux),
    # proj(=proj_gemm), comm(=allreduce+allgather+all2all).
    m_attn = measured["attention"]
    m_moe = measured["moe_expert_gemm"] + measured.get("moe_aux", 0.0)
    m_proj = measured["proj_gemm"] + measured.get("norm_elementwise", 0.0)
    m_comm = measured["comm"]
    m_wall = measured["wall"]
    m_gpu = measured["gpu_busy"]

    def ratio(a: float, b: float) -> str:
        return f"{a / b:6.2f}x" if b else "   n/a"

    print(f"\n  --- category: cb_sim(DB {version}) vs measured (profiler) ---")
    hdr = f"  {'category':14s} {'cb_sim ms':>10s} {'measured ms':>12s} {'sim/meas':>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for cname, sv, mv in [
        ("attention", attn, m_attn),
        ("moe", moe, m_moe),
        ("proj_gemm+norm", proj, m_proj),
        ("comm", comm, m_comm),
    ]:
        print(f"  {cname:14s} {sv:>10.3f} {mv:>12.3f} {ratio(sv, mv):>10s}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  {'SUM (serial)':14s} {sim_serial:>10.3f} {m_gpu:>12.3f} {ratio(sim_serial, m_gpu):>10s}  (vs gpu-busy)")
    print(f"  {'':14s} {'':>10s} {m_wall:>12.3f} {ratio(sim_serial, m_wall):>10s}  (vs decode wall)")
    # phase397m: the ep8 decode overhead is sourced from the backend constant.
    # Pre-phase397m it was the fictional 90ms placeholder for unmodeled TP comm;
    # it is now 0 because communication is modeled structurally by the re-enabled
    # generation_ar CustomAllReduce ops (see the comm category above).
    ep8_overhead = VLLMBackend._CB_SIM_8GPU_EP_DECODE_OVERHEAD_MS
    print(f"\n  cb_sim decode total = serial-sum {sim_serial:.3f} + ep8_overhead {ep8_overhead:.1f} = "
          f"{sim_serial + ep8_overhead:.3f} ms  (real wall {m_wall:.3f} ms => "
          f"{(sim_serial + ep8_overhead) / m_wall:.2f}x)")
    if ep8_overhead > 0:
        print(f"  measured comm is {m_comm:.3f} ms/iter; the {ep8_overhead:.0f} ms overhead is "
              f"{ep8_overhead / m_comm:.1f}x the real comm and {ep8_overhead / m_wall:.1f}x the whole iteration")
    else:
        print(f"  comm now structural: cb_sim comm {comm:.3f} ms vs measured {m_comm:.3f} ms "
              f"({ratio(comm, m_comm).strip()}); residual gap is attention {ratio(attn, m_attn).strip()} + "
              f"proj {ratio(proj, m_proj).strip()} (phase C)")

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["category", "cb_sim_ms", "measured_ms", "sim_over_meas"])
            for cname, sv, mv in [
                ("attention", attn, m_attn), ("moe", moe, m_moe),
                ("proj_gemm+norm", proj, m_proj), ("comm", comm, m_comm),
            ]:
                w.writerow([cname, round(sv, 4), round(mv, 4),
                            round(sv / mv, 4) if mv else ""])
            w.writerow(["serial_sum", round(sim_serial, 4), round(m_gpu, 4),
                        round(sim_serial / m_gpu, 4) if m_gpu else ""])
            w.writerow(["decode_wall_measured", "", round(m_wall, 4), ""])
            w.writerow(["ep8_overhead_ms", ep8_overhead, "", ""])
            w.writerow(["cb_sim_total_with_overhead", round(sim_serial + ep8_overhead, 4),
                        round(m_wall, 4), round((sim_serial + ep8_overhead) / m_wall, 4)])
        print(f"\n  wrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default="0.19.0")
    ap.add_argument("--kv-len", type=int, default=9000,
                    help="representative decode KV length (ISL + ~OSL/2)")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO_ROOT / "docs/iter_gap_investigation/phase397l_decode_profile")
    args = ap.parse_args()

    # Measured rollups from the profiler op_breakdown.csv files.
    measured_tp8 = {
        "attention": 16.898, "moe_expert_gemm": 8.020, "moe_aux": 1.509,
        "proj_gemm": 4.042, "norm_elementwise": 1.117, "comm": 6.181,
        "gpu_busy": 39.386, "wall": 36.847,
    }
    measured_tp4dp2 = {
        "attention": 13.649, "moe_expert_gemm": 14.156, "moe_aux": 1.202,
        "proj_gemm": 6.169, "norm_elementwise": 1.234, "comm": 11.998,
        "gpu_busy": 49.422, "wall": 46.680,
    }

    analyze("tp8ep8-8k2k", tp=8, dp=1, moe_tp=1, moe_ep=8, version=args.version,
            decode_bs=128, kv_len=args.kv_len, measured=measured_tp8,
            out_csv=args.out_dir / "K2.5-tp8ep8-8k2k" / "db_align.csv")
    # tp4dp2: cb_sim models a single DP replica; the profiled replica saw bs=69.
    analyze("tp4dp2ep8-8k2k", tp=4, dp=2, moe_tp=1, moe_ep=8, version=args.version,
            decode_bs=69, kv_len=args.kv_len, measured=measured_tp4dp2,
            out_csv=args.out_dir / "K2.5-tp4ep8dp2-8k2k" / "db_align.csv")


if __name__ == "__main__":
    main()

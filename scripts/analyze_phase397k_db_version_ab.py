#!/usr/bin/env python3
"""Phase397k: perf-DB version A/B for cb_sim aggregate throughput.

Runs the cb_sim aggregate predictor for each MULTI_CONFIG_DATA point against
both the legacy 0.12.0 perf database and the freshly-collected 0.19.0 perf
database, and compares both against the freshly-measured vLLM 0.19 serving
throughput. The goal is to confirm that swapping the simulator DB 0.12.0 ->
0.19.0 closes the absolute-magnitude under-prediction gap (phase397i/j).

Report only. Does NOT modify validate_cb_simulator.py gate constants.

Modes
-----
build-measured
    Turn the run_phase397k_agg_bench.sh output tree (per-point meta.json +
    bench_result.json) into a flat measured manifest CSV.

ab
    Run cb_sim on 0.12.0 and 0.19.0 for every point, join with the measured
    manifest, and emit the A/B convergence table + verdict.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from aiconfigurator.sdk import common  # noqa: E402
from aiconfigurator.sdk.backends.vllm_backend import VLLMBackend  # noqa: E402
from aiconfigurator.sdk.config import ModelConfig, RuntimeConfig  # noqa: E402
from aiconfigurator.sdk.models import get_model  # noqa: E402
from aiconfigurator.sdk.perf_database import PerfDatabase  # noqa: E402

from validate_cb_simulator import (  # noqa: E402
    BACKEND,
    MODEL_PATH,
    MULTI_CONFIG_DATA,
    SYSTEM,
    _abs_error,
    _make_cb_config,
)

SYSTEMS_ROOT = str(REPO_ROOT / "src" / "aiconfigurator" / "systems")
DB_VERSIONS = ("0.12.0", "0.19.0")
OVERLAP_FACTOR = 0.0
# phase397m: source the ep8 decode overhead from the backend constant (now 0.0)
# so this A/B reflects the current structural model — communication is carried by
# the re-enabled generation_ar CustomAllReduce ops instead of the fictional 90ms.
# Override with --ep8-overhead-ms 90 to reproduce the pre-phase397m comparison.
EP8_OVERHEAD_MS = VLLMBackend._CB_SIM_8GPU_EP_DECODE_OVERHEAD_MS


def _load_model_db(tp: int, dp: int, moe_tp: int, moe_ep: int, version: str):
    model_config = ModelConfig(
        tp_size=tp,
        pp_size=1,
        moe_tp_size=moe_tp,
        moe_ep_size=moe_ep,
        attention_dp_size=dp,
    )
    model = get_model(MODEL_PATH, model_config, backend_name=BACKEND)
    db = PerfDatabase(
        system=SYSTEM, backend=BACKEND, version=version, systems_root=SYSTEMS_ROOT
    )
    # phase397m: per-op queries fall back to db._default_database_mode (SILICON),
    # which hard-asserts when a kernel table is absent. K2.5 now declares w4a16, but
    # the legacy 0.12.0 DB predates 4-bit MoE and has no int4_wo table. Use HYBRID so
    # the missing-table case degrades to the empirical estimate (matching the
    # database_mode=HYBRID we pass to run_agg). 0.19.0 has the real int4 table, so its
    # numbers are unaffected; only 0.12.0's int4 queries take the empirical path.
    db._default_database_mode = common.DatabaseMode.HYBRID
    return model, db


def _sim_point(pt, version: str, ep8_overhead_ms: float = EP8_OVERHEAD_MS) -> dict:
    model, db = _load_model_db(pt.tp, pt.dp, pt.moe_tp, pt.moe_ep, version)
    # Force the standard kernel-table modeling path for BOTH DB versions.
    #
    # operations._is_vllm_module_scope() auto-activates an experimental Kimi
    # module-runtime route only for (version==0.19.0 AND topology==tp4dp2ep8),
    # which enforces discrete token buckets and therefore cannot run arbitrary
    # aggregate chunk sizes. That route is an orthogonal phase342-style module
    # perf table, not the kernel DB this phase re-collected. To keep the A/B a
    # like-for-like comparison of the kernel tables (attention/mla/gemm/moe/
    # allreduce), we bypass the module route by overriding model_name (the model
    # has no model_name attr by default, so this only affects the 0.19.0
    # tp4dp2ep8 case; 0.12.0 and 0.19.0-tp8ep8 never trigger it anyway).
    model.model_name = "phase397k-stdkernel"
    backend = VLLMBackend()
    cb_config = _make_cb_config(
        pt.isl,
        pt.batch_size,
        max_num_batched_tokens=pt.max_num_batched_tokens,
        overlap_factor=OVERLAP_FACTOR,
        per_iteration_overhead_ms=ep8_overhead_ms,
    )
    summary = backend.run_agg(
        model,
        db,
        RuntimeConfig(batch_size=pt.batch_size, isl=pt.isl, osl=pt.osl),
        ctx_tokens=pt.max_num_batched_tokens,
        database_mode=common.DatabaseMode.HYBRID,
        method="cb_sim",
        cb_config=cb_config,
    )
    d = summary.get_result_dict() or {}
    return {
        "output_tok_s_gpu": float(d.get("tokens/s/gpu", 0.0) or 0.0),
        "ttft_ms": float(d.get("ttft", 0.0) or 0.0),
        "tpot_ms": float(d.get("tpot", 0.0) or 0.0),
    }


def build_measured(results_dir: Path, out_csv: Path) -> None:
    rows: list[dict] = []
    for meta_path in sorted(results_dir.glob("*/meta.json")):
        point_dir = meta_path.parent
        bench_path = point_dir / "bench_result.json"
        if not bench_path.exists():
            print(f"skip (no bench_result.json): {point_dir.name}", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        bench = json.loads(bench_path.read_text(encoding="utf-8"))
        world = int(meta["world_size"])
        isl = int(meta["isl"])
        osl = int(meta["osl"])
        ok = int(bench.get("ok_requests", 0))
        fail = int(bench.get("failed_requests", -1))
        output_tok_s = float(bench.get("output_tok_s", 0.0) or 0.0)
        total_tok_s = float(bench.get("total_tok_s", 0.0) or 0.0)
        output_tok_s_gpu = output_tok_s / world if world else 0.0
        total_tok_s_gpu = total_tok_s / world if world else 0.0
        # Same derivation the frozen MULTI_CONFIG_DATA baseline uses.
        output_tok_s_gpu_derived = total_tok_s_gpu * osl / (isl + osl)
        rows.append(
            {
                "name": meta["name"],
                "tp": int(meta["tp"]),
                "dp": int(meta["dp"]),
                "ep": int(meta["ep"]),
                "isl": isl,
                "osl": osl,
                "max_num_batched_tokens": int(meta["max_num_batched_tokens"]),
                "batch_size": int(meta["batch_size"]),
                "world_size": world,
                "ok_requests": ok,
                "failed_requests": fail,
                "wall_s": round(float(bench.get("wall_s", 0.0) or 0.0), 3),
                "output_tok_s": round(output_tok_s, 3),
                "total_tok_s": round(total_tok_s, 3),
                "measured_output_tok_s_gpu": round(output_tok_s_gpu, 4),
                "measured_total_tok_s_gpu": round(total_tok_s_gpu, 4),
                "measured_output_tok_s_gpu_derived": round(output_tok_s_gpu_derived, 4),
                "mean_latency_ms": round(float(bench.get("mean_latency_ms", 0.0) or 0.0), 3),
                "p99_latency_ms": round(float(bench.get("p99_latency_ms", 0.0) or 0.0), 3),
            }
        )
    if not rows:
        raise SystemExit(f"no measured points found under {results_dir}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} points)")


def _read_measured(measured_csv: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with measured_csv.open() as f:
        for row in csv.DictReader(f):
            out[row["name"]] = row
    return out


def run_ab(measured_csv: Path | None, out_csv: Path, ep8_overhead_ms: float = EP8_OVERHEAD_MS) -> None:
    measured = _read_measured(measured_csv) if measured_csv else {}
    print(f"[ab] ep8 decode per-iteration overhead = {ep8_overhead_ms} ms")
    rows: list[dict] = []
    for pt in MULTI_CONFIG_DATA:
        sim = {ver: _sim_point(pt, ver, ep8_overhead_ms) for ver in DB_VERSIONS}
        m = measured.get(pt.name)
        real = float(m["measured_output_tok_s_gpu"]) if m else 0.0
        row = {
            "name": pt.name,
            "tp": pt.tp,
            "dp": pt.dp,
            "ep": pt.moe_ep,
            "max_bt": pt.max_num_batched_tokens,
            "isl": pt.isl,
            "osl": pt.osl,
            "measured_output_tok_s_gpu": round(real, 4),
            "frozen_baseline_output_tok_s_gpu": round(pt.real_output_tok_s_gpu, 4),
            "sim012_output_tok_s_gpu": round(sim["0.12.0"]["output_tok_s_gpu"], 4),
            "sim019_output_tok_s_gpu": round(sim["0.19.0"]["output_tok_s_gpu"], 4),
            "err_sim012_vs_measured": (
                round(_abs_error(sim["0.12.0"]["output_tok_s_gpu"], real), 4) if real else ""
            ),
            "err_sim019_vs_measured": (
                round(_abs_error(sim["0.19.0"]["output_tok_s_gpu"], real), 4) if real else ""
            ),
            "sim019_vs_sim012_ratio": (
                round(
                    sim["0.19.0"]["output_tok_s_gpu"] / sim["0.12.0"]["output_tok_s_gpu"],
                    4,
                )
                if sim["0.12.0"]["output_tok_s_gpu"]
                else ""
            ),
            "sim012_ttft_ms": round(sim["0.12.0"]["ttft_ms"], 2),
            "sim019_ttft_ms": round(sim["0.19.0"]["ttft_ms"], 2),
            "sim012_tpot_ms": round(sim["0.12.0"]["tpot_ms"], 3),
            "sim019_tpot_ms": round(sim["0.19.0"]["tpot_ms"], 3),
        }
        rows.append(row)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _print_table(rows)
    print(f"\nwrote {out_csv}")


def _print_table(rows: list[dict]) -> None:
    hdr = (
        f"{'point':32s} {'measured':>10s} {'sim@0.12':>10s} {'sim@0.19':>10s} "
        f"{'err012':>8s} {'err019':>8s} {'19/12':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    e012: list[float] = []
    e019: list[float] = []
    for r in rows:
        print(
            f"{r['name']:32s} {r['measured_output_tok_s_gpu']:>10} "
            f"{r['sim012_output_tok_s_gpu']:>10} {r['sim019_output_tok_s_gpu']:>10} "
            f"{str(r['err_sim012_vs_measured']):>8} {str(r['err_sim019_vs_measured']):>8} "
            f"{str(r['sim019_vs_sim012_ratio']):>7}"
        )
        if r["err_sim012_vs_measured"] != "":
            e012.append(float(r["err_sim012_vs_measured"]))
        if r["err_sim019_vs_measured"] != "":
            e019.append(float(r["err_sim019_vs_measured"]))
    if e012 and e019:
        print("-" * len(hdr))
        print(
            f"{'MAX symmetric error':32s} {'':>10s} {'':>10s} {'':>10s} "
            f"{max(e012):>8.3f} {max(e019):>8.3f}"
        )
        print(
            f"{'MEAN symmetric error':32s} {'':>10s} {'':>10s} {'':>10s} "
            f"{sum(e012)/len(e012):>8.3f} {sum(e019)/len(e019):>8.3f}"
        )
        verdict = "CLOSES" if max(e019) < max(e012) else "DOES NOT CLOSE"
        print(f"\nverdict: swapping 0.12.0 -> 0.19.0 DB {verdict} the absolute-magnitude gap")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_build = sub.add_parser("build-measured", help="build measured manifest from node results")
    p_build.add_argument("--results-dir", type=Path, required=True)
    p_build.add_argument("--out", type=Path, required=True)

    p_ab = sub.add_parser("ab", help="run cb_sim 0.12.0 vs 0.19.0 A/B")
    p_ab.add_argument("--measured", type=Path, default=None)
    p_ab.add_argument("--out", type=Path, required=True)
    p_ab.add_argument(
        "--ep8-overhead-ms",
        type=float,
        default=EP8_OVERHEAD_MS,
        help="ep8 decode per-iteration overhead ms (default: backend constant, "
        "now 0.0; pass 90 to reproduce the pre-phase397m A/B)",
    )

    args = parser.parse_args()
    if args.mode == "build-measured":
        build_measured(args.results_dir, args.out)
    elif args.mode == "ab":
        run_ab(args.measured, args.out, args.ep8_overhead_ms)


if __name__ == "__main__":
    main()

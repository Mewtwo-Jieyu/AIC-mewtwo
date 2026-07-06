import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ["PHASE431_OUT"])
MODEL_PATH = os.environ["MODEL_PATH"]
BATCHES = [8, 16, 32, 64, 128]
MLA_RAW = OUT / "phase431_generation_mla_raw.csv"
MOE_RAW = OUT / "phase431_moe_int4_wo_raw.csv"
SUMMARY = OUT / "phase431_mla_moe_summary.json"

from collector.vllm.collect_mla import run_attention_torch
from collector.vllm.collect_moe import run_moe_marlin_wna16


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _latencies_by_batch(path: Path, batch_field: str) -> dict[int, float]:
    rows = _read_rows(path)
    return {int(row[batch_field]): float(row["latency"]) for row in rows}


def _monotonic(values: dict[int, float]) -> bool:
    ordered = [values[b] for b in sorted(values)]
    return all(right >= left for left, right in zip(ordered, ordered[1:]))


def _collect_mla_once() -> None:
    if MLA_RAW.exists():
        MLA_RAW.unlink()
    for batch in BATCHES:
        run_attention_torch(
            batch_size=batch,
            input_len=8191,
            num_heads=128,
            tp_size=8,
            q_lora_rank=1536,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            qk_nope_head_dim=128,
            v_head_dim=128,
            block_size=16,
            model_name=MODEL_PATH,
            use_fp8_kv_cache=False,
            is_context_phase=False,
            perf_filename=str(MLA_RAW),
            device="cuda:0",
        )


def _collect_moe_once() -> None:
    if MOE_RAW.exists():
        MOE_RAW.unlink()
    run_moe_marlin_wna16(
        "int4_wo",
        BATCHES,
        hidden_size=7168,
        inter_size=2048,
        topk=8,
        num_experts=384,
        moe_tp_size=1,
        moe_ep_size=8,
        model_name=MODEL_PATH,
        perf_filename=str(MOE_RAW),
        distributed="power_law",
        power_law_alpha=1.01,
        device="cuda:0",
    )


def _run_with_retry(name: str, collect, path: Path, batch_field: str, *, require_monotonic: bool) -> dict:
    attempts = []
    for attempt in range(1, 4):
        collect()
        values = _latencies_by_batch(path, batch_field)
        complete = sorted(values) == BATCHES
        monotonic = _monotonic(values)
        attempts.append(
            {
                "attempt": attempt,
                "complete": complete,
                "monotonic": monotonic,
                "values": values,
            }
        )
        if complete and (monotonic or not require_monotonic):
            return {
                "name": name,
                "ok": True,
                "attempts": attempts,
                "selected_attempt": attempt,
            }
    return {
        "name": name,
        "ok": False,
        "attempts": attempts,
        "selected_attempt": None,
    }


def main() -> None:
    summary = {
        "source": "phase431_perfdb_recollect",
        "worker": subprocess.check_output(["hostname"], text=True).strip(),
        "timestamp_unix": time.time(),
        "committed_source_root": os.environ.get("SOURCE_ROOT", ""),
        "model_path": MODEL_PATH,
        "mla": _run_with_retry("mla", _collect_mla_once, MLA_RAW, "batch_size", require_monotonic=False),
        "moe": _run_with_retry("moe", _collect_moe_once, MOE_RAW, "num_tokens", require_monotonic=True),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not summary["mla"]["ok"] or not summary["moe"]["ok"]:
        raise SystemExit("phase431 MLA/MoE collection self-check failed")


if __name__ == "__main__":
    main()

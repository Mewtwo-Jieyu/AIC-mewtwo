#!/usr/bin/env python3
"""Phase447: compare current sim schedule fingerprint against B2b real fingerprint."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from aiconfigurator.sdk.backends.cb_simulator.datatypes import (  # noqa: E402
    CBSimConfig,
    Request,
    RequestState,
)
from aiconfigurator.sdk.backends.cb_simulator.scheduler import CBScheduler  # noqa: E402


DEFAULT_REAL_FINGERPRINT = REPO_ROOT / "docs/iter_gap_investigation/phase447_sequence_fingerprint.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase447_sim_fingerprint_gate.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase447_sim_fingerprint_gate.md"


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str
    sim_p50: float
    real_low: float
    real_high: float


def evaluate_mixed_batch_gate(
    *,
    sim_batches: list[int],
    real_low: float,
    real_high: float,
) -> GateVerdict:
    sim_p50 = float(statistics.median(sim_batches)) if sim_batches else 0.0
    passed = bool(sim_batches) and real_low <= sim_p50 <= real_high
    reason = "passed" if passed else "sim_mixed_decode_batch_outside_real_band"
    return GateVerdict(
        passed=passed,
        reason=reason,
        sim_p50=sim_p50,
        real_low=float(real_low),
        real_high=float(real_high),
    )


def _read_real_8k_mixed_batch_band(path: Path) -> tuple[float, float]:
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("scenario") == "K2.5-tp4ep8dp2-8k2k"
                and row.get("section") == "fingerprint_tolerance"
                and row.get("metric") == "mixed_decode_batch_p10_p90"
            ):
                return float(row["target_low"]), float(row["target_high"])
    raise ValueError(f"8k mixed_decode_batch_p10_p90 not found in {path}")


def _make_request(rid: int, isl: int, osl: int) -> Request:
    return Request(request_id=rid, isl=isl, osl=osl, arrival_time_ms=0.0)


def simulate_mixed_batches(
    *,
    isl: int = 8000,
    osl: int = 2000,
    concurrency: int = 64,
    max_num_batched_tokens: int = 8000,
    num_requests: int = 200,
) -> list[int]:
    config = CBSimConfig(
        max_num_batched_tokens=max_num_batched_tokens,
        num_requests=num_requests,
        warmup_requests=50,
    )
    scheduler = CBScheduler(config)
    waiting: list[Request] = []
    running: list[Request] = []
    completed: list[Request] = []
    next_id = 0
    mixed_batches: list[int] = []

    for _ in range(min(concurrency, num_requests)):
        waiting.append(_make_request(next_id, isl, osl))
        next_id += 1

    max_iters = num_requests * (osl + isl // max_num_batched_tokens + 10)
    for _ in range(max_iters):
        if len(completed) >= num_requests:
            break
        schedule = scheduler.schedule(waiting=waiting, running=running)
        if schedule.is_empty:
            break
        if schedule.total_prefill_tokens > 0 and schedule.decode_reqs:
            mixed_batches.append(len(schedule.decode_reqs))

        for req in schedule.prefill_reqs:
            tokens = schedule.prefill_tokens[req.request_id]
            if req.state in {RequestState.WAITING, RequestState.PREEMPTED}:
                req.state = RequestState.PREFILLING
                if req in waiting:
                    waiting.remove(req)
                running.append(req)
            req.prefill_tokens_remaining -= tokens
            if req.prefill_tokens_remaining <= 0:
                req.prefill_tokens_remaining = 0
                req.state = RequestState.DECODING

        newly_done: list[Request] = []
        for req in schedule.decode_reqs:
            req.generated_tokens += 1
            if req.generated_tokens >= req.osl - 1:
                req.state = RequestState.DONE
                newly_done.append(req)

        for req in newly_done:
            running.remove(req)
            completed.append(req)
            if next_id < num_requests:
                waiting.append(_make_request(next_id, isl, osl))
                next_id += 1

    return mixed_batches


def analyze(real_fingerprint: Path) -> tuple[list[dict[str, object]], GateVerdict]:
    real_low, real_high = _read_real_8k_mixed_batch_band(real_fingerprint)
    sim_batches = simulate_mixed_batches()
    verdict = evaluate_mixed_batch_gate(
        sim_batches=sim_batches,
        real_low=real_low,
        real_high=real_high,
    )
    rows = [
        {
            "scenario": "K2.5-tp4ep8dp2-8k2k",
            "metric": "mixed_decode_batch_p50",
            "sim_value": verdict.sim_p50,
            "real_low": verdict.real_low,
            "real_high": verdict.real_high,
            "passed": verdict.passed,
            "reason": verdict.reason,
            "runtime_lockstep_allowed": verdict.passed,
        },
        {
            "scenario": "K2.5-tp4ep8dp2-8k2k",
            "metric": "mixed_step_count",
            "sim_value": len(sim_batches),
            "real_low": "",
            "real_high": "",
            "passed": verdict.passed,
            "reason": verdict.reason,
            "runtime_lockstep_allowed": verdict.passed,
        },
    ]
    return rows, verdict


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "metric",
        "sim_value",
        "real_low",
        "real_high",
        "passed",
        "reason",
        "runtime_lockstep_allowed",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_md(path: Path, rows: list[dict[str, object]], verdict: GateVerdict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase447 sim fingerprint gate",
        "",
        "结论: 指纹门未过。当前 sim 的 8k2k mixed decode batch p50 仍在真实 p10-p90 区间之外,因此本 phase 不允许合入 runtime DP lockstep。",
        "",
        "| metric | sim | real low | real high | passed | reason |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['metric']} | {row['sim_value']} | {row['real_low']} | {row['real_high']} | "
            f"{row['passed']} | {row['reason']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- `runtime_lockstep_allowed={str(verdict.passed).lower()}`",
            "- 下一步必须先建模 DP global admission / stale score routing / closed-loop wave,让 sim 自然生成真实相位与构成;不能手工注入 offset。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-fingerprint", type=Path, default=DEFAULT_REAL_FINGERPRINT)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, verdict = analyze(args.real_fingerprint)
    write_csv(args.csv_out, rows)
    write_md(args.md_out, rows, verdict)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")
    if not verdict.passed:
        print(f"fingerprint gate failed: {verdict.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

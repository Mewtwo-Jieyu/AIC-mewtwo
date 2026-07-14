#!/usr/bin/env python3
"""Phase462 Step2a-3e fully predictive EngineCore report-only prototype."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

PRIMITIVE_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_arrival_admission_prototype.csv"
)
BENCH_META = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/meta.json"
)
ARRIVAL_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_arrival_observation/32k3k.jsonl.gz"
)
PREEMPT_SERVE_LOG = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/overhead_on/"
    "K2.5-tp8ep8-32k3k/serve.log.gz"
)
PREEMPT_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_preemption_observation/"
    "preemption_observation.jsonl"
)
DP_DIAGNOSTIC_JSONL = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase461_cost_recollect/"
    "dp2_bt65536_diagnostic/overhead_on/event_timing.jsonl.gz"
)
DEFAULT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_predictive_engine_loop.csv"
)
DEFAULT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_predictive_engine_loop.md"
)
CSV_FIELDS = ["section", "mode", "metric", "value", "target", "status", "note"]

REQUIRED_PREDICTION_DEPENDENCIES = (
    "bench_protocol",
    "deployment_config",
    "perfdb_iteration_cost",
    "tokenizer_primitive",
    "source_engine_loop_rules",
)
JUDGE_ONLY_DEPENDENCIES = {
    "target_run_timestamps",
    "target_run_drain_steps",
    "target_run_first_schedule_steps",
    "target_run_iteration_trace",
    "target_run_preemption_signature",
}
REPORT_BOUNDARIES = {
    "diagnostic_only": True,
    "valid_for_default": False,
    "perf_database": False,
    "default_aic": "No-Go",
}

FIRST_SCHEDULE_MIN = 0.85
FIRST_16_MIN_MATCHES = math.ceil(16 * 0.85)
PREEMPTION_TOLERANCE = 2
TOKENIZER_MAX_BATCH_SIZE = 32
TOKENIZER_WAIT_TIMEOUT_MS = 2.0
TOKENIZER_WORKERS = 1

FORMULA_RE = re.compile(
    r"^(?P<intercept>-?[0-9.]+) \+ "
    r"(?P<token>-?[0-9.]+)\*tokens \+ "
    r"\((?P<batch>-?[0-9.]+)\)\*batch$"
)


@dataclass(frozen=True)
class BenchProtocol:
    num_requests: int
    concurrency: int
    prompt_tokens: int


@dataclass(frozen=True)
class TokenizerRuntimeSpec:
    max_batch_size: int
    wait_timeout_ms: float
    workers: int


@dataclass(frozen=True)
class ApprovedTokenizerPrimitive:
    intercept_ms: float
    ms_per_prompt_token: float
    ms_per_request: float
    weighted_mape: float

    def predict(self, total_prompt_tokens: int, batch_size: int) -> float:
        return (
            self.intercept_ms
            + self.ms_per_prompt_token * total_prompt_tokens
            + self.ms_per_request * batch_size
        )


@dataclass(frozen=True)
class PredictedTokenizerBatch:
    batch_id: int
    request_ids: tuple[int, ...]
    start_ms: float
    service_ms: float
    complete_ms: float


@dataclass(frozen=True)
class TokenizerPrediction:
    batches: tuple[PredictedTokenizerBatch, ...]
    arrivals: tuple[object, ...]


def validate_prediction_dependencies(dependencies: Iterable[str]) -> None:
    values = set(dependencies)
    contaminated = values & JUDGE_ONLY_DEPENDENCIES
    if contaminated:
        raise ValueError(f"judge-only input used by predictor: {sorted(contaminated)}")
    unknown = values - set(REQUIRED_PREDICTION_DEPENDENCIES)
    if unknown:
        raise ValueError(f"unknown prediction dependencies: {sorted(unknown)}")
    missing = set(REQUIRED_PREDICTION_DEPENDENCIES) - values
    if missing:
        raise ValueError(f"missing prediction dependencies: {sorted(missing)}")


def load_approved_tokenizer_primitive(path: Path) -> ApprovedTokenizerPrimitive:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    formula_rows = [
        row
        for row in rows
        if row["section"] == "primitive_fit"
        and row["scenario"] == "8k2k+32k3k"
        and row["metric"] == "formula"
    ]
    wmape_rows = [
        row
        for row in rows
        if row["section"] == "primitive_fit"
        and row["scenario"] == "8k2k+32k3k"
        and row["metric"] == "tokenizer_weighted_mape"
    ]
    if len(formula_rows) != 1 or len(wmape_rows) != 1:
        raise ValueError("approved tokenizer primitive artifact is incomplete")
    formula = formula_rows[0]
    wmape = wmape_rows[0]
    match = FORMULA_RE.fullmatch(formula["value"])
    if match is None or formula["status"] != "measured" or wmape["status"] != "pass":
        raise ValueError("approved tokenizer primitive artifact did not pass")
    weighted_mape = float(wmape["value"])
    if weighted_mape > 0.10:
        raise ValueError(f"tokenizer primitive WMAPE exceeds gate: {weighted_mape}")
    return ApprovedTokenizerPrimitive(
        intercept_ms=float(match.group("intercept")),
        ms_per_prompt_token=float(match.group("token")),
        ms_per_request=float(match.group("batch")),
        weighted_mape=weighted_mape,
    )


def load_bench_protocol(path: Path) -> BenchProtocol:
    data = json.loads(path.read_text(encoding="utf-8"))
    return BenchProtocol(
        num_requests=int(data["num_prompts"]),
        concurrency=int(data["concurrency"]),
        prompt_tokens=int(data["isl"]),
    )


def build_predictive_tokenizer_arrivals(
    *,
    protocol: BenchProtocol,
    tokenizer: TokenizerRuntimeSpec,
    primitive: ApprovedTokenizerPrimitive,
) -> TokenizerPrediction:
    """Predict the initial closed-loop cohort without target-run timestamps."""
    import scripts.analyze_phase462_engine_loop_state_machine as engine

    if tokenizer.workers != 1:
        raise ValueError(f"unsupported tokenizer worker count: {tokenizer.workers}")
    if protocol.num_requests < 1 or protocol.concurrency < 1:
        raise ValueError("bench protocol must contain requests")
    cohort_size = min(protocol.num_requests, protocol.concurrency)
    pending = deque(range(cohort_size))
    worker_available_ms = 0.0
    batches: list[PredictedTokenizerBatch] = []
    arrivals: list[engine.TimedInput] = []

    while pending:
        request_ids = tuple(
            pending.popleft()
            for _ in range(min(tokenizer.max_batch_size, len(pending)))
        )
        start_ms = worker_available_ms
        if len(request_ids) < tokenizer.max_batch_size:
            start_ms += tokenizer.wait_timeout_ms
        service_ms = primitive.predict(
            protocol.prompt_tokens * len(request_ids), len(request_ids)
        )
        if service_ms <= 0:
            raise ValueError(f"non-positive tokenizer service time: {service_ms}")
        complete_ms = start_ms + service_ms
        batch = PredictedTokenizerBatch(
            batch_id=len(batches) + 1,
            request_ids=request_ids,
            start_ms=start_ms,
            service_ms=service_ms,
            complete_ms=complete_ms,
        )
        batches.append(batch)
        arrivals.extend(
            engine.TimedInput(arrival_ms=complete_ms, payload=request_id)
            for request_id in request_ids
        )
        worker_available_ms = complete_ms

    return TokenizerPrediction(
        batches=tuple(batches),
        arrivals=tuple(engine.normalize_timed_inputs(arrivals)),
    )


def predictive_gate(
    *,
    first_schedule_step_match: float,
    first_16_match: float,
    preemptions: int,
    self_preemptions: int,
    repeat_victim_events: int,
    target_preemptions: int,
) -> dict[str, object]:
    passed = (
        first_schedule_step_match >= FIRST_SCHEDULE_MIN
        and first_16_match * 16 >= FIRST_16_MIN_MATCHES
        and abs(preemptions - target_preemptions) <= PREEMPTION_TOLERANCE
        and self_preemptions == 0
        and repeat_victim_events == 0
    )
    return {
        "passed": passed,
        "first_schedule_min": FIRST_SCHEDULE_MIN,
        "first_16_min_matches": FIRST_16_MIN_MATCHES,
        "preemption_tolerance": PREEMPTION_TOLERANCE,
    }


def runtime_scope_verdict(*, prediction_gate_passed: bool) -> dict[str, object]:
    if prediction_gate_passed:
        return {
            "step2c": "candidate",
            "verdict": "prediction_gate_passed_report_only",
            "runtime_change_allowed": False,
        }
    return {
        "step2c": "locked",
        "verdict": "structure_correct_prediction_limited",
        "runtime_change_allowed": False,
    }


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    with _open_text(path) as source:
        return [json.loads(line) for line in source if line.strip()]


def _initial_target_batch_sizes(
    rows: Iterable[dict[str, object]], *, request_limit: int
) -> list[int]:
    sizes: list[int] = []
    observed = 0
    for row in rows:
        if row.get("kind") != "tokenizer_batch_enter":
            continue
        size = sum(
            int(str(trace_id).rsplit("-", 1)[-1]) < request_limit
            for trace_id in row.get("trace_ids", [])
        )
        if size:
            sizes.append(size)
            observed += size
        if observed >= request_limit:
            break
    return sizes


def _summarize_dp_stored_timing(path: Path) -> dict[str, object]:
    ranks: set[int] = set()
    timestamped_rows: Counter[int] = Counter()
    row_counts: Counter[int] = Counter()
    with _open_text(path) as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("dp_rank") is None:
                continue
            rank = int(row["dp_rank"])
            ranks.add(rank)
            row_counts[rank] += 1
            if "ts_ns" in row or "timestamp_ns" in row:
                timestamped_rows[rank] += 1
    return {
        "ranks": sorted(ranks),
        "row_counts": dict(sorted(row_counts.items())),
        "timestamped_rows": dict(sorted(timestamped_rows.items())),
        "has_per_rank_timeline": (
            len(ranks) >= 2 and all(timestamped_rows[rank] > 0 for rank in ranks)
        ),
    }


def _row(
    section: str,
    mode: str,
    metric: str,
    value: object,
    *,
    target: object = "",
    status: str = "",
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "mode": mode,
        "metric": metric,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def build_report() -> tuple[list[dict[str, object]], dict[str, object]]:
    import scripts.analyze_phase462_engine_loop_state_machine as engine
    import scripts.analyze_phase462_preemption_observation as preemption

    validate_prediction_dependencies(REQUIRED_PREDICTION_DEPENDENCIES)
    primitive = load_approved_tokenizer_primitive(PRIMITIVE_CSV)
    protocol = load_bench_protocol(BENCH_META)
    tokenizer = TokenizerRuntimeSpec(
        max_batch_size=TOKENIZER_MAX_BATCH_SIZE,
        wait_timeout_ms=TOKENIZER_WAIT_TIMEOUT_MS,
        workers=TOKENIZER_WORKERS,
    )
    deployment = engine.derive_deployment_queue_spec(
        executor="multiproc",
        pipeline_parallel_size=1,
        async_scheduling=True,
    )

    # Prediction is complete before any judge-only target timeline is loaded.
    prediction = build_predictive_tokenizer_arrivals(
        protocol=protocol,
        tokenizer=tokenizer,
        primitive=primitive,
    )
    predicted_result = engine._run_cb_queue_prototype(
        arrivals=list(prediction.arrivals),
        measured_iteration_latencies_ms=None,
        queue_depth=deployment.queue_depth,
    )

    arrival_rows = _load_jsonl(ARRIVAL_JSONL)
    real_first_schedule = engine.map_real_first_schedule_steps(
        arrival_rows, request_limit=protocol.concurrency
    )
    real_drain_steps = engine.map_real_drain_steps(
        arrival_rows, request_limit=protocol.concurrency
    )
    real_iterations = engine.parse_iterations(PREEMPT_SERVE_LOG)
    real_pairs = preemption.pair_records(_load_jsonl(PREEMPT_JSONL))
    real_preemption = preemption.summarize_real_decisions(real_pairs)
    target_preemptions = int(real_preemption["preemptions"])

    first_schedule_match = engine.exact_match_fraction(
        real_first_schedule, predicted_result["first_schedule_steps"]
    )
    first_16_match = engine.first_n_shape_match(
        real_iterations, predicted_result["trace"], count=16
    )
    drain_match = engine.exact_match_fraction(
        real_drain_steps, predicted_result["drain_steps"]
    )
    signature = dict(predicted_result["preemption"])
    gate = predictive_gate(
        first_schedule_step_match=first_schedule_match,
        first_16_match=first_16_match,
        preemptions=int(signature["preemptions"]),
        self_preemptions=int(signature["self_preemptions"]),
        repeat_victim_events=int(signature["repeat_victim_events"]),
        target_preemptions=target_preemptions,
    )
    scope = runtime_scope_verdict(prediction_gate_passed=bool(gate["passed"]))
    dp_timing = _summarize_dp_stored_timing(DP_DIAGNOSTIC_JSONL)
    target_batch_sizes = _initial_target_batch_sizes(
        arrival_rows, request_limit=protocol.concurrency
    )
    predicted_batch_sizes = [len(batch.request_ids) for batch in prediction.batches]

    rows: list[dict[str, object]] = []
    for dependency in REQUIRED_PREDICTION_DEPENDENCIES:
        rows.append(
            _row(
                "prediction_whitelist",
                "full_prediction",
                dependency,
                "eligible",
                target="eligible",
                status="pass",
            )
        )
    for dependency in sorted(JUDGE_ONLY_DEPENDENCIES):
        rows.append(
            _row(
                "prediction_whitelist",
                "full_prediction",
                dependency,
                "judge_only",
                target="not predictor input",
                status="pass",
            )
        )
    rows.extend(
        [
            _row(
                "prediction_input",
                "bench",
                "initial_cohort",
                min(protocol.num_requests, protocol.concurrency),
                target=protocol.concurrency,
                status="pass",
                note="num_prompts/concurrency/isl only; no request timestamp read",
            ),
            _row(
                "prediction_input",
                "tokenizer",
                "runtime_config",
                "32 requests / 2ms / 1 worker",
                target="runtime-observed configuration",
                status="pass",
                note="Phase462 arrival observation source/config audit",
            ),
            _row(
                "prediction_input",
                "engine_loop",
                "deployment_config",
                "multiproc / pp=1 / async=true / queue_depth=2",
                target="runtime-observed configuration + source rule",
                status="pass",
                note="Phase462 Step2a-3d deployment audit",
            ),
            _row(
                "prediction_input",
                "tokenizer",
                "weighted_mape",
                primitive.weighted_mape,
                target="<=0.10",
                status="pass",
                note="approved Phase462 measured primitive artifact",
            ),
            _row(
                "prediction_input",
                "tokenizer",
                "predicted_initial_batch_sizes",
                json.dumps(predicted_batch_sizes),
                target="source-defined from bench/config",
                status="pass",
                note="computed before judge timeline load",
            ),
            _row(
                "judge_comparison",
                "tokenizer",
                "target_initial_batch_sizes",
                json.dumps(target_batch_sizes),
                target=json.dumps(predicted_batch_sizes),
                status="pass" if target_batch_sizes == predicted_batch_sizes else "fail",
                note="judge only; never fed back into prediction",
            ),
            _row(
                "predictive_signature",
                "full_prediction",
                "drain_step_match",
                drain_match,
                target="diagnostic only",
                status="measured",
            ),
            _row(
                "predictive_signature",
                "full_prediction",
                "first_schedule_step_match",
                first_schedule_match,
                target=FIRST_SCHEDULE_MIN,
                status="pass" if first_schedule_match >= FIRST_SCHEDULE_MIN else "fail",
            ),
            _row(
                "predictive_signature",
                "full_prediction",
                "first_16_match",
                first_16_match,
                target=f">={FIRST_16_MIN_MATCHES}/16",
                status=(
                    "pass"
                    if first_16_match * 16 >= FIRST_16_MIN_MATCHES
                    else "fail"
                ),
            ),
        ]
    )
    for metric, target in (
        ("preemptions", f"{target_preemptions}±{PREEMPTION_TOLERANCE}"),
        ("self_preemptions", 0),
        ("repeat_victim_events", 0),
    ):
        value = int(signature[metric])
        if metric == "preemptions":
            passed = abs(value - target_preemptions) <= PREEMPTION_TOLERANCE
        else:
            passed = value == 0
        rows.append(
            _row(
                "predictive_signature",
                "full_prediction",
                metric,
                value,
                target=target,
                status="pass" if passed else "fail",
            )
        )
    rows.extend(
        [
            _row(
                "gate",
                "full_prediction",
                "step2a_3e",
                bool(gate["passed"]),
                target=True,
                status="pass" if gate["passed"] else "fail",
            ),
            _row(
                "gate",
                "full_prediction",
                "step2c",
                scope["step2c"],
                target="candidate",
                status="pass" if gate["passed"] else "blocked",
                note=str(scope["verdict"]),
            ),
            _row(
                "dp_evidence",
                "phase461_stored",
                "ranks",
                json.dumps(dp_timing["ranks"]),
                target="[0, 1]",
                status="pass" if dp_timing["ranks"] == [0, 1] else "fail",
            ),
            _row(
                "dp_evidence",
                "phase461_stored",
                "row_counts",
                json.dumps(dp_timing["row_counts"], sort_keys=True),
                target="non-zero for both ranks",
                status=(
                    "pass"
                    if all(dp_timing["row_counts"].get(rank, 0) > 0 for rank in (0, 1))
                    else "fail"
                ),
                note="composition and busy-time rows exist",
            ),
            _row(
                "dp_evidence",
                "phase461_stored",
                "timestamped_rows",
                json.dumps(dp_timing["timestamped_rows"], sort_keys=True),
                target="non-zero for both ranks",
                status="pass" if dp_timing["has_per_rank_timeline"] else "fail",
                note="DP short run remains deferred unless TP8 predictive gate passes",
            ),
        ]
    )
    for scenario in (
        "K2.5-tp8ep8-8k2k",
        "K2.5-tp4ep8dp2-8k2k",
        "K2.5-tp8ep8-8k2k-bt65536",
        "K2.5-tp4ep8dp2-8k2k-bt65536",
        "K2.5-tp4ep8dp2-32k3k",
        "K2.5-tp8ep8-32k3k",
    ):
        rows.append(
            _row(
                "six_point_sign_prediction",
                scenario,
                "throughput_direction",
                "not_identifiable" if not gate["passed"] else "requires_dynamic_ab",
                target="falsifiable direction",
                status="blocked" if not gate["passed"] else "pending",
                note="full predictive signature gate must pass before sign claims",
            )
        )
    for scenario in (
        "K2.5-tp8ep8-8k2k-bt65536",
        "K2.5-tp4ep8dp2-8k2k-bt65536",
        "K2.5-tp4ep8dp2-32k3k",
    ):
        rows.append(
            _row(
                "convergence_assessment",
                scenario,
                "step2c_convergence_claim",
                "eligible" if gate["passed"] else "not_identifiable",
                target="evidence-backed",
                status="pass" if gate["passed"] else "blocked",
                note=(
                    "full predictive gate passed"
                    if gate["passed"]
                    else "structure is correct but predictor misses target signatures"
                ),
            )
        )
    for key, value in REPORT_BOUNDARIES.items():
        rows.append(
            _row(
                "boundary",
                "all",
                key,
                value,
                status="blocked" if key == "default_aic" else "pass",
            )
        )

    details = {
        "gate": gate,
        "scope": scope,
        "primitive": primitive,
        "protocol": protocol,
        "tokenizer": tokenizer,
        "deployment": deployment,
        "prediction": prediction,
        "predicted_result": predicted_result,
        "signature": signature,
        "target_preemptions": target_preemptions,
        "first_schedule_match": first_schedule_match,
        "first_16_match": first_16_match,
        "drain_match": drain_match,
        "target_batch_sizes": target_batch_sizes,
        "predicted_batch_sizes": predicted_batch_sizes,
        "dp_timing": dp_timing,
    }
    return rows, details


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, details: dict[str, object]) -> None:
    gate = details["gate"]
    signature = details["signature"]
    scope = details["scope"]
    dp_timing = details["dp_timing"]
    passed = bool(gate["passed"])
    conclusion = (
        "全预测门通过，但本步仍是 report-only；2c 只能进入方案评审。"
        if passed
        else "全预测门失败：引擎环结构正确，但仅靠合规输入无法预测目标时序；2c 保持锁定。"
    )
    text = f"""# Phase462 Step 2a-3e 全预测引擎环

{conclusion}

| 判卷项 | 全预测 | 门 | 结果 |
|---|---:|---:|---|
| 首调度步精确吻合 | {details['first_schedule_match']:.2%} | >=85% | {'通过' if details['first_schedule_match'] >= FIRST_SCHEDULE_MIN else '失败'} |
| 前 16 步形状吻合 | {details['first_16_match']:.2%} | >=14/16 | {'通过' if details['first_16_match'] * 16 >= FIRST_16_MIN_MATCHES else '失败'} |
| 抢占 / 自抢占 / 重复 victim | {signature['preemptions']} / {signature['self_preemptions']} / {signature['repeat_victim_events']} | {details['target_preemptions']}±2 / 0 / 0 | {'通过' if passed else '失败'} |
| EngineCore drain 步吻合 | {details['drain_match']:.2%} | 诊断读数 | 不入门 |

## 输入边界

| 类别 | 内容 | 用法 |
|---|---|---|
| 预测白名单 | bench 协议、部署配置、PerfDB iteration cost、tokenizer measured primitive、源码引擎环规则 | 预测输入 |
| 目标运行时间戳、首调度步、iteration trace、抢占签名 | Phase462 target run | 预测完成后判卷，禁止回灌 |
| tokenizer 初始 batch | 预测 `{details['predicted_batch_sizes']}`；目标 `{details['target_batch_sizes']}` | 目标值只用于解释偏差 |

预测按 bench 的 128 并发初始 cohort 在 `t=0` 提交，并按运行时 `32 requests / 2ms / 1 worker` 与已过门 tokenizer 原语生成输入。若 batch 形态与目标不同，缺的是 bench 提交到 tokenizer queue 的 API 侧暴露时序；该量不在白名单内，不能拿目标时间戳补齐。

## 止损与后续

| 项目 | 结论 |
|---|---|
| Step 2c | `{scope['step2c']}`；`{scope['verdict']}` |
| 三个 fail 点可收敛度 | {'可进入动态 A/B' if passed else '当前不可识别，不声称符号或收益'} |
| 六点符号预测 | {'待动态 A/B' if passed else '六点均不可识别；不使用失败原型预测方向'} |
| Phase461 DP 存量 | ranks={dp_timing['ranks']}，row_counts={dp_timing['row_counts']}，timestamped_rows={dp_timing['timestamped_rows']}；{'可继承' if dp_timing['has_per_rank_timeline'] else '有构成/耗时行但无 per-rank 时序，不能继承'} |
| dp2 logging-only 短跑 | {'可提方案' if passed and not dp_timing['has_per_rank_timeline'] else '不触发'} |
| runtime / PerfDB / validate / gate | 均不改 |
| Default AIC | No-Go |

边界：`diagnostic_only=true`、`valid_for_default=false`、`perf_database=false`。
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows, details = build_report()
    write_csv(args.csv, rows)
    write_markdown(args.md, details)
    print(
        "phase462_predictive_engine_loop "
        f"gate={details['gate']['passed']} "
        f"schedule={details['first_schedule_match']:.6f} "
        f"first16={details['first_16_match']:.6f} "
        f"preemption={details['signature']['preemptions']}/"
        f"{details['signature']['self_preemptions']}/"
        f"{details['signature']['repeat_victim_events']} "
        f"step2c={details['scope']['step2c']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

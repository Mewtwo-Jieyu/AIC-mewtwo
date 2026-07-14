#!/usr/bin/env python3
"""Phase462 Step2c-1 report-only engine-loop integration design review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SIGN_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase462_exposure_sensitivity.csv"
PRIMITIVE_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase462_arrival_admission_prototype.csv"
)
DEFAULT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_engine_loop_integration_design.csv"
)
DEFAULT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase462_engine_loop_integration_design.md"
)
CSV_FIELDS = ["section", "item", "value", "target", "status", "note"]


def build_design_spec() -> dict[str, Any]:
    return {
        "runtime_paths": [
            {
                "path": "single_replica",
                "entry": "CBSimulator.run",
                "change": "engine_loop_state_machine",
                "delivery": "default_after_all_gates",
            },
            {
                "path": "multi_replica",
                "entry": "CBSimulator.run_multi_replica",
                "change": "engine_loop_state_machine",
                "delivery": "capability_then_step3_gate",
            },
            {
                "path": "dp_lockstep",
                "entry": "CBSimulator._run_multi_replica_lockstep",
                "change": "engine_loop_state_machine",
                "delivery": "preserve_lockstep_max",
            },
        ],
        "arrival_layer": {
            "workload_exposure": "t0_closed_loop",
            "initial_requests": "min(num_requests, concurrency)",
            "replacement_release": "on_request_completion",
            "tokenizer_primitive": "measured_32_requests_2ms",
            "primitive_support": "ISL 8k/32k; batch 1-32",
            "engine_receive": "tokenizer_completion_posts_input_event",
        },
        "engine_state": {
            "input_queue",
            "schedule_queue_depth",
            "execution_future",
            "sampled_output_tokens",
            "computed_output_tokens",
            "output_placeholders",
            "per_rank_clock",
        },
        "unchanged": {
            "scheduler_admission_policy",
            "preemption_trigger",
            "victim_selection",
            "max_bt_keying",
            "perf_database",
            "validate_protocol",
            "multi_config_gate",
        },
        "oracle_gate": {
            "input_drain_match": "100%",
            "first_schedule_match": ">=93.75%",
            "first_16_steps": "16/16",
            "purpose": "implementation_regression_anchor_only",
        },
        "steady_gate": {
            "preemptions": "10+/-2",
            "self_preemption": 0,
            "repeat_victim": 0,
            "failure_action": "stop_before_ab",
        },
        "dynamic_ab_gate": {
            "six_point_signs": "4_worsen_2_improve_until_crossing",
            "giant_bucket": "n512_trace_only",
            "unexpected_sign_action": "stop_before_cascade",
        },
        "stop_lines": {
            "oracle_structure_regression",
            "steady_self_preemption_nonzero",
            "unexpected_ab_sign",
            "out_of_scope_file_change",
        },
        "fallback": "none",
    }


def load_sign_predictions(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        signs = [
            {
                "scenario": row["scenario"],
                "prediction": row["value"],
                "status": row["status"],
                "note": row["note"],
            }
            for row in csv.DictReader(source)
            if row["section"] == "six_point_sign_prediction"
        ]
    if len(signs) != 6:
        raise ValueError(f"expected six sign predictions, found {len(signs)}")
    if sum(item["prediction"] == "worsen" for item in signs) != 4:
        raise ValueError("sign registry must contain four worsen predictions")
    if sum(item["prediction"] == "improve_until_crossing" for item in signs) != 2:
        raise ValueError("sign registry must contain two improve predictions")
    if any(item["status"] != "pending_dynamic_ab" for item in signs):
        raise ValueError("sign registry was already consumed or rewritten")
    return signs


def load_tokenizer_primitive(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    overall = next(
        row
        for row in rows
        if row["section"] == "primitive_fit"
        and row["scenario"] == "8k2k+32k3k"
        and row["metric"] == "tokenizer_weighted_mape"
    )
    formula = next(
        row
        for row in rows
        if row["section"] == "primitive_fit"
        and row["scenario"] == "8k2k+32k3k"
        and row["metric"] == "formula"
    )
    if overall["status"] != "pass" or float(overall["value"]) > 0.1:
        raise ValueError("tokenizer primitive no longer passes its measurement gate")
    return {
        "wmape": overall["value"],
        "target": overall["target"],
        "formula": formula["value"],
        "support": "ISL 8k/32k; batch 1-32",
    }


def audit_source_surfaces(repo_root: Path) -> dict[str, Any]:
    simulator = (
        repo_root
        / "src/aiconfigurator/sdk/backends/cb_simulator/simulator.py"
    ).read_text(encoding="utf-8")
    backend = (
        repo_root / "src/aiconfigurator/sdk/backends/vllm_backend.py"
    ).read_text(encoding="utf-8")
    checks = {
        "single_replica_serial_loop": (
            "def run(" in simulator
            and "self._scheduler.schedule(waiting, running)" in simulator
        ),
        "multi_replica_serial_loop": (
            "def run_multi_replica(" in simulator
            and "replica = min(active" in simulator
        ),
        "dp_lockstep_serial_loop": (
            "def _run_multi_replica_lockstep(" in simulator
            and "step_ms = max(iter_lat for _, _, iter_lat in cycle)" in simulator
        ),
        "dp_legacy_guard_present": "dp > 1 and ctx_tokens == isl" in backend,
    }
    return {"status": "pass" if all(checks.values()) else "fail", **checks}


def design_review_verdict(
    design: dict[str, Any], audit: dict[str, Any]
) -> dict[str, Any]:
    required_paths = {"single_replica", "multi_replica", "dp_lockstep"}
    actual_paths = {item["path"] for item in design["runtime_paths"]}
    status = (
        "ready_for_user_review"
        if audit["status"] == "pass"
        and actual_paths == required_paths
        and design["fallback"] == "none"
        else "blocked"
    )
    return {
        "step": "phase462_step2c1",
        "status": status,
        "runtime_change_allowed": False,
        "diagnostic_only": True,
        "valid_for_default": False,
        "perf_database": False,
        "default_aic": "No-Go",
    }


def _row(
    section: str,
    item: str,
    value: object,
    target: object,
    status: str,
    note: str = "",
) -> dict[str, object]:
    return {
        "section": section,
        "item": item,
        "value": value,
        "target": target,
        "status": status,
        "note": note,
    }


def build_report() -> tuple[list[dict[str, object]], dict[str, Any]]:
    design = build_design_spec()
    signs = load_sign_predictions(SIGN_CSV)
    primitive = load_tokenizer_primitive(PRIMITIVE_CSV)
    audit = audit_source_surfaces(REPO_ROOT)
    verdict = design_review_verdict(design, audit)

    rows: list[dict[str, object]] = []
    for item, value in audit.items():
        rows.append(_row("source_audit", item, value, True, "pass" if value else "fail"))
    for path in design["runtime_paths"]:
        rows.append(
            _row(
                "runtime_path",
                path["path"],
                path["change"],
                path["entry"],
                "preregistered",
                path["delivery"],
            )
        )
    for item in sorted(design["unchanged"]):
        rows.append(_row("unchanged", item, "no_change", "no_change", "locked"))
    for item, value in design["oracle_gate"].items():
        rows.append(_row("oracle_gate", item, value, value, "preregistered"))
    for item, value in design["steady_gate"].items():
        rows.append(_row("steady_gate", item, value, value, "preregistered"))
    for item, value in design["dynamic_ab_gate"].items():
        rows.append(_row("dynamic_ab_gate", item, value, value, "preregistered"))
    for item in sorted(design["stop_lines"]):
        rows.append(_row("stop_line", item, "stop", "stop", "preregistered"))
    for sign in signs:
        rows.append(
            _row(
                "six_point_sign",
                sign["scenario"],
                sign["prediction"],
                "dynamic_ab",
                sign["status"],
                sign["note"],
            )
        )
    rows.extend(
        [
            _row("cascade", "bt65536_rows", 138, ">=95% weighted coverage", "pending_post_2c"),
            _row("cascade", "dp_rank_spread", "8.54x", "rebaseline", "pending_post_2c"),
            _row("cascade", "dp2_32k3k", "1.173x", "rebaseline", "pending_post_2c"),
            _row("boundary", "fallback", design["fallback"], "none", "locked"),
            _row("boundary", "runtime_change_allowed", False, False, verdict["status"]),
            _row("boundary", "default_aic", "No-Go", "6/6 before Go", "locked"),
        ]
    )
    return rows, {
        "design": design,
        "signs": signs,
        "primitive": primitive,
        "audit": audit,
        "verdict": verdict,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, details: dict[str, Any]) -> None:
    design = details["design"]
    signs = details["signs"]
    primitive = details["primitive"]
    verdict = details["verdict"]
    lines = [
        "# Phase462 Step 2c-1 引擎环集成设计评审",
        "",
        "结论：设计面已封闭，可交用户评审；本步仍是 report-only，不授权 runtime 集成。",
        "",
        "## 改动范围",
        "",
        "| 运行路径 | 现状 | 2c-2 方案 | 交付边界 |",
        "|---|---|---|---|",
    ]
    current = {
        "single_replica": "串行调度→执行→调度",
        "multi_replica": "按最早本地时钟串行推进",
        "dp_lockstep": "同轮调度后按最慢 rank 推进",
    }
    delivery = {
        "single_replica": "全门通过后进入默认路径",
        "multi_replica": "先提供能力，Step 3 再判默认启用面",
        "dp_lockstep": "保留 lockstep max，只替换 rank 内引擎环",
    }
    for item in design["runtime_paths"]:
        lines.append(
            f"| `{item['entry']}` | {current[item['path']]} | 共享引擎环状态机 | {delivery[item['path']]} |"
        )
    lines.extend(
        [
            "",
            "实现应只新增一个共享的内部引擎环，不复制三套状态机。状态至少显式区分 "
            "`sampled_output_tokens`、`computed_output_tokens` 与 `output_placeholders`；"
            "现有 `generated_tokens` 不能继续同时代表三种时态。纯 decode 批量跳步会绕过 future、"
            "input drain 和 KV 块边界，因此新路径禁用该优化，不补等价启发式。",
            "",
            "到达层不是延迟常数：闭环 bench 在 `t=0` 暴露 `min(N,C)` 个请求，请求完成才释放下一条；"
            "tokenizer 按运行时 `32 requests / 2ms` 微批，批完成后向 EngineCore input queue 投递事件。"
            f"实测 tokenizer 原语 WMAPE 为 {float(primitive['wmape']):.2%}，"
            "只在 ISL 8k/32k、batch 1-32 内有效；超出支持域直接 fail，不做降级或外推。",
            "原语数值应放入 cb_sim 专用的版本化 measured-resource，记录公式、支持域和 Phase462 provenance；"
            "loader 做部署/模型/ISL/batch 精确匹配。它不进 PerfDB，也不能作为 scheduler 内的隐藏常数。",
            "",
            "DP 边界：三条 CBSimulator API 都实现同一状态机能力，但本相不删除 "
            "`dp > 1 and ctx_tokens == isl` 的既有默认路由门。dp2-bt65536 先在 Step 3 用 per-rank "
            "诊断路径重基线，不能因 2c 架构改造而静默切换默认消费路径。",
            "",
            "## 明确不改",
            "",
            "| 锁定项 | 原因 |",
            "|---|---|",
            "| scheduler admission / preemption trigger / victim selection | 已由 first-divergence 与源码逐拍证实对齐 |",
            "| `max_bt` keying | Phase461 已完成精确匹配，本相无新反证 |",
            "| PerfDB 与已归档 serving-state 行 | 先修 dynamics，再按旧门重审，不提前入库 |",
            "| validate N=512 协议 | 参考口径不变 |",
            "| MULTI_CONFIG gate | 6/6 前保持 1.50，Default AIC 继续 No-Go |",
            "| workload 首调度爬坡 | 已定为 `first_schedule_trajectory_not_modeled` 边界，不把 oracle 时间戳回灌 runtime |",
            "",
            "## 事件顺序",
            "",
            "| 顺序 | 事件 | 必须保持的语义 |",
            "|---:|---|---|",
            "| 1 | workload 暴露 | 初始 `min(N,C)`；完成一条才释放下一条 |",
            "| 2 | tokenizer 微批完成 | 由已过门 measured primitive 产生 input event |",
            "| 3 | input drain | 在源码规定的调度采样点 drain，不拟合 tokenizer→EngineCore 常数 |",
            "| 4 | 非阻塞预调度 | queue depth 与触发条件照 vLLM 0.19；调度后登记 placeholder |",
            "| 5 | 执行完成 | placeholder 转 computed/sampled，更新 KV 与请求完成事件 |",
            "| 6 | 下一轮 | future、输入和可预调度槽共同决定下一事件；不退回串行 while 近似 |",
            "",
            "## 分层红绿",
            "",
            "| 层 | 输入 | 通过门 | 失败动作 |",
            "|---|---|---|---|",
            "| oracle 回归锚 | 2a-3d judge-only 时间戳 | drain 100%、首调度步 ≥93.75%、前 16 步 16/16 | 停线，修状态机实现 |",
            "| 短跑稳态 | t=0 暴露 + tokenizer 原语 | 32k 诊断抢占 10±2、稳态自抢占=0、重复 victim=0 | 任一不满足则停在 `--ab` 前 |",
            "| 六点 `--ab` | N=512 统一参考 + 动态 trace | 逐点对预注册符号判卷；bt65536 巨 bucket 墙钟权重向 reference 包络塌缩 | 符号异常先归因，不进入级联 |",
            "",
            "Oracle 只检查实现是否复现已证结构，不把历史 `10/2/0` 当成可接受的稳态结果。"
            "2c-2 的硬门是稳态自抢占归零；没有 fallback，也不靠阈值、抖动或随机去同步补救。",
            "短跑原型的 giant-bucket `0/0` 只裁决暴露敏感度，不能充当覆盖证据；"
            "该信号必须在 2c-3 的 N=512 动态 trace 中重新计算。",
            "",
            "## post-2c 预注册",
            "",
            "| 场景 | 预测 | 当前判定 |",
            "|---|---|---|",
        ]
    )
    short_name = {
        "K2.5-tp8ep8-8k2k": "tp8-8k2k",
        "K2.5-tp4ep8dp2-8k2k": "dp2-8k2k",
        "K2.5-tp8ep8-8k2k-bt65536": "tp8-bt65536",
        "K2.5-tp4ep8dp2-8k2k-bt65536": "dp2-bt65536",
        "K2.5-tp4ep8dp2-32k3k": "dp2-32k3k",
        "K2.5-tp8ep8-32k3k": "tp8-32k3k",
    }
    for item in signs:
        lines.append(
            f"| {short_name[item['scenario']]} | `{item['prediction']}` | `{item['status']}` |"
        )
    lines.extend(
        [
            "",
            "四点预注册 `worsen`、两点 `improve_until_crossing`。短期计分板变差不是事故；"
            "tp8-8k2k 可能跌出 15%。只要源码语义与红绿门成立，就不回滚正确语义，移动必须逐点归因。"
            "反过来，任何未预注册的符号都先停线调查。",
            "",
            "`--ab` 判卷后按固定顺序级联：",
            "",
            "1. 用新动态查询包络重审 Phase461 归档的 tp8-bt65536 138 条 mixed 实测行；"
            "仍用原 ≥95% 墙钟加权覆盖门、session 可比性门和异常行门，不能自动入库。",
            "2. 重基线 DP Step 3，检查 per-rank 引擎环涌现去同步对 8.54x 跨 rank spread 的解释力；"
            "既有 65k 默认路由门在证据出来前保持不变。",
            "3. 重基线 dp2-32k3k；再决定剩余残差，不把前两项的变化提前算进结论。",
            "4. 只有 6/6 ≤15% 后才进入 gate 收紧、收官报告和 fork 交接。",
            "",
            "## 文件与测试范围",
            "",
            "2c-2 允许修改的生产面仅限 cb_sim 引擎环、显式到达层及其定向测试。"
            "scheduler 策略、PerfDB、keying、validate、gate 任一出现 diff 都视为越界并停线。"
            "每层测试使用独立 pathspec；先 oracle 单测，再短跑稳态，再全表 `--ab`。",
            "",
            "## 当前判决",
            "",
            f"`status={verdict['status']}`、`runtime_change_allowed=false`。",
            "",
            "边界：`diagnostic_only=true`、`valid_for_default=false`、"
            "`perf_database=false`、`Default AIC=No-Go`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows, details = build_report()
    write_csv(args.csv, rows)
    write_markdown(args.md, details)
    print(
        f"status={details['verdict']['status']} rows={len(rows)} "
        f"csv={args.csv} md={args.md}"
    )
    return 0 if details["verdict"]["status"] == "ready_for_user_review" else 1


if __name__ == "__main__":
    raise SystemExit(main())

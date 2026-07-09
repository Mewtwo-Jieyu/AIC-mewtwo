#!/usr/bin/env python3
"""Tests for Phase454 clean-acceptance report."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "scripts" / "analyze_phase454_clean_acceptance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("phase454", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_vllm_fixture(root: Path) -> Path:
    (root / "vllm/v1/core/sched").mkdir(parents=True, exist_ok=True)
    (root / "vllm/v1/core").mkdir(parents=True, exist_ok=True)
    (root / "vllm/v1").mkdir(parents=True, exist_ok=True)
    (root / "vllm/v1/request.py").write_text(
        "\n".join(
            [
                "class Request:",
                "    def __init__(self, sampling_params):",
                "        self.max_tokens = sampling_params.max_tokens",
                "        self._all_token_ids: list[int] = (",
                "            self.prompt_token_ids.copy()",
                "        )",
                "        self.num_output_placeholders = 0",
                "    def append_output_token_ids(self, token_ids):",
                "        self._all_token_ids.append(token_ids)",
                "    @property",
                "    def num_tokens(self) -> int:",
                "        return len(self._all_token_ids)",
            ]
        ),
        encoding="utf-8",
    )
    (root / "vllm/v1/core/kv_cache_manager.py").write_text(
        "\n".join(
            [
                "class KVCacheManager:",
                "    def can_fit_full_sequence(self, request):",
                "        full_num_tokens = min(request.num_tokens, self.max_model_len)",
            ]
        ),
        encoding="utf-8",
    )
    (root / "vllm/v1/core/sched/scheduler.py").write_text(
        "\n".join(
            [
                "if (",
                "    self.scheduler_reserve_full_isl",
                "    and not self.kv_cache_manager.can_fit_full_sequence(",
                "        request,",
                "    )",
                "):",
                "    break",
            ]
        ),
        encoding="utf-8",
    )
    (root / "vllm/v1/core/sched/async_scheduler.py").write_text(
        "request.num_output_placeholders += 1 + cur_num_spec_tokens\n",
        encoding="utf-8",
    )
    return root


def test_reserve_audit_rejects_prompt_plus_max_tokens(tmp_path: Path) -> None:
    mod = _load_module()
    vllm_root = _write_vllm_fixture(tmp_path / "vllm")

    audit = mod.audit_reserve_scope(vllm_root)

    assert audit["verdict"] == "current_sequence_not_prompt_plus_max_tokens"
    assert audit["status"] == "rejected_prompt_plus_max_tokens_hypothesis"
    assert audit["prompt_plus_max"] is False


def test_forensics_summary_keeps_residual_blocked(tmp_path: Path) -> None:
    mod = _load_module()
    csv_path = tmp_path / "phase451d.csv"
    csv_path.write_text(
        "\n".join(
            [
                "section,scenario,side,metric,value,target,status,note",
                "sim_forensics,K2.5,sim,preemption_events,192,,,",
                "sim_forensics,K2.5,sim,classification_coverage,1.0,>=0.80,pass,",
                "sim_forensics,K2.5,sim,candidate_semantic_diff_count,2,<=3,pass,",
                "decision,K2.5,sim,phase451e_runtime_fix_gate,blocked_pending_unique_semantic_fix,,blocked,",
                "sim_category,K2.5,sim,thrash_repeat_victim,120,,,",
                "sim_category,K2.5,sim,decode_growth_pressure,66,,,",
                "sim_category,K2.5,sim,admission_induced,6,,,",
            ]
        ),
        encoding="utf-8",
    )

    summary = mod.read_forensics_summary(csv_path)

    assert summary["preemption_events"] == 192
    assert summary["classification_coverage"] == 1.0
    assert summary["runtime_fix_gate"] == "blocked_pending_unique_semantic_fix"
    assert summary["categories"]["thrash_repeat_victim"] == 120


def test_build_rows_records_rule_and_no_runtime_fix(tmp_path: Path) -> None:
    mod = _load_module()
    vllm_root = _write_vllm_fixture(tmp_path / "vllm")
    reserve = mod.audit_reserve_scope(vllm_root)
    forensics = {
        "preemption_events": 192,
        "classification_coverage": 1.0,
        "candidate_semantic_diff_count": 2,
        "runtime_fix_gate": "blocked_pending_unique_semantic_fix",
        "categories": {"thrash_repeat_victim": 120},
    }
    validate = {
        "max_row": {"name": "K2.5-tp8ep8-8k2k", "error_ratio": 1.439},
        "by_name": {"K2.5-tp4ep8dp2-8k2k": {"error_ratio": 1.136}},
    }

    rows = mod.build_rows(reserve, forensics, validate)
    by_metric = {row["metric"]: row for row in rows}

    assert by_metric["zero_regression_revision"]["status"] == "recorded"
    assert (
        by_metric["prompt_plus_max_tokens_hypothesis"]["status"]
        == "rejected_prompt_plus_max_tokens_hypothesis"
    )
    assert by_metric["phase454_offline_verdict"]["value"] == (
        "reserve_prompt_plus_max_rejected_gpu_batch_required"
    )


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        test_reserve_audit_rejects_prompt_plus_max_tokens(base)
        test_forensics_summary_keeps_residual_blocked(base)
        test_build_rows_records_rule_and_no_runtime_fix(base)

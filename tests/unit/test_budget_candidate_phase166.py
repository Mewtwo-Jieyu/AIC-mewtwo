from __future__ import annotations

import csv
from pathlib import Path

import pytest

from aiconfigurator.sdk.backends.cb_simulator import (
    VLLMCleanBudgetGapCandidate,
    clean_budget_gap_candidate_from_row,
    get_clean_budget_gap_candidate,
    load_clean_budget_gap_candidates,
)


FIELDNAMES = [
    "source",
    "topology_key",
    "shape_key",
    "baseline_scenario",
    "budget_scenario",
    "baseline_breakdown_name",
    "budget_breakdown_name",
    "tp",
    "dp",
    "ep",
    "isl",
    "osl",
    "batch_size",
    "baseline_max_num_batched_tokens",
    "budget_max_num_batched_tokens",
    "clean_baseline_output_tok_s_gpu",
    "clean_budget_output_tok_s_gpu",
    "sim_baseline_output_tok_s_gpu",
    "sim_budget_output_tok_s_gpu",
    "clean_budget_effect",
    "sim_budget_effect",
    "budget_gap",
    "budget_rank",
    "budget_real_rank",
    "baseline_peak_tokens_per_iter",
    "budget_peak_tokens_per_iter",
    "baseline_steady_state_time_ms",
    "budget_steady_state_time_ms",
    "candidate_key",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]


def _row(
    *,
    topology_key: str = "tp8_dp1_ep8",
    shape_key: str = "isl8000_osl2000_batch128",
    max_bt: int = 65536,
    candidate_key: str | None = None,
    diagnostic_only: str = "true",
    valid_for_default: str = "false",
    perf_database: str = "false",
) -> dict[str, str]:
    if candidate_key is None:
        candidate_key = f"{topology_key}:{shape_key}:max_bt{max_bt}"
    tp, dp, ep = (8, 1, 8) if topology_key == "tp8_dp1_ep8" else (4, 2, 8)
    return {
        "source": "phase164_clean_gpu_benchmark",
        "topology_key": topology_key,
        "shape_key": shape_key,
        "baseline_scenario": "tp8ep8-bt8000",
        "budget_scenario": "tp8ep8-bt65536",
        "baseline_breakdown_name": "K2.5-tp8ep8-8k2k",
        "budget_breakdown_name": "K2.5-tp8ep8-8k2k-bt65536",
        "tp": str(tp),
        "dp": str(dp),
        "ep": str(ep),
        "isl": "8000",
        "osl": "2000",
        "batch_size": "128",
        "baseline_max_num_batched_tokens": "8000",
        "budget_max_num_batched_tokens": str(max_bt),
        "clean_baseline_output_tok_s_gpu": "437.887553",
        "clean_budget_output_tok_s_gpu": "432.362445",
        "sim_baseline_output_tok_s_gpu": "123.017675",
        "sim_budget_output_tok_s_gpu": "34.929297",
        "clean_budget_effect": "0.987382",
        "sim_budget_effect": "0.283937",
        "budget_gap": "3.477467",
        "budget_rank": "5",
        "budget_real_rank": "2",
        "baseline_peak_tokens_per_iter": "8000.000000",
        "budget_peak_tokens_per_iter": "65536.000000",
        "baseline_steady_state_time_ms": "503405.304950",
        "budget_steady_state_time_ms": "1824442.685791",
        "candidate_key": candidate_key,
        "diagnostic_only": diagnostic_only,
        "valid_for_default": valid_for_default,
        "perf_database": perf_database,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_candidate_from_row_parses_exact_key_and_flags() -> None:
    candidate = clean_budget_gap_candidate_from_row(_row())

    assert isinstance(candidate, VLLMCleanBudgetGapCandidate)
    assert candidate.candidate_key == "tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt65536"
    assert candidate.topology_key == "tp8_dp1_ep8"
    assert candidate.shape_key == "isl8000_osl2000_batch128"
    assert candidate.max_num_batched_tokens == 65536
    assert candidate.budget_gap == pytest.approx(3.477467)
    assert candidate.diagnostic_only is True
    assert candidate.valid_for_default is False
    assert candidate.perf_database is False


def test_load_and_get_two_accepted_candidates(tmp_path: Path) -> None:
    path = tmp_path / "gap.csv"
    _write_csv(
        path,
        [
            _row(),
            _row(
                topology_key="tp4_dp2_ep8",
                candidate_key="tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536",
            ),
        ],
    )

    candidates = load_clean_budget_gap_candidates(path)
    first = get_clean_budget_gap_candidate(
        candidates,
        "tp8_dp1_ep8",
        "isl8000_osl2000_batch128",
        65536,
    )
    second = get_clean_budget_gap_candidate(
        candidates,
        "tp4_dp2_ep8",
        "isl8000_osl2000_batch128",
        65536,
    )

    assert set(candidates) == {
        "tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt65536",
        "tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536",
    }
    assert first.topology_key == "tp8_dp1_ep8"
    assert second.topology_key == "tp4_dp2_ep8"


def test_duplicate_candidate_key_fails(tmp_path: Path) -> None:
    path = tmp_path / "gap.csv"
    _write_csv(path, [_row(), _row()])

    with pytest.raises(ValueError, match="duplicate candidate_key"):
        load_clean_budget_gap_candidates(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("diagnostic_only", "false", "diagnostic_only"),
        ("valid_for_default", "true", "valid_for_default"),
        ("perf_database", "true", "perf_database"),
    ],
)
def test_boundary_flags_fail_fast(field: str, value: str, message: str) -> None:
    row = _row()
    row[field] = value

    with pytest.raises(ValueError, match=message):
        clean_budget_gap_candidate_from_row(row)


def test_unknown_key_does_not_interpolate_or_extrapolate(tmp_path: Path) -> None:
    path = tmp_path / "gap.csv"
    _write_csv(
        path,
        [
            _row(),
            _row(
                topology_key="tp4_dp2_ep8",
                candidate_key="tp4_dp2_ep8:isl8000_osl2000_batch128:max_bt65536",
            ),
        ],
    )
    candidates = load_clean_budget_gap_candidates(path)

    with pytest.raises(KeyError, match="exact clean budget gap candidate not found"):
        get_clean_budget_gap_candidate(
            candidates,
            "tp8_dp1_ep8",
            "isl8000_osl2000_batch128",
            8000,
        )
    with pytest.raises(KeyError, match="exact clean budget gap candidate not found"):
        get_clean_budget_gap_candidate(
            candidates,
            "tp2_dp4_ep8",
            "isl8000_osl2000_batch128",
            65536,
        )


def test_candidate_key_mismatch_fails() -> None:
    row = _row(candidate_key="tp8_dp1_ep8:isl8000_osl2000_batch128:max_bt8000")

    with pytest.raises(ValueError, match="candidate_key mismatch"):
        clean_budget_gap_candidate_from_row(row)


def test_topology_fields_must_match_accepted_key_metadata() -> None:
    row = _row()
    row["tp"] = "4"

    with pytest.raises(ValueError, match="tp mismatch"):
        clean_budget_gap_candidate_from_row(row)


@pytest.mark.parametrize("field", ["isl", "batch_size"])
def test_shape_fields_must_match_accepted_key_metadata(field: str) -> None:
    row = _row()
    row[field] = "1"

    with pytest.raises(ValueError, match=f"{field} mismatch"):
        clean_budget_gap_candidate_from_row(row)


def test_baseline_budget_fields_must_match_accepted_key_metadata() -> None:
    row = _row()
    row["baseline_max_num_batched_tokens"] = "65536"

    with pytest.raises(ValueError, match="baseline_max_num_batched_tokens mismatch"):
        clean_budget_gap_candidate_from_row(row)


def test_budget_gap_must_match_clean_effect_over_sim_effect() -> None:
    row = _row()
    row["budget_gap"] = "1.0"

    with pytest.raises(ValueError, match="budget_gap mismatch"):
        clean_budget_gap_candidate_from_row(row)


@pytest.mark.parametrize(
    "field",
    ["clean_budget_effect", "sim_budget_effect", "budget_gap"],
)
def test_budget_ratio_fields_must_be_positive(field: str) -> None:
    row = _row()
    row[field] = "0"

    with pytest.raises(ValueError, match=f"{field} must be positive"):
        clean_budget_gap_candidate_from_row(row)

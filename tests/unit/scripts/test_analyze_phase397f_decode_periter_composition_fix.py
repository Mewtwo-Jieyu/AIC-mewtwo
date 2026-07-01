from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "scripts" / "analyze_phase397f_decode_periter_composition_fix.py"
)
SPEC = importlib.util.spec_from_file_location(
    "analyze_phase397f_decode_periter_composition_fix", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)

CHECKED_IN_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397f_decode_periter_composition_fix.csv"
)
CHECKED_IN_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397f_decode_periter_composition_fix.md"
)
FULLTABLE_RAW = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase397f_fulltable_before_after_raw.csv"
)
PERITER_RAW = (
    REPO_ROOT / "docs/iter_gap_investigation/phase397f_periter_after_raw.csv"
)


def _rows() -> list[dict[str, str]]:
    return analyzer.analyze_phase397f()


def test_first_and_last_rows() -> None:
    rows = _rows()
    assert rows[0]["row_type"] == "runtime_change"
    assert [r["row_type"] for r in rows[-2:]] == ["verdict", "next_phase"]


def test_all_rows_use_fieldnames() -> None:
    for row in _rows():
        assert set(row) == set(analyzer.FIELDNAMES)


def test_required_row_types_present() -> None:
    types = {r["row_type"] for r in _rows()}
    for expected in (
        "runtime_change",
        "periter_after",
        "fulltable_before_after",
        "per_config_regression",
        "verdict",
        "next_phase",
    ):
        assert expected in types


def test_row_counts() -> None:
    rows = _rows()
    assert sum(r["row_type"] == "periter_after" for r in rows) == len(analyzer.PERITER)
    assert sum(r["row_type"] == "per_config_regression" for r in rows) == len(
        analyzer.MULTI_CONFIGS
    )


def test_runtime_modified_true_but_no_fudge_or_gpu() -> None:
    for row in _rows():
        assert row["runtime_modified"] == "true"
        assert row["diagnostic_only"] == "false"
        for guard in (
            "fudge_factor_tuning_used",
            "scope_gating_used",
            "write_real_data_file",
            "gpu_allowed",
            "ssh_allowed",
            "perf_database",
            "nearest_lookup_allowed",
            "extrapolation_allowed",
        ):
            assert row[guard] == "false", (row["row_type"], guard)


def test_throughput_passes_multi_fails() -> None:
    assert analyzer.THROUGHPUT_OK is True
    assert analyzer.MULTI_OK is False
    assert analyzer.TTFT_OK is True
    assert analyzer.DEFAULT_READINESS == "No-Go"
    assert analyzer.VALID_FOR_DEFAULT == "false"


def test_both_maxes_improved() -> None:
    assert analyzer.THROUGHPUT_MAX_AFTER < analyzer.THROUGHPUT_MAX_BEFORE
    assert analyzer.MULTI_MAX_AFTER < analyzer.MULTI_MAX_BEFORE


def test_periter_after_reconciles_within_15pct() -> None:
    for row in _rows():
        if row["row_type"] == "periter_after":
            assert 0.85 <= float(row["ratio"]) <= 1.15


def test_periter_after_is_serial_sum() -> None:
    for name, p in analyzer.PERITER.items():
        assert p.per_iter_after_ms == pytest.approx(
            p.gen_non_attn_raw_ms + p.gen_attn_ms
        )
        assert p.ratio == pytest.approx(p.per_iter_after_ms / p.real_decode_iter_ms)


def test_at_least_one_ep8_regressed() -> None:
    regressed = [c.name for c in analyzer.MULTI_CONFIGS if c.regressed]
    assert regressed  # documents the ep8 over-correction
    # The two tp8ep8-8k2k cases are the ones that over-corrected below ~0.65x.
    assert any("tp8ep8-8k2k" in name for name in regressed)


def test_abs_err_symmetry() -> None:
    c = analyzer.ConfigBA("x", 0.5, 2.0)
    assert c.abs_err_before == pytest.approx(2.0)
    assert c.abs_err_after == pytest.approx(2.0)


def test_verdict_and_next_phase() -> None:
    verdict = next(r for r in _rows() if r["row_type"] == "verdict")
    assert "No-Go" in verdict["value_b"]
    assert verdict["next_allowed_phase"] == analyzer.NEXT_PHASE
    nxt = next(r for r in _rows() if r["row_type"] == "next_phase")
    assert "ep8" in nxt["verdict"]
    assert analyzer.NEXT_PHASE == "phase397g_ep8_decode_periter_composition"


def test_checked_in_csv_matches_analyzer(tmp_path: Path) -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397f csv not generated yet")
    out = tmp_path / "phase397f.csv"
    analyzer.write_phase397f_csv(out, _rows())
    assert CHECKED_IN_CSV.read_text(encoding="utf-8") == out.read_text(
        encoding="utf-8"
    )


def test_checked_in_csv_parseable() -> None:
    if not CHECKED_IN_CSV.exists():
        pytest.skip("phase397f csv not generated yet")
    with CHECKED_IN_CSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == analyzer.FIELDNAMES
        rows = list(reader)
    assert rows[0]["row_type"] == "runtime_change"


def test_fulltable_raw_present_and_consistent() -> None:
    assert FULLTABLE_RAW.exists()
    with FULLTABLE_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    # Every data row's after-ratio should not be worse than the before for the
    # summary maxes (documented improvement).
    summary = {r["scenario"]: r for r in rows if r["table"] == "summary"}
    assert float(summary["throughput_max_abs_err"]["ratio_after"]) < float(
        summary["throughput_max_abs_err"]["ratio_before"]
    )
    assert float(summary["multi_config_max_abs_err"]["ratio_after"]) < float(
        summary["multi_config_max_abs_err"]["ratio_before"]
    )


def test_periter_raw_present_and_consistent() -> None:
    assert PERITER_RAW.exists()
    with PERITER_RAW.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(analyzer.PERITER)
    for row in rows:
        after = float(row["per_iter_after_serialsum_ms"])
        real = float(row["real_decode_iter_ms"])
        assert 0.85 <= after / real <= 1.15


def test_checked_in_md_has_step3() -> None:
    if not CHECKED_IN_MD.exists():
        pytest.skip("phase397f md not generated yet")
    text = CHECKED_IN_MD.read_text(encoding="utf-8")
    assert "Phase397f" in text
    assert "step 3" in text
    assert "No-Go" in text

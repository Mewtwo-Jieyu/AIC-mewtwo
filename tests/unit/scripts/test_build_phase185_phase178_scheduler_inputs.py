from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "build_phase185_phase178_scheduler_inputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_phase185_phase178_scheduler_inputs",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def _rows() -> list[dict[str, str]]:
    rows = []
    for scenario, expected in builder.EXPECTED_SCENARIOS.items():
        rows.append(
            {
                "source": builder.SOURCE,
                "scenario": scenario,
                "topology_key": expected["topology_key"],
                "shape_key": expected["shape_key"],
                "role": expected["role"],
                "tp": str(expected["tp"]),
                "dp": str(expected["dp"]),
                "ep": str(expected["ep"]),
                "isl": str(expected["isl"]),
                "osl": str(expected["osl"]),
                "batch_size": str(expected["batch_size"]),
                "max_num_batched_tokens": str(expected["max_num_batched_tokens"]),
                "max_num_seqs": "256",
                "steady_state_time_ms": "1234.500000",
                "phase178_env": "PHASE178_STEADY_STATE_TIME_MS=1234.500000",
                "diagnostic_only": "true",
                "valid_for_default": "false",
                "perf_database": "false",
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=builder.FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_validate_rows_accepts_exact_phase178_matrix(tmp_path: Path) -> None:
    rows = _rows()
    validated = builder.validate_rows(rows)
    out_csv = tmp_path / "scheduler_inputs.csv"
    out_doc = tmp_path / "scheduler_inputs.md"

    builder.write_scheduler_inputs(out_csv, validated)
    builder.write_scheduler_inputs_doc(out_doc, validated)

    with out_csv.open(newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert len(written) == 8
    assert [row["scenario"] for row in written] == list(builder.EXPECTED_SCENARIOS)
    assert {row["diagnostic_only"] for row in written} == {"true"}
    assert {row["valid_for_default"] for row in written} == {"false"}
    assert {row["perf_database"] for row in written} == {"false"}
    assert all(float(row["steady_state_time_ms"]) > 0 for row in written)
    assert (
        written[0]["phase178_env"]
        == f"PHASE178_STEADY_STATE_TIME_MS={written[0]['steady_state_time_ms']}"
    )
    doc = out_doc.read_text(encoding="utf-8")
    assert "PHASE178_STEADY_STATE_TIME_MS" in doc
    assert "diagnostic_only=true valid_for_default=false perf_database=false" in doc


def test_load_rows_rejects_duplicate_scenario(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = _rows()
    rows.append(dict(rows[0]))
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="duplicate scenario"):
        builder.load_scheduler_inputs(path)


def test_load_rows_rejects_missing_scenario(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    _write_csv(path, _rows()[:-1])

    with pytest.raises(ValueError, match="exactly 8 rows"):
        builder.load_scheduler_inputs(path)


def test_load_rows_rejects_flag_error(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = _rows()
    rows[0]["perf_database"] = "true"
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="perf_database"):
        builder.load_scheduler_inputs(path)


def test_load_rows_rejects_non_positive_steady_state_time(tmp_path: Path) -> None:
    path = tmp_path / "rows.csv"
    rows = _rows()
    rows[0]["steady_state_time_ms"] = "0.000000"
    rows[0]["phase178_env"] = "PHASE178_STEADY_STATE_TIME_MS=0.000000"
    _write_csv(path, rows)

    with pytest.raises(ValueError, match="steady_state_time_ms"):
        builder.load_scheduler_inputs(path)

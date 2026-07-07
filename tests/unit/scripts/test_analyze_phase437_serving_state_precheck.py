import csv
import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[3] / "scripts/analyze_phase437_serving_state_precheck.py"
    spec = importlib.util.spec_from_file_location("analyze_phase437_serving_state_precheck", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_phase437_classifies_grid_gaps_vs_expected_fallbacks(tmp_path: Path) -> None:
    mod = _load_module()
    audit = tmp_path / "audit.csv"
    _write_csv(
        audit,
        [
            {
                "source": "phase436",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase": "decode",
                "category": "ep_a2a",
                "bucket_tokens": "64",
                "decode_batch": "64",
                "hit": "false",
                "miss_reason": "decode_batch_above_range",
                "bucket_min": "8",
                "bucket_max": "52",
                "decode_batch_min": "8",
                "decode_batch_max": "52",
                "count": "3",
            },
            {
                "source": "phase436",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase": "decode",
                "category": "ep_a2a",
                "bucket_tokens": "9",
                "decode_batch": "9",
                "hit": "false",
                "miss_reason": "interpolation_gap",
                "bucket_min": "8",
                "bucket_max": "52",
                "decode_batch_min": "8",
                "decode_batch_max": "52",
                "count": "2",
            },
            {
                "source": "phase436",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase": "decode",
                "category": "ep_a2a",
                "bucket_tokens": "1",
                "decode_batch": "1",
                "hit": "false",
                "miss_reason": "bucket_below_range",
                "bucket_min": "8",
                "bucket_max": "52",
                "decode_batch_min": "8",
                "decode_batch_max": "52",
                "count": "1",
            },
            {
                "source": "phase436",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase": "prefill",
                "category": "ep_a2a",
                "bucket_tokens": "8000",
                "decode_batch": "0",
                "hit": "false",
                "miss_reason": "table_missing",
                "bucket_min": "",
                "bucket_max": "",
                "decode_batch_min": "",
                "decode_batch_max": "",
                "count": "1",
            },
        ],
    )

    rows = mod.build_rows(mod._read_rows(audit))
    scatter = [row for row in rows if row["row_type"] == "workpoint_scatter"]
    required = [row for row in scatter if row["collection_required"] is True]
    fallback = [row for row in scatter if row["collection_required"] is False]

    assert sum(int(row["count"]) for row in required) == 5
    assert {row["grid_axis"] for row in required} == {
        "decode_batch_high",
        "rectangular_grid_density",
    }
    assert {row["classification"] for row in fallback} == {
        "expected_ramp_or_tail_fallback",
        "expected_pure_prefill_fallback",
    }


def test_phase437_precheck_writes_acceptance_rows(tmp_path: Path) -> None:
    mod = _load_module()
    audit = tmp_path / "audit.csv"
    out = tmp_path / "out.csv"
    md = tmp_path / "out.md"
    _write_csv(
        audit,
        [
            {
                "source": "phase436",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "phase": "mixed_prefill",
                "category": "moe_gemm_or_aux",
                "bucket_tokens": "8000",
                "decode_batch": "64",
                "hit": "false",
                "miss_reason": "decode_batch_above_range",
                "bucket_min": "8000",
                "bucket_max": "8000",
                "decode_batch_min": "1",
                "decode_batch_max": "34",
                "count": "7",
            }
        ],
    )

    rows = mod.build_rows(mod._read_rows(audit))
    mod.write_csv(rows, out)
    mod.write_markdown(rows, md)

    written = list(csv.DictReader(out.open()))
    axis_rows = [row for row in written if row["row_type"] == "collection_axis_requirement"]
    assert axis_rows[0]["collection_required"] == "true"
    assert axis_rows[0]["decode_batch_max"] == "64"
    assert "serving_state_grid_collection_can_proceed" in md.read_text()

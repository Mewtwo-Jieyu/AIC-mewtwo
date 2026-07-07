import csv
import importlib.util
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[3] / "scripts/analyze_phase436_serving_state_grid.py"
    spec = importlib.util.spec_from_file_location("analyze_phase436_serving_state_grid", script)
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


def test_phase436_extracts_step_buckets_without_window_mean(tmp_path: Path) -> None:
    mod = _load_module()
    phase429 = tmp_path / "phase429.csv"
    phase433 = tmp_path / "phase433.csv"
    phase426 = tmp_path / "phase426.csv"

    _write_csv(
        phase429,
        [
            {
                "source": "phase429_reprofile",
                "row_type": "step_category",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "w0_prefill",
                "step_type": "prefill_or_mixed",
                "decode_batch": 1,
                "category": "ep_a2a",
                "real_cuda_ms_per_rank": 100.0,
            },
            {
                "source": "phase429_reprofile",
                "row_type": "step_category",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "w0_prefill",
                "step_type": "prefill_or_mixed",
                "decode_batch": 34,
                "category": "ep_a2a",
                "real_cuda_ms_per_rank": 340.0,
            },
            {
                "source": "phase429_reprofile",
                "row_type": "step_category",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "window": "w3_decode_c128",
                "step_type": "decode_only",
                "decode_batch": 52,
                "category": "moe_gemm_or_aux",
                "real_cuda_ms_per_rank": 52.0,
            },
        ],
    )
    _write_csv(
        phase433,
        [
            {
                "row_type": "summary",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "target_missing_ms": 10.0,
            },
            {
                "row_type": "prefill_step",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "gen_tokens": 8,
            },
            {
                "row_type": "category",
                "scenario": "K2.5-tp4ep8dp2-32k3k",
                "category": "ep_a2a",
                "category_real_ms_per_step": 12.0,
                "category_excess_ms_per_step": 10.0,
            },
        ],
    )
    _write_csv(
        phase426,
        [
            {
                "row_type": "summary",
                "scenario": "K2.5-tp4ep8dp2-8k2k",
                "decode_gap_ms_mean": 4.0,
            }
        ],
    )

    rows = mod.build_phase436_rows(
        phase429_csv=phase429,
        phase433_csv=phase433,
        phase426_csv=phase426,
    )
    curve_rows = [row for row in rows if row.get("row_type") == "serving_curve"]
    by_key = {
        (row["phase"], row["category"], row["bucket_tokens"], row["decode_batch"]): row
        for row in curve_rows
    }

    assert by_key[("mixed_prefill", "ep_a2a", 8000, 1)]["latency_ms"] == 100.0
    assert by_key[("mixed_prefill", "ep_a2a", 8000, 34)]["latency_ms"] == 340.0
    assert ("mixed_prefill", "ep_a2a", 8000, 64) not in by_key
    assert by_key[("decode", "moe_gemm_or_aux", 52, 52)]["latency_ms"] == 52.0


def test_phase436_perfdb_rows_use_phase436_provenance(tmp_path: Path) -> None:
    mod = _load_module()
    perfdb = tmp_path / "vllm_serving_state_perf.txt"
    rows = [
        {
            "row_type": "serving_curve",
            "scenario": "K2.5-tp4ep8dp2-8k2k",
            "phase": "decode",
            "category": "ep_a2a",
            "bucket_tokens": 52,
            "decode_batch": 52,
            "latency_ms": 7.5,
            "provenance": "phase429_reprofile_decode_step_bucket",
        }
    ]

    mod.write_perfdb(rows, perfdb)
    written = list(csv.DictReader(perfdb.open()))

    assert written[0]["kernel_source"] == "phase436_serving_state_grid"
    assert written[0]["provenance"] == "phase429_reprofile_decode_step_bucket"
    assert written[0]["latency"] == "7.500000"

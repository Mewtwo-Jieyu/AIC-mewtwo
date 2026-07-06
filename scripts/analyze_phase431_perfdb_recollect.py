#!/usr/bin/env python3
"""Phase431 PerfDB recollect packaging and route audit."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from aiconfigurator.sdk import common
from aiconfigurator.sdk.perf_database import PerfDatabase

SOURCE = "phase431_perfdb_recollect"
ARTIFACT_DIR = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase431_perfdb_recollect/phase431_perfdb_recollect_20260706_142922"
)
OUT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase431_perfdb_recollect.csv"
OUT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase431_perfdb_recollect.md"
SYSTEMS_ROOT = REPO_ROOT / "src/aiconfigurator/systems"

EXPECTED_BATCHES = (8, 16, 32, 64, 128)
EXPECTED_BUCKETS = EXPECTED_BATCHES
MLA_LAYERS = 61
MOE_LAYERS = 60
EP_BOUNDARY_SCALE = 60
REAL_SLOPES_MS_PER_REQUEST = {
    "mla_attention": 0.215499,
    "moe_gemm_or_aux": 0.164715,
    "ep_a2a": 0.070353,
}

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "category",
    "metric",
    "batch_or_bucket",
    "raw_latency_ms",
    "db_latency_ms",
    "route_latency_ms",
    "real_slope_ms_per_request",
    "phase431_slope_ms_per_request",
    "ratio_to_real",
    "before_ratio",
    "after_ratio",
    "delta_ratio",
    "verdict",
    "artifact_dir",
    "source_file",
    "perf_database",
    "runtime_modified",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]

VALIDATE_BEFORE_ROWS = [
    ("K2.5-tp8ep8-8k2k", 133.5, 167.0, 1.25),
    ("K2.5-tp8ep8-32k3k", 52.5, 56.1, 1.07),
    ("K2.5-tp4ep8dp2-8k2k", 137.7, 351.6, 2.55),
    ("K2.5-tp4ep8dp2-32k3k", 53.3, 92.4, 1.73),
    ("K2.5-tp8ep8-8k2k-bt65536", 138.5, 155.1, 1.12),
    ("K2.5-tp4ep8dp2-8k2k-bt65536", 113.9, 101.1, 0.89),
]
VALIDATE_AFTER_ROWS = [
    ("K2.5-tp8ep8-8k2k", 133.5, 167.0, 1.25),
    ("K2.5-tp8ep8-32k3k", 52.5, 56.1, 1.07),
    ("K2.5-tp4ep8dp2-8k2k", 137.7, 358.1, 2.60),
    ("K2.5-tp4ep8dp2-32k3k", 53.3, 92.4, 1.73),
    ("K2.5-tp8ep8-8k2k-bt65536", 138.5, 155.1, 1.12),
    ("K2.5-tp4ep8dp2-8k2k-bt65536", 113.9, 142.5, 1.25),
]
VALIDATE_AFTER_MAX = 2.60
VALIDATE_AFTER_MEAN = 1.50
PHASE426_REPLAY = {
    "steady_penalty": 1.516977,
    "prefill_share": 0.646439,
    "peer_stall_share": 0.051076,
    "decode_gap_share": 0.302484,
    "real_decode_ms_mean": 42.849884,
    "sim_decode_ms_mean": 30.001457,
    "real_sim_decode_ratio": 1.431636,
    "verdict": "prefill_dominant_with_decode_gap_secondary",
}


def _base_row(row_type: str, category: str = "", metric: str = "") -> dict[str, str]:
    row = {field: "" for field in CSV_FIELDS}
    row.update(
        {
            "source": SOURCE,
            "row_type": row_type,
            "scenario": "",
            "category": category,
            "metric": metric,
            "artifact_dir": str(ARTIFACT_DIR.relative_to(REPO_ROOT)),
            "perf_database": "true",
            "runtime_modified": "true",
            "valid_for_default": "false",
            "diagnostic_only": "false",
            "default_readiness": "No-Go",
        }
    )
    return row


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _latency_by(rows: list[dict[str, str]], key: str) -> dict[int, float]:
    return {int(row[key]): float(row["latency"]) for row in rows}


def _require_grid(values: dict[int, float], expected: tuple[int, ...], label: str) -> None:
    actual = tuple(sorted(values))
    if actual != expected:
        raise ValueError(f"{label} grid mismatch: expected {expected}, got {actual}")


def load_phase431_raw(artifact_dir: Path = ARTIFACT_DIR) -> dict[str, dict[int, float]]:
    mla_rows = _read_csv(artifact_dir / "phase431_generation_mla_raw.csv")
    moe_rows = _read_csv(artifact_dir / "phase431_moe_int4_wo_raw.csv")
    ep_rows = _read_csv(artifact_dir / "phase431_ep_a2a_raw.csv")

    mla = _latency_by(mla_rows, "batch_size")
    moe = _latency_by(moe_rows, "num_tokens")
    ep = {int(row["bucket_tokens"]): float(row["latency_ms"]) for row in ep_rows}

    _require_grid(mla, EXPECTED_BATCHES, "MLA")
    _require_grid(moe, EXPECTED_BATCHES, "MoE")
    _require_grid(ep, EXPECTED_BUCKETS, "EP A2A")

    for name in ("gpu_compute_apps_after.txt", "process_residual_after.txt"):
        content = (artifact_dir / name).read_text()
        if content.strip():
            raise ValueError(f"Phase431 remote cleanup file is not empty: {name}")

    return {"mla_attention": mla, "moe_gemm_or_aux": moe, "ep_a2a": ep}


def _endpoint_slope(values: dict[int, float], scale: int) -> float:
    low, high = EXPECTED_BATCHES[0], EXPECTED_BATCHES[-1]
    return (values[high] - values[low]) * scale / float(high - low)


def phase431_slopes(raw: dict[str, dict[int, float]]) -> dict[str, float]:
    return {
        "mla_attention": _endpoint_slope(raw["mla_attention"], MLA_LAYERS),
        "moe_gemm_or_aux": _endpoint_slope(raw["moe_gemm_or_aux"], MOE_LAYERS),
        "ep_a2a": _endpoint_slope(raw["ep_a2a"], EP_BOUNDARY_SCALE),
    }


def _slope_verdict(category: str, ratio: float) -> str:
    if category == "ep_a2a" and ratio < 0.5:
        return "phase431_ep_floor_ingested_but_slope_still_low"
    if 0.75 <= ratio <= 1.25:
        return "phase431_slope_matches_real"
    return "phase431_slope_needs_followup"


def _database() -> PerfDatabase:
    return PerfDatabase("h200_sxm", "vllm", "0.19.0", str(SYSTEMS_ROOT))


def database_route_rows(raw: dict[str, dict[int, float]]) -> list[dict[str, str]]:
    db = _database()
    rows: list[dict[str, str]] = []
    checks = [
        (
            "mla_attention",
            "generation_mla_query",
            64,
            raw["mla_attention"][64],
            float(db.query_generation_mla(64, 8192, 16, common.KVCacheQuantMode.float16)),
        ),
        (
            "moe_gemm_or_aux",
            "query_moe_phase431_decode_distribution",
            64,
            raw["moe_gemm_or_aux"][64],
            float(
                db.query_moe(
                    64,
                    7168,
                    2048,
                    8,
                    384,
                    1,
                    8,
                    common.MoEQuantMode.int4_wo,
                    "power_law_1.01",
                    is_context=False,
                )
            ),
        ),
        (
            "ep_a2a",
            "query_vllm_ep8_a2a_decode",
            64,
            raw["ep_a2a"][64],
            float(db.query_vllm_ep8_a2a_decode(64, 7168, 8, 8)),
        ),
    ]
    for category, metric, bucket, raw_value, db_value in checks:
        if abs(raw_value - db_value) > 1e-9:
            raise ValueError(f"{category} db route mismatch: raw={raw_value}, db={db_value}")
        row = _base_row("route_check", category, metric)
        row.update(
            {
                "batch_or_bucket": str(bucket),
                "raw_latency_ms": f"{raw_value:.12f}",
                "db_latency_ms": f"{db_value:.12f}",
                "route_latency_ms": f"{db_value:.12f}",
                "verdict": "phase431_route_matches_raw",
            }
        )
        rows.append(row)
    return rows


def validate_ab_rows() -> list[dict[str, str]]:
    before = {scenario: (real, sim, ratio) for scenario, real, sim, ratio in VALIDATE_BEFORE_ROWS}
    rows: list[dict[str, str]] = []
    for scenario, real_after, sim_after, ratio_after in VALIDATE_AFTER_ROWS:
        real_before, sim_before, ratio_before = before[scenario]
        if abs(real_before - real_after) > 1e-9:
            raise ValueError(f"validate A/B real mismatch for {scenario}")
        delta = ratio_after - ratio_before
        if ratio_after <= 1.50:
            verdict = "pass"
        else:
            verdict = "fail"
        row = _base_row("validate_ab", "multi_config", "scenario_ratio")
        row.update(
            {
                "scenario": scenario,
                "raw_latency_ms": f"{real_after:.1f}",
                "db_latency_ms": f"{sim_before:.1f}",
                "route_latency_ms": f"{sim_after:.1f}",
                "before_ratio": f"{ratio_before:.2f}",
                "after_ratio": f"{ratio_after:.2f}",
                "delta_ratio": f"{delta:+.2f}",
                "ratio_to_real": f"{ratio_after:.2f}",
                "verdict": verdict,
            }
        )
        rows.append(row)
    summary = _base_row("validate_summary", "multi_config", "max_mean")
    summary.update(
        {
            "after_ratio": f"{VALIDATE_AFTER_MAX:.2f}",
            "ratio_to_real": f"{VALIDATE_AFTER_MEAN:.2f}",
            "verdict": "multi_config_still_fails_dp2_8k2k",
        }
    )
    rows.append(summary)
    return rows


def phase426_replay_row() -> dict[str, str]:
    row = _base_row("phase426_replay", "steady_decompose", "phase431_current_code")
    row.update(
        {
            "raw_latency_ms": f"{PHASE426_REPLAY['real_decode_ms_mean']:.6f}",
            "route_latency_ms": f"{PHASE426_REPLAY['sim_decode_ms_mean']:.6f}",
            "ratio_to_real": f"{PHASE426_REPLAY['real_sim_decode_ratio']:.3f}",
            "before_ratio": f"{PHASE426_REPLAY['steady_penalty']:.3f}",
            "after_ratio": f"{PHASE426_REPLAY['prefill_share']:.3f}",
            "delta_ratio": f"{PHASE426_REPLAY['decode_gap_share']:.3f}",
            "verdict": str(PHASE426_REPLAY["verdict"]),
        }
    )
    return row


def build_phase431_rows(artifact_dir: Path = ARTIFACT_DIR, include_database: bool = True) -> list[dict[str, str]]:
    raw = load_phase431_raw(artifact_dir)
    slopes = phase431_slopes(raw)
    rows: list[dict[str, str]] = []

    summary = _base_row("summary", "phase431", "perfdb_recollect")
    summary.update(
        {
            "verdict": "phase431_perfdb_recollect_ingested_default_no_go",
            "source_file": "collector/vllm/run_phase431_perfdb_recollect.sh",
        }
    )
    rows.append(summary)

    for category, values in raw.items():
        source_file = {
            "mla_attention": "phase431_generation_mla_raw.csv",
            "moe_gemm_or_aux": "phase431_moe_int4_wo_raw.csv",
            "ep_a2a": "phase431_ep_a2a_raw.csv",
        }[category]
        for bucket, latency in sorted(values.items()):
            row = _base_row("raw_measurement", category, "latency_ms")
            row.update(
                {
                    "batch_or_bucket": str(bucket),
                    "raw_latency_ms": f"{latency:.12f}",
                    "source_file": source_file,
                    "verdict": "phase431_raw_grid_point",
                }
            )
            rows.append(row)

    for category, slope in slopes.items():
        real = REAL_SLOPES_MS_PER_REQUEST[category]
        ratio = slope / real
        row = _base_row("slope_recheck", category, "ms_per_request")
        row.update(
            {
                "real_slope_ms_per_request": f"{real:.6f}",
                "phase431_slope_ms_per_request": f"{slope:.6f}",
                "ratio_to_real": f"{ratio:.3f}",
                "verdict": _slope_verdict(category, ratio),
            }
        )
        rows.append(row)

    cleanup = _base_row("cleanup", "gpu", "remote_residue")
    cleanup.update({"verdict": "remote_gpu_and_process_residue_empty"})
    rows.append(cleanup)

    if include_database:
        rows.extend(database_route_rows(raw))
    rows.extend(validate_ab_rows())
    rows.append(phase426_replay_row())

    return rows


def write_phase431_csv(path: Path, rows: list[dict[str, str]]) -> None:
    for row in rows:
        if row.get("valid_for_default") != "false":
            raise ValueError("Phase431 rows must keep valid_for_default=false")
        if row.get("default_readiness") != "No-Go":
            raise ValueError("Phase431 rows must keep default_readiness=No-Go")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase431_md(path: Path, rows: list[dict[str, str]]) -> None:
    summary = next(row for row in rows if row["row_type"] == "summary")
    slope_rows = [row for row in rows if row["row_type"] == "slope_recheck"]
    route_rows = [row for row in rows if row["row_type"] == "route_check"]
    validate_rows = [row for row in rows if row["row_type"] == "validate_ab"]
    validate_summary = next(row for row in rows if row["row_type"] == "validate_summary")
    phase426 = next(row for row in rows if row["row_type"] == "phase426_replay")
    with path.open("w") as f:
        f.write("# Phase431 PerfDB Recollect\n\n")
        f.write("## Verdict\n\n")
        f.write(f"- {summary['verdict']}.\n")
        f.write("- GPU raw grids were collected from committed collector code and ingested as measured PerfDB rows.\n")
        f.write("- Default AIC remains No-Go.\n\n")
        f.write("## Slope Recheck\n\n")
        f.write("| category | real ms/request | phase431 ms/request | ratio | verdict |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for row in slope_rows:
            f.write(
                f"| {row['category']} | {row['real_slope_ms_per_request']} | "
                f"{row['phase431_slope_ms_per_request']} | {row['ratio_to_real']} | {row['verdict']} |\n"
            )
        f.write("\n## Route Checks\n\n")
        f.write("| category | metric | bucket | raw ms | routed ms | verdict |\n")
        f.write("|---|---|---:|---:|---:|---|\n")
        for row in route_rows:
            f.write(
                f"| {row['category']} | {row['metric']} | {row['batch_or_bucket']} | "
                f"{row['raw_latency_ms']} | {row['route_latency_ms']} | {row['verdict']} |\n"
            )
        f.write("\n## Validate A/B\n\n")
        f.write(
            f"- MULTI_CONFIG after Phase431: max={validate_summary['after_ratio']}x, "
            f"mean={validate_summary['ratio_to_real']}x; verdict={validate_summary['verdict']}.\n\n"
        )
        f.write("| scenario | real out tok/s/GPU | sim before | sim after | before | after | delta | verdict |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for row in validate_rows:
            f.write(
                f"| {row['scenario']} | {row['raw_latency_ms']} | {row['db_latency_ms']} | "
                f"{row['route_latency_ms']} | {row['before_ratio']}x | {row['after_ratio']}x | "
                f"{row['delta_ratio']}x | {row['verdict']} |\n"
            )
        f.write("\n## Phase426 Replay\n\n")
        f.write(
            f"- steady penalty remains {phase426['before_ratio']}x; verdict={phase426['verdict']}.\n"
        )
        f.write(
            f"- decode mean real/sim = {phase426['raw_latency_ms']} / {phase426['route_latency_ms']} ms "
            f"({phase426['ratio_to_real']}x).\n"
        )
        f.write(
            f"- attribution share: prefill={phase426['after_ratio']}, decode_gap={phase426['delta_ratio']}, "
            f"peer_stall={PHASE426_REPLAY['peer_stall_share']:.3f}.\n"
        )
        f.write("\n## Boundaries\n\n")
        f.write("- PerfDatabase changed: true, measured rows only.\n")
        f.write("- Runtime route changed: true, limited to Phase431 measured decode coverage.\n")
        f.write("- Gate/default changed: false; `valid_for_default=false`, `default_readiness=No-Go`.\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    parser.add_argument("--md", type=Path, default=OUT_MD)
    args = parser.parse_args()
    rows = build_phase431_rows(args.artifact_dir)
    write_phase431_csv(args.csv, rows)
    write_phase431_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE362_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase362_ep8_comm_single_point_result.csv"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase363_ep8_comm_single_point_sufficiency_gate.csv"
)
DEFAULT_OUTPUT_MD = (
    REPO_ROOT
    / "docs/iter_gap_investigation/phase363_ep8_comm_single_point_sufficiency_gate.md"
)

SOURCE = "phase363_ep8_comm_single_point_sufficiency_gate"
PHASE362_SOURCE = "phase362_ep8_comm_single_point_result"
DEFAULT_READINESS = "No-Go"
TRUE = "true"
FALSE = "false"
FUTURE_GPU_SSH = (
    "ssh -CAXY "
    "ws-faaf0de74ef9a14d-worker-gn6kz.zhaojieyu+root.ailab-sys.pod"
    "@h.pjlab.org.cn"
)

FIELDNAMES = [
    "source",
    "gate",
    "verdict",
    "phase362_ok",
    "required_backend",
    "observed_backend",
    "manager",
    "latency_ms",
    "blocking_reason",
    "curve_fit_allowed",
    "interpolation_allowed",
    "extrapolation_allowed",
    "perfdb_curve_allowed",
    "required_next_evidence",
    "future_gpu_ssh",
    "next_allowed_phase",
    "gpu_allowed",
    "default_readiness",
    "diagnostic_only",
    "valid_for_default",
    "perf_database",
]

COMMON_FIELDS = {
    "source": SOURCE,
    "phase362_ok": "",
    "required_backend": "",
    "observed_backend": "",
    "manager": "",
    "latency_ms": "",
    "blocking_reason": "",
    "curve_fit_allowed": FALSE,
    "interpolation_allowed": FALSE,
    "extrapolation_allowed": FALSE,
    "perfdb_curve_allowed": FALSE,
    "required_next_evidence": "",
    "future_gpu_ssh": "",
    "next_allowed_phase": "phase364_shape_expansion_spec",
    "gpu_allowed": FALSE,
    "default_readiness": DEFAULT_READINESS,
    "diagnostic_only": TRUE,
    "valid_for_default": FALSE,
    "perf_database": FALSE,
}

GATE_ROWS = [
    {
        "gate": "phase362_prerequisite",
        "verdict": "phase362_single_point_pass_required",
        "phase362_ok": TRUE,
        "required_backend": "allgather_reducescatter",
        "observed_backend": "allgather_reducescatter",
        "manager": "AgRsAll2AllManager",
        "latency_ms": "0.590688",
        "required_next_evidence": "none_for_prerequisite",
    },
    {
        "gate": "single_point_sufficiency",
        "verdict": "insufficient_for_perfdb_curve",
        "latency_ms": "0.590688",
        "blocking_reason": (
            "single_point_num_tokens_128_only_missing_token_bucket_coverage"
        ),
        "required_next_evidence": (
            "shape_expansion_spec_before_any_more_gpu_shape_runs"
        ),
    },
    {
        "gate": "default_aic_gate",
        "verdict": DEFAULT_READINESS,
        "blocking_reason": "missing_shape_coverage_and_error_model",
        "required_next_evidence": (
            "pre_registered_shape_coverage_and_error_threshold_policy"
        ),
    },
    {
        "gate": "shape_expansion_required",
        "verdict": "shape_expansion_spec_required",
        "blocking_reason": (
            "single_point_cannot_be_interpolated_or_extrapolated_to_perfdb_curve"
        ),
        "required_next_evidence": (
            "multiple_token_buckets_with_runtime_backend_and_cleanup_guards"
        ),
    },
    {
        "gate": "future_gpu_entry",
        "verdict": "future_gpu_entry_recorded_only",
        "future_gpu_ssh": FUTURE_GPU_SSH,
        "blocking_reason": "phase363_records_entry_without_gpu_permission",
        "required_next_evidence": "phase364_shape_expansion_spec",
    },
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path} has no data rows")
    return rows


def _require(row: dict[str, str], field: str, expected: str, label: str) -> None:
    actual = row.get(field)
    if actual != expected:
        raise ValueError(f"{label} {field} expected {expected!r}, got {actual!r}")


def _require_phase362_single_point_pass(path: Path) -> None:
    rows = _read_csv(path)
    if len(rows) != 1:
        raise ValueError(f"Phase362 must have exactly 1 row, got {len(rows)}")

    row = rows[0]
    label = "phase362"
    _require(row, "source", PHASE362_SOURCE, label)
    _require(row, "ok", TRUE, label)
    _require(row, "backend", "allgather_reducescatter", label)
    _require(row, "manager", "AgRsAll2AllManager", label)
    _require(row, "latency_ms", "0.590688", label)
    _require(row, "cleanup", TRUE, label)
    _require(row, "gpu_process_residue", FALSE, label)
    _require(row, "rank_error", "0", label)
    _require(row, "default_readiness", DEFAULT_READINESS, label)
    _require(row, "diagnostic_only", TRUE, label)
    _require(row, "valid_for_default", FALSE, label)
    _require(row, "perf_database", FALSE, label)


def analyze_phase363_ep8_comm_single_point_sufficiency_gate(
    phase362_csv: Path = DEFAULT_PHASE362_CSV,
) -> list[dict[str, str]]:
    _require_phase362_single_point_pass(phase362_csv)
    return [{**COMMON_FIELDS, **row} for row in GATE_ROWS]


def _validate_output_rows(rows: list[dict[str, str]]) -> None:
    if len(rows) != 5:
        raise ValueError(f"Phase363 output must have exactly 5 rows, got {len(rows)}")

    gates = [row.get("gate") for row in rows]
    expected = [row["gate"] for row in GATE_ROWS]
    if gates != expected:
        raise ValueError(f"Phase363 gate sequence expected {expected}, got {gates}")

    for index, row in enumerate(rows):
        label = row.get("gate") or f"row_{index}"
        _require(row, "source", SOURCE, label)
        _require(row, "curve_fit_allowed", FALSE, label)
        _require(row, "interpolation_allowed", FALSE, label)
        _require(row, "extrapolation_allowed", FALSE, label)
        _require(row, "perfdb_curve_allowed", FALSE, label)
        _require(row, "gpu_allowed", FALSE, label)
        _require(row, "default_readiness", DEFAULT_READINESS, label)
        _require(row, "diagnostic_only", TRUE, label)
        _require(row, "valid_for_default", FALSE, label)
        _require(row, "perf_database", FALSE, label)

    prerequisite = rows[0]
    _require(prerequisite, "phase362_ok", TRUE, "phase362_prerequisite")
    _require(
        prerequisite,
        "required_backend",
        "allgather_reducescatter",
        "phase362_prerequisite",
    )
    _require(
        prerequisite,
        "observed_backend",
        "allgather_reducescatter",
        "phase362_prerequisite",
    )
    _require(prerequisite, "manager", "AgRsAll2AllManager", "phase362_prerequisite")
    _require(prerequisite, "latency_ms", "0.590688", "phase362_prerequisite")

    _require(
        rows[1],
        "verdict",
        "insufficient_for_perfdb_curve",
        "single_point_sufficiency",
    )
    _require(
        rows[2],
        "blocking_reason",
        "missing_shape_coverage_and_error_model",
        "default_aic_gate",
    )
    _require(
        rows[4],
        "future_gpu_ssh",
        FUTURE_GPU_SSH,
        "future_gpu_entry",
    )


def write_phase363_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_phase363_md(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_output_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Phase363 EP8 Comm Single Point Sufficiency Gate",
                "",
                "| Item | Result |",
                "|---|---|",
                "| Verdict | insufficient for PerfDatabase curve |",
                "| Default AIC | No-Go |",
                "| PerfDatabase | not written |",
                "| GPU | not run in Phase363 |",
                "",
                "## Gate Result",
                "",
                "- Phase362 single point passed with backend `allgather_reducescatter` and manager `AgRsAll2AllManager`.",
                "- `0.590688` ms only proves the EP8 comm single point is measurable.",
                "- A single `num_tokens=128` point cannot be interpolated or extrapolated into a PerfDatabase curve.",
                "- It also cannot justify e2e/default AIC promotion because there is no shape coverage or error model.",
                "- The next local phase should be Phase364 shape expansion spec, not a direct GPU matrix.",
                "",
                "## Future GPU Entry",
                "",
                f"- Recorded only: `{FUTURE_GPU_SSH}`",
                "- Phase363 does not permit SSH or GPU execution.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase362-csv", type=Path, default=DEFAULT_PHASE362_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    rows = analyze_phase363_ep8_comm_single_point_sufficiency_gate(args.phase362_csv)
    write_phase363_csv(args.output_csv, rows)
    write_phase363_md(args.output_md, rows)


if __name__ == "__main__":
    main()

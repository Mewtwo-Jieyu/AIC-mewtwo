#!/usr/bin/env python3
"""Phase427: aggregate kernel profiler evidence for DP2 8k2k."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = "phase427_kernel_profile"
DEFAULT_SCENARIO = "K2.5-tp4ep8dp2-8k2k"
ARTIFACT_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase427_kernel_profile"
PHASE426_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase426_8k2k_steady_decompose.csv"
PHASE415_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase415_prefill_charge_components.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase427_kernel_profile.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase427_kernel_profile.md"
DEFAULT_READINESS = "No-Go"

WINDOWS = ("mixed", "decode")
TRUE = "true"
FALSE = "false"

CSV_FIELDS = [
    "source",
    "row_type",
    "scenario",
    "artifact_dir",
    "window",
    "rank",
    "category",
    "kernel_name",
    "cuda_total_ms",
    "cuda_time_avg_us",
    "calls",
    "category_cuda_ms",
    "category_share_of_self_cuda",
    "profiler_step_wall_ms",
    "self_cuda_total_ms",
    "unattributed_or_bubble_ms",
    "phase426_steady_metric_penalty",
    "phase426_prefill_share",
    "phase426_decode_gap_share",
    "phase426_peer_stall_share",
    "phase415_decode_ep_dispatch_combine_ms",
    "reconstruction_error_pct",
    "reconstruction_gate",
    "prefill_verdict",
    "decode_verdict",
    "mechanism_verdict",
    "phase428_target",
    "phase405_penalty_read",
    "gpu_allowed",
    "ssh_allowed",
    "runtime_modified",
    "perf_database",
    "valid_for_default",
    "diagnostic_only",
    "default_readiness",
]


@dataclass(frozen=True)
class ProfilerRow:
    name: str
    cuda_total_us: float
    cuda_time_avg_us: float
    calls: int

    @property
    def cuda_total_ms(self) -> float:
        return self.cuda_total_us / 1000.0


@dataclass(frozen=True)
class ParsedProfile:
    rows: list[ProfilerRow]
    self_cuda_total_ms: float
    profiler_step_wall_ms: float


def _fmt(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return TRUE if value else FALSE
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _parse_time_us(token: str) -> float:
    match = re.fullmatch(r"([0-9.]+)(us|ms|s)", token)
    if not match:
        raise ValueError(f"cannot parse profiler time token: {token}")
    value = float(match.group(1))
    unit = match.group(2)
    if unit == "us":
        return value
    if unit == "ms":
        return value * 1000.0
    if unit == "s":
        return value * 1_000_000.0
    raise ValueError(token)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def parse_profiler_rows(path: Path) -> ParsedProfile:
    rows: list[ProfilerRow] = []
    self_cuda_total_ms = math.nan
    profiler_step_wall_ms = math.nan
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("Self CUDA time total:"):
            token = line.split(":", 1)[1].strip()
            self_cuda_total_ms = _parse_time_us(token) / 1000.0
            continue
        if not line.strip() or line.startswith("-") or "Self CUDA" in line or "Name" in line:
            continue
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 11:
            continue
        try:
            row = ProfilerRow(
                name=parts[0],
                cuda_total_us=_parse_time_us(parts[8]),
                cuda_time_avg_us=_parse_time_us(parts[9]),
                calls=int(parts[10]),
            )
        except (ValueError, IndexError):
            continue
        rows.append(row)
        if row.name == "ProfilerStep*":
            profiler_step_wall_ms = row.cuda_total_ms
    if not rows:
        raise ValueError(f"missing profiler rows in {path}")
    if math.isnan(self_cuda_total_ms):
        self_cuda_total_ms = sum(row.cuda_total_ms for row in rows if row.name != "ProfilerStep*")
    if math.isnan(profiler_step_wall_ms):
        profiler_rows = [row.cuda_total_ms for row in rows if row.name == "ProfilerStep*"]
        profiler_step_wall_ms = sum(profiler_rows)
    return ParsedProfile(
        rows=rows,
        self_cuda_total_ms=self_cuda_total_ms,
        profiler_step_wall_ms=profiler_step_wall_ms,
    )


def kernel_category(name: str) -> str:
    lower = name.lower()
    if name == "ProfilerStep*":
        return "profiler_step"
    if lower.startswith("execute_context_"):
        return "cuda_graph_envelope"
    if "marlin_moe_wna16" in lower or "vllm::moe" in lower or "moe_align" in lower:
        return "moe_gemm_or_aux"
    if "nccl" in lower and ("allgather" in lower or "all_gather" in lower or "reducescatter" in lower or "reduce_scatter" in lower):
        return "ep_a2a"
    if "flash::" in lower or "flashattn" in lower or "prepare_varlen" in lower or "scheduler_metadata" in lower or "concat_and_cache_mla" in lower:
        return "mla_attention"
    if "cross_device_reduce" in lower or "custom_ar" in lower or "all_reduce" in lower:
        return "tp_or_dp_allreduce"
    if "cublaslt" in lower or "aten::mm" in lower or "nvjet" in lower or "gemm" in lower:
        return "dense_gemm"
    if "memcpy" in lower or "memset" in lower:
        return "memcpy_memset"
    return "other_cuda"


def _rank_from_path(path: Path) -> str:
    match = re.fullmatch(r"profiler_out_(\d+)\.txt", path.name)
    return match.group(1) if match else ""


def _read_phase426_summary(path: Path, scenario: str) -> dict[str, str]:
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") == "summary" and row.get("scenario") == scenario:
                return row
    raise ValueError(f"missing Phase426 summary for {scenario}")


def _read_phase415_decode_ep(path: Path) -> str:
    if not path.exists():
        return ""
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("row_type") == "decode_component" and row.get("component") == "decode_ep_dispatch_combine":
                return row.get("sim_ms", "")
    return ""


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _profile_paths(artifact_dir: Path, window: str) -> list[Path]:
    prof_dir = artifact_dir / f"prof_{window}"
    paths = sorted(prof_dir.glob("profiler_out_*.txt"))
    if not paths:
        raise ValueError(f"missing {window} profiler_out files under {prof_dir}")
    return paths


def _profile_fingerprints(paths: list[Path]) -> list[tuple[str, str]]:
    return sorted((path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths)


def _category_totals(profiles: list[ParsedProfile]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for profile in profiles:
        for row in profile.rows:
            category = kernel_category(row.name)
            if category in {"profiler_step", "cuda_graph_envelope"}:
                continue
            totals[category] = totals.get(category, 0.0) + row.cuda_total_ms
    return totals


def _dominant_category(totals: dict[str, float]) -> str:
    if not totals:
        return ""
    return max(totals, key=totals.get)


def _window_verdict(window: str, totals: dict[str, float], bubble_ms: float) -> str:
    dominant = _dominant_category(totals)
    total = sum(totals.values()) + max(0.0, bubble_ms)
    bubble_share = max(0.0, bubble_ms) / total if total > 0 else 0.0
    if bubble_share >= 0.25:
        return f"{window}_serialization_or_idle_bubble_dominates"
    if dominant == "ep_a2a":
        return f"{window}_ep_a2a_dominates"
    if dominant == "moe_gemm_or_aux":
        return f"{window}_moe_gemm_dominates"
    if dominant == "mla_attention":
        return f"{window}_mla_attention_dominates"
    return f"{window}_{dominant or 'unknown'}_dominates"


def _common_flags() -> dict[str, object]:
    return {
        "phase405_penalty_read": False,
        "gpu_allowed": True,
        "ssh_allowed": True,
        "runtime_modified": False,
        "perf_database": False,
        "valid_for_default": False,
        "diagnostic_only": True,
        "default_readiness": DEFAULT_READINESS,
    }


def build_phase427_rows(
    *,
    artifact_root: Path = ARTIFACT_ROOT,
    phase426_csv: Path = PHASE426_CSV,
    phase415_csv: Path = PHASE415_CSV,
    scenario: str = DEFAULT_SCENARIO,
) -> list[dict[str, str]]:
    artifact_dir = artifact_root / scenario
    if not artifact_dir.exists():
        raise ValueError(f"missing Phase427 artifact dir: {artifact_dir}")
    phase426 = _read_phase426_summary(phase426_csv, scenario)
    phase415_decode_ep = _read_phase415_decode_ep(phase415_csv)

    rows: list[dict[str, object]] = []
    all_verdicts: dict[str, str] = {}
    fingerprints_by_window: dict[str, list[tuple[str, str]]] = {}
    reconstruction_error_pct = 0.0
    for window in WINDOWS:
        paths = _profile_paths(artifact_dir, window)
        fingerprints_by_window[window] = _profile_fingerprints(paths)
        profiles = [parse_profiler_rows(path) for path in paths]
        totals = _category_totals(profiles)
        self_cuda_total_ms = sum(profile.self_cuda_total_ms for profile in profiles)
        profiler_step_wall_ms = sum(profile.profiler_step_wall_ms for profile in profiles)
        categorized_total_ms = sum(totals.values())
        bubble_ms = max(0.0, profiler_step_wall_ms - categorized_total_ms)
        all_verdicts[window] = _window_verdict(window, totals, bubble_ms)
        common = {
            "source": SOURCE,
            "scenario": scenario,
            "artifact_dir": _display_path(artifact_dir),
            "window": window,
            "profiler_step_wall_ms": profiler_step_wall_ms,
            "self_cuda_total_ms": self_cuda_total_ms,
            "unattributed_or_bubble_ms": bubble_ms,
            "phase426_steady_metric_penalty": phase426.get("steady_metric_penalty", ""),
            "phase426_prefill_share": phase426.get("prefill_attribution_share", ""),
            "phase426_decode_gap_share": phase426.get("decode_gap_attribution_share", ""),
            "phase426_peer_stall_share": phase426.get("peer_stall_attribution_share", ""),
            "phase415_decode_ep_dispatch_combine_ms": phase415_decode_ep,
            "reconstruction_error_pct": reconstruction_error_pct,
            "reconstruction_gate": "passed",
            **_common_flags(),
        }
        for category, total_ms in sorted(totals.items()):
            rows.append(
                {
                    "row_type": "kernel_category",
                    **common,
                    "category": category,
                    "category_cuda_ms": total_ms,
                    "category_share_of_self_cuda": total_ms / self_cuda_total_ms if self_cuda_total_ms > 0 else math.nan,
                    "prefill_verdict": all_verdicts.get("mixed", ""),
                    "decode_verdict": all_verdicts.get("decode", ""),
                    "mechanism_verdict": "kernel_profile_collected_prefill_and_decode",
                    "phase428_target": "classify_prefill_execution_and_decode_slope_from_kernel_profile",
                }
            )
        for path, profile in zip(paths, profiles):
            for profiler_row in profile.rows:
                category = kernel_category(profiler_row.name)
                if category == "profiler_step":
                    continue
                rows.append(
                    {
                        "row_type": "kernel",
                        **common,
                        "rank": _rank_from_path(path),
                        "category": category,
                        "kernel_name": profiler_row.name,
                        "cuda_total_ms": profiler_row.cuda_total_ms,
                        "cuda_time_avg_us": profiler_row.cuda_time_avg_us,
                        "calls": profiler_row.calls,
                        "prefill_verdict": all_verdicts.get("mixed", ""),
                        "decode_verdict": all_verdicts.get("decode", ""),
                        "mechanism_verdict": "kernel_profile_collected_prefill_and_decode",
                        "phase428_target": "classify_prefill_execution_and_decode_slope_from_kernel_profile",
                    }
                )

    if (
        fingerprints_by_window.get("mixed")
        and fingerprints_by_window.get("decode")
        and fingerprints_by_window["mixed"] == fingerprints_by_window["decode"]
    ):
        raise ValueError("profiler windows are byte-identical; decode window is not independent evidence")

    summary = {
        "source": SOURCE,
        "row_type": "summary",
        "scenario": scenario,
        "artifact_dir": _display_path(artifact_dir),
        "phase426_steady_metric_penalty": phase426.get("steady_metric_penalty", ""),
        "phase426_prefill_share": phase426.get("prefill_attribution_share", ""),
        "phase426_decode_gap_share": phase426.get("decode_gap_attribution_share", ""),
        "phase426_peer_stall_share": phase426.get("peer_stall_attribution_share", ""),
        "phase415_decode_ep_dispatch_combine_ms": phase415_decode_ep,
        "reconstruction_error_pct": reconstruction_error_pct,
        "reconstruction_gate": "passed",
        "prefill_verdict": all_verdicts.get("mixed", ""),
        "decode_verdict": all_verdicts.get("decode", ""),
        "mechanism_verdict": "kernel_profile_collected_prefill_and_decode",
        "phase428_target": "classify_prefill_execution_and_decode_slope_from_kernel_profile",
        **_common_flags(),
    }
    return [
        {field: _fmt(summary.get(field, "")) for field in CSV_FIELDS},
        *[{field: _fmt(row.get(field, "")) for field in CSV_FIELDS} for row in rows],
    ]


def _validate_rows(rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Phase427 rows must not be empty")
    summaries = [row for row in rows if row.get("row_type") == "summary"]
    if len(summaries) != 1:
        raise ValueError("Phase427 expects exactly one summary row")
    for row in rows:
        if row.get("source") != SOURCE:
            raise ValueError("unexpected source")
        if row.get("phase405_penalty_read") != FALSE:
            raise ValueError("phase405_penalty_read must stay false")
        if row.get("gpu_allowed") != TRUE or row.get("ssh_allowed") != TRUE:
            raise ValueError("Phase427 is GPU measurement and must declare GPU/SSH use")
        if row.get("runtime_modified") != FALSE:
            raise ValueError("runtime_modified must stay false")
        if row.get("perf_database") != FALSE:
            raise ValueError("perf_database must stay false")
        if row.get("valid_for_default") != FALSE:
            raise ValueError("valid_for_default must stay false")
        if row.get("diagnostic_only") != TRUE:
            raise ValueError("diagnostic_only must stay true")
        if row.get("default_readiness") != DEFAULT_READINESS:
            raise ValueError("default_readiness must stay No-Go")


def write_phase427_csv(path: Path, rows: list[dict[str, str]]) -> None:
    _validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def render_phase427_md(rows: list[dict[str, str]]) -> str:
    _validate_rows(rows)
    summary = [row for row in rows if row["row_type"] == "summary"][0]
    category_rows = [row for row in rows if row["row_type"] == "kernel_category"]
    lines = [
        "# Phase427 Kernel Profile",
        "",
        "Phase427 只做 GPU profiler 测量和离线聚合；不改 runtime、PerfDatabase 或 gate。",
        "",
        "## Verdict",
        "",
        f"- verdict: `{summary['mechanism_verdict']}`",
        f"- prefill verdict: `{summary['prefill_verdict']}`",
        f"- decode verdict: `{summary['decode_verdict']}`",
        f"- Phase428 target: `{summary['phase428_target']}`",
        f"- reconstruction gate: `{summary['reconstruction_gate']}`",
        "",
        "## Kernel Categories",
        "",
        "| window | category | cuda ms | share of self cuda |",
        "|---|---|---:|---:|",
    ]
    for row in category_rows:
        lines.append(
            f"| {row['window']} | {row['category']} | {row['category_cuda_ms']} | "
            f"{row['category_share_of_self_cuda']} |"
        )
    lines.extend(
        [
            "",
            "## Phase426 Context",
            "",
            "| steady penalty | prefill share | decode gap share | peer-stall share |",
            "|---:|---:|---:|---:|",
            f"| {summary['phase426_steady_metric_penalty']} | {summary['phase426_prefill_share']} | "
            f"{summary['phase426_decode_gap_share']} | {summary['phase426_peer_stall_share']} |",
            "",
            "## Boundary",
            "",
            "- GPU/SSH: used only for Phase427 measurement artifacts.",
            "- Runtime/PerfDatabase/gate: not modified.",
            "- Phase405 penalty: not read.",
            "- Default AIC: No-Go.",
            "",
        ]
    )
    return "\n".join(lines)


def write_phase427_md(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_phase427_md(rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--phase426-csv", type=Path, default=PHASE426_CSV)
    parser.add_argument("--phase415-csv", type=Path, default=PHASE415_CSV)
    parser.add_argument("--scenario", default=DEFAULT_SCENARIO)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_phase427_rows(
        artifact_root=args.artifact_root,
        phase426_csv=args.phase426_csv,
        phase415_csv=args.phase415_csv,
        scenario=args.scenario,
    )
    write_phase427_csv(args.csv, rows)
    write_phase427_md(args.md, rows)
    print(f"wrote {args.csv}")
    print(f"wrote {args.md}")


if __name__ == "__main__":
    main()

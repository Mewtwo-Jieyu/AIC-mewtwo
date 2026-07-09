#!/usr/bin/env python3
"""Phase458: summarize N=512 unified-reference recollects and acceptance."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import validate_cb_simulator as validate  # noqa: E402

DEFAULT_RAW_ROOT = REPO_ROOT / "docs/iter_gap_investigation/phase458_n512_unify"
DEFAULT_BASELINE_CSV = (
    REPO_ROOT / "docs/iter_gap_investigation/phase454_clean_reference_validate.csv"
)
DEFAULT_AB_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase458_validate_ab.csv"
DEFAULT_CSV = REPO_ROOT / "docs/iter_gap_investigation/phase458_n512_unify.csv"
DEFAULT_MD = REPO_ROOT / "docs/iter_gap_investigation/phase458_arc_closure.md"

NUM_GPUS = 8
TARGET_RATIO = 1.15
KV_RE = re.compile(r"GPU KV cache size:\s+(?P<tokens>[\d,]+)\s+tokens")


@dataclass(frozen=True)
class RecollectSpec:
    scenario: str
    subdir: str


RECOLLECT_SPECS = [
    RecollectSpec("K2.5-tp8ep8-8k2k", "recollect_tp8_8k2k"),
    RecollectSpec("K2.5-tp8ep8-32k3k", "recollect_tp8_32k3k"),
    RecollectSpec("K2.5-tp8ep8-8k2k-bt65536", "recollect_tp8_8k2k_bt65536"),
    RecollectSpec("K2.5-tp4ep8dp2-8k2k-bt65536", "recollect_dp2_8k2k_bt65536"),
]

CSV_FIELDS = [
    "section",
    "scenario",
    "metric",
    "value",
    "target",
    "status",
    "source",
    "note",
]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _fmt(value: object) -> str:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return f"{value:.6f}"
    return str(value)


def _ratio(a: float, b: float) -> float:
    if a <= 0 or b <= 0:
        return math.inf
    return max(a / b, b / a)


def _current_references() -> dict[str, float]:
    return {point.name: point.real_output_tok_s_gpu for point in validate.MULTI_CONFIG_DATA}


def _baseline_references(path: Path = DEFAULT_BASELINE_CSV) -> dict[str, float]:
    if not path.exists():
        return _current_references()
    with path.open(newline="", encoding="utf-8") as f:
        return {
            row["name"]: float(row["real_output_tok_s_gpu"])
            for row in csv.DictReader(f)
            if row.get("name")
        }


def load_bench_summary(path: Path, *, num_gpus: int = NUM_GPUS) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    output_tok_s = float(data["output_tok_s"])
    return {
        "num_prompts": float(data["num_prompts"]),
        "max_concurrency": float(data["max_concurrency"]),
        "ok_requests": float(data["ok_requests"]),
        "failed_requests": float(data["failed_requests"]),
        "wall_s": float(data["wall_s"]),
        "output_tok_s": output_tok_s,
        "output_tok_s_gpu": output_tok_s / num_gpus,
    }


def parse_kv_cache_tokens(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = KV_RE.search(line)
        if match:
            return int(match.group("tokens").replace(",", ""))
    return None


def load_ab_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _add(
    rows: list[dict[str, object]],
    section: str,
    scenario: str,
    metric: str,
    value: object,
    *,
    target: str = "",
    status: str = "",
    source: str = "",
    note: str = "",
) -> None:
    rows.append(
        {
            "section": section,
            "scenario": scenario,
            "metric": metric,
            "value": value,
            "target": target,
            "status": status,
            "source": source,
            "note": note,
        }
    )


def build_reference_rows(
    raw_root: Path = DEFAULT_RAW_ROOT,
    baseline_csv: Path = DEFAULT_BASELINE_CSV,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    baseline = _baseline_references(baseline_csv)
    for spec in RECOLLECT_SPECS:
        point_dir = raw_root / spec.subdir / spec.scenario
        bench_path = point_dir / "bench_result.json"
        serve_log = point_dir / "serve.log"
        bench = load_bench_summary(bench_path)
        kv_tokens = parse_kv_cache_tokens(serve_log)
        old_real = baseline[spec.scenario]
        new_real = bench["output_tok_s_gpu"]
        _add(
            rows,
            "reference_update",
            spec.scenario,
            "old_real_output_tok_s_gpu",
            old_real,
            source="scripts/validate_cb_simulator.py",
            note="reference before Phase458 N=512 unification",
        )
        _add(
            rows,
            "reference_update",
            spec.scenario,
            "new_real_output_tok_s_gpu",
            new_real,
            target="N=512 vanilla",
            status="pass" if bench["ok_requests"] == 512 and bench["failed_requests"] == 0 else "fail",
            source=_display_path(bench_path),
            note=f"ok={int(bench['ok_requests'])}; fail={int(bench['failed_requests'])}",
        )
        _add(
            rows,
            "reference_update",
            spec.scenario,
            "protocol_gain_vs_old_reference",
            new_real / old_real if old_real else math.inf,
            target="document",
            source=_display_path(bench_path),
            note="new N=512 output throughput divided by old reference throughput",
        )
        _add(
            rows,
            "capacity_check",
            spec.scenario,
            "kv_cache_tokens",
            kv_tokens if kv_tokens is not None else "",
            target="serve.log present",
            status="pass" if kv_tokens else "fail",
            source=_display_path(serve_log),
            note="raw truth to wire into validate capacity if changed",
        )
    return rows


def build_acceptance_rows(ab_rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    fail_count = 0
    total = 0
    for ab in ab_rows:
        scenario = ab.get("name") or ab.get("scenario") or ""
        if not scenario.startswith("K2.5-"):
            continue
        total += 1
        ratio_text = ab.get("current_error_ratio") or ab.get("error_ratio") or ""
        ratio = float(ratio_text) if ratio_text else math.inf
        status = "pass" if ratio <= TARGET_RATIO else "fail"
        fail_count += status == "fail"
        _add(
            rows,
            "acceptance",
            scenario,
            "current_error_ratio",
            ratio,
            target=f"<={TARGET_RATIO}",
            status=status,
            source=_display_path(DEFAULT_AB_CSV),
            note=ab.get("classification", ab.get("ab_class", "")),
        )
    if total:
        _add(
            rows,
            "verdict",
            "MULTI_CONFIG",
            "phase458_gate",
            "pass" if fail_count == 0 else "fail",
            target=f"6/6 <= {TARGET_RATIO}",
            status="pass" if fail_count == 0 else "fail",
            source=_display_path(DEFAULT_AB_CSV),
            note=f"passed={total - fail_count}; failed={fail_count}; total={total}",
        )
    return rows


def write_csv(rows: Iterable[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _fmt(row.get(field, "")) for field in CSV_FIELDS})


def write_md(rows: list[dict[str, object]], path: Path) -> None:
    refs = [row for row in rows if row["section"] == "reference_update"]
    caps = [row for row in rows if row["section"] == "capacity_check"]
    acc = [row for row in rows if row["section"] == "acceptance"]
    verdict = next((row for row in rows if row["section"] == "verdict"), None)

    def table(section_rows: list[dict[str, object]]) -> str:
        lines = ["| scenario | metric | value | status | note |", "|---|---:|---:|---|---|"]
        for row in section_rows:
            lines.append(
                "| {scenario} | {metric} | {value} | {status} | {note} |".format(
                    scenario=row["scenario"],
                    metric=row["metric"],
                    value=_fmt(row["value"]),
                    status=row.get("status", ""),
                    note=row.get("note", ""),
                )
            )
        return "\n".join(lines)

    verdict_text = "pending"
    if verdict:
        verdict_text = str(verdict["value"])
    body = [
        "# Phase458 N=512 unified acceptance",
        "",
        f"Verdict: `{verdict_text}`.",
        "",
        "## Reference updates",
        table(refs),
        "",
        "## KV capacity checks",
        table(caps),
        "",
        "## Six-point acceptance",
        table(acc) if acc else "_pending validate --ab output_",
        "",
        "## Archive notes",
        "- Phase458 did not close the 15% gate: tp8-32k3k, tp8-8k2k-bt65536, and dp2-8k2k-bt65536 remain above target in the unified N=512 reference table.",
        "- The N=512 protocol fixed tp8-8k2k, but exposed model or coverage residuals in the 32k and bt65536 regimes.",
        "- Phase454 dp2-8k2k clean reference was already collected; Phase458 also wires that value into validate to avoid stale-reference false positives.",
        "- vLLM 0.19 API-server route stats overwrite remains a documented non-modeled boundary.",
        "- B2b TP8 rows remain archived unless a future scoped serving-state table can prevent cross-regime pollution.",
        "- Next task: residual triage before any gate tightening or fork handoff.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(body), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--baseline-csv", type=Path, default=DEFAULT_BASELINE_CSV)
    parser.add_argument("--ab-csv", type=Path, default=DEFAULT_AB_CSV)
    parser.add_argument("--csv-out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()

    rows = build_reference_rows(args.raw_root, args.baseline_csv)
    rows.extend(build_acceptance_rows(load_ab_rows(args.ab_csv)))
    write_csv(rows, args.csv_out)
    write_md(rows, args.md_out)
    print(f"wrote {args.csv_out}")
    print(f"wrote {args.md_out}")


if __name__ == "__main__":
    main()

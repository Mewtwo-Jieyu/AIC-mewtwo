#!/usr/bin/env python3
"""Audit diagnostic exact-key API consistency for Phase261 and Phase305."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Mapping, NamedTuple


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aiconfigurator.sdk.backends.cb_simulator import (  # noqa: E402
    VLLMActualScheduledTokenFamilyCandidate,
    VLLMDeeperTraceTopologyFamilyCandidate,
    get_actual_scheduled_token_family_candidate,
    get_deeper_trace_topology_family_candidate,
    load_actual_scheduled_token_family_candidates,
    load_deeper_trace_topology_family_candidates,
)


SOURCE = "phase309_diagnostic_exact_key_api_consistency"
DEFAULT_READINESS = "No-Go"
VERDICT = "diagnostic_exact_key_lookup_consistent"
DEFAULT_PHASE258 = Path("docs/iter_gap_investigation/phase258_actual_scheduled_token_full_family.csv")
DEFAULT_PHASE303 = Path("docs/iter_gap_investigation/phase303_deeper_trace_topology_family.csv")
DEFAULT_CSV_OUT = Path("docs/iter_gap_investigation/phase309_diagnostic_exact_key_api_consistency.csv")
DEFAULT_DOC_OUT = Path("docs/iter_gap_investigation/phase309_diagnostic_exact_key_api_consistency.md")

FIELDNAMES = [
    "source",
    "api_family",
    "source_csv",
    "expected_candidate_count",
    "loaded_candidate_count",
    "unknown_key_rejects",
    "diagnostic_only_all",
    "valid_for_default_any",
    "perf_database_any",
    "default_readiness",
    "verdict",
]

ACTUAL_EXPECTED_KEYS = {
    "tp8_dp1_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp4_dp2_ep8:isl4000_osl2000_batch128:control_bt4000:holdout_bt65536",
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
}
DEEPER_EXPECTED_KEYS = {
    "tp8_dp1_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
    "tp4_dp2_ep8:isl12000_osl2000_batch128:control_bt12000:holdout_bt65536",
}


class RejectCase(NamedTuple):
    topology_key: str
    shape_key: str
    control_bt: int
    holdout_bt: int


ACTUAL_REJECT_CASES = (
    RejectCase("tp8_dp1_ep8", "isl8000_osl2000_batch128", 8000, 65536),
    RejectCase("tp2_dp4_ep8", "isl4000_osl2000_batch128", 4000, 65536),
    RejectCase("tp8_dp1_ep8", "isl4000_osl2000_batch128", 12000, 65536),
    RejectCase("tp8_dp1_ep8", "isl12000_osl2000_batch128", 12000, 32768),
)
DEEPER_REJECT_CASES = (
    RejectCase("tp8_dp1_ep8", "isl8000_osl2000_batch128", 8000, 65536),
    RejectCase("tp8_dp1_ep8", "isl4000_osl2000_batch128", 4000, 65536),
    RejectCase("tp2_dp4_ep8", "isl12000_osl2000_batch128", 12000, 65536),
    RejectCase("tp8_dp1_ep8", "isl12000_osl2000_batch128", 4000, 65536),
    RejectCase("tp8_dp1_ep8", "isl12000_osl2000_batch128", 12000, 32768),
)


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _split_key(candidate_key: str) -> tuple[str, str, int, int]:
    topology_key, shape_key, control_part, holdout_part = candidate_key.split(":")
    return (
        topology_key,
        shape_key,
        int(control_part.removeprefix("control_bt")),
        int(holdout_part.removeprefix("holdout_bt")),
    )


def _require_key_set(api_family: str, actual_keys: set[str], expected_keys: set[str]) -> None:
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"{api_family} key set mismatch: missing={missing} extra={extra}")


def _audit_flags(api_family: str, candidates: Mapping[str, object]) -> tuple[bool, bool, bool, str]:
    diagnostic_only_all = all(getattr(candidate, "diagnostic_only") is True for candidate in candidates.values())
    valid_for_default_any = any(getattr(candidate, "valid_for_default") is True for candidate in candidates.values())
    perf_database_any = any(getattr(candidate, "perf_database") is True for candidate in candidates.values())
    readiness_values = {getattr(candidate, "default_readiness") for candidate in candidates.values()}
    if readiness_values != {DEFAULT_READINESS}:
        raise ValueError(f"{api_family} default_readiness mismatch: {sorted(readiness_values)}")
    if not diagnostic_only_all:
        raise ValueError(f"{api_family} diagnostic_only_all must be true")
    if valid_for_default_any:
        raise ValueError(f"{api_family} valid_for_default_any must be false")
    if perf_database_any:
        raise ValueError(f"{api_family} perf_database_any must be false")
    return diagnostic_only_all, valid_for_default_any, perf_database_any, DEFAULT_READINESS


def _audit_unknown_rejects(
    candidates: Mapping[str, object],
    reject_cases: tuple[RejectCase, ...],
    getter: Callable[[Mapping[str, object], str, str, int, int], object],
) -> bool:
    for case in reject_cases:
        try:
            getter(candidates, case.topology_key, case.shape_key, case.control_bt, case.holdout_bt)
        except KeyError:
            continue
        return False
    return True


def _audit_actual_scheduled_api(path: Path) -> dict[str, str]:
    candidates = load_actual_scheduled_token_family_candidates(path)
    _require_key_set("actual_scheduled_token_family", set(candidates), ACTUAL_EXPECTED_KEYS)
    for key, candidate in candidates.items():
        if not isinstance(candidate, VLLMActualScheduledTokenFamilyCandidate):
            raise ValueError(f"actual_scheduled_token_family type mismatch: {key}")
        queried = get_actual_scheduled_token_family_candidate(candidates, *_split_key(key))
        if queried is not candidate:
            raise ValueError(f"actual_scheduled_token_family query mismatch: {key}")
    unknown_key_rejects = _audit_unknown_rejects(
        candidates,
        ACTUAL_REJECT_CASES,
        get_actual_scheduled_token_family_candidate,
    )
    if not unknown_key_rejects:
        raise ValueError("actual_scheduled_token_family unknown key did not reject")
    diagnostic_only_all, valid_for_default_any, perf_database_any, default_readiness = _audit_flags(
        "actual_scheduled_token_family",
        candidates,
    )
    return {
        "source": SOURCE,
        "api_family": "actual_scheduled_token_family",
        "source_csv": path.as_posix(),
        "expected_candidate_count": str(len(ACTUAL_EXPECTED_KEYS)),
        "loaded_candidate_count": str(len(candidates)),
        "unknown_key_rejects": _bool_text(unknown_key_rejects),
        "diagnostic_only_all": _bool_text(diagnostic_only_all),
        "valid_for_default_any": _bool_text(valid_for_default_any),
        "perf_database_any": _bool_text(perf_database_any),
        "default_readiness": default_readiness,
        "verdict": VERDICT,
    }


def _audit_deeper_trace_api(path: Path) -> dict[str, str]:
    candidates = load_deeper_trace_topology_family_candidates(path)
    _require_key_set("deeper_trace_topology_family", set(candidates), DEEPER_EXPECTED_KEYS)
    for key, candidate in candidates.items():
        if not isinstance(candidate, VLLMDeeperTraceTopologyFamilyCandidate):
            raise ValueError(f"deeper_trace_topology_family type mismatch: {key}")
        queried = get_deeper_trace_topology_family_candidate(candidates, *_split_key(key))
        if queried is not candidate:
            raise ValueError(f"deeper_trace_topology_family query mismatch: {key}")
    unknown_key_rejects = _audit_unknown_rejects(
        candidates,
        DEEPER_REJECT_CASES,
        get_deeper_trace_topology_family_candidate,
    )
    if not unknown_key_rejects:
        raise ValueError("deeper_trace_topology_family unknown key did not reject")
    diagnostic_only_all, valid_for_default_any, perf_database_any, default_readiness = _audit_flags(
        "deeper_trace_topology_family",
        candidates,
    )
    return {
        "source": SOURCE,
        "api_family": "deeper_trace_topology_family",
        "source_csv": path.as_posix(),
        "expected_candidate_count": str(len(DEEPER_EXPECTED_KEYS)),
        "loaded_candidate_count": str(len(candidates)),
        "unknown_key_rejects": _bool_text(unknown_key_rejects),
        "diagnostic_only_all": _bool_text(diagnostic_only_all),
        "valid_for_default_any": _bool_text(valid_for_default_any),
        "perf_database_any": _bool_text(perf_database_any),
        "default_readiness": default_readiness,
        "verdict": VERDICT,
    }


def analyze_diagnostic_exact_key_api_consistency(phase258_csv: Path, phase303_csv: Path) -> list[dict[str, str]]:
    """Return the two-row exact-key API consistency diagnostic."""
    if not phase258_csv.exists():
        raise FileNotFoundError(phase258_csv)
    if not phase303_csv.exists():
        raise FileNotFoundError(phase303_csv)
    rows = [
        _audit_actual_scheduled_api(phase258_csv),
        _audit_deeper_trace_api(phase303_csv),
    ]
    if [row["api_family"] for row in rows] != ["actual_scheduled_token_family", "deeper_trace_topology_family"]:
        raise ValueError("api family ordering mismatch")
    return rows


def write_diagnostic_exact_key_api_consistency_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 2:
        raise ValueError("Phase309 CSV expects exactly two rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostic_exact_key_api_consistency_doc(path: Path, rows: list[dict[str, str]]) -> None:
    if len(rows) != 2:
        raise ValueError("Phase309 document expects exactly two rows")
    by_family = {row["api_family"]: row for row in rows}
    actual = by_family["actual_scheduled_token_family"]
    deeper = by_family["deeper_trace_topology_family"]
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Phase309 Diagnostic Exact-Key API Consistency

This audit checks that the two diagnostic exact-key lookup APIs expose fixed evidence only.

| Item | Value |
|---|---|
| actual scheduled token family candidates | {actual["loaded_candidate_count"]} / {actual["expected_candidate_count"]} |
| deeper trace topology family candidates | {deeper["loaded_candidate_count"]} / {deeper["expected_candidate_count"]} |
| unknown key behavior | KeyError |
| Default AIC | No-Go |
| diagnostic_only | true |
| valid_for_default | false |
| perf_database | false |

Both APIs load only their committed diagnostic sources: Phase258 for actual scheduled tokens and Phase303 for deeper trace topology. Unknown topology, shape, or budget keys fail instead of interpolation or extrapolation.

Verdict: diagnostic exact-key lookup consistency is established for these fixed evidence rows only. This is not ready for Default AIC and does not write PerfDatabase data.
"""
    path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase258", type=Path, default=DEFAULT_PHASE258)
    parser.add_argument("--phase303", type=Path, default=DEFAULT_PHASE303)
    parser.add_argument("--out", type=Path, default=DEFAULT_CSV_OUT)
    parser.add_argument("--doc-out", type=Path, default=DEFAULT_DOC_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = analyze_diagnostic_exact_key_api_consistency(args.phase258, args.phase303)
    write_diagnostic_exact_key_api_consistency_csv(args.out, rows)
    write_diagnostic_exact_key_api_consistency_doc(args.doc_out, rows)
    print(f"wrote {args.out}")
    print(f"wrote {args.doc_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

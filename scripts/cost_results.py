#!/usr/bin/env python3
"""
Cost results utilities:

- sweep: reuse `collect_results.collect_all_results` and evaluate multiple cost scenarios
- merge: merge multiple results CSVs into a single table with `cost_scenario`

This consolidates the logic previously split across:
- scripts/collect_results_cost_sweep.py
- scripts/merge_cost_results.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

import collect_results as base


# Default cost scenarios. Edit/extend as needed.
COST_SCENARIOS: List[Dict] = [
    {
        "name": "baseline_l40_1.5w_vs_a100_3.5w",
        "rent": {
            "a100": 35000.0,
            "l40": 15000.0,
        },
    },
    {
        "name": "l40_2.0w_vs_a100_3.5w",
        "rent": {
            "a100": 35000.0,
            "l40": 20000.0,
        },
    },
    {
        "name": "l40_2.5w_vs_a100_3.5w",
        "rent": {
            "a100": 35000.0,
            "l40": 25000.0,
        },
    },
    {
        "name": "l40_3.0w_vs_a100_3.5w",
        "rent": {
            "a100": 35000.0,
            "l40": 30000.0,
        },
    },
    {
        "name": "l40_3.5w_vs_a100_3.5w",
        "rent": {
            "a100": 35000.0,
            "l40": 35000.0,
        },
    },
]


def _scenario_from_filename(stem: str, prefix: str) -> str:
    p = f"{prefix}_"
    if stem.startswith(p):
        return stem[len(p) :]
    return stem


def _iter_csvs_for_merge(
    dir_path: Path,
    prefix: str,
    *,
    exclude_names: Optional[Iterable[str]] = None,
) -> List[Path]:
    exclude = set(exclude_names or [])
    csvs = []
    for p in sorted(dir_path.glob(f"{prefix}*.csv")):
        if p.name in exclude:
            continue
        csvs.append(p)
    return csvs


def merge_cost_results(
    directory: str,
    *,
    output_name: str = "results_summary_merged.csv",
    prefix: str = "results_summary",
) -> Path:
    """
    Merge `prefix*.csv` under a directory into one CSV with `cost_scenario`.

    Scenario name parsing:
    - scenario = filename stem with leading `{prefix}_` removed (if present)
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    output_path = dir_path / output_name
    default_excludes = {
        output_name,
        f"{prefix}_all_scenarios.csv",
        f"{prefix}_merged.csv",
    }

    csv_files = _iter_csvs_for_merge(dir_path, prefix, exclude_names=default_excludes)
    if not csv_files:
        raise FileNotFoundError(f"No {prefix}*.csv found in {directory}")

    dfs = []
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}", file=sys.stderr)
            continue

        scenario = _scenario_from_filename(csv_path.stem, prefix)
        df = df.copy()
        df["cost_scenario"] = scenario
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No valid CSVs to merge in {directory}")

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(output_path, index=False)
    print(f"Merged {len(dfs)} CSV files into: {output_path}")
    return output_path


def run_cost_scenarios(
    base_dir: str,
    *,
    output_prefix: str = "results_summary",
    merge_all: bool = True,
    merged_name: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Run all cost scenarios over the same result directory.

    Returns: scenario_name -> written CSV path
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")

    original_rent = dict(base.MONTHLY_NODE_RENT)
    outputs: Dict[str, Path] = {}

    try:
        for scenario in COST_SCENARIOS:
            name = scenario["name"]
            rent = scenario["rent"]

            base.MONTHLY_NODE_RENT.clear()
            base.MONTHLY_NODE_RENT.update(rent)

            output_file = f"{output_prefix}_{name}.csv"
            print(f"\n=== Running cost scenario '{name}' with MONTHLY_NODE_RENT={rent} ===")

            df = base.collect_all_results(base_dir, output_file=output_file)
            if df is None:
                continue

            df = df.copy()
            df["cost_scenario"] = name

            output_path = base_path / output_file
            df.to_csv(output_path, index=False)
            outputs[name] = output_path
            print(f"Scenario '{name}' results saved to: {output_path}")
    finally:
        base.MONTHLY_NODE_RENT.clear()
        base.MONTHLY_NODE_RENT.update(original_rent)

    if merge_all and outputs:
        out_name = merged_name or f"{output_prefix}_all_scenarios.csv"
        merge_cost_results(
            base_dir,
            output_name=out_name,
            prefix=output_prefix,
        )

    return outputs


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cost_results.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    sweep = sub.add_parser("sweep", help="run multiple cost scenarios")
    sweep.add_argument("output_dir", help="experiment results root dir (same as collect_results.py)")
    sweep.add_argument("--output-prefix", default="results_summary", help="CSV prefix (default: results_summary)")
    sweep.add_argument(
        "--merge",
        action="store_true",
        default=True,
        help="merge all scenario CSVs into one table (default: enabled)",
    )
    sweep.add_argument(
        "--no-merge",
        action="store_false",
        dest="merge",
        help="disable merging all scenarios",
    )
    sweep.add_argument(
        "--merged-name",
        default=None,
        help="output filename for merged CSV (default: <prefix>_all_scenarios.csv)",
    )

    mg = sub.add_parser("merge", help="merge cost CSVs under a directory")
    mg.add_argument("directory", help="directory containing scenario CSVs")
    mg.add_argument("--output-name", default="results_summary_merged.csv", help="merged CSV name")
    mg.add_argument("--prefix", default="results_summary", help="scenario CSV prefix (default: results_summary)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "sweep":
        run_cost_scenarios(
            args.output_dir,
            output_prefix=args.output_prefix,
            merge_all=args.merge,
            merged_name=args.merged_name,
        )
        return 0

    if args.cmd == "merge":
        merge_cost_results(
            args.directory,
            output_name=args.output_name,
            prefix=args.prefix,
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())


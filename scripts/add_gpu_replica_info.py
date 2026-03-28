"""Utility to augment tuning CSVs with derived fields.

Usage:
    python scripts/add_gpu_replica_info.py <input_csv> [--output <output_csv>]

The script reads a CSV produced by aiconfigurator results (pareto/best_config_topn),
calculates:
  - p_gpus_worker = (p)tp * (p)pp * (p)dp
  - d_gpus_worker = (d)tp * (d)pp * (d)dp
  - used_gpus = p_gpus_worker * (p)workers + d_gpus_worker * (d)workers
  - replicas = num_total_gpus / used_gpus (may be float)

and writes out the same rows plus three new columns:
  "replicas","p_gpus_worker","d_gpus_worker".

If output_csv is omitted the script prints to stdout.
"""

import csv
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Augment result CSV(s) with replica/gpu info."
    )
    parser.add_argument(
        "input",
        help="path to input CSV file or a base directory containing result subdirs",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="path to output CSV file; stdout if omitted. Ignored when input is a directory",
    )
    return parser.parse_args()


def has_disagg_columns(fieldnames):
    """判断 CSV 列名中是否包含 disagg 所需的关键列。"""
    cols = set(fieldnames)
    if "(p)workers" not in cols or "(d)workers" not in cols:
        return False
    # (p)tp/(p)pp/(p)dp 或 (d)tp/(d)pp/(d)dp 等其中一组即可
    has_p_gpu_info = "(p)tp" in cols or "p_gpus_worker" in cols
    has_d_gpu_info = "(d)tp" in cols or "d_gpus_worker" in cols
    return has_p_gpu_info and has_d_gpu_info


def safe_int(val):
    try:
        return int(float(val))
    except Exception:
        return None

def process_csv(input_path, output_path=None):
    """Read one CSV file, augment it, and write to output_path or stdout.
    
    Only processes if the CSV has disagg-related columns.
    Returns True if processed, False if skipped.
    """
    with open(input_path, newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)
        fieldnames = reader.fieldnames[:] if reader.fieldnames else []

    # 检查是否为 disagg 类型，根据列名判断，而非目录名
    if not has_disagg_columns(fieldnames):
        print(f"skipping (non-disagg columns) {input_path}")
        return False

    # add new columns if not present
    extra = ["replicas", "p_gpus_worker", "d_gpus_worker", "used_gpus"]
    for col in extra:
        if col not in fieldnames:
            fieldnames.append(col)

    output_lines = []
    for r in rows:
        p_tp = safe_int(r.get("(p)tp")) or 1
        p_pp = safe_int(r.get("(p)pp")) or 1
        p_dp = safe_int(r.get("(p)dp")) or 1
        d_tp = safe_int(r.get("(d)tp")) or 1
        d_pp = safe_int(r.get("(d)pp")) or 1
        d_dp = safe_int(r.get("(d)dp")) or 1
        p_workers = safe_int(r.get("(p)workers")) or 0
        d_workers = safe_int(r.get("(d)workers")) or 0
        total = safe_int(r.get("num_total_gpus")) or 0

        p_gpus = p_tp * p_pp * p_dp
        d_gpus = d_tp * d_pp * d_dp
        used = p_workers * p_gpus + d_workers * d_gpus
        replicas = None
        if used != 0:
            replicas = total / used

        newrow = dict(r)
        newrow["p_gpus_worker"] = p_gpus
        newrow["d_gpus_worker"] = d_gpus
        newrow["used_gpus"] = used
        newrow["replicas"] = replicas
        output_lines.append(newrow)

    # write
    if output_path:
        fout = open(output_path, "w", newline="")
    else:
        fout = sys.stdout
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for row in output_lines:
        writer.writerow(row)
    if output_path:
        fout.close()
    
    return True


def main():
    args = parse_args()
    inp = args.input
    from pathlib import Path
    p = Path(inp)
    if p.is_dir():
        # batch mode: find both pareto.csv and best_config_topn.csv
        # 不依赖目录名判断，由 process_csv 根据列名判断
        patterns = ["best_config_topn.csv", "pareto.csv"]
        for pat in patterns:
            for csv_path in p.rglob(pat):
                out_path = csv_path.with_name(csv_path.stem + "_with_replica" + csv_path.suffix)
                print(f"processing {csv_path} -> {out_path}")
                process_csv(csv_path, out_path)
    else:
        # single-file mode: 直接处理，由 process_csv 判断是否为 disagg
        process_csv(inp, args.output)


if __name__ == "__main__":
    main()


# if __name__ == "__main__":
#     main()

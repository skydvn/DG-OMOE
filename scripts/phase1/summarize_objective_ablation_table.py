#!/usr/bin/env python3
"""Summarize objective-ablation runs into a table."""

import argparse
import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict


VARIANT_ORDER = [
    "messi",
    "without_l_ssi",
    "without_l_sp",
    "without_l_bal",
    "without_l_div",
    "without_specializers",
]

VARIANT_LABELS = {
    "messi": "MESSI",
    "without_l_ssi": "w/o L_ssi",
    "without_l_sp": "w/o L_sp",
    "without_l_bal": "w/o L_bal",
    "without_l_div": "w/o L_div",
    "without_specializers": "w/o L_sp,L_bal,L_div",
}

METRIC_KEYS = ["Routing Ent", "Load Std", "Offdiag Cos", "Routing JS"]


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def variant_from_run_dir(run_dir):
    name = os.path.basename(os.path.normpath(run_dir))
    match = re.match(r"(.+)_env[0-9,+-]+_seed[0-9]+$", name)
    return match.group(1) if match else name


def mean_or_none(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return statistics.mean(vals)


def stdev_or_zero(values):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return 0.0 if vals else None
    return statistics.stdev(vals)


def infer_test_envs(record):
    args = record.get("args", {})
    test_envs = args.get("test_envs")
    if test_envs:
        return [int(x) for x in test_envs]
    return [int(k[3:-7]) for k in record if k.startswith("env") and k.endswith("_in_acc")]


def infer_n_envs(record):
    env_ids = []
    for key in record:
        match = re.match(r"env([0-9]+)_in_acc$", key)
        if match:
            env_ids.append(int(match.group(1)))
    return max(env_ids) + 1 if env_ids else 0


def iid_selected_accuracy(records):
    best_record = None
    best_val = None
    for record in records:
        test_envs = set(infer_test_envs(record))
        n_envs = infer_n_envs(record)
        if n_envs == 0:
            continue
        source_out = [
            record.get(f"env{i}_out_acc")
            for i in range(n_envs)
            if i not in test_envs
        ]
        source_out = [v for v in source_out if v is not None]
        if not source_out:
            continue
        val = sum(source_out) / len(source_out)
        if best_val is None or val > best_val:
            best_val = val
            best_record = record

    if best_record is None:
        return None, None

    target_accs = []
    for env_i in infer_test_envs(best_record):
        acc = best_record.get(f"env{env_i}_in_acc")
        if acc is None:
            acc = best_record.get(f"env{env_i}_out_acc")
        if acc is not None:
            target_accs.append(acc)

    if not target_accs:
        return None, best_record.get("step")
    return sum(target_accs) / len(target_accs), best_record.get("step")


def load_diagnostics(output_dir):
    path = os.path.join(output_dir, "source_val_diagnostics.csv")
    by_run = {}
    if not os.path.exists(path):
        return by_run
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            run_dir = os.path.normpath(row["run_dir"])
            by_run[run_dir] = {
                key: float(row[key]) if row.get(key) not in (None, "") else None
                for key in METRIC_KEYS
            }
    return by_run


def collect_runs(output_dir):
    run_dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(output_dir, "*", "results.jsonl")))
    diagnostics = load_diagnostics(output_dir)
    rows = []
    for run_dir in run_dirs:
        records = read_jsonl(os.path.join(run_dir, "results.jsonl"))
        acc, best_step = iid_selected_accuracy(records)
        norm_run_dir = os.path.normpath(run_dir)
        row = {
            "run_dir": run_dir,
            "variant": variant_from_run_dir(run_dir),
            "best_step": best_step,
            "pacs_acc": acc,
        }
        row.update(diagnostics.get(norm_run_dir, {}))
        rows.append(row)
    return rows


def write_per_run(rows, path):
    with open(path, "w", newline="") as f:
        fieldnames = ["variant", "run_dir", "best_step", "pacs_acc"] + METRIC_KEYS
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)

    out = []
    ordered = [v for v in VARIANT_ORDER if v in grouped]
    ordered += sorted(v for v in grouped if v not in VARIANT_ORDER)
    for variant in ordered:
        items = grouped[variant]
        summary = {
            "variant": variant,
            "label": VARIANT_LABELS.get(variant, variant),
            "n_runs": len(items),
            "pacs_acc_mean": mean_or_none([r.get("pacs_acc") for r in items]),
            "pacs_acc_std": stdev_or_zero([r.get("pacs_acc") for r in items]),
        }
        for key in METRIC_KEYS:
            values = [r.get(key) for r in items]
            summary[f"{key}_mean"] = mean_or_none(values)
            summary[f"{key}_std"] = stdev_or_zero(values)
        out.append(summary)
    return out


def fmt_acc(value):
    return "-" if value is None else f"{value * 100:.2f}"


def fmt_metric(value):
    return "-" if value is None else f"{value:.4f}"


def write_summary_csv(rows, path):
    fieldnames = [
        "variant",
        "label",
        "n_runs",
        "pacs_acc_mean",
        "pacs_acc_std",
    ]
    for key in METRIC_KEYS:
        fieldnames += [f"{key}_mean", f"{key}_std"]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(rows, path, dataset):
    header = [
        "Variant",
        f"{dataset} Acc. (up)",
        "Routing Ent. (down)",
        "Load Std. (down)",
        "Offdiag Cos. (down)",
        "Routing JS (down)",
        "n",
    ]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in rows:
        cells = [
            row["label"],
            fmt_acc(row["pacs_acc_mean"]),
            fmt_metric(row["Routing Ent_mean"]),
            fmt_metric(row["Load Std_mean"]),
            fmt_metric(row["Offdiag Cos_mean"]),
            fmt_metric(row["Routing JS_mean"]),
            str(row["n_runs"]),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset", default="PACS")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    per_run = collect_runs(args.output_dir)
    summary = aggregate(per_run)

    per_run_path = os.path.join(args.output_dir, "objective_ablation_per_run.csv")
    csv_path = os.path.join(args.output_dir, "objective_ablation_table.csv")
    md_path = os.path.join(args.output_dir, "objective_ablation_table.md")
    write_per_run(per_run, per_run_path)
    write_summary_csv(summary, csv_path)
    write_markdown(summary, md_path, args.dataset)

    print(f"Wrote {per_run_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()

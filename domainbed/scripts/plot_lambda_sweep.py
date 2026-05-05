import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from domainbed.lib.invariance_metrics import (
    class_conditional_domain_probe_accuracy,
    pairwise_class_conditional_discrepancy,
)


def _load_meta(npz_path):
    meta_path = os.path.splitext(npz_path)[0] + "_meta.json"
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        return json.load(f)


def _lambda_from_meta(meta, fallback=None):
    value = meta.get("lambda_inv", fallback)
    if value is None:
        return np.nan
    return float(value)


def _test_env_from_meta(meta):
    test_envs = meta.get("test_envs") or []
    if len(test_envs) == 1:
        return int(test_envs[0])
    return "+".join(str(int(x)) for x in test_envs)


def _compute_row(run_dir, args):
    source_path = os.path.join(run_dir, args.source_features_name)
    target_path = os.path.join(run_dir, args.target_features_name)
    if not os.path.exists(source_path):
        raise FileNotFoundError(source_path)
    if not os.path.exists(target_path):
        raise FileNotFoundError(target_path)

    source = np.load(source_path)
    target = np.load(target_path)
    source_meta = _load_meta(source_path)

    disc = pairwise_class_conditional_discrepancy(
        source["z"],
        source["y"],
        source["d"],
        distance=args.distance,
        min_count=args.min_count,
        mmd_sigmas=tuple(args.mmd_sigmas),
    )
    probe = class_conditional_domain_probe_accuracy(
        source["z"],
        source["y"],
        source["d"],
        min_count=args.min_count,
        test_size=args.probe_test_size,
        seed=args.probe_seed,
        max_iter=args.probe_max_iter,
    )

    lambda_inv = _lambda_from_meta(source_meta)
    target_acc = float(np.asarray(target["correct"]).astype(bool).mean() * 100.0)
    row = {
        "lambda_inv": lambda_inv,
        "pairwise_discrepancy": disc["pairwise_discrepancy"],
        "domain_probe_acc": probe["domain_probe_acc"] * 100.0,
        "target_acc": target_acc,
        "seed": source_meta.get("seed"),
        "test_env": _test_env_from_meta(source_meta),
        "run_dir": run_dir,
        "num_discrepancy_terms": disc["num_terms"],
        "num_probe_classes": probe["num_classes"],
    }
    if args.write_details:
        details_path = os.path.join(run_dir, "lambda_sweep_metric_details.json")
        with open(details_path, "w") as f:
            json.dump({"discrepancy": disc, "probe": probe}, f, indent=2, sort_keys=True)
    return row


def _fmt_mean_std(mean, std):
    if pd.isna(std):
        return "{:.4g}".format(mean)
    return "{:.4g} $\\pm$ {:.3g}".format(mean, std)


def _write_outputs(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "lambda_sweep_results.csv")
    tex_path = os.path.join(output_dir, "lambda_sweep_results.tex")
    df.to_csv(csv_path, index=False)

    agg = (
        df.groupby("lambda_inv", dropna=False)
        .agg(
            pairwise_discrepancy_mean=("pairwise_discrepancy", "mean"),
            pairwise_discrepancy_std=("pairwise_discrepancy", "std"),
            domain_probe_acc_mean=("domain_probe_acc", "mean"),
            domain_probe_acc_std=("domain_probe_acc", "std"),
            target_acc_mean=("target_acc", "mean"),
            target_acc_std=("target_acc", "std"),
            n=("target_acc", "count"),
        )
        .reset_index()
        .sort_values("lambda_inv")
    )
    display = pd.DataFrame({
        "lambda_inv": agg["lambda_inv"],
        "pairwise_discrepancy": [
            _fmt_mean_std(m, s)
            for m, s in zip(agg["pairwise_discrepancy_mean"], agg["pairwise_discrepancy_std"])
        ],
        "domain_probe_acc": [
            _fmt_mean_std(m, s)
            for m, s in zip(agg["domain_probe_acc_mean"], agg["domain_probe_acc_std"])
        ],
        "target_acc": [
            _fmt_mean_std(m, s)
            for m, s in zip(agg["target_acc_mean"], agg["target_acc_std"])
        ],
        "n": agg["n"],
    })
    display.to_latex(tex_path, index=False, escape=False)
    return csv_path, tex_path, agg


def _plot_panel(ax, x, mean, std, title, ylabel):
    ax.errorbar(x, mean, yerr=std, marker="o", capsize=3, linewidth=1.8)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def _plot(agg, output_dir, chance=None):
    agg = agg.sort_values("lambda_inv")
    labels = ["0" if v == 0 else "{:g}".format(v) for v in agg["lambda_inv"].to_numpy()]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    _plot_panel(
        axes[0],
        x,
        agg["pairwise_discrepancy_mean"],
        agg["pairwise_discrepancy_std"].fillna(0.0),
        "(a) Pairwise discrepancy",
        "discrepancy",
    )
    _plot_panel(
        axes[1],
        x,
        agg["domain_probe_acc_mean"],
        agg["domain_probe_acc_std"].fillna(0.0),
        "(b) Domain probe accuracy",
        "accuracy (%)",
    )
    if chance is not None:
        axes[1].axhline(float(chance) * 100.0, linestyle="--", color="black", linewidth=1.0)
    _plot_panel(
        axes[2],
        x,
        agg["target_acc_mean"],
        agg["target_acc_std"].fillna(0.0),
        "(c) Target accuracy",
        "accuracy (%)",
    )

    for ax in axes:
        ax.set_xlabel("lambda_inv")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    path = os.path.join(output_dir, "lambda_sweep.pdf")
    fig.savefig(path)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and plot pairwise-to-global invariance lambda sweeps."
    )
    parser.add_argument("--run_dirs", nargs="*", default=None,
                        help="Run directories containing features_source_val.npz and features_target.npz.")
    parser.add_argument("--csv", default=None,
                        help="Existing lambda_sweep_results.csv to plot without recomputing metrics.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--source_features_name", default="features_source_val.npz")
    parser.add_argument("--target_features_name", default="features_target.npz")
    parser.add_argument("--distance", choices=["mean", "coral", "mmd", "ed", "energy"], default="mmd")
    parser.add_argument("--mmd_sigmas", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0, 16.0])
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--probe_test_size", type=float, default=0.3)
    parser.add_argument("--probe_seed", type=int, default=0)
    parser.add_argument("--probe_max_iter", type=int, default=1000)
    parser.add_argument("--chance", type=float, default=None,
                        help="Domain probe chance level as a fraction, e.g. 0.333 for PACS source domains.")
    parser.add_argument("--write_details", action="store_true")
    args = parser.parse_args()

    if args.csv:
        df = pd.read_csv(args.csv)
        os.makedirs(args.output_dir, exist_ok=True)
        csv_path = os.path.join(args.output_dir, "lambda_sweep_results.csv")
        if os.path.abspath(args.csv) != os.path.abspath(csv_path):
            df.to_csv(csv_path, index=False)
        _, tex_path, agg = _write_outputs(df, args.output_dir)
    else:
        if not args.run_dirs:
            raise ValueError("Provide --run_dirs or --csv.")
        rows = [_compute_row(run_dir, args) for run_dir in args.run_dirs]
        df = pd.DataFrame(rows)
        csv_path, tex_path, agg = _write_outputs(df, args.output_dir)

    fig_path = _plot(agg, args.output_dir, chance=args.chance)
    print(json.dumps({
        "csv": csv_path,
        "tex": tex_path,
        "figure": fig_path,
        "num_rows": int(len(df)),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

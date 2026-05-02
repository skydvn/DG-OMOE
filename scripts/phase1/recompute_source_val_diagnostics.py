#!/usr/bin/env python3
"""Recompute GMoE routing diagnostics from saved checkpoints on source-val.

This script intentionally avoids target-domain samples. For each run directory
with a DomainBed ``model.pkl``, it rebuilds the same holdout split as
``domainbed/scripts/train.py`` and evaluates only the out-splits of the source
domains.
"""

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# Keep this before importing algorithms: algorithms imports vision/Tutel code.
import domainbed.tutel_patch  # noqa: F401,E402
from domainbed import algorithms, datasets  # noqa: E402
from domainbed.lib import misc  # noqa: E402
from domainbed.lib.fast_data_loader import FastDataLoader  # noqa: E402


METRIC_KEYS = ["Routing Ent", "Load Std", "Offdiag Cos", "Routing JS"]


def patch_cuda_to_cpu():
    """Allow constructors with hard-coded .cuda() calls to run on CPU."""
    torch.nn.Module.cuda = lambda self, device=None: self
    torch.Tensor.cuda = lambda self, device=None, non_blocking=False: self


def patch_pretrained_init():
    """Do not download/load ImageNet weights before checkpoint state_dict load."""
    original = algorithms.DeiTFeaturizer

    def no_pretrained_featurizer(model_name="deit_small_patch16_224", pretrained=True):
        return original(model_name=model_name, pretrained=False)

    algorithms.DeiTFeaturizer = no_pretrained_featurizer


def read_last_jsonl(path):
    if not os.path.exists(path):
        return {}
    last = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last


def infer_run_dirs(output_dir):
    pattern = os.path.join(output_dir, "*", "model.pkl")
    return sorted(os.path.dirname(p) for p in glob.glob(pattern))


def build_dataset_and_splits(ckpt, data_dir=None, deterministic_val=True):
    ckpt_args = ckpt["args"]
    hparams = dict(ckpt["model_hparams"])
    if deterministic_val:
        hparams["data_augmentation"] = False

    dataset_name = ckpt_args["dataset"]
    data_dir = data_dir or ckpt_args["data_dir"]
    dataset = datasets.get_dataset_class(dataset_name)(
        data_dir, ckpt_args["test_envs"], hparams
    )

    test_envs = set(ckpt_args["test_envs"])
    train_envs = ckpt_args.get("train_envs")
    if train_envs is None:
        train_envs = [i for i in range(len(dataset)) if i not in test_envs]

    out_splits = []
    for env_i, env in enumerate(dataset):
        out, _in = misc.split_dataset(
            env,
            int(len(env) * ckpt_args.get("holdout_fraction", 0.2)),
            misc.seed_hash(ckpt_args.get("trial_seed", 0), env_i),
        )
        out_splits.append(out)

    return dataset, train_envs, out_splits


def js_divergence(p, q, eps=1e-8):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * ((p * (p / m).log()).sum() + (q * (q / m).log()).sum())


def class_conditional_routing_js(domain_class_pi_sums, domain_class_counts, min_count):
    """Unweighted mean JS(p_{d,y}, pbar_y) over valid source domain-class pairs."""
    valid_by_class = defaultdict(list)
    for env_i, class_sums in domain_class_pi_sums.items():
        for class_i, pi_sum in class_sums.items():
            count = domain_class_counts[env_i][class_i]
            if count >= min_count:
                proto = pi_sum / count
                proto = proto / proto.sum().clamp_min(1e-8)
                valid_by_class[class_i].append((env_i, proto))

    js_vals = []
    valid_pairs = []
    classes_used = []
    domains_per_class = {}
    for class_i in sorted(valid_by_class):
        items = valid_by_class[class_i]
        if not items:
            continue
        template = torch.stack([proto for _, proto in items]).mean(dim=0)
        template = template / template.sum().clamp_min(1e-8)
        classes_used.append(class_i)
        domains_per_class[str(class_i)] = [int(env_i) for env_i, _ in items]
        for env_i, proto in items:
            js_vals.append(js_divergence(proto, template))
            valid_pairs.append({"domain": int(env_i), "class": int(class_i)})

    if js_vals:
        value = torch.stack(js_vals).mean().item()
    else:
        value = 0.0

    return value, {
        "min_count": min_count,
        "num_valid_pairs": len(valid_pairs),
        "classes_used": [int(c) for c in classes_used],
        "domains_per_class": domains_per_class,
        "valid_pairs": valid_pairs,
    }


def evaluate_source_val(
    algorithm,
    dataset,
    train_envs,
    out_splits,
    device,
    batch_size,
    num_workers,
    routing_js_min_count,
):
    algorithm.eval()

    total_n = 0
    correct = 0
    entropy_sum = 0.0
    offdiag_sum = 0.0
    pi_sum = None
    domain_pi_sums = {}
    domain_counts = {}
    domain_class_counts = {}
    domain_class_pi_sums = {}

    with torch.no_grad():
        for env_i in train_envs:
            split = out_splits[env_i]
            loader = FastDataLoader(
                dataset=split,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            env_pi_sum = None
            env_count = 0
            class_counts = defaultdict(int)
            class_pi_sums = {}

            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                logits, pi, h_stack = algorithm._forward(x)

                n = x.size(0)
                total_n += n
                env_count += n
                correct += (logits.argmax(dim=1) == y).sum().item()

                entropy = -(pi * (pi + 1e-8).log()).sum(dim=-1)
                entropy_sum += entropy.sum().item()

                if pi_sum is None:
                    pi_sum = torch.zeros(pi.size(1), device=device)
                pi_sum += pi.sum(dim=0)

                if env_pi_sum is None:
                    env_pi_sum = torch.zeros(pi.size(1), device=device)
                env_pi_sum += pi.sum(dim=0)

                for cls in torch.unique(y.detach()).tolist():
                    cls = int(cls)
                    mask = y == cls
                    cnt = int(mask.sum().item())
                    class_counts[cls] += cnt
                    if cls not in class_pi_sums:
                        class_pi_sums[cls] = torch.zeros(pi.size(1), device=device)
                    class_pi_sums[cls] += pi[mask].sum(dim=0)

                num_experts = h_stack.size(1)
                if num_experts > 1:
                    h_norm = F.normalize(h_stack, dim=-1)
                    cos = torch.bmm(h_norm, h_norm.transpose(1, 2))
                    offdiag = ~torch.eye(num_experts, dtype=torch.bool, device=device)
                    per_sample = cos[:, offdiag].mean(dim=1)
                    offdiag_sum += per_sample.sum().item()

            if env_count:
                domain_pi_sums[env_i] = env_pi_sum.detach().cpu()
                domain_counts[env_i] = env_count
                domain_class_counts[env_i] = dict(sorted(class_counts.items()))
                domain_class_pi_sums[env_i] = {
                    cls: pi_sum.detach().cpu()
                    for cls, pi_sum in sorted(class_pi_sums.items())
                }

    if total_n == 0:
        raise RuntimeError("No source-validation samples were evaluated.")

    load = (pi_sum / total_n).detach().cpu()
    domain_routes = [
        (domain_pi_sums[env_i] / domain_counts[env_i]).detach().cpu()
        for env_i in train_envs
        if env_i in domain_pi_sums and domain_counts[env_i] > 0
    ]

    domain_js_vals = []
    for i in range(len(domain_routes)):
        for j in range(i + 1, len(domain_routes)):
            domain_js_vals.append(js_divergence(domain_routes[i], domain_routes[j]))

    routing_js, routing_js_meta = class_conditional_routing_js(
        domain_class_pi_sums,
        domain_class_counts,
        routing_js_min_count,
    )

    metrics = {
        "Routing Ent": entropy_sum / total_n,
        "Load Std": load.std(unbiased=False).item(),
        "Offdiag Cos": offdiag_sum / total_n,
        "Routing JS": routing_js,
        "Routing JS domain-only": torch.stack(domain_js_vals).mean().item() if domain_js_vals else 0.0,
        "Routing JS meta": routing_js_meta,
        "source_val_acc": correct / total_n,
        "source_val_samples": total_n,
        "source_val_domains": train_envs,
        "source_val_domain_counts": {str(k): v for k, v in sorted(domain_counts.items())},
        "source_val_domain_class_counts": {
            str(k): {str(c): n for c, n in sorted(v.items())}
            for k, v in sorted(domain_class_counts.items())
        },
        "source_val_load": [float(v) for v in load.tolist()],
    }
    return metrics


def load_algorithm(ckpt, device):
    algorithm_class = algorithms.get_algorithm_class(ckpt["args"]["algorithm"])
    algorithm = algorithm_class(
        ckpt["model_input_shape"],
        ckpt["model_num_classes"],
        ckpt["model_num_domains"],
        ckpt["model_hparams"],
    )
    algorithm.load_state_dict(ckpt["model_dict"])
    algorithm.to(device)
    return algorithm


def row_for_run(run_dir, args):
    ckpt_path = os.path.join(run_dir, "model.pkl")
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    last_result = read_last_jsonl(os.path.join(run_dir, "results.jsonl"))

    random.seed(ckpt["args"].get("seed", 0))
    np.random.seed(ckpt["args"].get("seed", 0))
    torch.manual_seed(ckpt["args"].get("seed", 0))

    dataset, train_envs, out_splits = build_dataset_and_splits(
        ckpt,
        data_dir=args.data_dir,
        deterministic_val=not args.use_train_augmentation,
    )
    algorithm = load_algorithm(ckpt, args.device)
    metrics = evaluate_source_val(
        algorithm,
        dataset,
        train_envs,
        out_splits,
        args.device,
        args.batch_size,
        args.num_workers,
        args.routing_js_min_count,
    )

    test_envs = ckpt["args"]["test_envs"]
    test_env = test_envs[0] if len(test_envs) == 1 else "+".join(map(str, test_envs))
    test_domain = "+".join(dataset.ENVIRONMENTS[i] for i in test_envs)
    row = {
        "run_dir": run_dir,
        "checkpoint": ckpt_path,
        "algorithm": ckpt["args"]["algorithm"],
        "dataset": ckpt["args"]["dataset"],
        "test_env": test_env,
        "test_domain": test_domain,
        "checkpoint_step": last_result.get("step"),
        "deterministic_val_transform": not args.use_train_augmentation,
    }
    row.update(metrics)
    return row


def write_outputs(rows, output_dir):
    csv_path = os.path.join(output_dir, "source_val_diagnostics.csv")
    json_path = os.path.join(output_dir, "source_val_diagnostics.json")
    jsonl_path = os.path.join(output_dir, "source_val_diagnostics_per_run.jsonl")

    fieldnames = [
        "run_dir",
        "checkpoint",
        "algorithm",
        "dataset",
        "test_env",
        "test_domain",
        "checkpoint_step",
        "deterministic_val_transform",
        "source_val_acc",
        "source_val_samples",
        "source_val_domains",
        "source_val_domain_counts",
        "source_val_domain_class_counts",
        "source_val_load",
        "Routing JS meta",
        "Routing JS domain-only",
    ] + METRIC_KEYS

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in [
                "source_val_domains",
                "source_val_domain_counts",
                "source_val_domain_class_counts",
                "source_val_load",
                "Routing JS meta",
            ]:
                flat[key] = json.dumps(flat[key], sort_keys=True)
            writer.writerow(flat)

    with open(jsonl_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    mean = {
        key: float(np.mean([row[key] for row in rows]))
        for key in ["source_val_acc"] + METRIC_KEYS
    }
    summary = {
        "n_runs": len(rows),
        "mean": mean,
        "rows": rows,
        "outputs": {
            "csv": csv_path,
            "jsonl": jsonl_path,
        },
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return csv_path, json_path, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/mnt/disk1/backup_user/dat.tt2/DG-OMOE/train_output/L_inv_and_L_sp",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Override checkpoint args['data_dir'] when PACS is stored elsewhere.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--routing_js_min_count",
        type=int,
        default=5,
        help="Minimum samples required for a source domain-class prototype p_{d,y}.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--use_train_augmentation",
        action="store_true",
        help="Use checkpoint hparams as-is. Default disables augmentation for deterministic source-val.",
    )
    parser.add_argument(
        "--load_pretrained_init",
        action="store_true",
        help="Load ImageNet init before state_dict. Default skips it to avoid downloads.",
    )
    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu":
        patch_cuda_to_cpu()
    if not args.load_pretrained_init:
        patch_pretrained_init()

    run_dirs = infer_run_dirs(args.output_dir)
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with model.pkl under {args.output_dir}")

    rows = []
    for run_dir in run_dirs:
        print(f"[source-val diagnostics] {run_dir}", flush=True)
        rows.append(row_for_run(run_dir, args))
        brief = {k: rows[-1][k] for k in METRIC_KEYS}
        print(json.dumps(brief, sort_keys=True), flush=True)

    csv_path, json_path, summary = write_outputs(rows, args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(summary["mean"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Benchmark PACS training-time table entries.

This script intentionally avoids domainbed.scripts.train so it can measure only
training-step cost and inference cost without checkpointing or evaluation.
"""

import argparse
import csv
import json
import os
import random
import statistics
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def _early_cuda_device_arg():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda_device", default="0")
    args, _ = parser.parse_known_args()
    if args.cuda_device not in (None, "", "none", "None"):
        explicit = any(
            arg == "--cuda_device" or arg.startswith("--cuda_device=")
            for arg in sys.argv[1:]
        )
        if explicit:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
        else:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.cuda_device))


_early_cuda_device_arg()
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import domainbed.tutel_patch  # noqa: F401,E402
from domainbed import algorithms, datasets, hparams_registry  # noqa: E402
from domainbed.lib import misc  # noqa: E402
from domainbed.lib.fast_data_loader import InfiniteDataLoader  # noqa: E402


METHOD_ORDER = ["ERM", "SAGM", "GMoE", "GMoE+SAGM", "MESSI-MMD", "MESSI-OT"]
BACKBONES = {
    "ERM": "ResNet-50",
    "SAGM": "ResNet-50",
    "GMoE": "DeiT-S/16",
    "GMoE+SAGM": "DeiT-S/16",
    "MESSI-MMD": "DeiT-S/16",
    "MESSI-OT": "DeiT-S/16",
}
BACKBONE_LABELS = {
    "deit_small_patch16_224": "DeiT-S/16",
    "deit_tiny_patch16_224": "DeiT-Ti/16",
}
ALGORITHM_BY_METHOD = {
    "ERM": "ERM",
    "SAGM": "ERM",
    "GMoE": "GMOE",
    "GMoE+SAGM": "GMOE",
    "MESSI-MMD": "GMOE_InvMMD",
    "MESSI-OT": "GMOE_InvOT",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark PACS training time, memory, and inference time."
    )
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "domainbed" / "data"))
    parser.add_argument("--output_dir", default=str(REPO_ROOT / "train_output" / "benchmark_training_time"))
    parser.add_argument("--test_env", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--cuda_device", default="0")
    parser.add_argument("--methods", nargs="+", default=["all"])
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--inf_batches", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument(
        "--model",
        default=None,
        help="Override DeiT model for MoE methods, e.g. deit_tiny_patch16_224.",
    )
    parser.add_argument(
        "--relative_erm_seconds",
        type=float,
        default=None,
        help="ERM train/1k seconds to use when summarizing a method-only run.",
    )
    return parser.parse_args()


def expand_methods(methods):
    if len(methods) == 1 and methods[0].lower() == "all":
        return list(METHOD_ORDER)
    unknown = [method for method in methods if method not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}. Valid: {METHOD_ORDER} or all")
    return methods


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def preserve_bn_running_stats(model):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            momenta[module] = module.momentum
            module.momentum = 0
    try:
        yield
    finally:
        for module, momentum in momenta.items():
            module.momentum = momentum


class LinearScheduler:
    def __init__(self, max_value, min_value=None, t_max=5000, optimizer=None):
        self.max_value = max_value
        self.min_value = max_value if min_value is None else min_value
        self.t_max = max(1, t_max)
        self.optimizer = optimizer
        self.t = 0

    def step(self):
        ratio = min(self.t / self.t_max, 1.0)
        value = self.max_value + ratio * (self.min_value - self.max_value)
        self.t += 1
        if self.optimizer is not None:
            for group in self.optimizer.param_groups:
                group["lr"] = value
        return value


class LocalSAGM:
    def __init__(self, params, base_optimizer, model, alpha=0.001, rho=0.05, perturb_eps=1e-12):
        self.base_optimizer = base_optimizer
        self.model = model
        self.param_groups = self.base_optimizer.param_groups
        self.alpha = alpha
        self.perturb_eps = perturb_eps
        self.rho_scheduler = LinearScheduler(max_value=rho, min_value=rho, t_max=5000)
        self.rho_t = self.rho_scheduler.step()
        self.state = {p: {} for group in self.param_groups for p in group["params"]}
        list(params)  # consume for API symmetry; param_groups are owned by base_optimizer

    @torch.no_grad()
    def _grad_norm(self, key=None):
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if key is None:
                    grad = p.grad
                else:
                    grad = self.state[p].get(key)
                if grad is not None:
                    norms.append(grad.norm(p=2))
        if not norms:
            return torch.tensor(0.0, device=self.param_groups[0]["params"][0].device)
        return torch.norm(torch.stack(norms), p=2)

    @torch.no_grad()
    def perturb_weights(self):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_g"] = p.grad.detach().clone()
                scale = self.rho_t / (grad_norm + self.perturb_eps) - self.alpha
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def unperturb(self):
        for group in self.param_groups:
            for p in group["params"]:
                e_w = self.state[p].pop("e_w", None)
                if e_w is not None:
                    p.sub_(e_w)

    @torch.no_grad()
    def gradient_decompose(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                old_g = self.state[p].get("old_g")
                if old_g is not None:
                    p.grad.add_(old_g * 0.5 - p.grad * 0.5)

    def step(self, closure):
        self.base_optimizer.zero_grad()
        with torch.enable_grad():
            loss = closure()
        loss_value = loss.detach()
        loss.backward()

        self.perturb_weights()
        with preserve_bn_running_stats(self.model):
            self.base_optimizer.zero_grad()
            with torch.enable_grad():
                loss_perturbed = closure()
            loss_perturbed.backward()
            self.gradient_decompose()
            self.unperturb()

        self.base_optimizer.step()
        self.rho_t = self.rho_scheduler.step()
        return loss_value


class SAGMERM(torch.nn.Module):
    def __init__(self, base, hparams):
        super().__init__()
        self.base = base
        self.network = base.network
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.lr_scheduler = LinearScheduler(
            max_value=hparams["lr"],
            min_value=hparams["lr"],
            t_max=5000,
            optimizer=self.optimizer,
        )
        self.sagm = LocalSAGM(
            self.network.parameters(),
            self.optimizer,
            self.network,
            alpha=hparams.get("alpha", 0.001),
            rho=hparams.get("rho", 0.05),
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        def closure():
            return F.cross_entropy(self.predict(all_x), all_y)

        loss = self.sagm.step(closure)
        self.lr_scheduler.step()
        return {"loss": float(loss.item())}

    def predict(self, x):
        return self.base.predict(x)


class SAGMGMOE(torch.nn.Module):
    def __init__(self, base, hparams):
        super().__init__()
        self.base = base
        self.model = base.model
        self.hparams = hparams
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )
        self.lr_scheduler = LinearScheduler(
            max_value=hparams["lr"],
            min_value=hparams["lr"],
            t_max=5000,
            optimizer=self.optimizer,
        )
        self.sagm = LocalSAGM(
            self.model.parameters(),
            self.optimizer,
            self.model,
            alpha=hparams.get("alpha", 0.001),
            rho=hparams.get("rho", 0.05),
        )

    def _loss(self, all_x, all_y):
        loss = F.cross_entropy(self.predict(all_x), all_y)
        aux_loss = loss.new_tensor(0.0)
        ortho_loss = loss.new_tensor(0.0)
        variance_loss = loss.new_tensor(0.0)
        for block in self.model.blocks:
            if getattr(block, "aux_loss", None) is not None:
                aux_loss = aux_loss + block.aux_loss
                if (
                    getattr(block, "expert_outputs", None) is not None
                    and self.hparams.get("ortho_loss_weight", 0.0) > 0
                ):
                    ortho_loss = ortho_loss + self.base.ortho_loss_fn(block.expert_outputs)
                if (
                    getattr(block, "routing_scores", None) is not None
                    and self.hparams.get("variance_loss_weight", 0.0) > 0
                ):
                    variance_loss = variance_loss + self.base.variance_loss_fn(block.routing_scores)
        return (
            loss
            + aux_loss
            + self.hparams.get("ortho_loss_weight", 0.0) * ortho_loss
            + self.hparams.get("variance_loss_weight", 0.0) * variance_loss
        )

    def update(self, minibatches, unlabeled=None):
        all_x = torch.cat([x for x, _ in minibatches])
        all_y = torch.cat([y for _, y in minibatches])

        def closure():
            return self._loss(all_x, all_y)

        loss = self.sagm.step(closure)
        self.lr_scheduler.step()
        return {"loss": float(loss.item())}

    def predict(self, x, *args, **kwargs):
        return self.base.predict(x, *args, **kwargs)


def backbone_for_method(method, hparams=None):
    if hparams is not None and "model" in hparams:
        return BACKBONE_LABELS.get(hparams["model"], hparams["model"])
    return BACKBONES[method]


def default_hparams_for_method(method, model_override=None):
    algorithm_name = ALGORITHM_BY_METHOD[method]
    hparams = hparams_registry.default_hparams(algorithm_name, "PACS")
    hparams["batch_size"] = 32
    if method in {"GMoE", "GMoE+SAGM", "MESSI-MMD", "MESSI-OT"}:
        hparams.update(
            {
                "model": model_override or "deit_small_patch16_224",
                "num_experts": 6,
                "gate_k": 1,
                "expert_depth": 2,
                "mlp_ratio": 4.0,
                "expert_prune_ratio": 0.0,
            }
        )
    if method in {"SAGM", "GMoE+SAGM"}:
        hparams.update({"alpha": 0.001, "rho": 0.05})
    if method in {"MESSI-MMD", "MESSI-OT"}:
        hparams.update({"lambda_inv": 0.1, "alpha": 4.0})
    return hparams


def build_dataset_and_loaders(args, method):
    hparams = default_hparams_for_method(method, args.model)
    hparams["batch_size"] = args.batch_size
    dataset = datasets.PACS(args.data_dir, [args.test_env], hparams)
    train_envs = [i for i in range(len(dataset)) if i != args.test_env]
    in_splits = []
    for env_i, env in enumerate(dataset):
        _, in_split = misc.split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(0, env_i),
        )
        in_splits.append(in_split)

    num_workers = dataset.N_WORKERS if args.num_workers is None else args.num_workers
    train_loaders = [
        InfiniteDataLoader(
            dataset=in_splits[env_i],
            weights=None,
            batch_size=hparams["batch_size"],
            num_workers=num_workers,
        )
        for env_i in train_envs
    ]
    return dataset, hparams, zip(*train_loaders)


def build_algorithm(method, dataset, hparams, device):
    algorithm_name = ALGORITHM_BY_METHOD[method]
    algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    base = algorithm_class(dataset.input_shape, dataset.num_classes, 3, hparams).to(device)
    if method == "SAGM":
        return SAGMERM(base, hparams).to(device)
    if method == "GMoE+SAGM":
        return SAGMGMOE(base, hparams).to(device)
    return base


def next_minibatches(iterator, device):
    return [(x.to(device, non_blocking=True), y.to(device, non_blocking=True)) for x, y in next(iterator)]


def run_training_steps(algorithm, iterator, device, n_steps, timed):
    step_times = []
    algorithm.train()
    for _ in range(n_steps):
        minibatches = next_minibatches(iterator, device)
        if timed:
            sync(device)
            start = time.perf_counter()
        algorithm.update(minibatches)
        if timed:
            sync(device)
            step_times.append(time.perf_counter() - start)
    return step_times


def run_inference(algorithm, iterator, device, n_batches):
    algorithm.eval()
    batch_times = []
    with torch.no_grad():
        for _ in range(n_batches):
            minibatches = next_minibatches(iterator, device)
            x = torch.cat([batch_x for batch_x, _ in minibatches])
            sync(device)
            start = time.perf_counter()
            _ = algorithm.predict(x)
            sync(device)
            batch_times.append(time.perf_counter() - start)
    return batch_times


def run_one(args, method, repeat, device):
    set_seed(repeat)
    dataset, hparams, train_iterator = build_dataset_and_loaders(args, method)
    algorithm = build_algorithm(method, dataset, hparams, device)

    run_training_steps(algorithm, train_iterator, device, args.warmup_steps, timed=False)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    step_times = run_training_steps(algorithm, train_iterator, device, args.steps, timed=True)
    peak_mem_gib = (
        torch.cuda.max_memory_allocated(device) / (1024.0 ** 3)
        if device.type == "cuda"
        else 0.0
    )
    inf_times = run_inference(algorithm, train_iterator, device, args.inf_batches)

    record = {
        "method": method,
        "backbone": backbone_for_method(method, hparams),
        "algorithm": ALGORITHM_BY_METHOD[method],
        "repeat": repeat,
        "seed": repeat,
        "test_env": args.test_env,
        "batch_size": args.batch_size,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "mean_step_seconds": statistics.mean(step_times),
        "train_per_1k_seconds": statistics.mean(step_times) * 1000.0,
        "step_seconds_std": statistics.stdev(step_times) if len(step_times) > 1 else 0.0,
        "peak_mem_gib": peak_mem_gib,
        "inf_ms_per_batch": statistics.mean(inf_times) * 1000.0,
        "inf_ms_per_batch_std": statistics.stdev(inf_times) * 1000.0 if len(inf_times) > 1 else 0.0,
    }
    del algorithm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def summarize(records, relative_erm_seconds=None):
    by_method = {method: [] for method in METHOD_ORDER}
    for record in records:
        by_method[record["method"]].append(record)

    summaries = []
    for method in METHOD_ORDER:
        method_records = by_method.get(method, [])
        if not method_records:
            continue
        train = [r["train_per_1k_seconds"] for r in method_records]
        mem = [r["peak_mem_gib"] for r in method_records]
        inf = [r["inf_ms_per_batch"] for r in method_records]
        summaries.append(
            {
                "method": method,
                "backbone": method_records[0].get("backbone", BACKBONES[method]),
                "n": len(method_records),
                "train_per_1k_mean_s": statistics.mean(train),
                "train_per_1k_std_s": statistics.stdev(train) if len(train) > 1 else 0.0,
                "peak_mem_mean_gib": statistics.mean(mem),
                "peak_mem_std_gib": statistics.stdev(mem) if len(mem) > 1 else 0.0,
                "inf_mean_ms_per_batch": statistics.mean(inf),
                "inf_std_ms_per_batch": statistics.stdev(inf) if len(inf) > 1 else 0.0,
            }
        )

    erm = next((row for row in summaries if row["method"] == "ERM"), None)
    erm_time = erm["train_per_1k_mean_s"] if erm is not None else relative_erm_seconds
    for row in summaries:
        row["relative_time"] = (
            row["train_per_1k_mean_s"] / erm_time
            if erm_time is not None
            else None
        )
    return summaries


def write_raw(path, records):
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def write_csv(path, rows):
    fieldnames = [
        "method",
        "backbone",
        "n",
        "train_per_1k_mean_s",
        "train_per_1k_std_s",
        "relative_time",
        "peak_mem_mean_gib",
        "peak_mem_std_gib",
        "inf_mean_ms_per_batch",
        "inf_std_ms_per_batch",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def pm(mean, std, unit="", digits=2):
    suffix = f" {unit}" if unit else ""
    return f"{mean:.{digits}f}$\\pm${std:.{digits}f}{suffix}"


def write_latex(path, rows):
    row_by_method = {row["method"]: row for row in rows}
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\caption{",
        "Training-time comparison on PACS under the same input resolution, batch size,",
        "hardware, and training schedule. Train time is measured as wall-clock time per",
        "1,000 optimization steps after warm-up. Relative time is normalized by ERM.",
        "Peak memory is measured during training. Inference time is measured with",
        "training-only losses disabled.",
        "}",
        "\\label{tab:training_time}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\begin{tabular*}{0.98\\linewidth}{@{\\extracolsep{\\fill}}llcccc}",
        "\\toprule",
        "Method & Backbone & Train / 1k steps $\\downarrow$ & Rel. time $\\downarrow$ & Peak mem. $\\downarrow$ & Inf. time $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for method in METHOD_ORDER:
        if method not in row_by_method:
            continue
        row = row_by_method[method]
        prefix = "\\rowcolor{gray!10}\n" if method == "MESSI-OT" else ""
        if method == "ERM":
            rel = "1.00$\\times$"
        elif row["relative_time"] is None:
            rel = "NA"
        else:
            rel = f"{row['relative_time']:.2f}$\\times$"
        lines.append(
            prefix
            + f"{method} & {row['backbone']} & "
            + pm(row["train_per_1k_mean_s"], row["train_per_1k_std_s"], "s")
            + f" & {rel} & "
            + pm(row["peak_mem_mean_gib"], row["peak_mem_std_gib"], "GiB")
            + " & "
            + pm(row["inf_mean_ms_per_batch"], row["inf_std_ms_per_batch"], "ms/batch")
            + " \\\\"
        )
        if method != "MESSI-OT":
            lines.append("")
    lines.extend(["\\bottomrule", "\\end{tabular*}", "\\end{table}", ""])
    path.write_text("\n".join(lines))


def load_existing_raw(path):
    if not path.exists():
        return []
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    args = parse_args()
    methods = expand_methods(args.methods)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw_runs.jsonl"
    csv_path = output_dir / "summary.csv"
    latex_path = output_dir / "training_time_table.tex"

    if not (Path(args.data_dir) / "PACS").is_dir():
        raise FileNotFoundError(f"PACS not found at {Path(args.data_dir) / 'PACS'}")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this benchmark. DG-OMOE algorithms move "
            "models to CUDA internally, and the table requires CUDA peak "
            "memory measurements."
        )
    device = torch.device("cuda")

    records = load_existing_raw(raw_path) if args.skip_existing else []
    existing = {(r["method"], r["repeat"]) for r in records}

    for method in methods:
        for repeat in range(args.repeats):
            if args.skip_existing and (method, repeat) in existing:
                print(f"Skipping existing {method} repeat {repeat}")
                continue
            print(f"Running {method} repeat {repeat} on {device}...")
            record = run_one(args, method, repeat, device)
            records.append(record)
            write_raw(raw_path, records)
            print(
                f"  train/1k={record['train_per_1k_seconds']:.2f}s "
                f"mem={record['peak_mem_gib']:.2f}GiB "
                f"inf={record['inf_ms_per_batch']:.2f}ms/batch"
            )

    summaries = summarize(records, args.relative_erm_seconds)
    write_csv(csv_path, summaries)
    write_latex(latex_path, summaries)
    print(f"Wrote {raw_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()

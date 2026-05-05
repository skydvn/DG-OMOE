import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import domainbed.tutel_patch  # noqa: F401
from domainbed import algorithms
from domainbed import datasets
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import FastDataLoader


def _load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _batch_to_xy(batch):
    if isinstance(batch, dict):
        return batch["x"], batch["y"]
    return batch


def _split_dataset(dataset, ckpt_args):
    in_splits = []
    out_splits = []
    test_envs = ckpt_args.get("test_envs", [0])
    trial_seed = ckpt_args.get("trial_seed", 0)
    holdout_fraction = ckpt_args.get("holdout_fraction", 0.2)
    uda_holdout_fraction = ckpt_args.get("uda_holdout_fraction", 0)
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * holdout_fraction),
            misc.seed_hash(trial_seed, env_i),
        )
        if env_i in test_envs and uda_holdout_fraction:
            _, in_ = misc.split_dataset(
                in_,
                int(len(in_) * uda_holdout_fraction),
                misc.seed_hash(trial_seed, env_i),
            )
        in_splits.append(in_)
        out_splits.append(out)
    return in_splits, out_splits


def _source_envs(n_envs, test_envs, ckpt_args):
    if ckpt_args.get("train_envs") is not None:
        return list(ckpt_args["train_envs"])
    return [i for i in range(n_envs) if i not in test_envs]


def _extract_z_logits(algorithm, x):
    """Return final representation z and logits without changing model weights."""
    if hasattr(algorithm, "moe_head") and hasattr(algorithm, "featurizer"):
        if hasattr(algorithm, "_preprocess"):
            x = algorithm._preprocess(x)
        backbone_z = algorithm.featurizer(x)
        logits, pi, h_stack = algorithm.moe_head(backbone_z)
        z = (pi.unsqueeze(-1) * h_stack).sum(dim=1)
        return z, logits

    if hasattr(algorithm, "featurizer") and hasattr(algorithm, "classifier"):
        z = algorithm.featurizer(x)
        logits = algorithm.classifier(z)
        return z, logits

    if hasattr(algorithm, "network") and isinstance(algorithm.network, torch.nn.Sequential):
        modules = list(algorithm.network.children())
        if len(modules) >= 2:
            z = modules[0](x)
            logits = modules[1](z)
            return z, logits

    raise RuntimeError(
        "Could not infer a representation extractor for {}. Add an explicit "
        "case in _extract_z_logits().".format(type(algorithm).__name__)
    )


def _collect_split(algorithm, split_items, batch_size, num_workers, device, max_examples):
    zs, ys, ds, logits_list, correct = [], [], [], [], []
    total = 0
    with torch.no_grad():
        for env_i, dataset in split_items:
            loader = FastDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
            )
            for batch in loader:
                x, y = _batch_to_xy(batch)
                if max_examples is not None:
                    remaining = max_examples - total
                    if remaining <= 0:
                        break
                    x = x[:remaining]
                    y = y[:remaining]
                x = x.to(device)
                y = y.to(device)
                z, logits = _extract_z_logits(algorithm, x)
                pred = logits.argmax(dim=1) if logits.size(1) > 1 else logits.gt(0).long().view(-1)
                zs.append(z.detach().cpu().numpy())
                ys.append(y.detach().cpu().numpy())
                ds.append(np.full(y.numel(), env_i, dtype=np.int64))
                logits_list.append(logits.detach().cpu().numpy())
                correct.append(pred.eq(y).detach().cpu().numpy())
                total += int(y.numel())
            if max_examples is not None and total >= max_examples:
                break

    if not zs:
        raise RuntimeError("No examples were collected.")
    return {
        "z": np.concatenate(zs, axis=0).astype(np.float32),
        "y": np.concatenate(ys, axis=0).astype(np.int64),
        "d": np.concatenate(ds, axis=0).astype(np.int64),
        "logits": np.concatenate(logits_list, axis=0).astype(np.float32),
        "correct": np.concatenate(correct, axis=0).astype(bool),
    }


def _save_npz(path, arrays, meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **arrays)
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main():
    parser = argparse.ArgumentParser(
        description="Collect frozen representations for lambda-invariance sweeps."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to DomainBed model.pkl.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["source_val", "target"],
        default=["source_val", "target"],
        help="source_val uses source out-splits; target uses held-out target out-split.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_examples", type=int, default=None)
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ckpt = _load_checkpoint(args.checkpoint, args.device)
    ckpt_args = ckpt["args"]
    hparams = dict(ckpt["model_hparams"])
    model_keys = ckpt["model_dict"].keys()
    if any(".mlp.gate_proj.weight" in key for key in model_keys):
        hparams["force_custom_moe"] = True

    dataset_name = ckpt_args["dataset"]
    test_envs = ckpt_args.get("test_envs", [0])
    data_dir = ckpt_args.get("data_dir", "./domainbed/data")
    dataset = vars(datasets)[dataset_name](data_dir, test_envs, hparams)
    in_splits, out_splits = _split_dataset(dataset, ckpt_args)
    source_envs = _source_envs(len(dataset), test_envs, ckpt_args)

    algorithm_class = algorithms.get_algorithm_class(ckpt_args["algorithm"])
    algorithm = algorithm_class(
        ckpt["model_input_shape"],
        ckpt["model_num_classes"],
        ckpt["model_num_domains"],
        hparams,
    )
    algorithm.load_state_dict(ckpt["model_dict"])
    algorithm.to(args.device)
    algorithm.eval()

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    split_defs = {
        "source_val": [(env_i, out_splits[env_i]) for env_i in source_envs],
        "target": [(env_i, out_splits[env_i]) for env_i in test_envs],
    }

    written = {}
    for split in args.splits:
        arrays = _collect_split(
            algorithm,
            split_defs[split],
            args.batch_size,
            args.num_workers,
            args.device,
            args.max_examples,
        )
        filename = "features_raw.npz" if len(args.splits) == 1 else f"features_{split}.npz"
        path = os.path.join(output_dir, filename)
        meta = {
            "checkpoint": args.checkpoint,
            "dataset": dataset_name,
            "algorithm": ckpt_args["algorithm"],
            "lambda_inv": hparams.get("lambda_inv"),
            "seed": ckpt_args.get("seed"),
            "trial_seed": ckpt_args.get("trial_seed"),
            "test_envs": [int(env_i) for env_i in test_envs],
            "split": split,
            "domains": [int(env_i) for env_i, _ in split_defs[split]],
            "num_examples": int(arrays["y"].shape[0]),
            "feature_dim": int(arrays["z"].shape[1]),
            "accuracy": float(arrays["correct"].mean()),
        }
        _save_npz(path, arrays, meta)
        written[split] = path

    print(json.dumps({"written": written}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

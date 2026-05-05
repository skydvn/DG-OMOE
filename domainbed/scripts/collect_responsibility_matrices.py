import argparse
import csv
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
from domainbed.lib.responsibility_metrics import (
    compute_A_m,
    mean_router_mass,
    pairwise_responsibility,
    responsibility_summary,
    sanity_check_responsibility,
)


def _load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _batch_to_xy(batch):
    if isinstance(batch, dict):
        return batch['x'], batch['y']
    return batch


def _split_dataset(dataset, args):
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(
            env,
            int(len(env) * args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i),
        )
        if env_i in args.test_envs and args.uda_holdout_fraction:
            _, in_ = misc.split_dataset(
                in_,
                int(len(in_) * args.uda_holdout_fraction),
                misc.seed_hash(args.trial_seed, env_i),
            )
        in_splits.append(in_)
        out_splits.append(out)
    return in_splits, out_splits


def _write_summary_csv(path, rows, aggregate):
    fieldnames = ['expert', 'mean', 'max', 'std', 'sparsity', 'active']
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    aggregate_path = os.path.splitext(path)[0] + '_aggregate.json'
    with open(aggregate_path, 'w') as f:
        json.dump(aggregate, f, indent=2, sort_keys=True)


def _resolve_algorithm_name(requested_name, checkpoint_name):
    if requested_name is None:
        return checkpoint_name
    if requested_name in algorithms.ALGORITHMS or requested_name in vars(algorithms):
        return requested_name
    if requested_name.upper() == 'MESSI':
        if checkpoint_name in algorithms.ALGORITHMS or checkpoint_name in vars(algorithms):
            return checkpoint_name
        return 'GMOE_InvMMD'
    return requested_name


def main():
    parser = argparse.ArgumentParser(
        description='Collect expert-specific source-domain responsibility matrices.')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to DomainBed model.pkl.')
    parser.add_argument('--data_dir', default=None,
                        help='Override checkpoint data_dir.')
    parser.add_argument('--dataset', default=None,
                        help='Override checkpoint dataset.')
    parser.add_argument('--algorithm', default=None,
                        help='Override checkpoint algorithm.')
    parser.add_argument('--test_env', type=int, default=None,
                        help='Single held-out environment override.')
    parser.add_argument('--test_envs', type=int, nargs='*', default=None,
                        help='Held-out environment override.')
    parser.add_argument('--train_envs', type=int, nargs='*', default=None,
                        help='Source environments. Defaults to checkpoint train_envs or non-test envs.')
    parser.add_argument('--split',
                        choices=['source_val', 'source_train', 'out', 'in'],
                        default='source_val',
                        help='Use source_val/out for held-out source split or source_train/in for training split.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--alpha', type=float, default=None,
                        help='Routing responsibility temperature. Defaults to hparams alpha or 4.0.')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Optional total cap for smoke tests.')
    parser.add_argument('--threshold', type=float, default=1e-6,
                        help='Threshold for sparsity/activity summary.')
    parser.add_argument('--allow_missing_router', action='store_true')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ckpt = _load_checkpoint(args.checkpoint, args.device)
    ckpt_args = ckpt['args']
    hparams = dict(ckpt['model_hparams'])

    model_keys = ckpt['model_dict'].keys()
    if any('.mlp.gate_proj.weight' in key for key in model_keys):
        hparams['force_custom_moe'] = True

    data_dir = args.data_dir or ckpt_args.get('data_dir', './domainbed/data')
    dataset_name = args.dataset or ckpt_args['dataset']
    algorithm_name = _resolve_algorithm_name(args.algorithm, ckpt_args['algorithm'])

    if args.test_env is not None:
        test_envs = [args.test_env]
    elif args.test_envs is not None:
        test_envs = args.test_envs
    else:
        test_envs = ckpt_args.get('test_envs', [0])

    dataset = vars(datasets)[dataset_name](data_dir, test_envs, hparams)
    train_envs = args.train_envs
    if train_envs is None:
        train_envs = ckpt_args.get('train_envs')
    if train_envs is None:
        train_envs = [i for i in range(len(dataset)) if i not in test_envs]

    split_args = argparse.Namespace(
        holdout_fraction=ckpt_args.get('holdout_fraction', 0.2),
        uda_holdout_fraction=ckpt_args.get('uda_holdout_fraction', 0),
        test_envs=test_envs,
        trial_seed=ckpt_args.get('trial_seed', 0),
    )
    in_splits, out_splits = _split_dataset(dataset, split_args)
    selected_splits = out_splits if args.split in ('source_val', 'out') else in_splits

    algorithm_class = algorithms.get_algorithm_class(algorithm_name)
    algorithm = algorithm_class(
        ckpt['model_input_shape'],
        ckpt['model_num_classes'],
        ckpt.get('model_num_domains', len(train_envs)),
        hparams,
    )
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.to(args.device)
    algorithm.eval()

    probs, labels, domains = [], [], []
    with torch.no_grad():
        for compact_domain, env_i in enumerate(train_envs):
            loader = FastDataLoader(
                dataset=selected_splits[env_i],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            for batch in loader:
                x, y = _batch_to_xy(batch)
                if args.max_examples is not None:
                    remaining = args.max_examples - sum(len(a) for a in labels)
                    if remaining <= 0:
                        break
                    x = x[:remaining]
                    y = y[:remaining]

                x = x.to(args.device)
                logits, aux = algorithm.predict(x, return_router=True)
                pi = aux.get('router_probs') if isinstance(aux, dict) else None
                if pi is None:
                    if args.allow_missing_router:
                        continue
                    raise RuntimeError(
                        'No router_probs were returned. Use a MESSI/GMoE '
                        'checkpoint with routing exposed.'
                    )

                probs.append(pi.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
                domains.append(np.full(y.numel(), compact_domain, dtype=np.int64))

            if args.max_examples is not None and sum(len(a) for a in labels) >= args.max_examples:
                break

    if not probs:
        raise RuntimeError('No routing batches were collected.')

    router_probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0).astype(np.int64)
    domains_compact = np.concatenate(domains, axis=0).astype(np.int64)
    alpha = args.alpha if args.alpha is not None else hparams.get('alpha', 4.0)

    rho, counts, domain_values, class_values = mean_router_mass(
        router_probs,
        labels,
        domains_compact,
        num_classes=ckpt['model_num_classes'],
        domain_values=np.arange(len(train_envs)),
    )
    a = pairwise_responsibility(rho, counts=counts, alpha=alpha)
    A = compute_A_m(a, mask_diagonal=True, normalize=None)
    checks = sanity_check_responsibility(a, A)
    rows, aggregate = responsibility_summary(A, threshold=args.threshold)

    os.makedirs(args.output_dir, exist_ok=True)
    raw_path = os.path.join(args.output_dir, 'responsibility_raw.npz')
    np.savez(
        raw_path,
        router_probs=router_probs,
        labels=labels,
        domains=domains_compact,
        rho=rho,
        counts=counts,
        a=a,
        A=A,
        domain_values=domain_values,
        domain_envs=np.asarray(train_envs, dtype=np.int64),
        class_values=class_values,
        alpha=np.asarray(alpha, dtype=np.float64),
    )

    summary_csv = os.path.join(args.output_dir, 'optional_responsibility_summary.csv')
    _write_summary_csv(summary_csv, rows, aggregate)

    meta = {
        'checkpoint': args.checkpoint,
        'dataset': dataset_name,
        'algorithm': algorithm_name,
        'split': args.split,
        'test_envs': test_envs,
        'source_envs': train_envs,
        'num_examples': int(labels.shape[0]),
        'num_experts': int(router_probs.shape[1]),
        'num_source_domains': int(len(train_envs)),
        'num_classes': int(ckpt['model_num_classes']),
        'alpha': float(alpha),
        'sanity_checks': checks,
        **aggregate,
    }
    with open(os.path.join(args.output_dir, 'responsibility_raw_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

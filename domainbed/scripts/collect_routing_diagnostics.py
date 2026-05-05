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
from domainbed.lib.routing_metrics import sanity_check_routing


def _load_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _split_dataset(dataset, args, hparams):
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


def _batch_to_xy(batch):
    if isinstance(batch, dict):
        return batch['x'], batch['y']
    return batch


def main():
    parser = argparse.ArgumentParser(description='Collect raw MoE routing outputs.')
    parser.add_argument('--checkpoint', required=True, help='Path to DomainBed model.pkl.')
    parser.add_argument('--output_dir', default=None)
    parser.add_argument('--split', choices=['out', 'in'], default='out',
                        help='Held-out target split is DomainBed out by default.')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--target_envs', type=int, nargs='*', default=None,
                        help='Override checkpoint test_envs when collecting.')
    parser.add_argument('--max_examples', type=int, default=None,
                        help='Optional cap for quick smoke tests.')
    parser.add_argument('--allow_missing_router', action='store_true')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    ckpt = _load_checkpoint(args.checkpoint, args.device)
    ckpt_args = ckpt['args']
    hparams = ckpt['model_hparams']

    data_dir = ckpt_args.get('data_dir', './domainbed/data')
    dataset_name = ckpt_args['dataset']
    test_envs = args.target_envs or ckpt_args.get('test_envs', [0])
    hparams = dict(hparams)
    model_keys = ckpt['model_dict'].keys()
    if any('.mlp.gate_proj.weight' in key for key in model_keys):
        hparams['force_custom_moe'] = True

    dataset = vars(datasets)[dataset_name](data_dir, test_envs, hparams)
    split_args = argparse.Namespace(
        holdout_fraction=ckpt_args.get('holdout_fraction', 0.2),
        uda_holdout_fraction=ckpt_args.get('uda_holdout_fraction', 0),
        test_envs=test_envs,
        trial_seed=ckpt_args.get('trial_seed', 0),
    )
    in_splits, out_splits = _split_dataset(dataset, split_args, hparams)
    selected_splits = out_splits if args.split == 'out' else in_splits

    algorithm_class = algorithms.get_algorithm_class(ckpt_args['algorithm'])
    algorithm = algorithm_class(
        ckpt['model_input_shape'],
        ckpt['model_num_classes'],
        ckpt['model_num_domains'],
        hparams,
    )
    algorithm.load_state_dict(ckpt['model_dict'])
    algorithm.to(args.device)
    algorithm.eval()

    probs, labels, domains, preds, correct = [], [], [], [], []
    with torch.no_grad():
        for env_i in test_envs:
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
                y = y.to(args.device)
                logits, aux = algorithm.predict(x, return_router=True)
                pi = aux.get('router_probs') if isinstance(aux, dict) else None
                if pi is None:
                    if args.allow_missing_router:
                        continue
                    raise RuntimeError(
                        'No router_probs were returned. Use a checkpoint with '
                        'ExplicitMoEHead or custom MoE routing exposed.'
                    )
                pred = logits.argmax(dim=1)
                probs.append(pi.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
                domains.append(np.full(y.numel(), env_i, dtype=np.int64))
                preds.append(pred.detach().cpu().numpy())
                correct.append(pred.eq(y).detach().cpu().numpy())
            if args.max_examples is not None and sum(len(a) for a in labels) >= args.max_examples:
                break

    if not probs:
        raise RuntimeError('No routing batches were collected.')

    arrays = {
        'probs': np.concatenate(probs, axis=0),
        'labels': np.concatenate(labels, axis=0).astype(np.int64),
        'domains': np.concatenate(domains, axis=0).astype(np.int64),
        'preds': np.concatenate(preds, axis=0).astype(np.int64),
        'correct': np.concatenate(correct, axis=0).astype(bool),
    }

    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'routing_raw.npz'), **arrays)

    checks = sanity_check_routing(arrays['probs'])
    meta = {
        'checkpoint': args.checkpoint,
        'dataset': dataset_name,
        'algorithm': ckpt_args['algorithm'],
        'split': args.split,
        'target_envs': test_envs,
        'num_examples': int(arrays['labels'].shape[0]),
        'num_experts': int(arrays['probs'].shape[1]),
        'sanity_checks': checks,
    }
    with open(os.path.join(output_dir, 'routing_raw_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2, sort_keys=True)

    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()

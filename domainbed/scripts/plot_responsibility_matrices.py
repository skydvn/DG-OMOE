import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np

from domainbed import datasets
from domainbed.lib.responsibility_metrics import compute_A_m


_DOMAIN_NAME_OVERRIDES = {
    'PACS': ['A', 'C', 'P', 'S'],
    'DomainNet': ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch'],
}


def _domain_names(dataset_name, domain_envs=None, override=None):
    if override:
        names = list(override)
    elif dataset_name in _DOMAIN_NAME_OVERRIDES:
        names = _DOMAIN_NAME_OVERRIDES[dataset_name]
    elif dataset_name in vars(datasets):
        names = list(vars(datasets)[dataset_name].ENVIRONMENTS)
    else:
        names = None

    if names is None:
        size = len(domain_envs) if domain_envs is not None else 0
        return [f'd{i}' for i in range(size)]

    if domain_envs is not None:
        selected = []
        for env_i in domain_envs:
            env_i = int(env_i)
            if 0 <= env_i < len(names):
                selected.append(names[env_i])
            else:
                selected.append(f'd{env_i}')
        return selected
    return names


def _load_A(npz, normalize):
    if 'A' in npz and normalize in (None, 'none'):
        A = np.asarray(npz['A'], dtype=np.float64).copy()
        for i in range(A.shape[1]):
            A[:, i, i] = np.nan
        return A
    if 'a' not in npz:
        raise ValueError('Input must contain either A or a.')
    return compute_A_m(npz['a'], mask_diagonal=True, normalize=normalize)


def main():
    parser = argparse.ArgumentParser(
        description='Plot expert-specific responsibility matrices.')
    parser.add_argument('--input', required=True,
                        help='Path to responsibility_raw.npz.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output', required=True,
                        help='Output PDF path.')
    parser.add_argument('--domain_names', nargs='*', default=None)
    parser.add_argument('--normalize', choices=['none', 'global', 'per_expert'],
                        default='none')
    parser.add_argument('--title_prefix', default='Expert')
    parser.add_argument('--cbar_label', default=r'$A^{(m)}_{ij}$')
    args = parser.parse_args()

    arrays = np.load(args.input)
    normalize = None if args.normalize == 'none' else args.normalize
    A = _load_A(arrays, normalize)

    domain_envs = arrays['domain_envs'] if 'domain_envs' in arrays else None
    names = _domain_names(args.dataset, domain_envs=domain_envs,
                          override=args.domain_names)
    if len(names) != A.shape[1]:
        names = [f'd{i}' for i in range(A.shape[1])]

    num_experts = A.shape[0]
    if num_experts == 6:
        rows, cols = 2, 3
    else:
        cols = int(math.ceil(math.sqrt(num_experts)))
        rows = int(math.ceil(num_experts / cols))

    vmax = np.nanmax(A)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='#f2f2f2')

    fig_w = max(3.0 * cols, 1.15 * A.shape[1] * cols)
    fig_h = max(3.2 * rows, 1.2 * A.shape[1] * rows)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), squeeze=False)
    fig.subplots_adjust(hspace=0.48, wspace=0.24)

    im = None
    for m, ax in enumerate(axes.ravel()):
        if m >= num_experts:
            ax.axis('off')
            continue
        im = ax.imshow(A[m], vmin=0.0, vmax=vmax, interpolation='nearest',
                       cmap=cmap)
        ax.set_title(f'{args.title_prefix} {m}')
        ax.set_xticks(np.arange(A.shape[2]))
        ax.set_yticks(np.arange(A.shape[1]))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        row = m // cols
        col = m % cols
        ax.set_xlabel('source domain' if row == rows - 1 else '')
        ax.set_ylabel('source domain' if col == 0 else '')
        ax.tick_params(axis='both', length=0)

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
        cbar.set_label(args.cbar_label)

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    fig.savefig(args.output, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

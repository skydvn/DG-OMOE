import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib.pyplot as plt
import numpy as np

from domainbed.lib.routing_metrics import p_e_given_d, p_e_given_d_y, p_e_given_y


def _labels(prefix, values, names=None):
    out = []
    for value in values:
        value = int(value)
        if names and 0 <= value < len(names):
            out.append(str(names[value]))
        else:
            out.append(f'{prefix}{value}')
    return out


def _heatmap(matrix, row_labels, title, path):
    fig_w = max(5.0, matrix.shape[1] * 0.65)
    fig_h = max(2.5, matrix.shape[0] * 0.35)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix, aspect='auto', interpolation='nearest', vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel('Expert')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([f'e{i}' for i in range(matrix.shape[1])])
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Plot routing diagnostic heatmaps.')
    parser.add_argument('--routing_raw', required=True, nargs='+',
                        help='One or more routing_raw.npz files.')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--class_names', nargs='*', default=None)
    parser.add_argument('--domain_names', nargs='*', default=None)
    parser.add_argument('--max_classes', type=int, default=20)
    args = parser.parse_args()

    arrays = [np.load(path) for path in args.routing_raw]
    probs = np.concatenate([a['probs'] for a in arrays], axis=0)
    labels = np.concatenate([a['labels'] for a in arrays], axis=0)
    domains = np.concatenate([a['domains'] for a in arrays], axis=0)
    os.makedirs(args.output_dir, exist_ok=True)

    ped, domain_values, _ = p_e_given_d(probs, domains)
    _heatmap(
        ped,
        _labels('d', domain_values, args.domain_names),
        'P(e|d)',
        os.path.join(args.output_dir, 'p_e_given_d.pdf'),
    )

    class_values = np.unique(labels)
    if len(class_values) > args.max_classes:
        counts = np.array([(labels == c).sum() for c in class_values])
        class_values = class_values[np.argsort(counts)[::-1][:args.max_classes]]
        class_values = np.sort(class_values)

    pey, class_values, _ = p_e_given_y(probs, labels, class_values)
    _heatmap(
        pey,
        _labels('y', class_values, args.class_names),
        'P(e|y)',
        os.path.join(args.output_dir, 'p_e_given_y.pdf'),
    )

    pedy, domain_values, class_values, _ = p_e_given_d_y(
        probs, domains, labels, class_values=class_values)
    rows = []
    matrix = []
    domain_labels = _labels('d', domain_values, args.domain_names)
    class_labels = _labels('y', class_values, args.class_names)
    for i, d_name in enumerate(domain_labels):
        for j, y_name in enumerate(class_labels):
            if not np.isnan(pedy[i, j]).all():
                rows.append(f'{d_name}/{y_name}')
                matrix.append(pedy[i, j])
    _heatmap(
        np.stack(matrix, axis=0),
        rows,
        'P(e|d,y)',
        os.path.join(args.output_dir, 'p_e_given_d_y.pdf'),
    )


if __name__ == '__main__':
    main()

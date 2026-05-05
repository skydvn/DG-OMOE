import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from domainbed.lib.routing_metrics import compute_routing_metrics


METRIC_COLUMNS = ['routing_entropy', 'load_std', 'routing_js', 'accuracy']
DISPLAY_COLUMNS = {
    'routing_entropy': 'Routing Entropy',
    'load_std': 'Load Std.',
    'routing_js': 'Routing JS',
    'accuracy': 'Accuracy',
}


def _fmt(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size > 1:
        return f'{values.mean():.4f} +/- {values.std(ddof=1):.4f}'
    return f'{values.mean():.4f}'


def _latex_table(rows):
    headers = ['Method'] + [DISPLAY_COLUMNS[c] for c in METRIC_COLUMNS]
    lines = [
        r'\begin{tabular}{lrrrr}',
        r'\toprule',
        ' & '.join(headers) + r' \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(' & '.join([row['method']] + [row[c] for c in METRIC_COLUMNS]) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', ''])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Create routing diagnostics table.')
    parser.add_argument('--routing_raw', required=True, nargs='+',
                        help='routing_raw.npz files. Use --method per file or parent dir names.')
    parser.add_argument('--method', nargs='*', default=None,
                        help='Method label for each routing_raw file.')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    if args.method is not None and len(args.method) != len(args.routing_raw):
        raise ValueError('--method must have one label per --routing_raw path.')

    grouped = defaultdict(list)
    for i, path in enumerate(args.routing_raw):
        label = args.method[i] if args.method else os.path.basename(os.path.dirname(path))
        data = np.load(path)
        metrics = compute_routing_metrics(
            data['probs'], data['labels'], data['domains'], data['correct'])
        grouped[label].append(metrics)

    rows = []
    for method in sorted(grouped):
        row = {'method': method}
        for column in METRIC_COLUMNS:
            row[column] = _fmt([m[column] for m in grouped[method]])
        rows.append(row)

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'routing_table.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method'] + METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    tex_path = os.path.join(args.output_dir, 'routing_table.tex')
    with open(tex_path, 'w') as f:
        f.write(_latex_table(rows))

    print(f'Wrote {csv_path}')
    print(f'Wrote {tex_path}')


if __name__ == '__main__':
    main()

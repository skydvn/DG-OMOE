import itertools

import numpy as np


EPS = 1e-12


def _as_int_array(x):
    return np.asarray(x).astype(np.int64)


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def mean_router_mass(router_probs, labels, domains, num_classes=None,
                     domain_values=None):
    """
    Compute rho[m, d, c] = E[pi_m(x) | domain=d, class=c].

    Args:
        router_probs: [N, M] routing probabilities.
        labels: [N] integer class labels.
        domains: [N] compact or raw domain ids.
        num_classes: optional total class count. Defaults to max(labels)+1.
        domain_values: optional ordered domain ids to include.

    Returns:
        rho: [M, D, C]
        counts: [D, C]
        domain_values: [D]
        class_values: [C]
    """
    pi = np.asarray(router_probs, dtype=np.float64)
    labels = _as_int_array(labels)
    domains = _as_int_array(domains)
    if pi.ndim != 2:
        raise ValueError(f'router_probs must be [N, M], got {pi.shape}')
    if labels.shape[0] != pi.shape[0] or domains.shape[0] != pi.shape[0]:
        raise ValueError('router_probs, labels, and domains must have the same N')

    if domain_values is None:
        domain_values = np.unique(domains)
    domain_values = _as_int_array(domain_values)

    if num_classes is None:
        num_classes = int(labels.max()) + 1 if labels.size else 0
    class_values = np.arange(num_classes, dtype=np.int64)

    num_experts = pi.shape[1]
    rho = np.zeros((num_experts, len(domain_values), num_classes),
                   dtype=np.float64)
    counts = np.zeros((len(domain_values), num_classes), dtype=np.int64)

    for d_idx, domain in enumerate(domain_values):
        mask_d = domains == domain
        for c in class_values:
            mask = mask_d & (labels == c)
            counts[d_idx, c] = int(mask.sum())
            if counts[d_idx, c] > 0:
                rho[:, d_idx, c] = pi[mask].mean(axis=0)

    return rho, counts, domain_values, class_values


def pairwise_responsibility(rho, counts=None, alpha=4.0, zero_diagonal=True):
    """
    Compute a[m, i, j, c] from rho[m, i, c].

    The training loss uses sigmoid(alpha * rho_i) * sigmoid(alpha * rho_j).
    Missing domain/class cells are assigned zero responsibility because no
    alignment pair exists for that cell.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 3:
        raise ValueError(f'rho must be [M, D, C], got {rho.shape}')

    left = sigmoid(alpha * rho)[:, :, None, :]
    right = sigmoid(alpha * rho)[:, None, :, :]
    a = left * right

    if counts is not None:
        counts = np.asarray(counts)
        if counts.shape != rho.shape[1:]:
            raise ValueError(f'counts must be [D, C], got {counts.shape}')
        valid = (counts > 0)
        pair_valid = valid[None, :, None, :] & valid[None, None, :, :]
        a = np.where(pair_valid, a, 0.0)

    if zero_diagonal:
        for i in range(rho.shape[1]):
            a[:, i, i, :] = 0.0
    return a


def pairwise_weight(rho, counts=None, alpha=4.0, zero_diagonal=True):
    """Alias matching older plan/prototype names."""
    return pairwise_responsibility(rho, counts=counts, alpha=alpha,
                                   zero_diagonal=zero_diagonal)


def compute_A_m(a, mask_diagonal=True, normalize=None, eps=EPS):
    """
    Average a[m, i, j, c] over classes to get A[m, i, j].

    normalize:
        None/'none'  - raw values
        'global'     - divide all experts by the global off-diagonal max
        'per_expert' - divide each expert by its own off-diagonal max
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 4:
        raise ValueError(f'a must be [M, D, D, C], got {a.shape}')
    A = a.mean(axis=-1)

    if mask_diagonal:
        for i in range(A.shape[1]):
            A[:, i, i] = np.nan

    if normalize in (None, 'none'):
        return A
    if normalize == 'global':
        denom = np.nanmax(A)
        return A / (denom + eps)
    if normalize == 'per_expert':
        out = A.copy()
        for m in range(out.shape[0]):
            denom = np.nanmax(out[m])
            out[m] = out[m] / (denom + eps)
        return out
    raise ValueError(f'Unknown normalize mode: {normalize}')


def responsibility_summary(A, threshold=1e-6):
    """Per-expert mean/max/std plus aggregate diversity/activity metrics."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 3:
        raise ValueError(f'A must be [M, D, D], got {A.shape}')

    rows = []
    means = []
    flats = []
    for m in range(A.shape[0]):
        vals = A[m][~np.isnan(A[m])]
        mean = float(vals.mean()) if vals.size else 0.0
        means.append(mean)
        rows.append({
            'expert': int(m),
            'mean': mean,
            'max': float(vals.max()) if vals.size else 0.0,
            'std': float(vals.std()) if vals.size else 0.0,
            'sparsity': float((vals < threshold).mean()) if vals.size else 1.0,
            'active': bool(mean > threshold),
        })
        flats.append(np.nan_to_num(A[m], nan=0.0).reshape(-1))

    distances = []
    for i, j in itertools.combinations(range(A.shape[0]), 2):
        distances.append(float(np.linalg.norm(flats[i] - flats[j])))

    aggregate = {
        'responsibility_sparsity': float(np.mean([r['sparsity'] for r in rows])),
        'expert_responsibility_diversity': (
            float(np.mean(distances)) if distances else 0.0
        ),
        'active_expert_ratio': float(np.mean([r['active'] for r in rows])),
    }
    return rows, aggregate


def sanity_check_responsibility(a, A, allow_masked_diagonal=True):
    a = np.asarray(a, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    if a.ndim != 4:
        raise ValueError(f'a must be [M, D, D, C], got {a.shape}')
    if A.shape != a.shape[:3]:
        raise ValueError(f'A shape {A.shape} does not match a[:3] {a.shape[:3]}')

    offdiag = ~np.eye(A.shape[1], dtype=bool)[None, :, :]
    A_offdiag = A[offdiag.repeat(A.shape[0], axis=0)].reshape(A.shape[0], -1)
    checks = {
        'a_nonnegative': bool((a >= -1e-12).all()),
        'A_shape_is_MDD': True,
        'A_symmetric': bool(np.allclose(
            np.nan_to_num(A), np.nan_to_num(np.swapaxes(A, 1, 2)), atol=1e-8)),
        'some_offdiag_nonzero': bool((A_offdiag > 0).any()),
    }

    if allow_masked_diagonal:
        diag = np.stack([A[:, i, i] for i in range(A.shape[1])], axis=1)
        non_diag = A[:, offdiag[0]]
        checks['nan_only_on_diagonal'] = bool(
            np.isnan(non_diag).sum() == 0 and np.isnan(diag).all())
    else:
        checks['no_nan'] = bool(not np.isnan(A).any())

    return checks

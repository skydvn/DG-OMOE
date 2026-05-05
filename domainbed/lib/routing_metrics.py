import math

import numpy as np


EPS = 1e-12


def _as_int_array(x):
    return np.asarray(x).astype(np.int64)


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    denom = x.sum(axis=1, keepdims=True)
    return x / np.clip(denom, EPS, None)


def _means_by_group(probs, group_ids, group_values=None):
    probs = np.asarray(probs, dtype=np.float64)
    group_ids = _as_int_array(group_ids)
    if group_values is None:
        group_values = np.unique(group_ids)
    group_values = _as_int_array(group_values)

    out = np.full((len(group_values), probs.shape[1]), np.nan, dtype=np.float64)
    counts = np.zeros(len(group_values), dtype=np.int64)
    for i, value in enumerate(group_values):
        mask = group_ids == value
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            out[i] = probs[mask].mean(axis=0)
    return out, group_values, counts


def p_e_given_d(probs, domains, domain_values=None):
    return _means_by_group(probs, domains, domain_values)


def p_e_given_y(probs, labels, class_values=None):
    return _means_by_group(probs, labels, class_values)


def p_e_given_d_y(probs, domains, labels, domain_values=None, class_values=None):
    probs = np.asarray(probs, dtype=np.float64)
    domains = _as_int_array(domains)
    labels = _as_int_array(labels)
    if domain_values is None:
        domain_values = np.unique(domains)
    if class_values is None:
        class_values = np.unique(labels)
    domain_values = _as_int_array(domain_values)
    class_values = _as_int_array(class_values)

    out = np.full(
        (len(domain_values), len(class_values), probs.shape[1]),
        np.nan,
        dtype=np.float64,
    )
    counts = np.zeros((len(domain_values), len(class_values)), dtype=np.int64)
    for i, domain in enumerate(domain_values):
        for j, label in enumerate(class_values):
            mask = (domains == domain) & (labels == label)
            counts[i, j] = int(mask.sum())
            if counts[i, j] > 0:
                out[i, j] = probs[mask].mean(axis=0)
    return out, domain_values, class_values, counts


def entropy(probs, normalize=True):
    probs = np.asarray(probs, dtype=np.float64)
    ent = -(probs * np.log(np.clip(probs, EPS, None))).sum(axis=1)
    if normalize and probs.shape[1] > 1:
        ent = ent / math.log(probs.shape[1])
    return ent


def load_std(probs):
    probs = np.asarray(probs, dtype=np.float64)
    return float(probs.mean(axis=0).std())


def js_divergence(distributions):
    p = _normalize_rows(distributions)
    m = p.mean(axis=0)
    return float((p * (np.log(np.clip(p, EPS, None)) - np.log(np.clip(m, EPS, None)))).sum(axis=1).mean())


def routing_js(probs, domains, labels):
    """Class-conditional routing variation across domains."""
    probs = np.asarray(probs, dtype=np.float64)
    domains = _as_int_array(domains)
    labels = _as_int_array(labels)

    values = []
    for label in np.unique(labels):
        class_mask = labels == label
        per_domain = []
        for domain in np.unique(domains):
            mask = class_mask & (domains == domain)
            if mask.any():
                per_domain.append(probs[mask].mean(axis=0))
        if len(per_domain) > 1:
            values.append(js_divergence(np.stack(per_domain, axis=0)))
    if not values:
        return 0.0
    return float(np.mean(values))


def compute_routing_metrics(probs, labels, domains, correct, normalize_entropy=True):
    probs = np.asarray(probs, dtype=np.float64)
    labels = _as_int_array(labels)
    domains = _as_int_array(domains)
    correct = np.asarray(correct).astype(bool)

    return {
        'routing_entropy': float(entropy(probs, normalize=normalize_entropy).mean()),
        'load_std': load_std(probs),
        'routing_js': routing_js(probs, domains, labels),
        'accuracy': float(correct.mean()),
    }


def sanity_check_routing(probs, normalized_entropy=True):
    probs = np.asarray(probs, dtype=np.float64)
    checks = {
        'no_nan': bool(not np.isnan(probs).any()),
        'row_sum_close': bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)),
        'nonnegative': bool((probs >= -1e-8).all()),
    }
    ent = entropy(probs, normalize=normalized_entropy)
    checks['entropy_in_range'] = bool((ent >= -1e-8).all() and (ent <= 1.0 + 1e-8).all())
    return checks

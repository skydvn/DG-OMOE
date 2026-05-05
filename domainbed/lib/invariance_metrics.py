import itertools
import math

import numpy as np


EPS = 1e-12


def _as_array(x, dtype=None):
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _sq_euclidean(x, y):
    x = _as_array(x, np.float64)
    y = _as_array(y, np.float64)
    return (
        np.sum(x * x, axis=1, keepdims=True)
        + np.sum(y * y, axis=1, keepdims=True).T
        - 2.0 * np.matmul(x, y.T)
    )


def mean_feature_distance(x, y):
    """Squared L2 distance between class/domain feature means."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    diff = np.mean(x, axis=0) - np.mean(y, axis=0)
    return float(np.dot(diff, diff))


def coral_distance(x, y):
    """CORAL-style mean + covariance discrepancy."""
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x = _as_array(x, np.float64)
    y = _as_array(y, np.float64)
    mean_term = mean_feature_distance(x, y)
    cx = np.cov(x, rowvar=False)
    cy = np.cov(y, rowvar=False)
    cov_term = np.mean((cx - cy) ** 2)
    return float(mean_term + cov_term)


def mmd_rbf_distance(x, y, sigmas=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Biased multi-bandwidth RBF MMD^2."""
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x = _as_array(x, np.float64)
    y = _as_array(y, np.float64)
    dxx = _sq_euclidean(x, x)
    dyy = _sq_euclidean(y, y)
    dxy = _sq_euclidean(x, y)
    value = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * float(sigma) ** 2)
        value += (
            np.exp(-gamma * dxx).mean()
            + np.exp(-gamma * dyy).mean()
            - 2.0 * np.exp(-gamma * dxy).mean()
        )
    return float(max(value / len(sigmas), 0.0))


def energy_distance(x, y):
    if len(x) == 0 or len(y) == 0:
        return np.nan
    x = _as_array(x, np.float64)
    y = _as_array(y, np.float64)
    dxx = np.sqrt(np.maximum(_sq_euclidean(x, x), 0.0))
    dyy = np.sqrt(np.maximum(_sq_euclidean(y, y), 0.0))
    dxy = np.sqrt(np.maximum(_sq_euclidean(x, y), 0.0))
    return float(max(2.0 * dxy.mean() - dxx.mean() - dyy.mean(), 0.0))


def pairwise_class_conditional_discrepancy(
    z,
    y,
    d,
    distance="mean",
    min_count=5,
    class_values=None,
    domain_values=None,
    mmd_sigmas=(1.0, 2.0, 4.0, 8.0, 16.0),
):
    """Average discrepancy over valid source-domain pairs and classes."""
    z = _as_array(z, np.float64)
    y = _as_array(y, np.int64)
    d = _as_array(d, np.int64)
    if class_values is None:
        class_values = np.unique(y)
    if domain_values is None:
        domain_values = np.unique(d)

    distance_fns = {
        "mean": mean_feature_distance,
        "coral": coral_distance,
        "mmd": lambda a, b: mmd_rbf_distance(a, b, mmd_sigmas),
        "ed": energy_distance,
        "energy": energy_distance,
    }
    if distance not in distance_fns:
        raise ValueError("distance must be one of {}".format(sorted(distance_fns)))
    distance_fn = distance_fns[distance]

    values = []
    details = []
    for cls in class_values:
        for di, dj in itertools.combinations(domain_values, 2):
            mask_i = (y == cls) & (d == di)
            mask_j = (y == cls) & (d == dj)
            ni = int(mask_i.sum())
            nj = int(mask_j.sum())
            if ni < min_count or nj < min_count:
                continue
            value = distance_fn(z[mask_i], z[mask_j])
            if np.isfinite(value):
                values.append(value)
                details.append((int(di), int(dj), int(cls), ni, nj, float(value)))

    return {
        "pairwise_discrepancy": float(np.mean(values)) if values else math.nan,
        "num_terms": int(len(values)),
        "details": details,
    }


def class_conditional_domain_probe_accuracy(
    z,
    y,
    d,
    min_count=5,
    test_size=0.3,
    seed=0,
    max_iter=1000,
):
    """Train one frozen-feature logistic probe per class and average accuracy."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError("scikit-learn is required for domain probes") from exc

    z = _as_array(z, np.float64)
    y = _as_array(y, np.int64)
    d = _as_array(d, np.int64)

    accs = []
    details = []
    for cls in np.unique(y):
        mask = y == cls
        domains, counts = np.unique(d[mask], return_counts=True)
        valid_domains = domains[counts >= min_count]
        class_mask = mask & np.isin(d, valid_domains)
        if len(valid_domains) < 2 or int(class_mask.sum()) < 2 * len(valid_domains):
            continue
        zc = z[class_mask]
        dc = d[class_mask]
        stratify = dc if np.min(np.unique(dc, return_counts=True)[1]) >= 2 else None
        z_train, z_test, d_train, d_test = train_test_split(
            zc,
            dc,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=max_iter, class_weight="balanced"),
        )
        probe.fit(z_train, d_train)
        acc = float(probe.score(z_test, d_test))
        accs.append(acc)
        details.append({
            "class": int(cls),
            "accuracy": acc,
            "num_examples": int(class_mask.sum()),
            "num_domains": int(len(valid_domains)),
        })

    return {
        "domain_probe_acc": float(np.mean(accs)) if accs else math.nan,
        "num_classes": int(len(accs)),
        "details": details,
    }

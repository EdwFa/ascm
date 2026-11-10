"""Cluster bootstrap by document, paired sign-flip permutation test, Holm correction."""
from __future__ import annotations
import itertools
import numpy as np


def cluster_bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0):
    """values: per-document metric values. Returns (mean, lo, hi)."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    m = v[idx].mean(axis=1)
    return float(v.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_permutation_p(diffs, n_perm=100000, seed=0):
    """Two-sided sign-flip test on per-document differences. Exact when 2^n <= n_perm."""
    d = np.asarray(diffs, float)
    n = len(d)
    obs = abs(d.mean())
    if 2 ** n <= n_perm:
        signs = np.array(list(itertools.product([-1, 1], repeat=n)))
    else:
        signs = np.random.default_rng(seed).choice([-1, 1], size=(n_perm, n))
    stat = np.abs((signs * d).mean(axis=1))
    return float((stat >= obs - 1e-12).mean())


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()

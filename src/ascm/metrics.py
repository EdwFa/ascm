"""Chunk-level metrics that need no questions."""
from __future__ import annotations
import numpy as np


def chunks(starts, n):
    b = list(starts) + [n]
    return [(b[k], b[k + 1]) for k in range(len(starts))]


def chunk_of(starts, n):
    """sentence index -> chunk index."""
    out = np.zeros(n, dtype=int)
    for k, (a, b) in enumerate(chunks(starts, n)):
        out[a:b] = k
    return out


def size_stats(starts, tokens, t_max, atoms_exceptions=0):
    sz = np.array([sum(tokens[a:b]) for a, b in chunks(starts, len(tokens))])
    return dict(n_chunks=len(sz), mean_tok=float(sz.mean()), median_tok=float(np.median(sz)),
                max_tok=int(sz.max()), over_tmax=int((sz > t_max).sum()),
                small_lt64=float((sz < 64).mean()))


def erl(starts, pairs, n):
    """Share of recommendation-LoC/LoR pairs kept in one chunk: adjacent sentence pair, and the whole recommendation block."""
    if not pairs:
        return dict(erl_adjacent=float('nan'), erl_block=float('nan'))
    c = chunk_of(starts, n)
    adj = np.mean([c[p['adjacent']] == c[p['loc']] for p in pairs])
    blk = np.mean([all(c[k] == c[p['loc']] for k in p['block']) for p in pairs])
    return dict(erl_adjacent=float(adj), erl_block=float(blk))


def list_split_rate(starts, why):
    """Share of list-continuity protected boundaries that were cut."""
    idx = [j for j, w in enumerate(why) if w == 'list']
    if not idx:
        return float('nan')
    s = set(starts)
    return float(np.mean([j in s for j in idx]))


def f_violations(starts, prot):
    """Number of cuts placed on protected boundaries (must be 0 for ASCM)."""
    return int(sum(1 for j in starts if j < len(prot) and prot[j]))


def scs(starts, E_eval):
    """Semantic coherence: mean cosine of sentences to the chunk centroid, averaged over sentences."""
    tot, cnt = 0.0, 0
    for a, b in chunks(starts, len(E_eval)):
        m = E_eval[a:b].mean(axis=0)
        nm = np.linalg.norm(m)
        if nm < 1e-9:
            continue
        m = m / nm
        tot += float((E_eval[a:b] @ m).sum())
        cnt += b - a
    return tot / max(cnt, 1)

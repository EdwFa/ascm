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
                p5_tok=float(np.percentile(sz, 5)), p95_tok=float(np.percentile(sz, 95)),
                cv_tok=float(sz.std() / sz.mean()), max_tok=int(sz.max()), over_tmax=int((sz > t_max).sum()),
                small_lt64=float((sz < 64).mean()), small_lt100=float((sz < 100).mean()))


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


def structure_metrics(starts, S):
    """sec_cross: share of chunks that contain a section start (depth <= 2) other than at their first sentence
    (a chunk that straddles two sections; lower is better).
    section_cut: share of chunk starts that are section starts (higher = cuts follow the document structure).
    comment_sep: share of 'Комментарии' blocks that begin a chunk, i.e. were separated from their recommendation.
    These depend on the same structure signal as the hierarchical cost, so they are partly optimised by construction."""
    n = len(S)
    bounds = chunks(starts, n)
    sec = [i for i, s in enumerate(S) if 0 < s.get('sec_level', 0) <= 2]
    cross = sum(1 for a, b in bounds if any(a < i < b for i in sec))
    sset = set(sec)
    cuts = [a for a, _ in bounds if a > 0]
    com = [i for i, s in enumerate(S) if s.get('comment') and i > 0]
    cset = set(starts)
    return dict(sec_cross=cross / max(len(bounds), 1),
                section_cut=(sum(1 for a in cuts if a in sset) / len(cuts)) if cuts else float('nan'),
                comment_sep=(sum(1 for i in com if i in cset) / len(com)) if com else float('nan'))

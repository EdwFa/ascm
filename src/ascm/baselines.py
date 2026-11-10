"""Baselines. A chunking is a sorted list of sentence indices where chunks start (always begins with 0).
All baselines here are sentence-aligned, which favours them over character-level fixed-size splitting.
Not yet implemented: B5 parent-document, B6 TextTiling (listed in the protocol)."""
from __future__ import annotations
import numpy as np


def fixed_size(tokens, tau):
    """B1: accumulate sentences until the chunk reaches tau tokens. Ignores F."""
    starts, cur = [0], 0
    for i, t in enumerate(tokens):
        if cur >= tau:
            starts.append(i)
            cur = 0
        cur += t
    return starts


def recursive(tokens, para, tau):
    """B2: keep whole paragraphs together while they fit; a paragraph larger than tau is split by sentences. Ignores F."""
    n = len(tokens)
    starts, cur = [0], 0
    for i in range(n):
        new_par = i == 0 or para[i] != para[i - 1]
        if i and new_par and cur >= tau * 0.5:
            starts.append(i)
            cur = 0
        elif i and cur + tokens[i] > tau and cur > 0:
            starts.append(i)
            cur = 0
        cur += tokens[i]
    return starts


def header_based(tokens, heading_start, tau, t_max):
    """B4: every heading starts a chunk; oversized sections are split by sentences at tau. Ignores F."""
    starts, cur = [0], 0
    for i, t in enumerate(tokens):
        if i and (heading_start[i] or (cur + t > t_max) or (cur >= tau)):
            starts.append(i)
            cur = 0
        cur += t
    return starts


def semantic(E, tokens, tau, t_max, pct=85.0):
    """B3: cut where the cosine distance between neighbouring sentences exceeds a percentile; cap by t_max. Ignores F."""
    n = len(tokens)
    d = np.array([1 - float(E[i - 1] @ E[i]) for i in range(1, n)])
    thr = np.percentile(d, pct) if len(d) else 1.0
    starts, cur = [0], 0
    for i in range(n):
        if i and ((d[i - 1] > thr and cur >= tau * 0.25) or cur + tokens[i] > t_max):
            starts.append(i)
            cur = 0
        cur += tokens[i]
    return starts


def atom_fill(atom_tokens, tau):
    """Ablation: pack atoms greedily to tau (respects F, no semantics). Returns atom-level starts."""
    starts, cur = [0], 0
    for a, t in enumerate(atom_tokens):
        if cur >= tau:
            starts.append(a)
            cur = 0
        cur += t
    return starts


def header_atoms(atom_tokens, section_start, tau, t_max):
    """B8: header-based chunking that respects F. A chunk starts at every section start; sections larger than tau are packed
    from atoms. Returns atom-level starts."""
    starts, cur = [0], 0
    for a, t in enumerate(atom_tokens):
        if a and (section_start[a] or cur >= tau or (cur + t > t_max and cur > 0)):
            starts.append(a)
            cur = 0
        cur += t
    return starts

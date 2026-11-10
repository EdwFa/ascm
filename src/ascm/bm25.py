"""Tiny BM25 (Okapi) with a crude prefix 'stemmer' (works for Russian inflection and for English). No external dependencies."""
from __future__ import annotations
import math, re
from collections import Counter
import numpy as np

_W = re.compile(r'\w+', re.U)


def tokenize(text: str, stem: int = 6):
    return [w[:stem] if len(w) > stem else w for w in _W.findall(text.lower())]


class BM25:
    def __init__(self, texts, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(t)) for t in texts]
        self.len = np.array([sum(c.values()) for c in self.tf], dtype=float)
        self.avg = max(self.len.mean(), 1.0)
        df = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(texts)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query: str) -> np.ndarray:
        q = set(tokenize(query))
        out = np.zeros(len(self.tf))
        norm = self.k1 * (1 - self.b + self.b * self.len / self.avg)
        for w in q:
            idf = self.idf.get(w)
            if idf is None:
                continue
            f = np.array([c.get(w, 0) for c in self.tf], dtype=float)
            out += idf * f * (self.k1 + 1) / (f + norm)
        return out


def rrf(score_lists, k: int = 60) -> np.ndarray:
    """Reciprocal rank fusion of several score arrays (higher = better)."""
    out = np.zeros(len(score_lists[0]))
    for s in score_lists:
        rank = np.empty(len(s)); rank[np.argsort(-s, kind='stable')] = np.arange(1, len(s) + 1)
        out += 1.0 / (k + rank)
    return out

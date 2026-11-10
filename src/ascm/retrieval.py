"""Retrieval evaluation against a physician-written question set (JSONL).

Question format: {"id", "doc", "type", "question", "answerable": true/false,
                  "gold_spans": [[start, end], ...],   # character offsets in data/processed/<doc>.txt
                  "evidence_span": [start, end] | null}  # the LoC/LoR line for evidence-level questions
"""
from __future__ import annotations
import json
import numpy as np


def load_questions(path):
    return [json.loads(l) for l in open(path, encoding='utf-8') if l.strip()]


def _cover(span, ranges):
    s, e = span
    got = 0
    for a, b in ranges:
        got += max(0, min(e, b) - max(s, a))
    return got / max(e - s, 1)


def evaluate(q, chunk_spans, chunk_tokens, chunk_vecs, q_vec, budget):
    """chunk_spans: [(start,end)] char spans of chunks; returns metrics for one question."""
    order = np.argsort(-(chunk_vecs @ q_vec))
    sel, used = [], 0
    for k in order:
        if used + chunk_tokens[k] > budget and sel:
            break
        sel.append(int(k))
        used += chunk_tokens[k]
    ranges = [chunk_spans[k] for k in sel]
    gold = q['gold_spans']
    out = dict(context_tokens=used)
    if not q.get('answerable', True) or not gold:
        return out
    out['span_recall'] = float(np.mean([_cover(g, ranges) for g in gold]))
    out['hit'] = float(any(_cover(g, ranges) >= 0.5 for g in gold))
    rank = next((r + 1 for r, k in enumerate(order)
                 if any(_cover(g, [chunk_spans[k]]) > 0 for g in gold)), None)
    out['mrr'] = 1.0 / rank if rank else 0.0
    ev = q.get('evidence_span')
    if ev:
        together = any(all(_cover(g, [chunk_spans[k]]) >= 0.9 for g in gold) and _cover(ev, [chunk_spans[k]]) >= 0.9
                       for k in sel)
        out['evidence_linked'] = float(together)
    out['context_precision'] = float(sum(max(0, min(e, b) - max(s, a)) for g in gold for a, b in ranges
                                          for s, e in [g]) / max(sum(b - a for a, b in ranges), 1))
    return out

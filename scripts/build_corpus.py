#!/usr/bin/env python3
"""Raw guidelines -> sentences, protected boundaries, atoms (data/processed) + results/corpus_stats.csv"""
import argparse, glob, os, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ascm import pipeline, tokens, loc

ap = argparse.ArgumentParser()
ap.add_argument('--raw', default='data/raw'); ap.add_argument('--out', default='data/processed')
ap.add_argument('--tokenizer', default='proxy'); ap.add_argument('--tau', type=int, default=512)
a = ap.parse_args()
count = tokens.get_counter(a.tokenizer)
rows, tmax = [], int(1.5 * a.tau)
for f in sorted(glob.glob(os.path.join(a.raw, '*.md'))):
    slug = os.path.basename(f)[:-3]
    raw = open(f, encoding='utf-8').read()
    d = pipeline.process_document(raw, count)
    pipeline.save_document(d, a.out, slug)
    S = d['sentences']; tok = [s['tokens'] for s in S]
    at_tok = [sum(tok[i] for i in at) for at in d['atoms']]
    m, anchors = loc.count_matches(d['text'])
    rows.append(dict(doc=slug, words=len(d['text'].split()), sentences=len(S), tokens=sum(tok), atoms=len(d['atoms']),
                     protected=sum(d['protected']), loc_anchors=anchors, loc_matched=m, pairs=len(d['pairs']),
                     atoms_over_tmax=sum(t > tmax for t in at_tok), max_atom_tok=max(at_tok)))
os.makedirs('results', exist_ok=True)
with open('results/corpus_stats.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    tot = {k: (sum(r[k] for r in rows) if k != 'doc' and k != 'max_atom_tok' else None) for k in rows[0]}
    tot['doc'] = 'TOTAL'; tot['max_atom_tok'] = max(r['max_atom_tok'] for r in rows); w.writerow(tot)
for r in rows + [tot]:
    print(' '.join(f"{k}={v}" for k, v in r.items()))

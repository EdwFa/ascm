#!/usr/bin/env python3
"""data/manifest.csv -> sentences, protected boundaries, atoms (data/processed) + results/corpus_stats.csv.
English texts are produced by scripts/extract_pdfs.py first."""
import argparse, csv, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ascm import pipeline, tokens, loc

ap = argparse.ArgumentParser()
ap.add_argument('--manifest', default='data/manifest.csv'); ap.add_argument('--out', default='data/processed')
ap.add_argument('--tokenizer', default='proxy'); ap.add_argument('--tau', type=int, default=512)
a = ap.parse_args()
count = tokens.get_counter(a.tokenizer)
rows, tmax = [], int(1.5 * a.tau)
for m in csv.DictReader(open(a.manifest, encoding='utf-8')):
    if not os.path.exists(m['path']):
        print('SKIP (file missing):', m['slug'], m['path']); continue
    raw = open(m['path'], encoding='utf-8').read()
    d = pipeline.process_document(raw, count, m['lang'], m['family'],
                                  meta={k: m[k] for k in ('kind', 'publisher', 'role')})
    pipeline.save_document(d, a.out, m['slug'])
    S = d['sentences']; tok = [s['tokens'] for s in S]
    at_tok = [sum(tok[i] for i in at) for at in d['atoms']]
    rows.append(dict(doc=m['slug'], lang=m['lang'], kind=m['kind'], words=len(d['text'].split()), sentences=len(S),
                     tokens=sum(tok), atoms=len(d['atoms']), protected=sum(d['protected']),
                     evidence_sentences=sum(s['loc'] for s in S), pairs=len(d['pairs']),
                     sections=sum(1 for s in S if s.get('sec_level')), atoms_over_tmax=sum(t > tmax for t in at_tok),
                     max_atom_tok=max(at_tok)))
os.makedirs('results', exist_ok=True)
with open('results/corpus_stats.csv', 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    print(' '.join(f"{k}={v}" for k, v in r.items() if k != 'kind'))
for lang in sorted({r['lang'] for r in rows}):
    sub = [r for r in rows if r['lang'] == lang]
    print(f"TOTAL[{lang}] docs={len(sub)} sentences={sum(r['sentences'] for r in sub)} tokens={sum(r['tokens'] for r in sub)} "
          f"atoms={sum(r['atoms'] for r in sub)} pairs={sum(r['pairs'] for r in sub)}")

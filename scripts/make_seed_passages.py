#!/usr/bin/env python3
"""Sample seed passages for physicians to write questions about (written BEFORE and independently of any chunker).
Output: results/annotation/seed_passages.csv  (columns to fill: type, question, answerable, answer, notes)."""
import argparse, csv, glob, os, sys, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ascm import pipeline, loc

ap = argparse.ArgumentParser()
ap.add_argument('--proc', default='data/processed'); ap.add_argument('--per-doc', type=int, default=30)
ap.add_argument('--plain-per-doc', type=int, default=6); ap.add_argument('--seed', type=int, default=13)
ap.add_argument('--out', default='results/annotation/seed_passages.csv')
ap.add_argument('--lang', default='all', choices=['all', 'ru', 'en'])
a = ap.parse_args()
rnd = random.Random(a.seed)
rows = []
for f in sorted(glob.glob(os.path.join(a.proc, '*.json'))):
    slug = os.path.basename(f)[:-5]
    d = pipeline.load_document(a.proc, slug); S = d['sentences']
    if a.lang != 'all' and d.get('meta', {}).get('lang', 'ru') != a.lang:
        continue
    pairs = d['pairs']; rnd.shuffle(pairs)
    for p in pairs[:a.per_doc]:
        blk = [S[k] for k in p['block']]; l = S[p['loc']]
        rows.append(dict(id=f"{slug}-{len(rows):04d}", doc=slug, kind='recommendation',
                         start=blk[0]['start'], end=blk[-1]['end'], evidence_start=l['start'], evidence_end=l['end'],
                         passage=' '.join(b['text'] for b in blk), evidence=l['text']))
    plain = [s for s in S if not s['loc'] and not s['heading'] and 80 < len(s['text']) < 400]
    for s in rnd.sample(plain, min(a.plain_per_doc, len(plain))):
        rows.append(dict(id=f"{slug}-{len(rows):04d}", doc=slug, kind='plain', start=s['start'], end=s['end'],
                         evidence_start='', evidence_end='', passage=s['text'], evidence=''))
for r in rows:
    r.update(type='', question='', answerable='yes', answer='', notes='')
os.makedirs(os.path.dirname(a.out), exist_ok=True)
with open(a.out, 'w', newline='', encoding='utf-8') as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(rows), 'seed passages ->', a.out)

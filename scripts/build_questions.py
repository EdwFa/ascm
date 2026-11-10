#!/usr/bin/env python3
"""Merge hand-written (in-session) question sources results/qsrc/*.json -> results/questions_synth.jsonl.
Source format: {passage_id: [question_about_recommendation, question_about_evidence_or_null]}.
Rule check (instead of a second model): a question sharing a 6-word sequence with its passage/evidence line is rejected."""
import argparse, csv, glob, json, os, re, sys
ap = argparse.ArgumentParser()
ap.add_argument('--seeds', default='results/annotation/seed_passages.csv'); ap.add_argument('--src', default='results/qsrc/[!_]*.json')
ap.add_argument('--out', default='results/questions_synth.jsonl'); ap.add_argument('--swap', default='', help='JSON list of seed ids whose passage is only a label: the evidence span (rec + inline grade) is used as gold')
ap.add_argument('--amb', default='results/qsrc/_ambiguous.json')
a = ap.parse_args()
rows = {r['id']: r for r in csv.DictReader(open(a.seeds, encoding='utf-8'))}
swap = set(json.load(open(a.swap))) if a.swap else set()
def grams(t, n=6):
    w = re.findall(r'\w+', t.lower()); return {tuple(w[i:i+n]) for i in range(len(w)-n+1)}
amb = set(json.load(open(a.amb))) if os.path.exists(a.amb) else set()  # passage text occurs more than once in the corpus -> gold span ambiguous
out, rej, seen = [], [], set()
for f in sorted(glob.glob(a.src)):
    for i, (rq, eq) in json.load(open(f, encoding='utf-8')).items():
        r = dict(rows[i])
        if i in swap:
            r['passage'], r['start'], r['end'] = r['evidence'], r['evidence_start'], r['evidence_end']
        if i in amb: rej.append((i, 'ambiguous gold')); continue
        for typ, q in (('recommendation', rq), ('evidence', eq)):
            if not q: continue
            if typ == 'evidence' and not r['evidence']: rej.append((i, 'no evidence line')); continue
            if grams(q) & grams(r['passage'] + ' ' + r['evidence']): rej.append((i, typ, 'copied')); continue
            if q in seen: rej.append((i, typ, 'duplicate')); continue
            seen.add(q)
            out.append(dict(id=f'{i}-{typ[:3]}', doc=r['doc'], type=typ, question=q, answerable=True, source='synthetic',
                            gen_model='claude-sonnet-5-5 (in-session)', val_model='none (rule check: no shared 6-gram)',
                            gold_spans=[[int(r['start']), int(r['end'])]],
                            evidence_span=[int(r['evidence_start']), int(r['evidence_end'])] if typ == 'evidence' else None))
open(a.out, 'w', encoding='utf-8').write(''.join(json.dumps(o, ensure_ascii=False) + '\n' for o in out))
import collections
print(len(out), 'questions;', dict(collections.Counter(o['type'] for o in out)), '; rejected', len(rej), rej[:12])
print('passages covered:', len({o['id'].rsplit('-', 1)[0] for o in out}), 'of', len(rows))

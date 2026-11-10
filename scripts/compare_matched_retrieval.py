#!/usr/bin/env python3
"""Retrieval metrics at MATCHED mean chunk size. Chunkers differ in how far their real mean size is from the nominal tau
(semantic/recursive give much smaller chunks), so comparing at equal tau favours small chunks. For every ASCM row (tau) each
other method is taken at the tau of the grid whose mean chunk size is closest to ASCM's (within --tol). Paired by document.
Input: results/retrieval_questions[_pool].csv produced with a tau grid, e.g.  --tau 192 256 320 384 448 512 640 768
Usage: python scripts/compare_matched_retrieval.py --csv results/retrieval_questions.csv --budget 1024"""
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ascm import stats

ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='results/retrieval_questions.csv'); ap.add_argument('--ref', default='ASCM')
ap.add_argument('--budget', type=int, default=1024); ap.add_argument('--tol', type=float, default=0.10)
ap.add_argument('--metrics', nargs='*', default=['evidence_linked', 'hit', 'mrr']); ap.add_argument('--out')
a = ap.parse_args()
df = pd.read_csv(a.csv); df = df[df.budget == a.budget]
mets = [m for m in a.metrics if m in df]
dm = df.groupby(['method', 'tau', 'doc'])[mets + ['mean_chunk_tok']].mean().reset_index()
size = dm.groupby(['method', 'tau']).mean_chunk_tok.mean().reset_index()
out = []
for tau, rs in size[size.method == a.ref].groupby('tau'):
    s_ref = float(rs.mean_chunk_tok.iloc[0]); ref = dm[(dm.method == a.ref) & (dm.tau == tau)].set_index('doc')
    for m in sorted(set(size.method) - {a.ref}):
        c = size[size.method == m]
        k = (c.mean_chunk_tok - s_ref).abs().idxmin(); t2, s2 = int(c.loc[k, 'tau']), float(c.loc[k, 'mean_chunk_tok'])
        if abs(s2 - s_ref) / s_ref > a.tol: continue
        oth = dm[(dm.method == m) & (dm.tau == t2)].set_index('doc')
        row = dict(ref_tau=tau, other=m, other_tau=t2, size_ref=round(s_ref), size_other=round(s2))
        for q in mets:
            d = (ref[q] - oth[q]).dropna()
            if len(d) < 2: continue
            mean, lo, hi = stats.cluster_bootstrap_ci(d.values)
            row.update({q + '_ref': round(float(ref[q].mean()), 3), q + '_other': round(float(oth[q].mean()), 3),
                        q + '_diff': round(mean, 3), q + '_lo': round(lo, 3), q + '_hi': round(hi, 3),
                        q + '_p': round(stats.paired_permutation_p(d.values), 3)})
        out.append(row)
res = pd.DataFrame(out)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
cols = ['ref_tau', 'other', 'other_tau', 'size_ref', 'size_other'] + [c for q in mets for c in (q + '_ref', q + '_other', q + '_diff', q + '_lo', q + '_hi', q + '_p') if c in res]
print(res[cols].to_string(index=False))
if a.out: res.to_csv(a.out, index=False)
print('\\n(diff = ASCM minus other; CI = cluster bootstrap over documents; p = paired sign-flip test, NOT Holm-adjusted; '
      'with', df.doc.nunique(), 'documents the CIs are wide)')

#!/usr/bin/env python3
"""Compare methods at matched mean chunk size. For each (method, tau) the mean over documents is computed; every ASCM
setting is paired with the setting of each other method whose mean chunk size is closest (within --tol, default 8%).
Run run_experiment.py with several --tau values first, e.g.  --tau 192 256 320 384 448 512 640 768"""
import argparse, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument('--csv', default='results/chunk_metrics.csv'); ap.add_argument('--tol', type=float, default=0.08)
ap.add_argument('--ref', default='ASCM')
ap.add_argument('--out', default='results/matched.csv')
a = ap.parse_args()
df = pd.read_csv(a.csv)
cols = ['mean_tok', 'cv_tok', 'small_lt100', 'erl_block', 'list_split', 'scs', 'sec_cross', 'section_cut', 'comment_sep']
g = df.groupby(['method', 'tau'])[[c for c in cols if c in df]].mean().reset_index()
rows = []
for _, r in g[g.method == a.ref].iterrows():
    for m, gm in g[g.method != a.ref].groupby('method'):
        j = (gm.mean_tok - r.mean_tok).abs().idxmin()
        o = gm.loc[j]
        if abs(o.mean_tok - r.mean_tok) / r.mean_tok <= a.tol:
            row = dict(ref_tau=int(r.tau), other=m, other_tau=int(o.tau), size_ref=round(r.mean_tok), size_other=round(o.mean_tok))
            for c in cols[1:]:
                if c in g:
                    row[c + '_ref'] = round(r[c], 3); row[c + '_other'] = round(o[c], 3)
            rows.append(row)
out = pd.DataFrame(rows)
out.to_csv(a.out, index=False)
pd.set_option('display.width', 250, 'display.max_columns', 40)
print(out[['ref_tau', 'other', 'other_tau', 'size_ref', 'size_other', 'cv_tok_ref', 'cv_tok_other', 'scs_ref', 'scs_other',
           'sec_cross_ref', 'sec_cross_other', 'comment_sep_ref', 'comment_sep_other']].to_string(index=False))

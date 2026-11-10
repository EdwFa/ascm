#!/usr/bin/env python3
"""Run all chunking methods on all documents; write results/chunk_metrics.csv and results/summary.csv.
Offline debug:  python scripts/run_experiment.py --embedder hash --eval-embedder hash4
Paper run:      python scripts/build_corpus.py --tokenizer bge-m3
                python scripts/run_experiment.py --embedder bge-m3 --eval-embedder e5
(token counts come from data/processed, so the tokenizer is chosen in build_corpus.py)"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ascm import pipeline, dp, stats
from ascm.embed import get_embedder, cached_encode

ap = argparse.ArgumentParser()
ap.add_argument('--proc', default='data/processed'); ap.add_argument('--config', default='configs/default.json')
ap.add_argument('--embedder', default='hash'); ap.add_argument('--eval-embedder', default='hash4')
ap.add_argument('--out', default='results'); ap.add_argument('--cache', default='results/cache')
ap.add_argument('--tau', type=int, nargs='*')
ap.add_argument('--manifest', default='data/manifest.csv')
ap.add_argument('--lang', default='all', choices=['all', 'ru', 'en'], help='restrict to documents of one language')
ap.add_argument('--structure', default='full', choices=['full', 'no_headings', 'flat'],
                help='hide document structure from the methods (evaluation keeps the real structure)')
a = ap.parse_args()
cfg = json.load(open(a.config))
taus = a.tau or cfg['tau_star']
emb, emb_eval = get_embedder(a.embedder), get_embedder(a.eval_embedder)
base = dp.Params(tau_star=512, t_max=int(512 * cfg['t_max_factor']), lam_b=cfg['lam_b'], lam_len=cfg['lam_len'],
                 lam_coh=cfg['lam_coh'], alpha=cfg['alpha'], beta=cfg['beta'])
import csv
active = {m['slug'] for m in csv.DictReader(open(a.manifest, encoding='utf-8')) if m['role'] != 'excluded'}
rows = []
for f in sorted(glob.glob(os.path.join(a.proc, '*.json'))):
    slug = os.path.basename(f)[:-5]
    if slug not in active:
        continue          # stale processed file or an excluded document
    d = pipeline.load_document(a.proc, slug)
    if a.lang != 'all' and d.get('meta', {}).get('lang', 'ru') != a.lang:
        continue
    texts = [s['text'] for s in d['sentences']]
    E = cached_encode(emb, texts, a.cache); Ee = cached_encode(emb_eval, texts, a.cache)
    for tau in taus:
        res = pipeline.run_methods(pipeline.strip_structure(d, a.structure), tau, base, E)
        tmax = int(tau * cfg['t_max_factor'])
        for m, starts in res.items():
            if m.startswith('_'): continue
            r = pipeline.evaluate_chunking(d, starts, tau, tmax, Ee)
            r.update(doc=slug, method=m, tau=tau, structure=a.structure, embedder=emb.name, eval_embedder=emb_eval.name)
            if m == 'ASCM':
                r['objective'] = res['_objective']; r['atoms_over_tmax'] = res['_atoms_over_tmax']
            rows.append(r)
    print('done', slug, flush=True)
os.makedirs(a.out, exist_ok=True)
df = pd.DataFrame(rows)
suffix = '' if a.structure == 'full' else '_' + a.structure
df.to_csv(os.path.join(a.out, f'chunk_metrics{suffix}.csv'), index=False)

# summary: mean over documents with cluster bootstrap CI; paired permutation ASCM vs each baseline (Holm per metric/tau)
metrics = ['n_chunks', 'mean_tok', 'median_tok', 'p5_tok', 'p95_tok', 'cv_tok', 'small_lt100', 'erl_adjacent', 'erl_block', 'list_split', 'f_violations', 'scs',
           'sec_cross', 'section_cut', 'comment_sep']
summ = []
for tau, g in df.groupby('tau'):
    for m, gm in g.groupby('method'):
        row = dict(tau=tau, method=m)
        for k in metrics:
            v = gm.sort_values('doc')[k].dropna().values
            mean, lo, hi = stats.cluster_bootstrap_ci(v) if len(v) else (float('nan'),) * 3
            row.update({k: mean, k + '_lo': lo, k + '_hi': hi})
        summ.append(row)
summ = pd.DataFrame(summ)
tests = []
for tau, g in df.groupby('tau'):
    asc = g[g.method == 'ASCM'].sort_values('doc')
    for k in ['erl_block', 'scs', 'sec_cross', 'comment_sep']:
        ps, names, diffs = [], [], []
        for m in g.method.unique():
            if m == 'ASCM': continue
            other = g[g.method == m].set_index('doc')[k]
            dd = (asc.set_index('doc')[k] - other).dropna()
            if len(dd) < 2: continue
            ps.append(stats.paired_permutation_p(dd.values)); names.append(m); diffs.append(float(dd.mean()))
        for n, p, ph, df_ in zip(names, ps, stats.holm(ps), diffs):
            tests.append(dict(tau=tau, metric=k, baseline=n, p=p, p_holm=ph, diff=df_))
summ.to_csv(os.path.join(a.out, f'summary{suffix}.csv'), index=False)
pd.DataFrame(tests).to_csv(os.path.join(a.out, f'tests{suffix}.csv'), index=False)
pd.set_option('display.width', 200)
print(summ[summ.tau == 512][['method', 'n_chunks', 'mean_tok', 'erl_block', 'list_split', 'scs', 'sec_cross', 'section_cut', 'comment_sep']].round(3).to_string(index=False))

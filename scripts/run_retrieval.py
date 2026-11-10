#!/usr/bin/env python3
"""Retrieval evaluation on a question set (JSONL, see src/ascm/retrieval.py). For every chunking method the chunks of the
question's document are embedded, the top chunks are taken until a TOKEN BUDGET is full, and we measure
  evidence_possible: diagnostic - some chunk holds recommendation + evidence together (chunking only, no ranking)
  evidence_linked : recommendation AND its evidence line are both inside ONE retrieved chunk (>=90% of each)  [main metric]
  hit / span_recall / mrr / context_precision.
Retrieval is inside one document (not a pooled index). Questions of one document are one cluster for the bootstrap.
Offline debug: python scripts/run_retrieval.py --embedder hash --questions results/questions_synth_pilot.jsonl
Paper run:     python scripts/run_retrieval.py --embedder bge-m3 --questions results/questions_synth_pilot.jsonl"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from ascm import pipeline, dp, stats, retrieval
from ascm.embed import get_embedder, cached_encode

ap = argparse.ArgumentParser()
ap.add_argument('--questions', required=True); ap.add_argument('--proc', default='data/processed')
ap.add_argument('--config', default='configs/default.json'); ap.add_argument('--embedder', default='hash')
ap.add_argument('--tau', type=int, nargs='*'); ap.add_argument('--budget', type=int, nargs='*', default=[1024])
ap.add_argument('--structure', default='full', choices=['full', 'no_headings', 'flat'])
ap.add_argument('--out', default='results'); ap.add_argument('--cache', default='results/cache')
ap.add_argument('--methods', nargs='*')
ap.add_argument('--pool', action='store_true', help='one index over the chunks of ALL selected documents (realistic); default: retrieval inside the question\'s own document')
ap.add_argument('--pool-docs', default='questions', choices=['questions', 'all'], help='pooled index: only documents that have questions, or every active document of the manifest (distractors)')
a = ap.parse_args()
cfg = json.load(open(a.config)); taus = a.tau or cfg['tau_star']
emb = get_embedder(a.embedder)
base = dp.Params(tau_star=512, t_max=int(512 * cfg['t_max_factor']), lam_b=cfg['lam_b'], lam_len=cfg['lam_len'],
                 lam_coh=cfg['lam_coh'], alpha=cfg['alpha'], beta=cfg['beta'])
Q = retrieval.load_questions(a.questions)
by_doc = {}
for q in Q: by_doc.setdefault(q['doc'], []).append(q)
rows = []
import csv as _csv
active = [m['slug'] for m in _csv.DictReader(open('data/manifest.csv', encoding='utf-8')) if m['role'] != 'excluded']
docs = sorted(set(by_doc) | (set(active) if (a.pool and a.pool_docs == 'all') else set()))
OFF = 10 ** 9
cache_doc = {}
def chunks_for(slug, tau, res_cache={}):
    d = cache_doc.get(slug)
    if d is None:
        d = pipeline.load_document(a.proc, slug)
        text = open(os.path.join(a.proc, slug + '.txt'), encoding='utf-8').read()
        E = cached_encode(emb, [x['text'] for x in d['sentences']], a.cache)
        cache_doc[slug] = d = (d, text, E)
    return d
for tau in taus:
    # per method: chunk spans / tokens / vectors for each document
    per = {}   # method -> {slug: (spans, toks, CV)}
    for slug in docs:
        d, text, E = chunks_for(slug, tau)
        S = d['sentences']
        res = pipeline.run_methods(pipeline.strip_structure(d, a.structure), tau, base, E, methods=a.methods)
        for m, starts in res.items():
            if m.startswith('_'): continue
            ends = list(starts[1:]) + [len(S)]
            spans = [(S[i]['start'], S[j - 1]['end']) for i, j in zip(starts, ends)]
            toks = [sum(S[k]['tokens'] for k in range(i, j)) for i, j in zip(starts, ends)]
            CV = cached_encode(emb, [text[x:y] for x, y in spans], a.cache)
            per.setdefault(m, {})[slug] = (spans, toks, CV)
    print('tau', tau, 'chunked', len(docs), 'docs', flush=True)
    for m, dd in per.items():
        if a.pool:
            order = sorted(dd); k_of = {sl: n for n, sl in enumerate(order)}
            P_sp = [(x + k_of[sl] * OFF, y + k_of[sl] * OFF) for sl in order for x, y in dd[sl][0]]
            P_tk = [t for sl in order for t in dd[sl][1]]
            P_cv = np.vstack([dd[sl][2] for sl in order])
        for slug, qs in sorted(by_doc.items()):
            spans, toks, CV = dd[slug]
            qv = cached_encode(emb, [q['question'] for q in qs], a.cache)
            mean_tok = float(np.mean(toks))
            for qi, q in enumerate(qs):
                for b in a.budget:
                    if a.pool:
                        o = k_of[slug] * OFF
                        qq = dict(q, gold_spans=[[g0 + o, g1 + o] for g0, g1 in q['gold_spans']],
                                  evidence_span=[q['evidence_span'][0] + o, q['evidence_span'][1] + o] if q.get('evidence_span') else None)
                        r = retrieval.evaluate(qq, P_sp, P_tk, P_cv, qv[qi], b)
                    else:
                        r = retrieval.evaluate(q, spans, toks, CV, qv[qi], b)
                    r.update(doc=slug, qid=q['id'], qtype=q['type'], method=m, tau=tau, budget=b, structure=a.structure,
                             mean_chunk_tok=mean_tok, pooled=a.pool)
                    rows.append(r)
df = pd.DataFrame(rows)
os.makedirs(a.out, exist_ok=True)
sfx = ('' if a.structure == 'full' else '_' + a.structure) + ('_pool' if a.pool else '')
df.to_csv(os.path.join(a.out, f'retrieval_questions{sfx}.csv'), index=False)

mets = ['mean_chunk_tok', 'evidence_linked', 'evidence_possible', 'hit', 'span_recall', 'mrr', 'context_precision', 'context_tokens']
doc_mean = df.groupby(['tau', 'budget', 'method', 'doc'])[[x for x in mets if x in df]].mean().reset_index()
summ, tests = [], []
for (tau, b), g in doc_mean.groupby(['tau', 'budget']):
    for m, gm in g.groupby('method'):
        row = dict(tau=tau, budget=b, method=m, n_docs=len(gm), n_questions=int(((df.tau == tau) & (df.budget == b) & (df.method == m)).sum()))
        for k in mets:
            if k not in gm: continue
            v = gm.sort_values('doc')[k].dropna().values
            mean, lo, hi = stats.cluster_bootstrap_ci(v) if len(v) else (float('nan'),) * 3
            row.update({k: mean, k + '_lo': lo, k + '_hi': hi})
        summ.append(row)
    if 'ASCM' in set(g.method):
        asc = g[g.method == 'ASCM'].set_index('doc')
        for k in ['evidence_linked', 'evidence_possible', 'hit', 'mrr']:
            if k not in g: continue
            ps, nm, df_ = [], [], []
            for m in g.method.unique():
                if m == 'ASCM': continue
                dd = (asc[k] - g[g.method == m].set_index('doc')[k]).dropna()
                if len(dd) < 2: continue
                ps.append(stats.paired_permutation_p(dd.values)); nm.append(m); df_.append(float(dd.mean()))
            for n, p, ph, x in zip(nm, ps, stats.holm(ps), df_):
                tests.append(dict(tau=tau, budget=b, metric=k, baseline=n, diff=x, p=p, p_holm=ph))
summ = pd.DataFrame(summ)
summ.to_csv(os.path.join(a.out, f'retrieval_summary{sfx}.csv'), index=False)
pd.DataFrame(tests).to_csv(os.path.join(a.out, f'retrieval_tests{sfx}.csv'), index=False)
pd.set_option('display.width', 200)
show = summ.sort_values(['tau', 'budget', 'method'])
print(show[['tau', 'budget', 'method', 'n_docs', 'n_questions', 'mean_chunk_tok', 'evidence_possible', 'evidence_linked', 'hit', 'mrr', 'context_precision']].round(3).to_string(index=False))
print('\nIMPORTANT: question set is', os.path.basename(a.questions), '- check its source (synthetic vs physician) before quoting any number.')

"""Document processing and running all chunking methods on one document."""
from __future__ import annotations
import json, os
import numpy as np
from . import atoms as A, baselines as B, dp, metrics as M, loc as L
from .embed import cached_encode, unit_mean


def process_document(raw: str, count_tokens) -> dict:
    norm, S = A.build_sentences(raw, count_tokens)
    prot, why = A.protected_boundaries(S)
    atoms = A.build_atoms(S, prot)
    d = A.to_json(norm, S, prot, why, atoms)
    d['pairs'] = A.loc_pairs(S)
    return d


def save_document(d: dict, out_dir: str, slug: str):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(d, open(os.path.join(out_dir, f"{slug}.json"), 'w', encoding='utf-8'), ensure_ascii=False)
    open(os.path.join(out_dir, f"{slug}.txt"), 'w', encoding='utf-8').write(d['text'])


def load_document(proc_dir: str, slug: str) -> dict:
    return json.load(open(os.path.join(proc_dir, f"{slug}.json"), encoding='utf-8'))


def atom_inputs(d: dict, E_sent):
    S, atoms = d['sentences'], d['atoms']
    tok = np.array([sum(S[i]['tokens'] for i in a) for a in atoms])
    struct = np.array([0.0 if (k == 0 or S[a[0]]['para'] != S[atoms[k - 1][-1]]['para']) else 1.0
                       for k, a in enumerate(atoms)])
    E = None if E_sent is None else np.vstack([unit_mean(E_sent[a]) for a in atoms])
    return tok, E, struct


def run_methods(d: dict, tau: int, params: dp.Params, E_sent, methods=None):
    """Return {method: sentence-start list}. E_sent: embeddings of sentences from the chunker's embedder."""
    S, atoms = d['sentences'], d['atoms']
    tokens = [s['tokens'] for s in S]
    para = [s['para'] for s in S]
    heading = [bool(s['heading']) and (i == 0 or S[i - 1]['para'] != s['para']) for i, s in enumerate(S)]
    p = dp.Params(**{**params.__dict__, 'tau_star': tau, 't_max': int(round(params.t_max / params.tau_star * tau))
                     if params.tau_star else int(1.5 * tau)})
    tok_a, E_a, struct = atom_inputs(d, E_sent)
    c_sem, c_struct = dp.boundary_inputs(E_a, struct)
    cm = dp.CostModel(tok_a, E_a, c_sem, c_struct, p)
    to_sent = lambda cuts: [atoms[a][0] for a in cuts]
    out = {}
    want = methods or ['B1_fixed', 'B2_recursive', 'B3_semantic', 'B4_header', 'B7_greedy', 'F_atom_fill', 'ASCM']
    if 'B1_fixed' in want: out['B1_fixed'] = B.fixed_size(tokens, tau)
    if 'B2_recursive' in want: out['B2_recursive'] = B.recursive(tokens, para, tau)
    if 'B3_semantic' in want: out['B3_semantic'] = B.semantic(E_sent, tokens, tau, p.t_max)
    if 'B4_header' in want: out['B4_header'] = B.header_based(tokens, heading, tau, p.t_max)
    if 'B7_greedy' in want: out['B7_greedy'] = to_sent(dp.greedy(cm))
    if 'F_atom_fill' in want: out['F_atom_fill'] = to_sent(B.atom_fill(tok_a, tau))
    if 'ASCM' in want:
        cuts, obj = dp.solve(cm)
        out['ASCM'] = to_sent(cuts)
        out['_objective'] = obj
        out['_atoms_over_tmax'] = int((tok_a > p.t_max).sum())
    return out


def evaluate_chunking(d: dict, starts, tau, t_max, E_eval):
    S = d['sentences']
    tokens = [s['tokens'] for s in S]
    n = len(S)
    r = M.size_stats(starts, tokens, t_max)
    r.update(M.erl(starts, d['pairs'], n))
    r['list_split'] = M.list_split_rate(starts, d['why'])
    r['f_violations'] = M.f_violations(starts, d['protected'])
    if E_eval is not None:
        r['scs'] = M.scs(starts, E_eval)
    return r

"""Document processing and running all chunking methods on one document."""
from __future__ import annotations
import json, os
import numpy as np
from . import atoms as A, baselines as B, dp, metrics as M, loc as L
from .embed import cached_encode, unit_mean, center


def process_document(raw: str, count_tokens, lang: str = 'ru', family: str = 'ru_uur', meta: dict | None = None) -> dict:
    norm, S = A.build_sentences(raw, count_tokens, lang, family)
    prot, why = A.protected_boundaries(S)
    atoms = A.build_atoms(S, prot)
    d = A.to_json(norm, S, prot, why, atoms)
    d['pairs'] = A.loc_pairs(S)
    d['meta'] = {**(meta or {}), 'lang': lang, 'family': family}
    return d


def save_document(d: dict, out_dir: str, slug: str):
    os.makedirs(out_dir, exist_ok=True)
    json.dump(d, open(os.path.join(out_dir, f"{slug}.json"), 'w', encoding='utf-8'), ensure_ascii=False)
    open(os.path.join(out_dir, f"{slug}.txt"), 'w', encoding='utf-8').write(d['text'])


def load_document(proc_dir: str, slug: str) -> dict:
    return json.load(open(os.path.join(proc_dir, f"{slug}.json"), encoding='utf-8'))


def struct_costs(d: dict, mode: str = 'hier'):
    """Cost of cutting before each atom. 'binary': 0 at a paragraph start, 1 inside a paragraph.
    'hier': section starts are cheap (by depth), a 'Комментарии' block is expensive to separate from its recommendation,
    a cut right after a completed recommendation (LoC line) is cheaper than other paragraph breaks."""
    S, atoms = d['sentences'], d['atoms']
    out = []
    for k, a in enumerate(atoms):
        if k == 0:
            out.append(0.0); continue
        s, prev = S[a[0]], S[atoms[k - 1][-1]]
        newpar = s['para'] != prev['para']
        if mode == 'binary':
            out.append(0.0 if newpar else 1.0); continue
        lvl = s.get('sec_level', 0)
        if lvl:
            out.append({1: 0.0, 2: 0.05}.get(lvl, 0.15))
        elif s.get('comment'):
            out.append(0.8)
        elif newpar:
            out.append(0.3 if prev['loc'] else 0.5)
        else:
            out.append(1.0)
    return np.array(out)


def atom_inputs(d: dict, E_sent, struct_mode: str = 'binary'):
    S, atoms = d['sentences'], d['atoms']
    tok = np.array([sum(S[i]['tokens'] for i in a) for a in atoms])
    struct = struct_costs(d, struct_mode)
    E = None if E_sent is None else np.vstack([unit_mean(E_sent[a]) for a in atoms])
    return tok, E, struct


VARIANTS = {            # name -> (centre embeddings, structure mode)
    'ASCM_basic': (False, 'binary'),
    'ASCM_struct': (False, 'hier'),
    'ASCM_center': (True, 'binary'),
    'ASCM': (True, 'hier'),
}


def run_methods(d: dict, tau: int, params: dp.Params, E_sent, methods=None):
    """Return {method: sentence-start list}. E_sent: embeddings of sentences from the chunker's embedder.
    ASCM = centred embeddings + hierarchical structure cost; the other ASCM_* are ablations; B7 uses the full ASCM cost."""
    S, atoms = d['sentences'], d['atoms']
    tokens = [s['tokens'] for s in S]
    para = [s['para'] for s in S]
    heading = [bool(s['heading']) and (i == 0 or S[i - 1]['para'] != s['para']) for i, s in enumerate(S)]
    p = dp.Params(**{**params.__dict__, 'tau_star': tau, 't_max': int(round(params.t_max / params.tau_star * tau))
                     if params.tau_star else int(1.5 * tau)})
    E_c = center(E_sent)
    to_sent = lambda cuts: [atoms[a][0] for a in cuts]

    def make_cm(centre, mode):
        tok_a, E_a, struct = atom_inputs(d, E_c if centre else E_sent, mode)
        c_sem, c_struct = dp.boundary_inputs(E_a, struct)
        return dp.CostModel(tok_a, E_a, c_sem, c_struct, p), tok_a

    want = methods or ['B1_fixed', 'B2_recursive', 'B3_semantic', 'B4_header', 'B7_greedy', 'B8_header_F', 'F_atom_fill',
                       'ASCM_basic', 'ASCM_struct', 'ASCM_center', 'ASCM']
    out = {}
    if 'B1_fixed' in want: out['B1_fixed'] = B.fixed_size(tokens, tau)
    if 'B2_recursive' in want: out['B2_recursive'] = B.recursive(tokens, para, tau)
    if 'B3_semantic' in want: out['B3_semantic'] = B.semantic(E_sent, tokens, tau, p.t_max)
    if 'B4_header' in want: out['B4_header'] = B.header_based(tokens, heading, tau, p.t_max)
    cm_full, tok_a = make_cm(*VARIANTS['ASCM'])
    if 'B7_greedy' in want: out['B7_greedy'] = to_sent(dp.greedy(cm_full))
    if 'B8_header_F' in want:
        sec = [bool(S[a[0]].get('sec_level', 0)) for a in atoms]
        out['B8_header_F'] = to_sent(B.header_atoms(tok_a, sec, tau, p.t_max))
    if 'F_atom_fill' in want: out['F_atom_fill'] = to_sent(B.atom_fill(tok_a, tau))
    for name, (centre, mode) in VARIANTS.items():
        if name in want:
            cm = cm_full if name == 'ASCM' else make_cm(centre, mode)[0]
            cuts, obj = dp.solve(cm)
            out[name] = to_sent(cuts)
            if name == 'ASCM':
                out['_objective'] = obj
                out['_atoms_over_tmax'] = int((tok_a > p.t_max).sum())
    return out


def strip_structure(d: dict, mode: str = 'full') -> dict:
    """Simulate documents with weak structure for the methods (evaluation still uses the original structure).
    'no_headings': section numbers, headings and 'Комментарии' markers are hidden.
    'flat': additionally all paragraph breaks are hidden. Protected boundaries F and atoms are unchanged."""
    if mode == 'full':
        return d
    S = []
    for s in d['sentences']:
        t = dict(s)
        t['sec_level'], t['comment'], t['heading'] = 0, False, False
        if mode == 'flat':
            t['para'] = 0
        S.append(t)
    return {**d, 'sentences': S}


def evaluate_chunking(d: dict, starts, tau, t_max, E_eval):
    S = d['sentences']
    tokens = [s['tokens'] for s in S]
    n = len(S)
    r = M.size_stats(starts, tokens, t_max)
    r.update(M.erl(starts, d['pairs'], n))
    r['list_split'] = M.list_split_rate(starts, d['why'])
    r['f_violations'] = M.f_violations(starts, d['protected'])
    r.update(M.structure_metrics(starts, S))
    if E_eval is not None:
        r['scs'] = M.scs(starts, E_eval)
    return r

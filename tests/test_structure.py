import numpy as np
from ascm import text as T, atoms as A, tokens, pipeline, dp
from ascm.embed import center

def test_section_levels():
    assert T.section_level("2.1. Жалобы и анамнез Рекомендуется сбор жалоб.") == 2
    assert T.section_level("1.5.3. Макроскопическая классификация РЖ") == 3
    assert T.section_level("2. Диагностика") == 1
    assert T.section_level("3. Смешанный тип: аденокарцинома представлена") == 0   # list item, not a section
    assert T.section_level("1. данные анамнеза;") == 0
    assert T.is_comment("Комментарии: эндосонография позволяет")

DOC = ("2. Диагностика\n\n2.1. Жалобы Рекомендуется сбор жалоб.\n\n"
       "Уровень убедительности рекомендаций – C (уровень достоверности доказательств – 5).\n\n"
       "Комментарии: сбор жалоб необходим.\n\nВторая рекомендация нужна.\n\n"
       "Уровень убедительности рекомендаций – B (уровень достоверности доказательств – 2).\n\n"
       "2.2. Физикальное обследование Осмотр пациента.")

def doc():
    norm, S = A.build_sentences(DOC, tokens.proxy_count)
    prot, why = A.protected_boundaries(S)
    atoms = A.build_atoms(S, prot)
    return dict(sentences=[s.__dict__ for s in S], atoms=atoms)

def test_hier_costs_order():
    d = doc()
    c = pipeline.struct_costs(d, 'hier')
    S, atoms = d['sentences'], d['atoms']
    by_text = {S[a[0]]['text'][:12]: c[k] for k, a in enumerate(atoms)}
    assert by_text['2.2. Физикал'] < by_text['Комментарии:']          # section start cheaper than detaching a comment
    assert by_text['Комментарии:'] == 0.8
    assert by_text['Вторая реком'] == 0.5                             # plain paragraph break (previous block is a comment)
    assert pipeline.struct_costs(d, 'binary')[0] == 0.0

def test_center_removes_mean():
    rng = np.random.default_rng(0)
    E = rng.normal(size=(50, 16)) + 5.0
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    C = center(E)
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0)
    assert (E[:-1] * E[1:]).sum(1).mean() > 0.9 > np.abs((C[:-1] * C[1:]).sum(1).mean()) + 0.5

def test_all_variants_respect_F():
    d = doc()
    rng = np.random.default_rng(1)
    E = rng.normal(size=(len(d['sentences']), 8)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    d['protected'] = A.protected_boundaries([A.Sentence(**s) for s in d['sentences']])[0]
    res = pipeline.run_methods(d, 64, dp.Params(tau_star=64, t_max=96), E)
    for name in ('ASCM', 'ASCM_basic', 'ASCM_struct', 'ASCM_center', 'B7_greedy'):
        assert not any(d['protected'][j] for j in res[name] if j < len(d['protected']))

def test_header_atoms_cuts_at_sections_and_respects_cap():
    from ascm import baselines as B
    tok = [100] * 12
    sec = [False] * 12; sec[4] = sec[9] = True
    st = B.header_atoms(tok, sec, 250, 380)
    assert 4 in st and 9 in st and st[0] == 0
    b = st + [12]
    assert all(sum(tok[b[k]:b[k + 1]]) <= 380 for k in range(len(st)))

import numpy as np, pytest
from ascm import dp

def make(seed, n, use_emb):
    rng = np.random.default_rng(seed)
    tok = rng.integers(40, 300, size=n)
    E = None
    if use_emb:
        E = rng.normal(size=(n, 8)); E /= np.linalg.norm(E, axis=1, keepdims=True)
    struct = rng.integers(0, 2, size=n).astype(float)
    p = dp.Params(tau_star=300, t_max=450, lam_b=0.5, lam_len=1.0, lam_coh=0.5 if use_emb else 0.0)
    c_sem, c_struct = dp.boundary_inputs(E, struct)
    return dp.CostModel(tok, E, c_sem, c_struct, p)

@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("use_emb", [False, True])
def test_dp_equals_brute_force(seed, use_emb):
    cm = make(seed, 9, use_emb)
    cuts, obj = dp.solve(cm)
    bcuts, bobj = dp.brute_force(cm)
    assert obj == pytest.approx(bobj, abs=1e-9)
    assert dp.objective(cm, cuts) == pytest.approx(obj, abs=1e-9)

def test_hard_constraint_and_oversize_atom():
    cm = make(1, 30, True)
    cm.T = np.concatenate([[0], np.cumsum(np.r_[100, 900, np.full(28, 100)])]).astype(float)  # one atom > t_max
    cuts, _ = dp.solve(cm)
    b = list(cuts) + [cm.N]
    for k in range(len(cuts)):
        assert cm.feasible(b[k], b[k + 1])      # only the single oversize atom may exceed t_max

def test_dp_never_worse_than_greedy():
    for s in range(10):
        cm = make(s, 40, True)
        _, obj = dp.solve(cm)
        assert obj <= dp.objective(cm, dp.greedy(cm)) + 1e-9

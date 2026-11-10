import numpy as np
from ascm import stats, metrics

def test_holm_monotone():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and max(adj) <= 1

def test_permutation_exact():
    p = stats.paired_permutation_p([1, 1, 1, 1, 1, 1, 1, 1])
    assert abs(p - 2 / 256) < 1e-9

def test_bootstrap_ci_contains_mean():
    m, lo, hi = stats.cluster_bootstrap_ci([0.8, 0.9, 1.0, 0.7, 0.85], n_boot=2000)
    assert lo <= m <= hi

def test_erl():
    pairs = [dict(loc=3, adjacent=2, block=[1, 2])]
    assert metrics.erl([0, 3], pairs, 6)['erl_adjacent'] == 0.0
    assert metrics.erl([0, 1], pairs, 6)['erl_block'] == 1.0

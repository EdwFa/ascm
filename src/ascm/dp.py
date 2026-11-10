"""Exact constrained chunking by dynamic programming over atoms.

Units are atoms (runs of sentences joined by protected boundaries F), so every unit boundary is an admissible cut.
Phi(i,j) = lam_b*C_j + lam_len*((tau-tau*)/tau*)^2 + lam_coh*(1 - ||E[j]-E[i]|| / (j-i)),  C_N = 0.
Hard constraint H2: tau <= t_max, except for a single atom (logged exception).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Params:
    tau_star: int = 512
    t_max: int = 768
    lam_b: float = 0.5
    lam_len: float = 1.0
    lam_coh: float = 0.5
    alpha: float = 0.7
    beta: float = 0.3


class CostModel:
    def __init__(self, tokens, E, c_sem, c_struct, p: Params):
        """tokens: tokens per atom (N,). E: unit embeddings (N,d) or None. c_sem, c_struct: (N+1,) boundary cost inputs
        for a cut before atom j (index 0 and N unused/zero)."""
        self.p, self.N = p, len(tokens)
        self.T = np.concatenate([[0], np.cumsum(tokens)]).astype(np.float64)
        self.E = None if E is None else np.vstack([np.zeros((1, E.shape[1])), np.cumsum(E, axis=0)])
        C = p.alpha * np.asarray(c_sem, float) + p.beta * np.asarray(c_struct, float)
        C[0] = 0.0
        C[self.N] = 0.0
        self.C = C

    def tau(self, i, j):
        return self.T[j] - self.T[i]

    def feasible(self, i, j):
        return j == i + 1 or self.tau(i, j) <= self.p.t_max

    def phi(self, i, j):
        p = self.p
        tau = self.tau(i, j)
        c = p.lam_b * self.C[j] + p.lam_len * ((tau - p.tau_star) / p.tau_star) ** 2
        if self.E is not None and p.lam_coh:
            c += p.lam_coh * (1.0 - np.linalg.norm(self.E[j] - self.E[i]) / (j - i))
        return float(c)


def boundary_inputs(E, struct_cut):
    """c_sem[j] = max(0, cos(e_{j-1}, e_j)); c_struct[j] given (1 = cut inside a paragraph, 0 = at a paragraph start)."""
    N = len(struct_cut)
    c_sem = np.zeros(N + 1)
    if E is not None:
        for j in range(1, N):
            c_sem[j] = max(0.0, float(E[j - 1] @ E[j]))
    c_struct = np.zeros(N + 1)
    c_struct[:N] = struct_cut
    return c_sem, c_struct


def solve(cm: CostModel):
    """Return (cuts, objective). cuts = sorted atom indices where chunks start (including 0)."""
    N = cm.N
    best = np.full(N + 1, np.inf)
    prev = np.full(N + 1, -1, dtype=int)
    best[0] = 0.0
    for j in range(1, N + 1):
        i = j - 1
        while i >= 0:
            if not cm.feasible(i, j):
                break
            v = best[i] + cm.phi(i, j)
            if v < best[j] - 1e-12:
                best[j], prev[j] = v, i
            i -= 1
    cuts, j = [], N
    while j > 0:
        cuts.append(prev[j])
        j = prev[j]
    return sorted(cuts), float(best[N])


def objective(cm: CostModel, cuts):
    b = list(cuts) + [cm.N]
    return sum(cm.phi(b[k], b[k + 1]) for k in range(len(cuts)))


def brute_force(cm: CostModel):
    """Enumerate all feasible partitions (small N only). For tests."""
    N = cm.N
    best = [np.inf, None]

    def rec(i, cuts, acc):
        if i == N:
            if acc < best[0]:
                best[0], best[1] = acc, list(cuts)
            return
        for j in range(i + 1, N + 1):
            if not cm.feasible(i, j):
                break
            rec(j, cuts + [j] if j < N else cuts, acc + cm.phi(i, j))
    rec(0, [0], 0.0)
    return sorted(best[1]), float(best[0])


def greedy(cm: CostModel):
    """B7: from each start choose the end minimising Phi(i,j) locally (same Phi and F as ASCM)."""
    N, i, cuts = cm.N, 0, []
    while i < N:
        cuts.append(i)
        bj, bv = i + 1, np.inf
        for j in range(i + 1, N + 1):
            if not cm.feasible(i, j):
                break
            v = cm.phi(i, j)
            if v < bv - 1e-12:
                bj, bv = j, v
        i = bj
    return cuts

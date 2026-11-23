#!/usr/bin/env python3
"""Derivative of chlorine's CI energy with respect to the exponents of its
correlating d and f functions, computed from the functions themselves.

Forward-mode automatic differentiation, the mechanism behind PyTorch's
`jvp`, written out with dual numbers: every quantity carries its value and
its derivative, and each operation (+, *, exp, sqrt, the Boys function)
propagates both. The integrals over Gaussians are closed formulas in the
exponents and positions, so their derivatives come out exact, with no step
size.

What is differentiated, and what is not:
- The integrals S, h and (ab|cd): by dual numbers. A d or f function is a
  set of s-Gaussians at distance delta / sqrt(exponent) from the nucleus, so
  both the exponent and the positions move.
- The orbitals: by the rule that the energy of a full CI depends only on the
  space the orbitals span. It is enough to let the orbitals follow the
  moving functions and stay orthonormal (the terms with dS below).
- The CI vector: not at all. It is at a minimum, so its change does not
  enter the first derivative (Hellmann-Feynman).

    dE/dx = <C| H[dh, dg] |C>

Exact for full CI. For a selected list it is an approximation, because the
energy then also depends on which orthonormal orbitals span the space.

    python3 analyzer/exponent_gradient.py - 0.1,0.8 0.4 -      # compare with central differences
"""
import sys
import time

import numpy as np

import cl_affinity
import quantum
from he2_dispersion import contracted_eri, one_electron


class Dual:
    """A value and its derivative along one direction."""
    __array_priority__ = 100

    def __init__(self, v, t=None):
        self.v = np.asarray(v, float)
        self.t = np.zeros_like(self.v) if t is None else np.asarray(t, float)

    @staticmethod
    def of(x):
        return x if isinstance(x, Dual) else Dual(x)

    def __add__(self, o):
        o = Dual.of(o)
        return Dual(self.v + o.v, self.t + o.t)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.v, -self.t)

    def __sub__(self, o):
        o = Dual.of(o)
        return Dual(self.v - o.v, self.t - o.t)

    def __rsub__(self, o):
        return Dual.of(o) - self

    def __mul__(self, o):
        o = Dual.of(o)
        return Dual(self.v * o.v, self.t * o.v + self.v * o.t)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = Dual.of(o)
        q = self.v / o.v
        return Dual(q, (self.t - q * o.t) / o.v)

    def __rtruediv__(self, o):
        return Dual.of(o) / self

    def __getitem__(self, k):
        return Dual(self.v[k], self.t[k])

    def sum(self, axis):
        return Dual(self.v.sum(axis), self.t.sum(axis))


def d_exp(x):
    e = np.exp(x.v)
    return Dual(e, e * x.t)


def d_sqrt(x):
    r = np.sqrt(x.v)
    return Dual(r, 0.5 * x.t / r)


def d_boys0(x):
    """F0 and its derivative F0' = -F1 = (exp(-x) - F0) / (2x)."""
    f0 = quantum.boys0(x.v)
    small = x.v < 1e-6
    safe = np.where(small, 1.0, x.v)
    f1 = np.where(small, 1.0 / 3.0 - x.v / 5.0, (f0 - np.exp(-x.v)) / (2.0 * safe))
    return Dual(f0, -f1 * x.t)


def pair_data(a, A):
    """Gaussian product of every pair of primitives: p, K, P."""
    ai, aj = a[:, None], a[None, :]
    p = ai + aj
    red = ai * aj / p
    diff = A[:, None, :] - A[None, :, :]
    d2 = (diff * diff).sum(-1)
    K = d_exp(-(red * d2))
    P = (a[:, None, None] * A[:, None, :] + a[None, :, None] * A[None, :, :]) / p[:, :, None]
    return p, red, d2, K, P


def integral_derivatives(space, shell):
    """dS, dh and d(ab|cd) over the contracted functions with respect to
    ln(exponent) of one shell of correlating functions. `shell` is
    (primitive indices, function indices) of that shell."""
    prims, funcs = shell
    exps, cen, W = space["exponents"], space["centres"], space["weights"]
    N, nc = W.shape
    seed = np.zeros(N)
    seed[prims] = 1.0
    a = Dual(exps, exps * seed)                             # d exponent / d ln(exponent) = exponent
    A = Dual(cen, -0.5 * cen * seed[:, None])               # positions scale as exponent^(-1/2)
    p, red, d2, K, P = pair_data(a, A)
    S = d_exp(Dual(1.5 * np.log(np.pi)) - 1.5 * Dual(np.log(p.v), p.t / p.v)) * K
    r2 = (P * P).sum(-1)                                    # the nucleus is at the origin
    h = red * (3.0 - 2.0 * red * d2) * S - 17.0 * 2.0 * np.pi / p * K * d_boys0(p * r2)
    dS, dh = W.T @ S.t @ W, W.T @ h.t @ W
    # two-electron part: only integrals with at least one function of the shell change.
    # slab[m, n, l, s] = d (f_m n | l s) for f_m in the shell; the rest follows by symmetry.
    slab = np.zeros((len(funcs), nc, nc, nc))
    where = {f: k for k, f in enumerate(funcs)}
    for i in prims:
        p1, K1, P1 = p[i], K[i], P[i]                       # pairs (i, b) for all b
        pq = p1[:, None, None] * p[None, :, :]
        tot = p1[:, None, None] + p[None, :, :]
        diff = P1[:, None, None, :] - P[None, :, :, :]
        R2 = (diff * diff).sum(-1)
        block = 2.0 * np.pi ** 2.5 / (pq * d_sqrt(tot)) * K1[:, None, None] * K[None, :, :] * d_boys0(pq / tot * R2)
        part = np.einsum("bcd,bn,cl,ds->nls", block.t, W, W, W, optimize=True)
        for f in np.nonzero(W[i])[0]:
            slab[where[f]] += W[i, f] * part
    d_eri = np.zeros((nc, nc, nc, nc))
    d_eri[funcs] = slab
    d_eri[:, funcs] = slab.transpose(1, 0, 2, 3)
    d_eri[:, :, funcs] = slab.transpose(2, 3, 0, 1)
    d_eri[:, :, :, funcs] = slab.transpose(2, 3, 1, 0)
    return dS, dh, d_eri


def shells(space, extra_d, extra_f):
    """(primitive indices, function indices) of each d and f shell, in order."""
    W = space["weights"]
    out, f0 = [], space["n_sp"]
    for count in [5] * len(extra_d) + [7] * len(extra_f):
        funcs = np.arange(f0, f0 + count)
        prims = np.nonzero(np.abs(W[:, funcs]).sum(1))[0]
        out.append((prims, funcs))
        f0 += count
    return out


def transform(g, c1, c2, c3, c4):
    g = np.einsum("pi,pqrs->iqrs", c1, g)
    g = np.einsum("qj,iqrs->ijrs", c2, g)
    g = np.einsum("rk,ijrs->ijks", c3, g)
    return np.einsum("sl,ijks->ijkl", c4, g)


def energy_and_gradient(n_alpha, n_beta, occupations, extra_s, extra_p, extra_d, extra_f=()):
    """Full CI energy of the outer electrons and its derivative with respect
    to ln(exponent) of every d and f shell."""
    sp = cl_affinity.active_space(occupations, extra_s, extra_p, extra_d, extra_f, details=True)
    S, eri, A, O, Dc, h_eff = sp["S"], sp["eri"], sp["orbitals"], sp["occupied"], sp["core_density"], sp["h_eff"]
    h_act, g_act = A.T @ h_eff @ A, quantum._transform(eri, A)
    energy, C = quantum.fci(h_act, g_act, n_alpha, n_beta, e_nuc=sp["e_core"], tol=1e-8, low_memory=True,
                            max_vectors=4, return_vector=True)
    # the space of the active orbitals: the valence orbitals and the chosen functions, made orthogonal to the occupied
    B = np.hstack([sp["valence"], sp["raw"] - O @ (O.T @ S @ sp["raw"])])
    M0 = np.linalg.solve(B.T @ S @ B, B.T @ S @ A)
    gradient = []
    for shell in shells(sp, extra_d, extra_f):
        dS, dh, d_eri = integral_derivatives(sp, shell)
        dh_eff = dh + np.einsum("ijkl,kl->ij", d_eri, Dc) - 0.5 * np.einsum("ikjl,kl->ij", d_eri, Dc)
        dB = np.hstack([np.zeros_like(sp["valence"]), -O @ (O.T @ dS @ sp["raw"])])
        dA = dB @ M0                                        # follow the moving space ...
        dA = dA - 0.5 * A @ (A.T @ dS @ A + dA.T @ S @ A + A.T @ S @ dA)   # ... and stay orthonormal
        dh_act = A.T @ dh_eff @ A + dA.T @ h_eff @ A + A.T @ h_eff @ dA
        dg_act = (transform(d_eri, A, A, A, A) + transform(eri, dA, A, A, A) + transform(eri, A, dA, A, A)
                  + transform(eri, A, A, dA, A) + transform(eri, A, A, A, dA))
        gradient.append(quantum.ci_expectation(dh_act, dg_act, n_alpha, n_beta, C))
    return energy, np.array(gradient), sp


def main():
    parse = lambda t: () if t == "-" else tuple(float(x) for x in t.split(","))
    extra_s, extra_p, extra_d, extra_f = (parse(t) for t in sys.argv[1:5])
    closed = (2,) * 9
    t0 = time.time()
    e, g, sp = energy_and_gradient(4, 4, closed, extra_s, extra_p, extra_d, extra_f)
    t1 = time.time()
    print(f"Cl-: energy {e:.8f}   dE/dln(exponent) from the functions: {np.round(1e3 * g, 6)} mEh  ({t1 - t0:.0f} s)", flush=True)
    # the same by central differences of the whole calculation, and of the integrals alone
    step = 1e-3
    shell = shells(sp, extra_d, extra_f)[0]
    dS, dh, d_eri = integral_derivatives(sp, shell)
    both = []
    for sign in (1, -1):
        d = (extra_d[0] * np.exp(sign * step),) + tuple(extra_d[1:])
        s2 = cl_affinity.active_space(closed, extra_s, extra_p, d, extra_f, details=True)
        both.append(s2)
    fd = lambda key: (both[0][key] - both[1][key]) / (2 * step)
    print("integrals, largest difference from central differences:  dS %.1e  dh %.1e  d(ab|cd) %.1e  (largest values %.1e %.1e %.1e)"
          % (np.abs(dS - fd("S")).max(), np.abs(dh - fd("h")).max(), np.abs(d_eri - fd("eri")).max(),
             np.abs(dS).max(), np.abs(dh).max(), np.abs(d_eri).max()), flush=True)
    es = []
    for s2 in both:
        A = s2["orbitals"]
        es.append(quantum.fci(A.T @ s2["h_eff"] @ A, quantum._transform(s2["eri"], A), 4, 4, e_nuc=s2["e_core"],
                              tol=1e-9, low_memory=True, max_vectors=4))
    print(f"energy, central difference of the whole calculation: {1e3 * (es[0] - es[1]) / (2 * step):.6f} mEh  ({time.time() - t1:.0f} s)")


if __name__ == "__main__":
    main()

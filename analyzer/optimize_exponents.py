#!/usr/bin/env python3
"""Exponents of chlorine's correlating d and f functions, found by following
the derivative of the energy instead of being chosen by hand.

The objective is E(Cl-) + E(Cl) with the inner ten electrons frozen. Both
energies are upper bounds, so their sum is one too and can be minimised
freely; the electron affinity (their difference) is then read off, not fitted.
The derivative with respect to the logarithm of each exponent is computed
from the functions themselves by automatic differentiation
(exponent_gradient.py) when the energies are full CI, and by central
differences otherwise. The search is quasi-Newton (BFGS) on those derivatives.

    python3 analyzer/optimize_exponents.py S P D F [determinants]
    python3 analyzer/optimize_exponents.py - 0.1,0.8 0.4 -        # one d exponent, full CI: the small test

S and P are fixed ("-" for none); the D and F values are starting points.
With more than 16 orbitals (or FULL_CI_ORBITALS in the environment) the energies come from mesh_ci with a fixed number
of determinants (default 200000) instead of full CI.
"""
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

import cl_affinity
import quantum

FULL_CI_ORBITALS = int(os.environ.get("FULL_CI_ORBITALS", 16))   # 20 needs about 5 GB and 6 minutes per energy
STEP = 0.05          # step in ln(exponent) for the central difference
CLOSED, OPEN = (2,) * 9, (2,) * 6 + (5 / 3,) * 3


def energies(extra_s, extra_p, extra_d, extra_f, determinants):
    """(E(Cl-), E(Cl)) in hartree."""
    out = []
    for (na, nb), occ in (((4, 4), CLOSED), ((4, 3), OPEN)):
        h, g, e_core, _ = cl_affinity.active_space(occ, extra_s, extra_p, extra_d, extra_f)
        if h.shape[0] <= FULL_CI_ORBITALS:
            out.append(quantum.fci(h, g, na, nb, e_nuc=e_core, tol=1e-8, low_memory=True, max_vectors=4))
        else:
            import mesh_ci
            out.append(mesh_ci.mesh_ci(h, g, na, nb, e_nuc=e_core, max_determinants=determinants, tol=1e-9)[0])
    return out


def main():
    parse = lambda t: () if t == "-" else tuple(float(x) for x in t.split(","))
    extra_s, extra_p, d0, f0 = (parse(t) for t in sys.argv[1:5])
    determinants = int(sys.argv[5]) if len(sys.argv) > 5 else 200000
    nd = len(d0)
    t0 = time.time()
    seen = {}

    import exponent_gradient
    exact = 4 + 3 * len(extra_p) + len(extra_s) + 5 * nd + 7 * len(f0) <= FULL_CI_ORBITALS     # full CI, derivative from the functions

    def both(x):
        """((E(Cl-), E(Cl)), gradient of their sum or None)."""
        key = tuple(np.round(x, 10))
        if key not in seen:
            e = np.exp(x)
            d, f = tuple(e[:nd]), tuple(e[nd:])
            if exact:
                em, gm, _ = exponent_gradient.energy_and_gradient(4, 4, CLOSED, extra_s, extra_p, d, f)
                e0, g0, _ = exponent_gradient.energy_and_gradient(4, 3, OPEN, extra_s, extra_p, d, f)
                seen[key] = ((em, e0), gm + g0)
            else:
                seen[key] = (energies(extra_s, extra_p, d, f, determinants), None)
        return seen[key]

    def objective(x):
        return sum(both(x)[0])

    def gradient(x):
        if exact:
            return both(x)[1]
        g = np.zeros(len(x))
        for k in range(len(x)):
            up, down = x.copy(), x.copy()
            up[k] += STEP
            down[k] -= STEP
            g[k] = (objective(up) - objective(down)) / (2 * STEP)
        return g

    def show(x, label):
        em, e0 = both(x)[0]
        g = gradient(x)
        print(f"{label}: exponents {np.round(np.exp(x), 4)}  E(Cl-) {em:.6f}  E(Cl) {e0:.6f}  sum {em + e0:.6f}  "
              f"dE/dln(exponent) {np.round(1e3 * g, 3)} mEh  electron affinity {(e0 - em) * cl_affinity.HARTREE_EV:.4f} eV  "
              f"({len(seen)} evaluations, {time.time() - t0:.0f} s)", flush=True)

    x0 = np.log(np.array(d0 + f0))
    show(x0, "start")
    step = [0]

    def callback(x):
        step[0] += 1
        show(x, f"step {step[0]}")

    result = minimize(objective, x0, jac=gradient, method="BFGS", callback=callback,
                      options={"gtol": 2e-5, "maxiter": 12})
    show(result.x, "final")


if __name__ == "__main__":
    main()

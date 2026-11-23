#!/usr/bin/env python3
"""The small molecules of diffusion_mc.py computed a second way, with nothing taken from the
literature: full configuration interaction in Gaussian functions (quantum.py).

The two methods fail differently. Full CI is an upper bound whose error is the finite set of
functions (Lean: `rayleigh_ritz_bound`); the walkers have no functions and, for two electrons,
no bias except the statistical one. So for each species
    walkers  <=  full CI   (within the walkers' error), and the gap is the basis error.
A published value is then a third opinion, not the ground the other two stand on.

    python3 analyzer/own_reference.py            # about a minute, below 1.5 GB
"""
import json
import sys
from pathlib import Path

import numpy as np

import quantum
import diffusion_mc as dmc

S_FUNCTIONS = [0.03, 0.09, 0.27, 0.8, 2.4, 7.5, 28.0]      # for charge 1; scaled by charge^2
LOBES = [(0.45, 0.35), (1.4, 0.2)]                         # p functions as pairs of displaced s functions
BOND = [0.35, 1.0, 2.5]                                    # s functions between each pair of nuclei


def basis(nuclei):
    exps, cen = [], []
    for z, r in nuclei:
        r = np.array(r, float)
        exps += [a * z * z for a in S_FUNCTIONS]
        cen += [r] * len(S_FUNCTIONS)
        for a, d in LOBES:
            for axis in range(3):
                for sign in (1, -1):
                    step = np.zeros(3)
                    step[axis] = sign * d / z
                    exps.append(a * z * z)
                    cen.append(r + step)
    for i in range(len(nuclei)):
        for j in range(i):
            mid = (np.array(nuclei[i][1], float) + np.array(nuclei[j][1], float)) / 2
            if not any(np.allclose(mid, r) for _, r in nuclei):
                exps += BOND
                cen += [mid] * len(BOND)
    return exps, np.array(cen)


def integrals(exps, cen, nuclei):
    """As quantum.integrals, with the two-electron part built one row at a time (quantum's
    version holds several arrays of n^4 x 3 numbers at once)."""
    a, A, n = np.asarray(exps, float), np.asarray(cen, float), len(exps)
    S, h = np.zeros((n, n)), np.zeros((n, n))
    for i in range(n):                                           # one-electron parts from pairs of functions
        for j in range(i + 1):
            s2, h2, _ = quantum.integrals([a[i], a[j]], [A[i], A[j]], nuclei)
            S[i, j] = S[j, i] = s2[0, 1]
            h[i, j] = h[j, i] = h2[0, 1]
    p = a[:, None] + a[None, :]
    K = np.exp(-a[:, None] * a[None, :] / p * ((A[:, None] - A[None, :]) ** 2).sum(-1))
    P = (a[:, None, None] * A[:, None] + a[None, :, None] * A[None, :]) / p[:, :, None]
    eri = np.zeros((n, n, n, n))
    for i in range(n):
        pq = p[i, :, None, None] + p[None, :, :]
        R2 = ((P[i, :, None, None, :] - P[None, :, :, :]) ** 2).sum(-1)
        eri[i] = (2.0 * np.pi ** 2.5 / (p[i, :, None, None] * p[None, :, :] * np.sqrt(pq))
                  * K[i, :, None, None] * K[None, :, :] * quantum.boys0(p[i, :, None, None] * p[None, :, :] / pq * R2))
    return S, h, eri


def full_ci(nuclei, n_up, n_dn):
    exps, cen = basis(nuclei)
    S, h, eri = integrals(exps, cen, nuclei)
    e_nuc = sum(nuclei[i][0] * nuclei[j][0] / np.linalg.norm(np.subtract(nuclei[i][1], nuclei[j][1]))
                for i in range(len(nuclei)) for j in range(i))
    if n_up + n_dn == 1:
        X = quantum.orthogonalizer(S, 1e-8)
        return float(np.linalg.eigvalsh(X.T @ h @ X)[0]) + e_nuc, len(exps)
    if (n_up, n_dn) == (1, 1):
        return quantum.fci_two_electrons(S, h, eri, e_nuc=e_nuc), len(exps)
    X = quantum.orthogonalizer(S, 1e-8)
    g = np.einsum("pi,pqrs->iqrs", X, eri)
    g = np.einsum("qj,iqrs->ijrs", X, g)
    g = np.einsum("rk,ijrs->ijks", X, g)
    g = np.einsum("sl,ijks->ijkl", X, g)
    return quantum.fci(X.T @ h @ X, g, n_up, n_dn, e_nuc=e_nuc, tol=1e-6, max_iter=1000, low_memory=True), len(exps)


def main():
    store = {}
    for path, key in ((dmc.STORE, "walkers, simple guess"), (dmc.IMPROVED_STORE, "walkers, improved guess")):
        if path.exists():
            for name, v in json.loads(path.read_text()).items():
                store.setdefault(name, {})[key] = (v["energy"], v["error"]) if isinstance(v, dict) else tuple(v)
    for name, (nuclei, (n_up, n_dn), *_, published, source) in dmc.CASES.items():
        e, n = full_ci(nuclei, n_up, n_dn)
        line = f"{name}: full CI in {n} functions {e:.6f}"
        if name == "He":
            hy = float(quantum.hylleraas_helium())
            line += f"; our Hylleraas calculation {hy:.6f}"
        for key, (w, s) in store.get(name, {}).items():
            line += f"; {key} {w:.5f} ± {s:.5f} (full CI is {1e3 * (e - w):+.2f} mEh above)"
        print(line + f"; published {published:.6f} (full CI is {1e3 * (e - published):+.2f} mEh above)", flush=True)


if __name__ == "__main__":
    main()

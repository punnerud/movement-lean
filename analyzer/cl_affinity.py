#!/usr/bin/env python3
"""Electron affinity of chlorine with correlation between the outer electrons.

Hartree-Fock gives chlorine's extra electron only 2.6 eV of binding; the
measured value is 3.61 eV. The difference is correlation: the eight outer
electrons of Cl- avoid each other better than an average field allows.

Here the inner ten electrons (1s, 2s, 2p) are frozen in their Hartree-Fock
orbitals, and the outer seven (Cl) or eight (Cl-) electrons are treated by
full CI in the 3s and 3p orbitals plus a set of correlating functions.
Each species uses its own Hartree-Fock orbitals.

    python3 analyzer/cl_affinity.py                 # 15 orbitals, 4 GB, 90 s on an M3; writes analyzer/precomputed.json
    python3 analyzer/cl_affinity.py 0.25 0.3 0.6    # s', p', d exponents ("-" for none): test run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

import quantum
from he2_dispersion import contracted_eri, one_electron
from nacl_hf import CL_P, CL_S, build, functions

HARTREE_EV = 27.211386245981
EXTRA_S, EXTRA_P, EXTRA_D = (), (0.1, 0.8), (0.35,)


def spherical_hf(S, h, eri, occupations):
    """Hartree-Fock with given occupation numbers of the lowest orbitals
    (fractions allowed: Cl has 5/3 electrons in each 3p orbital, which keeps
    the atom round). Returns (energy for closed shells, orbitals)."""
    X = quantum.orthogonalizer(S)
    occ = np.asarray(occupations, float)
    D = np.zeros_like(S)
    for it in range(300):
        F = h + np.einsum("ijkl,kl->ij", eri, D) - 0.5 * np.einsum("ikjl,kl->ij", eri, D)
        _, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C
        new = (C[:, :len(occ)] * occ) @ C[:, :len(occ)].T
        if np.abs(new - D).max() < 1e-9:
            break
        D = 0.5 * (D + new) if it < 40 else new
    return float(0.5 * np.sum(D * (h + F))), C


def valence_ci(n_alpha, n_beta, occupations, extra_s, extra_p, extra_d):
    """Frozen-core full CI energy of the outer electrons of chlorine."""
    funcs = functions(CL_S, CL_P)
    n_sp = len(funcs)
    extra = (functions((1.0, 1.0, 0), (1.0, 1.0, 0), extra_d) if extra_d else [])
    exps, cen, W = build([(0.0, funcs + extra)])
    eri = contracted_eri(exps, cen, W)
    S, h = one_electron(exps, cen, W, [(17, (0, 0, 0))])
    sp = np.arange(n_sp)
    e_hf, C_sp = spherical_hf(S[np.ix_(sp, sp)], h[np.ix_(sp, sp)], eri[np.ix_(sp, sp, sp, sp)], occupations)
    n = S.shape[0]
    C = np.zeros((n, 9)); C[:n_sp] = C_sp[:, :9]
    core, valence = C[:, :5], C[:, 5:9]
    Dc = 2.0 * core @ core.T
    G = np.einsum("ijkl,kl->ij", eri, Dc) - 0.5 * np.einsum("ikjl,kl->ij", eri, Dc)
    e_core = float(np.sum(Dc * (h + 0.5 * G)))
    h_eff = h + G
    # correlating functions: chosen primitives, made orthogonal to core and valence
    s_exps, p_exps = quantum.even_tempered(*CL_S), quantum.even_tempered(*CL_P)
    cols = []
    for a in extra_s:
        k = int(np.argmin(np.abs(np.log(np.array(s_exps) / a))))
        e = np.zeros(n); e[k] = 1.0; cols.append(e)
    for a in extra_p:
        k = int(np.argmin(np.abs(np.log(np.array(p_exps) / a))))
        for x in range(3):
            e = np.zeros(n); e[len(s_exps) + 3 * k + x] = 1.0; cols.append(e)
    for k in range(n_sp, n):
        e = np.zeros(n); e[k] = 1.0; cols.append(e)
    E = np.array(cols).T
    occ = C[:, :9]
    E = E - occ @ (occ.T @ S @ E)
    E = E @ quantum.orthogonalizer(E.T @ S @ E, 1e-8)
    A = np.hstack([valence, E])
    # canonical orbitals of the average field, for the CI preconditioner
    Dv = (valence * np.asarray(occupations[5:9], float)) @ valence.T
    F = h_eff + np.einsum("ijkl,kl->ij", eri, Dv) - 0.5 * np.einsum("ikjl,kl->ij", eri, Dv)
    _, U = np.linalg.eigh(A.T @ F @ A)
    A = A @ U
    m = A.shape[1]
    e_ci = quantum.fci(A.T @ h_eff @ A, quantum._transform(eri, A), n_alpha, n_beta, e_nuc=e_core, tol=1e-6)
    return e_hf, e_ci, m


def main():
    t0 = time.time()
    if len(sys.argv) > 1:
        parse = lambda t: () if t == "-" else tuple(float(x) for x in t.split(","))
        extra_s, extra_p, extra_d = (parse(t) for t in sys.argv[1:4])
    else:
        extra_s, extra_p, extra_d = EXTRA_S, EXTRA_P, EXTRA_D
    closed = (2,) * 9
    open_ = (2,) * 6 + (5 / 3,) * 3
    hf_m, ci_m, m = valence_ci(4, 4, closed, extra_s, extra_p, extra_d)
    print(f"Cl-  {m} active orbitals  HF {hf_m:.6f}  valence CI {ci_m:.6f}  ({time.time() - t0:.0f} s)", flush=True)
    _, ci_0, _ = valence_ci(4, 3, open_, extra_s, extra_p, extra_d)
    print(f"Cl   valence CI {ci_0:.6f}  ({time.time() - t0:.0f} s)", flush=True)
    ea = (ci_0 - ci_m) * HARTREE_EV
    print(f"electron affinity {ea:.4f} eV   (measured 3.6127, Hartree-Fock about 2.58)")
    if len(sys.argv) == 1:
        out = Path(__file__).resolve().parent / "precomputed.json"
        data = json.loads(out.read_text()) if out.exists() else {}
        data["Cl_affinity"] = {"extra_s": extra_s, "extra_p": extra_p, "extra_d": extra_d,
                               "active_orbitals": m, "E_Cl_minus_hartree": ci_m, "E_Cl_hartree": ci_0,
                               "E_Cl_minus_hartree_fock": hf_m, "seconds": round(time.time() - t0)}
        out.write_text(json.dumps(data, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Molecules with correlation between the outer electrons: NaCl and N2.

The inner shells stay doubly occupied in Hartree-Fock orbitals of the species.
The outer electrons are treated by selected CI on a free list of determinants
(mesh_ci) in all the other orbitals up to an orbital-energy cutoff. The
binding energy is E(atoms) - E(molecule), each atom computed in the basis of
the whole molecule (so that the atom gains nothing the molecule did not have).

Too large for the regular analyzer run (tables of size orbitals^4, several GB):

    python3 analyzer/molecule_ci.py nacl [determinants]
    python3 analyzer/molecule_ci.py n2 [determinants] [bond length / equilibrium]
    python3 analyzer/molecule_ci.py n2 20000 1.0 small     # small basis: quick test
"""
import sys
import time
from pathlib import Path

import numpy as np

import mesh_ci
import quantum
from he2_dispersion import contracted_eri, f_functions, one_electron
from nacl_hf import BOHR_A, CL_P, CL_S, NA_P, NA_S, build, functions

HARTREE_EV = 27.211386245981
N_S, N_P, N_D, N_F = (0.07, 12000.0, 13), (0.06, 40.0, 7), (0.35, 1.1), (0.6,)
CL_D2, CL_F = (0.15, 0.4, 1.0), (0.3, 0.8)
R_NACL, R_N2 = 2.3609, 1.09768          # measured equilibrium distances, angstrom
E_MAX = 10.0                            # orbitals above this energy (hartree) are left out


def orbitals(S, h, eri, occupations):
    """Hartree-Fock orbitals for given occupation numbers of the lowest
    orbitals (fractions allowed, to keep an open-shell atom round).
    Returns (orbitals, orbital energies)."""
    X = quantum.orthogonalizer(S, 1e-8)
    occ = np.asarray(occupations, float)
    D = np.zeros_like(S)
    for it in range(400):
        F = h + np.einsum("ijkl,kl->ij", eri, D) - 0.5 * np.einsum("ikjl,kl->ij", eri, D)
        eps, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C
        new = (C[:, :len(occ)] * occ) @ C[:, :len(occ)].T
        if np.abs(new - D).max() < 1e-9:
            break
        D = 0.5 * (D + new) if it < 60 else new
    return C, eps


def outer_space(S, h, eri, occupations, frozen, e_max=E_MAX):
    """Integrals for the electrons outside `frozen` doubly occupied orbitals,
    in the remaining Hartree-Fock orbitals with energy below e_max.
    Returns (h, eri, energy of the frozen part)."""
    C, eps = orbitals(S, h, eri, occupations)
    C = quantum.settle_shells(C, eps)
    core = C[:, :frozen]
    Dc = 2.0 * core @ core.T
    G = np.einsum("ijkl,kl->ij", eri, Dc) - 0.5 * np.einsum("ikjl,kl->ij", eri, Dc)
    A = C[:, frozen:][:, eps[frozen:] < e_max]
    return A.T @ (h + G) @ A, quantum._transform(eri, A), float(np.sum(Dc * (h + 0.5 * G)))


def outer_ci(name, S, h, eri, occupations, frozen, n_alpha, n_beta, e_nuc, determinants, tag, exact=False):
    t0 = time.time()
    ha, ga, e_core = outer_space(S, h, eri, occupations, frozen)
    if exact:                                              # small test cases only
        e = quantum.fci(ha, ga, n_alpha, n_beta, e_nuc=e_core + e_nuc, tol=1e-7, low_memory=True)
        print(f"{name}: {ha.shape[0]} orbitals, full CI {e:.6f}", flush=True)
        return e
    print(f"{name}: {n_alpha}+{n_beta} outer electrons in {ha.shape[0]} orbitals", flush=True)
    e, rest, _ = mesh_ci.mesh_ci(
        ha, ga, n_alpha, n_beta, e_nuc=e_core + e_nuc, max_determinants=determinants, store="content",
        report=lambda nd, en, r: print(f"    {nd:9d} determinants  {en:.6f}   with estimate of the rest {en + r:.6f}", flush=True))
    print(f"{name}: energy {e:.6f}  with estimate of the rest {e + rest:.6f}  ({time.time() - t0:.0f} s)", flush=True)
    return e, rest


def nacl(determinants, small=False, exact=False):
    R = R_NACL / BOHR_A
    if small:
        na, cl = functions((0.03, 300.0, 6), (0.1, 3.0, 2)), functions((0.1, 800.0, 7), (0.15, 20.0, 3))
    else:
        na = functions(NA_S, NA_P)
        cl = functions(CL_S, CL_P, CL_D2) + f_functions(CL_F)
    exps, cen, W = build([(0.0, na), (R, cl)])
    eri = contracted_eri(exps, cen, W)
    S, h = one_electron(exps, cen, W, [(11, (0, 0, 0)), (17, (0, 0, R))])
    _, h_na = one_electron(exps, cen, W, [(11, (0, 0, 0))])
    _, h_cl = one_electron(exps, cen, W, [(17, (0, 0, R))])
    print(f"NaCl at {R_NACL} angstrom, {W.shape[1]} functions", flush=True)
    tag = "NaCl_small" if small else "NaCl"
    run = lambda *a: outer_ci(*a, determinants, tag, exact)
    mol = run("NaCl", S, h, eri, (2,) * 14, 10, 4, 4, 11 * 17 / R)
    atom = run("Cl", S, h_cl, eri, (2,) * 6 + (5 / 3,) * 3, 5, 4, 3, 0.0)
    e_na = quantum.uhf(S, h_na, eri, 6, 5)                # one outer electron: Hartree-Fock is all there is
    print(f"Na: Hartree-Fock {e_na:.6f}", flush=True)
    report("NaCl", mol, [atom, (e_na, 0.0)], 4.2526, exact)


def n2(determinants, scale=1.0, small=False, exact=False):
    R = scale * R_N2 / BOHR_A
    if small:
        atom = functions((0.15, 400.0, 6), (0.2, 5.0, 2))
    else:
        atom = functions(N_S, N_P, N_D) + f_functions(N_F)
    exps, cen, W = build([(0.0, atom), (R, atom)])
    eri = contracted_eri(exps, cen, W)
    S, h = one_electron(exps, cen, W, [(7, (0, 0, 0)), (7, (0, 0, R))])
    _, h_a = one_electron(exps, cen, W, [(7, (0, 0, 0))])
    print(f"N2 at {scale:g} x {R_N2} angstrom, {W.shape[1]} functions", flush=True)
    tag = ("N2_small" if small else "N2") + f"_{scale:g}"
    run = lambda *a: outer_ci(*a, determinants, tag, exact)
    mol = run("N2", S, h, eri, (2,) * 7, 2, 5, 5, 49 / R)
    atom_e = run("N", S, h_a, eri, (2, 2, 1, 1, 1), 1, 4, 1, 0.0)
    report("N2", mol, [atom_e, atom_e], 9.905 if scale == 1.0 else None, exact)


def report(name, molecule, atoms, measured, exact):
    if exact:
        print(f"{name}: binding energy {(sum(a for a, _ in atoms) if isinstance(atoms[0], tuple) else sum(atoms)) - molecule:.6f} hartree (full CI)")
        return
    plain = (sum(a for a, _ in atoms) - molecule[0]) * HARTREE_EV
    with_rest = (sum(a + r for a, r in atoms) - molecule[0] - molecule[1]) * HARTREE_EV
    note = f"   (measured {measured})" if measured else ""
    print(f"{name}: binding energy {plain:.4f} eV, with estimate of the rest {with_rest:.4f} eV{note}")


def main():
    which = sys.argv[1]
    determinants = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    small = "small" in sys.argv
    exact = "exact" in sys.argv
    if which == "nacl":
        nacl(determinants, small, exact)
    else:
        n2(determinants, float(sys.argv[3]) if len(sys.argv) > 3 else 1.0, small, exact)


if __name__ == "__main__":
    main()

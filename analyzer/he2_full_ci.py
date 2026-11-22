#!/usr/bin/env python3
"""Full CI for two helium atoms (four electrons), to get the repulsion right.

Hartree-Fock misses part of the repulsion between two helium atoms: the
electrons inside each atom avoid each other, which changes how the two
clouds overlap. A four-electron full CI has all of that, but in any basis
small enough to run it also has only part of the attraction (dispersion).
So the result is split:

    short-range part = full CI interaction - second-order dispersion,
                       both in the same basis

and the analyzer adds the complete dispersion series from single-atom
calculations to that. Each atom is computed in the basis of the whole pair
(counterpoise), so borrowed basis functions do not show up as attraction.

Too large for the regular analyzer run (about 6 GB of memory and 80 s per
distance on an M3):

    python3 analyzer/he2_full_ci.py            # writes analyzer/precomputed.json
    python3 analyzer/he2_full_ci.py small 5.6  # quick test: small basis, one distance
    python3 analyzer/he2_full_ci.py custom 0,1,2,3,4,5 0.15,0.5,1.8 0.5 5.6   # s_keep, p, d, distances
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import quantum

S_KEEP = (0, 1, 2, 3, 4, 5)             # primitive s functions kept besides the 1s orbital
P_EXPONENTS = (0.15, 0.5, 1.8)
D_EXPONENTS = ()                       # one d shell (0.5) was tried at 5.6 bohr: 42 orbitals, 16 GB, 19 minutes
DISTANCES = (3.0, 3.5, 4.0, 4.5, 5.0, 5.6, 6.2, 7.0, 9.0)
SMALL = ((1, 3, 5), (0.3, 1.2), ())     # (s_keep, p, d) for test runs


def atom_functions(s_keep, p_exponents, d_exponents, delta_p=0.1, delta_d=0.3):
    """Basis functions of one atom at the origin, each a list of
    (exponent, position, weight) over s-Gaussians."""
    s_exps = quantum.even_tempered(*quantum.ATOM_BASIS[2])
    S, h, eri = quantum.atom_integrals(2)
    C, _ = quantum._average_hf(S, h, eri, len(s_exps), (2.0,))
    funcs = [[(a, (0.0, 0.0, 0.0), float(c)) for a, c in zip(s_exps, C[:, 0])]]
    funcs += [[(s_exps[k], (0.0, 0.0, 0.0), 1.0)] for k in s_keep]

    def at(axis, d, extra=None):
        pos = [0.0, 0.0, 0.0]
        pos[axis] = d
        if extra:
            pos[extra[0]] = extra[1]
        return tuple(pos)

    for a in p_exponents:
        d = delta_p / math.sqrt(a)
        funcs += [[(a, at(x, d), 1.0), (a, at(x, -d), -1.0)] for x in range(3)]
    for a in d_exponents:
        d = delta_d / math.sqrt(a)
        for u, v in ((0, 1), (0, 2), (1, 2)):
            funcs.append([(a, at(u, su * d, (v, sv * d)), su * sv) for su in (1, -1) for sv in (1, -1)])
        funcs.append([(a, at(0, d), 1.0), (a, at(0, -d), 1.0), (a, at(1, d), -1.0), (a, at(1, -d), -1.0)])
        funcs.append([(a, at(2, d), 2.0), (a, at(2, -d), 2.0)]
                     + [(a, at(x, s * d), -1.0) for x in (0, 1) for s in (1, -1)])
    return funcs


def pair_integrals(R, funcs):
    """Orthonormal orbitals for the pair: h with both nuclei, h with nucleus A
    only, h with nucleus B only, and (pq|rs)."""
    exps, cen, index, cols = [], [], {}, []
    for zc in (0.0, R):
        for f in funcs:
            col = {}
            for a, pos, w in f:
                key = (a, pos[0], pos[1], pos[2] + zc)
                if key not in index:
                    index[key] = len(exps)
                    exps.append(a); cen.append(key[1:])
                col[index[key]] = col.get(index[key], 0.0) + w
            cols.append(col)
    W = np.zeros((len(exps), len(cols)))
    for c, col in enumerate(cols):
        for k, v in col.items():
            W[k, c] = v
    cen = np.array(cen)
    A, B = (2, (0, 0, 0)), (2, (0, 0, R))
    S, h_ab, eri = quantum.contracted_integrals(exps, cen, [A, B], W)
    _, h_a, _ = quantum.contracted_integrals(exps, cen, [A], W, two_electron=False)
    _, h_b, _ = quantum.contracted_integrals(exps, cen, [B], W, two_electron=False)
    X = quantum.orthogonalizer(S, 1e-8)
    return S, eri, X, [X.T @ h @ X for h in (h_ab, h_a, h_b)], quantum._transform(eri, X), (h_ab, h_a)


def atom_states(h, g):
    """All singlet states of one two-electron atom in the pair's orbitals:
    (energies, transition densities from the ground state, shape (states, m*m))."""
    m = h.shape[0]
    eye = np.eye(m)
    I, J = np.triu_indices(m)
    norm = np.where(I == J, 0.5, math.sqrt(0.5))
    a, b, c, d = I[:, None], J[:, None], I[None, :], J[None, :]

    def element(i, j, k, l):
        return h[i, k] * eye[j, l] + eye[i, k] * h[j, l] + g[i, k, j, l]
    H = ((element(a, b, c, d) + element(b, a, c, d) + element(a, b, d, c) + element(b, a, d, c))
         * norm[:, None] * norm[None, :])
    w, V = np.linalg.eigh(H)
    coef = np.zeros((len(w), m, m))               # pair function of each state as a symmetric matrix
    scaled = V.T * np.where(I == J, 1.0, math.sqrt(0.5))[None, :]
    coef[:, I, J] = scaled
    coef[:, J, I] = scaled
    dens = 2.0 * np.einsum("pj,nqj->npq", coef[0], coef[1:])
    return w, dens.reshape(len(w) - 1, m * m)


def dispersion(states_a, states_b, g):
    """Second-order dispersion energy: each atom excited, coupled by the
    full Coulomb interaction between the two transition densities."""
    (wa, da), (wb, db) = states_a, states_b
    m2 = da.shape[1]
    coupling = da @ g.reshape(m2, m2) @ db.T
    gaps = (wa[1:] - wa[0])[:, None] + (wb[1:] - wb[0])[None, :]
    return -float(np.sum(coupling ** 2 / gaps))


def point(R, funcs):
    t0 = time.time()
    S, eri, X, (h_ab, h_a, h_b), g, raw = pair_integrals(R, funcs)
    m = X.shape[1]
    sa, sb = atom_states(h_a, g), atom_states(h_b, g)
    e_atoms = sa[0][0] + sb[0][0]
    e_disp = dispersion(sa, sb, g)
    hf = quantum.uhf(S, raw[0], eri, 2, 2, 4.0 / R) - 2.0 * quantum.uhf(S, raw[1], eri, 1, 1)
    e_pair = quantum.fci(h_ab, g, 2, 2, e_nuc=4.0 / R, tol=1e-6)
    out = {"R": R, "orbitals": m, "full_ci": e_pair - e_atoms, "dispersion": e_disp, "hartree_fock": hf,
           "seconds": round(time.time() - t0)}
    k = 315775.0248
    print(f"R = {R:4.1f}  {m} orbitals  full CI {out['full_ci'] * k:10.4f} K   dispersion {e_disp * k:9.4f} K   "
          f"HF {hf * k:10.4f} K   extra repulsion {(out['full_ci'] - e_disp - hf) * k:8.4f} K   "
          f"({out['seconds']} s)", flush=True)
    return out


def main():
    small = len(sys.argv) > 1 and sys.argv[1] == "small"
    s_keep, p_exps, d_exps = SMALL if small else (S_KEEP, P_EXPONENTS, D_EXPONENTS)
    distances = DISTANCES
    if small and len(sys.argv) > 2:
        distances = [float(x) for x in sys.argv[2:]]
    if len(sys.argv) > 1 and sys.argv[1] == "custom":      # custom s_keep p d distances...
        parse = lambda t, kind: () if t == "-" else tuple(kind(x) for x in t.split(","))
        s_keep, p_exps, d_exps = parse(sys.argv[2], int), parse(sys.argv[3], float), parse(sys.argv[4], float)
        distances = [float(x) for x in sys.argv[5:]]
    funcs = atom_functions(s_keep, p_exps, d_exps)
    points = [point(R, funcs) for R in distances]
    if len(sys.argv) == 1:
        out = Path(__file__).resolve().parent / "precomputed.json"
        data = json.loads(out.read_text()) if out.exists() else {}
        data["He2_full_ci"] = {"s_keep": s_keep, "p_exponents": p_exps, "d_exponents": d_exps,
                               "points": points}
        out.write_text(json.dumps(data, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()

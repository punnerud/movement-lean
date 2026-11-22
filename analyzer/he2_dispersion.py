#!/usr/bin/env python3
"""The attraction between two helium atoms, computed directly in a large basis.

Second-order dispersion with the full Coulomb interaction between the two
atoms (no expansion in powers of 1/R, no damping function):

    E_disp = - sum_{m,n} |<0 0| V |m n>|^2 / (w_m + w_n)

m and n run over every excited state of atom A and of atom B. The states come
from a full CI of each two-electron atom in one large shared basis: s, p, d
and f functions on both nuclei and extra functions at the midpoint.

Too large for the regular analyzer run (about 120 functions, 7000 states per
atom, several GB of memory, minutes per distance):

    python3 analyzer/he2_dispersion.py            # writes analyzer/precomputed.json
    python3 analyzer/he2_dispersion.py small 5.6  # quick test: s and p on the atoms only
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

import quantum
from he2_full_ci import atom_functions, atom_states, dispersion

S_KEEP = (0, 1, 2, 3, 4, 5, 6)
P_EXPONENTS = (0.06, 0.18, 0.5, 1.4, 4.0)
D_EXPONENTS = (0.1, 0.35, 1.1)
F_EXPONENTS = (0.15, 0.5)
MID_S, MID_P, MID_D = (0.1, 0.3, 0.9), (0.2, 0.6), (0.35,)
DISTANCES = (3.0, 3.5, 4.0, 4.5, 5.0, 5.6, 6.2, 7.0, 9.0)


def f_functions(exponents, delta=0.4):
    out = []
    for a in exponents:
        d = delta / math.sqrt(a)
        for shape in quantum._F_SHAPES:
            out.append([(a, (d * x, d * y, d * z), w)
                        for (x, y, z), w in quantum._difference_lobes(shape).items()])
    return out


def basis(R, small=False):
    """Exponents, centres and contraction matrix W for the shared basis."""
    if small:
        s_exps = quantum.even_tempered(*quantum.ATOM_BASIS[2])
        atom = [[(a, (0.0, 0.0, 0.0), 1.0)] for a in s_exps] + atom_functions((), quantum.HE_RESPONSE_P, ())[1:]
        mid = []
    else:
        atom = atom_functions(S_KEEP, P_EXPONENTS, D_EXPONENTS) + f_functions(F_EXPONENTS)
        mid = [[(a, (0.0, 0.0, 0.0), 1.0)] for a in MID_S] + atom_functions((), MID_P, MID_D)[1:]
    exps, cen, index, cols = [], [], {}, []
    for zc, funcs in ((0.0, atom), (R, atom), (R / 2.0, mid)):
        for f in funcs:
            col = {}
            for a, pos, w in f:
                key = (a, round(pos[0], 12), round(pos[1], 12), round(pos[2] + zc, 12))
                if key not in index:
                    index[key] = len(exps)
                    exps.append(a); cen.append(key[1:])
                col[index[key]] = col.get(index[key], 0.0) + w
            cols.append(col)
    W = np.zeros((len(exps), len(cols)))
    for c, col in enumerate(cols):
        for k, v in col.items():
            W[k, c] = v
    return np.array(exps), np.array(cen), W, len(atom)


def contracted_eri(exps, cen, W):
    """(ab|cd) for the contracted functions, using the symmetry of each pair
    and the sparsity of W, so that the primitive array is never stored."""
    n, nc = W.shape
    I, J = np.triu_indices(n)
    p = exps[I] + exps[J]
    K = np.exp(-exps[I] * exps[J] / p * ((cen[I] - cen[J]) ** 2).sum(-1))
    P = (exps[I, None] * cen[I] + exps[J, None] * cen[J]) / p[:, None]
    A, B = np.triu_indices(nc)
    pair_of = np.zeros((nc, nc), dtype=np.int64)
    pair_of[A, B] = pair_of[B, A] = np.arange(len(A))
    Ws = sparse.csr_matrix(W)
    rows, cols_, vals = [], [], []
    for u in range(len(I)):                      # primitive pair -> contracted pairs
        i, j = I[u], J[u]
        ai, vi = Ws.indices[Ws.indptr[i]:Ws.indptr[i + 1]], Ws.data[Ws.indptr[i]:Ws.indptr[i + 1]]
        aj, vj = Ws.indices[Ws.indptr[j]:Ws.indptr[j + 1]], Ws.data[Ws.indptr[j]:Ws.indptr[j + 1]]
        if len(ai) == 0 or len(aj) == 0:
            continue
        w = np.outer(vi, vj)
        a, b = np.meshgrid(ai, aj, indexing="ij")
        # ordered function pairs fold onto a <= b: see the weights T[a,b] = W_ia W_jb + W_ja W_ib
        w = w * (np.where(a == b, 2.0, 1.0) if i != j else np.where(a == b, 1.0, 0.5))
        rows.append(np.full(w.size, u)); cols_.append(pair_of[a, b].ravel()); vals.append(w.ravel())
    Wp = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols_))),
                           shape=(len(I), len(A)))
    WpT = Wp.T.tocsr()
    npair = len(I)
    sq = (P ** 2).sum(-1)
    out = np.zeros((len(A), len(A)))
    step = max(1, 6_000_000 // npair)
    for lo in range(0, npair, step):
        hi = min(npair, lo + step)
        pq = p[lo:hi, None] + p[None, :]
        R2 = np.maximum(sq[lo:hi, None] + sq[None, :] - 2.0 * P[lo:hi] @ P.T, 0.0)
        block = (2.0 * np.pi ** 2.5 / (p[lo:hi, None] * p[None, :] * np.sqrt(pq))
                 * K[lo:hi, None] * K[None, :] * quantum.boys0(p[lo:hi, None] * p[None, :] / pq * R2))
        half = (WpT @ block.T).T                  # (slice, contracted pairs)
        out += WpT[:, lo:hi] @ half
    # `out` holds sums over ordered function pairs folded onto a <= b; unfold to the full array
    return out[pair_of.ravel()][:, pair_of.ravel()].reshape(nc, nc, nc, nc)


def one_electron(exps, cen, W, nuclei, dipole=False):
    """Overlap and core Hamiltonian; with dipole=True also the matrix of z."""
    a, A = exps, cen
    p = a[:, None] + a[None, :]
    red = a[:, None] * a[None, :] / p
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    K = np.exp(-red * d2)
    P = (a[:, None, None] * A[:, None, :] + a[None, :, None] * A[None, :, :]) / p[:, :, None]
    S = (np.pi / p) ** 1.5 * K
    h = red * (3.0 - 2.0 * red * d2) * S
    for charge, pos in nuclei:
        r2 = ((P - np.asarray(pos, float)) ** 2).sum(-1)
        h = h - charge * 2.0 * np.pi / p * K * quantum.boys0(p * r2)
    if dipole:
        return W.T @ S @ W, W.T @ h @ W, W.T @ (P[:, :, 2] * S) @ W
    return W.T @ S @ W, W.T @ h @ W


def point(R, small=False):
    t0 = time.time()
    exps, cen, W, n_atom = basis(R, small)
    if small:                                  # each atom in its own functions only
        own = [np.arange(n_atom), np.arange(n_atom, 2 * n_atom)]
    eri = contracted_eri(exps, cen, W)
    t1 = time.time()
    S, h_a = one_electron(exps, cen, W, [(2, (0, 0, 0))])
    _, h_b = one_electron(exps, cen, W, [(2, (0, 0, R))])
    if small:
        states, X = [], []
        for idx, h in zip(own, (h_a, h_b)):
            x = quantum.orthogonalizer(S[np.ix_(idx, idx)], 1e-8)
            X.append((idx, x))
            states.append(atom_states(x.T @ h[np.ix_(idx, idx)] @ x,
                                      quantum._transform(eri[np.ix_(idx, idx, idx, idx)], x)))
        (ia, xa), (ib, xb) = X
        g = np.einsum("pi,qj,pqrs,rk,sl->ijkl", xa, xa, eri[np.ix_(ia, ia, ib, ib)], xb, xb, optimize=True)
        m = xa.shape[1]
        (wa, da), (wb, db) = states
        e = -float(np.sum((da @ g.reshape(m * m, -1) @ db.T) ** 2
                          / ((wa[1:] - wa[0])[:, None] + (wb[1:] - wb[0])[None, :])))
    else:
        X = quantum.orthogonalizer(S, 1e-8)
        m = X.shape[1]
        g = quantum._transform(eri, X)
        del eri
        e = dispersion(atom_states(X.T @ h_a @ X, g), atom_states(X.T @ h_b @ X, g), g)
    out = {"R": R, "functions": int(m), "primitives": int(len(exps)), "dispersion": e,
           "seconds": round(time.time() - t0)}
    print(f"R = {R:4.1f}  {m} functions  dispersion {e * 315775.0248:10.4f} K  "
          f"(integrals {t1 - t0:.0f} s, total {out['seconds']} s)", flush=True)
    print("JSON", json.dumps(out), flush=True)
    return out


def main():
    small = len(sys.argv) > 1 and sys.argv[1] == "small"
    args = sys.argv[2:] if small else sys.argv[1:]
    distances = [float(x) for x in args] if args else DISTANCES
    points = [point(R, small) for R in distances]
    if len(sys.argv) == 1:
        out = Path(__file__).resolve().parent / "precomputed.json"
        data = json.loads(out.read_text()) if out.exists() else {}
        data["He2_dispersion"] = {"s_keep": S_KEEP, "p_exponents": P_EXPONENTS, "d_exponents": D_EXPONENTS,
                                  "f_exponents": F_EXPONENTS, "midbond_s": MID_S, "midbond_p": MID_P,
                                  "midbond_d": MID_D, "points": points}
        out.write_text(json.dumps(data, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()

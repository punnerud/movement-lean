#!/usr/bin/env python3
"""NaCl from first principles: Hartree-Fock for all 28 electrons.

No measured input except the nuclear charges: the bond length, vibration
frequency, dipole moment and binding energy all come out of the energy
curve E(R). Basis: even-tempered s-Gaussians, p and d functions as
differences of shifted s-Gaussians, on both nuclei.

About five minutes and 2.5 GB, so it is run separately and stored:

    python3 analyzer/nacl_hf.py            # writes analyzer/precomputed.json
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import quantum
from he2_dispersion import contracted_eri, one_electron

BOHR_A = 0.529177210903
NA_S, NA_P = (0.012, 80000.0, 19), (0.05, 200.0, 8)
CL_S, CL_P, CL_D = (0.06, 200000.0, 19), (0.04, 600.0, 10), (0.25, 0.8)
DISTANCES_A = (2.20, 2.30, 2.36, 2.40, 2.45, 2.50, 2.60)
FIELD = 0.001


def functions(s, p, d=(), delta_p=0.1, delta_d=0.3):
    """Basis functions of one atom at the origin: lists of (exponent, position, weight)."""
    out = [[(a, (0.0, 0.0, 0.0), 1.0)] for a in quantum.even_tempered(*s)]

    def at(axis, dist, extra=None):
        pos = [0.0, 0.0, 0.0]
        pos[axis] = dist
        if extra:
            pos[extra[0]] = extra[1]
        return tuple(pos)

    for a in quantum.even_tempered(*p):
        dist = delta_p / math.sqrt(a)
        out += [[(a, at(x, dist), 1.0), (a, at(x, -dist), -1.0)] for x in range(3)]
    for a in d:
        dist = delta_d / math.sqrt(a)
        for u, v in ((0, 1), (0, 2), (1, 2)):
            out.append([(a, at(u, su * dist, (v, sv * dist)), su * sv) for su in (1, -1) for sv in (1, -1)])
        out.append([(a, at(0, dist), 1.0), (a, at(0, -dist), 1.0), (a, at(1, dist), -1.0), (a, at(1, -dist), -1.0)])
        out.append([(a, at(2, dist), 2.0), (a, at(2, -dist), 2.0)]
                   + [(a, at(x, s_ * dist), -1.0) for x in (0, 1) for s_ in (1, -1)])
    return out


def build(centres):
    exps, cen, cols = [], [], []
    for zc, funcs in centres:
        for f in funcs:
            col = {}
            for a, pos, w in f:
                exps.append(a); cen.append((pos[0], pos[1], pos[2] + zc)); col[len(exps) - 1] = w
            cols.append(col)
    W = np.zeros((len(exps), len(cols)))
    for c, col in enumerate(cols):
        for k, v in col.items():
            W[k, c] = v
    return np.array(exps), np.array(cen), W


def point(R, with_atoms=False):
    """Energies at distance R (bohr): molecule, molecule in a field of +-FIELD
    along the axis, and each ion alone in the basis of the whole molecule."""
    exps, cen, W = build([(0.0, functions(NA_S, NA_P)), (R, functions(CL_S, CL_P, CL_D))])
    eri = contracted_eri(exps, cen, W)
    na, cl = (11, (0, 0, 0)), (17, (0, 0, R))
    S, h, z = one_electron(exps, cen, W, [na, cl], dipole=True)
    e_nuc = 11 * 17 / R
    out = {"R_bohr": R, "functions": int(W.shape[1]),
           "molecule": quantum.uhf(S, h, eri, 14, 14, e_nuc)}
    # field F along z: electrons gain +F z, nuclei -F Z z; dipole = -dE/dF
    for sign, key in ((1.0, "field_plus"), (-1.0, "field_minus")):
        f = sign * FIELD
        out[key] = quantum.uhf(S, h + f * z, eri, 14, 14, e_nuc - f * 17 * R)
    _, h_na = one_electron(exps, cen, W, [na])
    _, h_cl = one_electron(exps, cen, W, [cl])
    out["Na_plus"] = quantum.uhf(S, h_na, eri, 5, 5)
    out["Cl_minus"] = quantum.uhf(S, h_cl, eri, 9, 9)
    if with_atoms:
        out["Na"] = quantum.uhf(S, h_na, eri, 6, 5)
        out["Cl"] = quantum.uhf(S, h_cl, eri, 9, 8)
    return out


def main():
    t0 = time.time()
    points = []
    for r in DISTANCES_A:
        p = point(r / BOHR_A, with_atoms=(r == 2.40))
        p["R_angstrom"] = r
        points.append(p)
        dip = -(p["field_plus"] - p["field_minus"]) / (2 * FIELD) * 2.541746473
        print(f"R = {r:.2f} Å  E = {p['molecule']:.6f}  binding to ions "
              f"{(p['Na_plus'] + p['Cl_minus'] - p['molecule']) * 27.211386:.4f} eV  dipole {dip:.3f} D  "
              f"({time.time() - t0:.0f} s)", flush=True)
    if len(sys.argv) == 1:
        out = Path(__file__).resolve().parent / "precomputed.json"
        data = json.loads(out.read_text()) if out.exists() else {}
        data["NaCl_hartree_fock"] = {"Na_s": NA_S, "Na_p": NA_P, "Cl_s": CL_S, "Cl_p": CL_P, "Cl_d": CL_D,
                                     "field_au": FIELD, "seconds": round(time.time() - t0), "points": points}
        out.write_text(json.dumps(data, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What the determinant list for chlorine leaves out: the second-order
estimate for a series of thresholds, from a stored mesh_ci state.

    python3 analyzer/cl_rest.py 0.15,0.6 0.1,0.4,1.2 0.15,0.4,1.0 0.3,0.8 [thresholds]
"""
import sys
from pathlib import Path

import numpy as np

import cl_affinity
import mesh_ci
import quantum


def main():
    parse = lambda t: () if t == "-" else tuple(float(x) for x in t.split(","))
    extra = [parse(t) for t in sys.argv[1:5]]
    thresholds = parse(sys.argv[5]) if len(sys.argv) > 5 else (1e-4, 3e-5, 1e-5, 3e-6, 1e-6)
    label = "_".join(",".join(f"{x:g}" for x in t) or "-" for t in extra)
    results = {}
    for name, n_alpha, n_beta, occ in (("Cl-", 4, 4, (2,) * 9), ("Cl", 4, 3, (2,) * 6 + (5 / 3,) * 3)):
        ha, ga, e_core, _ = cl_affinity.active_space(occ, *extra)
        with np.load(Path(__file__).resolve().parent / "ci_cache" / f"Cl_mesh_{n_alpha}{n_beta}_{label}.npz") as f:
            alpha, beta, c = f["alpha"], f["beta"], f["c"].astype(float)
        mesh = mesh_ci.Mesh(ha, ga, alpha, beta)
        mesh.plan()
        energy, c = mesh.solve(start=c)
        print(f"{name}: {len(c)} determinants, energy {energy + e_core:.6f}", flush=True)
        table = mesh_ci.rest_table(mesh, c, energy, thresholds, report=lambda eps, total, found, parts: print(
            f"    threshold {eps:7.0e}  rest {total:.6f}  energy with rest {energy + e_core + total:.6f}"
            f"  ({found} determinants, {parts} pieces)", flush=True))
        results[name] = (energy + e_core, table)
    for k in range(len(thresholds)):
        ea = (results["Cl"][0] + results["Cl"][1][k][1]) - (results["Cl-"][0] + results["Cl-"][1][k][1])
        print(f"threshold {thresholds[k]:7.0e}: electron affinity with rest {ea * cl_affinity.HARTREE_EV:.4f} eV")
    print(f"without rest: {(results['Cl'][0] - results['Cl-'][0]) * cl_affinity.HARTREE_EV:.4f} eV")


if __name__ == "__main__":
    main()

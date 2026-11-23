#!/usr/bin/env python3
"""Benchmark against PySCF on the same integrals: Cl- with the inner ten
electrons frozen, full CI in the default 20 orbitals (23 million determinants).

Needs PySCF (pip install pyscf); it is not used anywhere else in this repository.

    python3 analyzer/bench_pyscf.py            # PySCF full CI
    python3 analyzer/bench_pyscf.py own        # this repository's full CI
    python3 analyzer/bench_pyscf.py mesh 1500000   # determinant mesh
"""
import resource
import sys
import time

import cl_affinity
import quantum

mode = sys.argv[1] if len(sys.argv) > 1 else "pyscf"
t0 = time.time()
h, g, e_core, _ = cl_affinity.active_space((2,) * 9, cl_affinity.EXTRA_S, cl_affinity.EXTRA_P, cl_affinity.EXTRA_D)
t1 = time.time()
n = h.shape[0]
if mode == "pyscf":
    from pyscf import fci
    e, _ = fci.direct_spin1.kernel(h, g, n, (4, 4), conv_tol=1e-10, max_memory=20000)
    e += e_core
elif mode == "own":
    e = quantum.fci(h, g, 4, 4, e_nuc=e_core, tol=1e-6, low_memory=True)
else:
    import mesh_ci
    e, _, _ = mesh_ci.mesh_ci(h, g, 4, 4, e_nuc=e_core, max_determinants=int(sys.argv[2]))
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss = rss / 1e9 if sys.platform == "darwin" else rss / 1e6
print(f"{mode}: {n} orbitals  energy {e:.8f}  integrals {t1 - t0:.0f} s  CI {time.time() - t1:.0f} s  peak memory {rss:.1f} GB")

#!/usr/bin/env python3
"""Check of the 128-bit kernel: more than 64 orbitals.

1. Cl- in 20 orbitals with 64-bit and with 128-bit strings: same energy.
2. The same 20 orbitals placed among 70, with the real ones above bit 64 and
   the other 50 uncoupled and far up in energy: same energy again.

Needs about 2 GB (tables of size orbitals^4); run it on a machine with room.

    python3 analyzer/test_wide.py [determinants]
"""
import sys
import time

import numpy as np

import cl_affinity
import mesh_ci

count = int(sys.argv[1]) if len(sys.argv) > 1 else 60000
h, g, e_core, _ = cl_affinity.active_space((2,) * 9, cl_affinity.EXTRA_S, cl_affinity.EXTRA_P, cl_affinity.EXTRA_D)
n = h.shape[0]
energies = []
for wide in (False, True):
    t = time.time()
    e, rest, _ = mesh_ci.mesh_ci(h, g, 4, 4, e_nuc=e_core, max_determinants=count, wide=wide)
    energies.append(e)
    print(f"{n} orbitals, {'128' if wide else '64'}-bit strings: energy {e:.10f}  rest {rest:.10f}  {time.time() - t:.1f} s", flush=True)
big = 70
real = list(range(4)) + list(range(big - (n - 4), big))       # the four occupied stay lowest, the rest go to the top
dummy = [i for i in range(big) if i not in real]
H = np.zeros((big, big)); H[np.ix_(real, real)] = h
H[dummy, dummy] = 50.0
G = np.zeros((big,) * 4); G[np.ix_(real, real, real, real)] = g
t = time.time()
e, rest, _ = mesh_ci.mesh_ci(H, G, 4, 4, e_nuc=e_core, max_determinants=count)
energies.append(e)
print(f"{big} orbitals, real ones above bit 64: energy {e:.10f}  rest {rest:.10f}  {time.time() - t:.1f} s", flush=True)
spread = max(energies) - min(energies)
print("OK" if spread < 1e-8 else "MISMATCH", f"(spread {spread:.1e})")
sys.exit(0 if spread < 1e-8 else 1)

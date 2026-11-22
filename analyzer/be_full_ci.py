#!/usr/bin/env python3
"""All-electron full CI for beryllium with s, p and d orbitals.

Too large for the regular analyzer run: about 550 000 determinants and roughly
15 GB of memory. Run it on a machine with 32 GB or more:

    python3 analyzer/be_full_ci.py            # writes analyzer/precomputed.json

The analyzer reads the stored result and labels the row as precomputed.
"""
import json
import sys
import time
from pathlib import Path

import quantum

S_KEEP = (1, 2, 3, 4, 5, 7, 9)
P_EXPONENTS = (0.06, 0.2, 0.7, 2.5, 8.0)
D_EXPONENTS = (0.2, 0.7, 2.5)


def main():
    d_exps = tuple(float(x) for x in sys.argv[1].split(",")) if len(sys.argv) > 1 else D_EXPONENTS
    t0 = time.time()
    h, eri = quantum.atom_sp_orbitals(4, (2, 2), S_KEEP, P_EXPONENTS, d_exps)
    n = h.shape[0]
    print(f"{n} orbitals, {(n * (n - 1) // 2) ** 2} determinants; integrals {time.time() - t0:.0f} s", flush=True)
    ion = quantum.fci(h, eri, 2, 1)
    print(f"Be+ {ion:.8f}  ({time.time() - t0:.0f} s)", flush=True)
    neutral = quantum.fci(h, eri, 2, 2)
    print(f"Be  {neutral:.8f}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"ionization energy {(ion - neutral) * 27.211386245981:.5f} eV (infinite nuclear mass)")
    if len(sys.argv) == 1:
        out = Path(__file__).resolve().parent / "precomputed.json"
        out.write_text(json.dumps({"Be_full_ci_spd": {
            "E_Be_hartree": neutral, "E_BePlus_hartree": ion, "orbitals": n,
            "determinants": (n * (n - 1) // 2) ** 2, "s_keep": S_KEEP,
            "p_exponents": P_EXPONENTS, "d_exponents": d_exps,
            "seconds": round(time.time() - t0)}}, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()

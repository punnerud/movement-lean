#!/usr/bin/env python3
"""All-electron full CI for beryllium with s, p, d and f orbitals.

Too large for the regular analyzer run: about 1.07 million determinants and
roughly 21 GB of memory. Run it on a machine with 32 GB or more:

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
F_EXPONENTS = (0.35,)


def main():
    # optional arguments for smaller test runs: d exponents, then f exponents ("-" for none)
    def parse(i, default):
        if len(sys.argv) <= i:
            return default
        return () if sys.argv[i] == "-" else tuple(float(x) for x in sys.argv[i].split(","))
    d_exps, f_exps = parse(1, D_EXPONENTS), parse(2, F_EXPONENTS)
    t0 = time.time()
    h, eri = quantum.atom_sp_orbitals(4, (2, 2), S_KEEP, P_EXPONENTS, d_exps, f_exps)
    n = h.shape[0]
    print(f"{n} orbitals, {(n * (n - 1) // 2) ** 2} determinants; integrals {time.time() - t0:.0f} s", flush=True)
    ion = quantum.fci(h, eri, 2, 1)
    print(f"Be+ {ion:.8f}  ({time.time() - t0:.0f} s)", flush=True)
    neutral = quantum.fci(h, eri, 2, 2)
    print(f"Be  {neutral:.8f}  ({time.time() - t0:.0f} s)", flush=True)
    print(f"ionization energy {(ion - neutral) * 27.211386245981:.5f} eV (infinite nuclear mass)")
    if len(sys.argv) == 1:
        out = Path(__file__).resolve().parent / "precomputed.json"
        data = json.loads(out.read_text()) if out.exists() else {}
        data["Be_full_ci_spdf"] = {
            "E_Be_hartree": neutral, "E_BePlus_hartree": ion, "orbitals": n,
            "determinants": (n * (n - 1) // 2) ** 2, "s_keep": S_KEEP,
            "p_exponents": P_EXPONENTS, "d_exponents": d_exps, "f_exponents": f_exps,
            "seconds": round(time.time() - t0)}
        out.write_text(json.dumps(data, indent=2) + "\n")
        print("wrote", out)


if __name__ == "__main__":
    main()

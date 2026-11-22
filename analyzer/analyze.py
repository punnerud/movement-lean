#!/usr/bin/env python3
"""Atom and molecule analyzer.

Evaluates formulas whose derivations are proved in Lean (Movement/*.lean) and
compares the results with measured values (analyzer/reference_data.json).

    python3 analyzer/analyze.py            # print the report
    python3 analyzer/analyze.py --write    # also write REPORT.md and docs/analyzer.html
    python3 analyzer/analyze.py --check    # exit 1 if any row leaves its expected band

Every row names the Lean theorem behind its formula and states in advance
whether the model is expected to work ("ok") or is outside its domain
("limit"). --check fails if either expectation is violated, so the analyzer
cannot silently drift.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "analyzer" / "reference_data.json").read_text())

# CODATA 2018
C = 299792458.0                 # m/s
H = 6.62607015e-34              # J s
E_CH = 1.602176634e-19          # C
U = 1.66053906660e-27           # kg
R_GAS = 8.314462618             # J/(mol K)
ALPHA = 7.2973525693e-3
ME_U = 5.48579909065e-4         # electron mass in u
MC2_EV = 510998.95              # electron rest energy, eV
RY_EV = 13.605693122994         # Rydberg energy (infinite nuclear mass), eV
N_AIR = 1.000276                # refractive index of air (visible)

MASS = DATA["atomic_mass_u"]
ROWS = []


def row(group, system, quantity, model, lean, predicted, measured, unit, expect, tol, note=""):
    err = (predicted - measured) / measured
    ROWS.append(dict(group=group, system=system, quantity=quantity, model=model, lean=lean,
                     predicted=predicted, measured=measured, unit=unit, expect=expect,
                     tol=tol, err=err, note=note))


def nucleus(isotope, z):
    return MASS[isotope] - z * ME_U


def reduced(m1, m2):
    return m1 * m2 / (m1 + m2)


# ---------------------------------------------------------------- one-electron atoms
ONE_ELECTRON = [("H", 1, "1H"), ("He+", 2, "4He"), ("Li2+", 3, "7Li"),
                ("Be3+", 4, "9Be"), ("B4+", 5, "11B"), ("C5+", 6, "12C")]
for name, z, iso in ONE_ELECTRON:
    recoil = 1 / (1 + ME_U / nucleus(iso, z))          # reduced-mass factor
    measured = DATA["ionization_energy_eV"][name]
    row("One-electron atoms", name, "ionization energy", "Bohr levels, Z² scaling",
        "bohr_orbit, hydrogenlike_energy", RY_EV * z * z * recoil, measured, "eV", "ok", 1e-3)
    dirac = MC2_EV * (1 - math.sqrt(1 - (z * ALPHA) ** 2)) * recoil
    row("One-electron atoms", name, "ionization energy", "relativistic (Dirac) ground state",
        "dirac_binding_ge_bohr (inequality only)", dirac, measured, "eV", "ok", 5e-5)

R_H = RY_EV / (1 + ME_U / nucleus("1H", 1))
for n, lam in DATA["balmer_air_nm"].items():
    energy = R_H * (1 / 4 - 1 / int(n) ** 2)                       # eV
    vac_nm = H * C / (energy * E_CH) * 1e9
    row("One-electron atoms", f"H, n={n}→2", "Balmer wavelength (air)", "Rydberg formula",
        "rydberg_formula", vac_nm / N_AIR, lam, "nm", "ok", 5e-5)

# ---------------------------------------------------------------- more electrons
ie = DATA["ionization_energy_eV"]
he_total = -(ie["He"] + ie["He+"])
row("Atoms with several electrons", "He", "total ground-state energy",
    "two independent Bohr electrons (no repulsion)", "hydrogenlike_energy",
    -2 * 4 * RY_EV, he_total, "eV", "limit", 0.05, "ignores electron–electron repulsion")
row("Atoms with several electrons", "He", "total ground-state energy",
    "variational screening, ζ = Z − 5/16", "helium_variational",
    -2 * (2 - 5 / 16) ** 2 * RY_EV, he_total, "eV", "ok", 0.03)
row("Atoms with several electrons", "Li", "first ionization energy",
    "Bohr, n=2, no screening (Z=3)", "hydrogenlike_energy",
    9 * RY_EV / 4, ie["Li"], "eV", "limit", 0.05, "inner electrons screen the nucleus")
row("Atoms with several electrons", "Li", "first ionization energy",
    "Bohr, n=2, full screening (Z=1)", "hydrogenlike_energy",
    RY_EV / 4, ie["Li"], "eV", "limit", 0.05, "the 2s electron penetrates the inner shell")

# ---------------------------------------------------------------- rotating molecules
BE = DATA["rotational_constant_Be_cm1"]
MOL = {"H2": ("1H", "1H"), "D2": ("2H", "2H"), "HD": ("1H", "2H"), "N2": ("14N", "14N"),
       "O2": ("16O", "16O"), "12C16O": ("12C", "16O"), "13C16O": ("13C", "16O"),
       "H35Cl": ("1H", "35Cl"), "D35Cl": ("2H", "35Cl")}


def mu(name):
    a, b = MOL[name]
    return reduced(MASS[a], MASS[b])


for parent, child in (("H2", "D2"), ("H2", "HD"), ("H35Cl", "D35Cl")):
    row("Rotating molecules", f"{child} from {parent}", "rotational constant B",
        "isotope rule, B ∝ 1/μ at fixed bond length",
        "reduced_mass_inertia, isotope_rotational_constant",
        BE[parent] * mu(parent) / mu(child), BE[child], "cm⁻¹", "ok", 2e-3)

for name, r_e in DATA["bond_length_re_angstrom"].items():
    inertia = mu(name) * U * (r_e * 1e-10) ** 2
    b = H / (8 * math.pi ** 2 * C * inertia) / 100                 # cm⁻¹
    row("Rotating molecules", name, "rotational constant B", "rigid rotor, I = μr²",
        "reduced_mass_inertia", b, BE[name], "cm⁻¹", "ok", 1e-3,
        "consistency check: tabulated bond lengths are themselves derived from B")

row("Rotating molecules", "12C16O", "J=0→1 line frequency", "rigid rotor, line spacing 2B",
    "rotor_spacing", 2 * BE["12C16O"] * 100 * C / 1e9, DATA["CO_J0_to_1_GHz"], "GHz", "ok", 1e-2,
    "the real molecule also vibrates, which stretches the bond slightly")

# ---------------------------------------------------------------- heat capacity
FREEDOM = {"He": 3, "Ar": 3, "H2": 5, "N2": 5, "O2": 5, "CO": 5, "Cl2": 5, "CO2": 5}
LIMITS = {"Cl2": "vibration is already active at room temperature",
          "CO2": "bending vibrations are active at room temperature"}
for gas, cp in DATA["Cp_298K_J_per_molK"].items():
    f = FREEDOM[gas]
    kind = "translation only" if f == 3 else "translation + rotation"
    row("Heat capacity", gas, "C_p at 298 K", f"equipartition, f = {f} ({kind})",
        "heat_capacity_ratio; rotation: rot_share", (f / 2 + 1) * R_GAS, cp, "J/(mol·K)",
        "limit" if gas in LIMITS else "ok", 0.02, LIMITS.get(gas, ""))

T20 = 293.15
for gas, f, molar in (("air", 5, 0.0289647), ("He", 3, 0.004002602)):
    gamma = (f + 2) / f
    row("Heat capacity", gas, "speed of sound at 20 °C", f"√(γRT/M), γ = {f + 2}/{f}",
        "gamma_values", math.sqrt(gamma * R_GAS * T20 / molar),
        DATA["sound_speed_20C_m_per_s"][gas], "m/s", "ok", 5e-3)


# ---------------------------------------------------------------- reporting
def verdict(r):
    inside = abs(r["err"]) <= r["tol"]
    if r["expect"] == "ok":
        return ("matches", True) if inside else ("UNEXPECTED MISS", False)
    return ("fails, as expected", True) if not inside else ("UNEXPECTED MATCH", False)


def fmt(x):
    return f"{x:.6g}" if abs(x) < 1e5 else f"{x:.1f}"


def pct(e):
    a = abs(e) * 100
    return f"{e * 100:+.4f} %" if a < 0.1 else f"{e * 100:+.2f} %" if a < 10 else f"{e * 100:+.0f} %"


def markdown():
    out = ["# Analyzer report", "",
           "Generated by `python3 analyzer/analyze.py --write`. Do not edit by hand.", "",
           "Each formula's derivation is proved in Lean; the comparison with measurement is what "
           "tests the model. Rows marked *limit* are cases where the simple model is known to be "
           "outside its domain, kept to show where it breaks.", ""]
    groups = []
    for r in ROWS:
        if r["group"] not in groups:
            groups.append(r["group"])
    for g in groups:
        out += [f"## {g}", "",
                "| System | Quantity | Model | Lean theorem | Predicted | Measured | Error | Verdict |",
                "|---|---|---|---|---|---|---|---|"]
        for r in (x for x in ROWS if x["group"] == g):
            v, _ = verdict(r)
            note = f" ({r['note']})" if r["note"] else ""
            out.append(f"| {r['system']} | {r['quantity']} | {r['model']} | `{r['lean']}` | "
                       f"{fmt(r['predicted'])} {r['unit']} | {fmt(r['measured'])} {r['unit']} | "
                       f"{pct(r['err'])} | {v}{note} |")
        out.append("")
    ok = sum(1 for r in ROWS if r["expect"] == "ok")
    out += [f"**{ok} of {len(ROWS)} comparisons are inside the model's domain and match; "
            f"{len(ROWS) - ok} are documented limits.**", ""]
    return "\n".join(out)


def html():
    css = """:root{--bg:#fbfbfa;--card:#fff;--ink:#1f2933;--muted:#616e7c;--line:#e4e7eb;--a:#2563eb;--b:#dc2626;--c:#059669;--code:#f3f4f6}
@media (prefers-color-scheme: dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e5e7eb;--muted:#9aa5b1;--line:#2a2f3a;--a:#60a5fa;--b:#f87171;--c:#34d399;--code:#1f232c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1100px;margin:0 auto;padding:32px 16px 80px}h1{font-size:clamp(26px,5vw,36px);margin:8px 0}h2{margin:44px 0 8px;font-size:22px}
.lede{color:var(--muted);font-size:18px}.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:820px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12px}
.ok{color:var(--c);font-weight:600}.lim{color:var(--b);font-weight:600}.note{color:var(--muted);font-size:13px}a{color:var(--a)}"""
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           "<title>Atom and Molecule Analyzer</title>", f"<style>{css}</style></head><body><main>",
           "<p><a href='./'>← back to the explainer</a></p>",
           "<h1>Atom and molecule analyzer</h1>",
           "<p class='lede'>Formulas whose derivations are proved in Lean, evaluated with measured "
           "constants and compared with measured data. Rows marked as limits show where the simple "
           "models break.</p>"]
    groups = []
    for r in ROWS:
        if r["group"] not in groups:
            groups.append(r["group"])
    for g in groups:
        out.append(f"<h2>{g}</h2><div class='wrap'><table><tr><th>System</th><th>Quantity</th>"
                   "<th>Model</th><th>Lean theorem</th><th>Predicted</th><th>Measured</th>"
                   "<th>Error</th><th>Verdict</th></tr>")
        for r in (x for x in ROWS if x["group"] == g):
            v, _ = verdict(r)
            cls = "ok" if r["expect"] == "ok" else "lim"
            note = f"<div class='note'>{r['note']}</div>" if r["note"] else ""
            out.append(f"<tr><td>{r['system']}</td><td>{r['quantity']}</td><td>{r['model']}</td>"
                       f"<td><code>{r['lean']}</code></td><td class='n'>{fmt(r['predicted'])} {r['unit']}</td>"
                       f"<td class='n'>{fmt(r['measured'])} {r['unit']}</td><td class='n'>{pct(r['err'])}</td>"
                       f"<td class='{cls}'>{v}{note}</td></tr>")
        out.append("</table></div>")
    out.append("<p class='note'>Generated by <code>analyzer/analyze.py</code>. Reference values: "
               "NIST Atomic Spectra Database, NIST Chemistry WebBook, NIST-JANAF tables, CODATA.</p></main></body></html>")
    return "\n".join(out)


def main():
    bad = [r for r in ROWS if not verdict(r)[1]]
    if "--write" in sys.argv:
        (ROOT / "analyzer" / "REPORT.md").write_text(markdown())
        (ROOT / "docs" / "analyzer.html").write_text(html())
    if "--check" not in sys.argv or bad:
        for r in ROWS:
            print(f"{verdict(r)[0]:20} {r['system']:16} {r['quantity']:28} {r['model'][:44]:44} "
                  f"{fmt(r['predicted']):>10} vs {fmt(r['measured']):>10} {r['unit']:10} {pct(r['err'])}")
    print(f"\n{len(ROWS)} comparisons, {len(bad)} outside their expected band")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

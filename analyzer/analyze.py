#!/usr/bin/env python3
"""Atom and molecule analyzer.

Evaluates formulas whose derivations are proved in Lean (Movement/*.lean),
plus numerical quantum mechanics (analyzer/quantum.py), and compares the
results with reference values (analyzer/reference_data.json).

    python3 analyzer/analyze.py            # print the report
    python3 analyzer/analyze.py --write    # also write REPORT.md and docs/analyzer.html
    python3 analyzer/analyze.py --check    # exit 1 if any row leaves its expected band

Every row states in advance whether the model is expected to work ("ok") or is
outside its domain ("limit"). --check fails if either expectation is violated,
so the analyzer cannot silently drift. Requires numpy for the numerical part.
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analyzer"))
DATA = json.loads((ROOT / "analyzer" / "reference_data.json").read_text())

_pre = ROOT / "analyzer" / "precomputed.json"
PRECOMPUTED = json.loads(_pre.read_text()) if _pre.exists() else {}

try:
    import quantum
except ImportError:                      # numpy missing
    quantum = None

# CODATA
C = 299792458.0                 # m/s
H = 6.62607015e-34              # J s
E_CH = 1.602176634e-19          # C
U = 1.66053906660e-27           # kg
R_GAS = 8.314462618             # J/(mol K)
ALPHA = 7.2973525693e-3
ME_U = 5.48579909065e-4         # electron mass in u
MC2_EV = 510998.95              # electron rest energy, eV
RY_EV = 13.605693122994         # Rydberg energy (infinite nuclear mass), eV
HARTREE_EV = 2 * RY_EV
HARTREE_CM = 219474.6313632     # hartree in cm^-1
BOHR_A = 0.529177210903         # bohr in angstrom
HC_K = 1.438776877              # h c / k in cm K
BETHE_LOG_1S = 2.984128556

MASS = DATA["atomic_mass_u"]
UNC = DATA["relative_uncertainty"]
ROWS = []


def row(group, system, quantity, model, lean, predicted, measured, unit, expect, tol,
        unc, note="", base=None, fix=""):
    ROWS.append(dict(group=group, system=system, quantity=quantity, model=model, lean=lean,
                     predicted=predicted, measured=measured, unit=unit, expect=expect, tol=tol,
                     err=(predicted - measured) / measured, unc=unc, note=note, base=base, fix=fix))
    return len(ROWS) - 1


def nucleus(isotope, z):
    return MASS[isotope] - z * ME_U


def recoil(isotope, z):
    """Reduced-mass factor mu/m_e for an electron bound to this nucleus."""
    return 1 / (1 + ME_U / nucleus(isotope, z))


def reduced(m1, m2):
    return m1 * m2 / (m1 + m2)


G_MAG = "Magnetism: the electron cloud as a tiny superconductor"
G_VDW = "Forces between neutral atoms: two helium atoms"
G_BOND = "Chemical bonds: shared electrons (H₂) and transferred electrons (NaCl)"
G_ONE, G_MANY, G_ROT, G_HEAT, G_NUM, G_VAL = (
    "One-electron atoms", "Atoms with several electrons: simple models", "Rotating molecules",
    "Heat capacity and sound", "Numerical quantum mechanics", "Validation of the numerical methods")

# ---------------------------------------------------------------- one-electron atoms
def dirac_level(z, n, j):
    """Dirac energy of level (n, j) minus the rest energy, in eV (negative)."""
    za = z * ALPHA
    k = j + 0.5
    return MC2_EV * ((1 + (za / (n - k + math.sqrt(k * k - za * za))) ** 2) ** -0.5 - 1)


def lamb_shift_1s(z):
    """Leading QED shift of the 1s level, in eV (positive: less bound)."""
    za = z * ALPHA
    bracket = (4 / 3 * (math.log(za ** -2) - BETHE_LOG_1S) + 10 / 9 - 4 / 15
               + 4 * math.pi * (139 / 128 + 5 / 192 - math.log(2) / 2) * za)
    return ALPHA * za ** 4 * MC2_EV / math.pi * bracket


ONE_ELECTRON = [("H", 1, "1H"), ("He+", 2, "4He"), ("Li2+", 3, "7Li"),
                ("Be3+", 4, "9Be"), ("B4+", 5, "11B"), ("C5+", 6, "12C")]
for name, z, iso in ONE_ELECTRON:
    rm = recoil(iso, z)
    measured = DATA["ionization_energy_eV"][name]
    u = UNC["ionization_one_electron"]
    b = row(G_ONE, name, "ionization energy", "Bohr levels, Z² scaling",
            "bohr_orbit, hydrogenlike_energy", RY_EV * z * z * rm, measured, "eV", "ok", 1e-3, u)
    d = row(G_ONE, name, "ionization energy", "relativistic (Dirac) ground state",
            "dirac_binding_ge_bohr (inequality only)", -dirac_level(z, 1, 0.5) * rm, measured, "eV",
            "ok", 5e-5, u, base=b, fix="relativity: the electron's mass grows with speed")
    row(G_ONE, name, "ionization energy", "Dirac + leading QED (Lamb shift)", "not formalised",
        -dirac_level(z, 1, 0.5) * rm - lamb_shift_1s(z) * rm ** 3, measured, "eV", "ok", 2e-6, u,
        base=d, fix="QED: the electron interacts with its own field (Lamb shift)")


def level_mean(n):
    """Dirac energy of shell n averaged over its states (statistical weights)."""
    total = weight = 0
    for twice_j in range(1, 2 * n, 2):
        j = twice_j / 2
        g = (2 * j + 1) * (1 if twice_j == 2 * n - 1 else 2)
        total += g * dirac_level(1, n, j)
        weight += g
    return total / weight


R_H = RY_EV * recoil("1H", 1)
for n, lam in DATA["balmer_air_nm"].items():
    index = 1 + DATA["air_index_minus_one_1e4"][n] * 1e-4
    simple = H * C / (R_H * (1 / 4 - 1 / int(n) ** 2) * E_CH) * 1e9
    b = row(G_ONE, f"H, n={n}→2", "Balmer wavelength (air)", "Rydberg formula", "rydberg_formula",
            simple / index, lam, "nm", "ok", 5e-5, UNC["balmer"])
    fine = H * C / ((level_mean(int(n)) - level_mean(2)) * recoil("1H", 1) * E_CH) * 1e9
    row(G_ONE, f"H, n={n}→2", "Balmer wavelength (air)", "Dirac levels (fine structure, averaged)",
        "not formalised", fine / index, lam, "nm", "ok", 1e-5, UNC["balmer"], base=b,
        fix="relativity: fine structure shifts the n=2 and upper levels",
        note="the observed line is an unresolved blend about 0.009 nm wide")

# ---------------------------------------------------------------- several electrons, simple models
ie = DATA["ionization_energy_eV"]
he_total = -(ie["He"] + ie["He+"])
u_n = UNC["ionization_neutral"]
he_bohr = row(G_MANY, "He", "total ground-state energy", "two independent Bohr electrons",
              "hydrogenlike_energy", -2 * 4 * RY_EV, he_total, "eV", "limit", 0.05, u_n,
              "ignores electron–electron repulsion")
he_var = row(G_MANY, "He", "total ground-state energy", "one screening parameter, ζ = Z − 5/16",
             "helium_variational", -2 * (2 - 5 / 16) ** 2 * RY_EV, he_total, "eV", "ok", 0.03, u_n,
             base=he_bohr, fix="average screening by the other electron")
li_a = row(G_MANY, "Li", "first ionization energy", "Bohr, n=2, no screening (Z=3)",
           "hydrogenlike_energy", 9 * RY_EV / 4, ie["Li"], "eV", "limit", 0.05, u_n,
           "inner electrons screen the nucleus")
li_b = row(G_MANY, "Li", "first ionization energy", "Bohr, n=2, full screening (Z=1)",
           "hydrogenlike_energy", RY_EV / 4, ie["Li"], "eV", "limit", 0.05, u_n,
           "the 2s electron penetrates the inner shell")

# ---------------------------------------------------------------- rotating molecules
BE = DATA["rotational_constant_Be_cm1"]
MOL = {"H2": ("1H", "1H"), "D2": ("2H", "2H"), "HD": ("1H", "2H"), "N2": ("14N", "14N"),
       "O2": ("16O", "16O"), "12C16O": ("12C", "16O"), "H35Cl": ("1H", "35Cl"),
       "D35Cl": ("2H", "35Cl")}


def mu(name):
    a, b = MOL[name]
    return reduced(MASS[a], MASS[b])


def rotational_constant(mu_u, r_angstrom):
    return H / (8 * math.pi ** 2 * C * mu_u * U * (r_angstrom * 1e-10) ** 2) / 100


u_b = UNC["rotational_constant"]
for parent, child in (("H2", "D2"), ("H2", "HD"), ("H35Cl", "D35Cl")):
    row(G_ROT, f"{child} from {parent}", "rotational constant B",
        "isotope rule, B ∝ 1/μ at fixed bond length",
        "reduced_mass_inertia, isotope_rotational_constant",
        BE[parent] * mu(parent) / mu(child), BE[child], "cm⁻¹", "ok", 2e-3, u_b)
for name, r_e in DATA["bond_length_re_angstrom"].items():
    row(G_ROT, name, "rotational constant B", "rigid rotor, I = μr²", "reduced_mass_inertia",
        rotational_constant(mu(name), r_e), BE[name], "cm⁻¹", "ok", 1e-3, u_b,
        "consistency check: tabulated bond lengths are themselves derived from B")
co = row(G_ROT, "12C16O", "J=0→1 line frequency", "rigid rotor, line spacing 2B", "rotor_spacing",
         2 * BE["12C16O"] * 100 * C / 1e9, DATA["CO_J0_to_1_GHz"], "GHz", "ok", 1e-2, UNC["CO_line"])
b0 = BE["12C16O"] - DATA["CO_alpha_e_cm1"] / 2
row(G_ROT, "12C16O", "J=0→1 line frequency", "vibrating, stretching rotor: 2B₀ − 4D",
    "rotor_spacing_distortion", (2 * b0 - 4 * DATA["CO_D_e_cm1"]) * 100 * C / 1e9,
    DATA["CO_J0_to_1_GHz"], "GHz", "ok", 2e-5, UNC["CO_line"], base=co,
    fix="zero-point vibration lengthens the bond; rotation stretches it")

# ---------------------------------------------------------------- heat capacity and sound
def einstein(wavenumber, t):
    """Heat capacity of one quantised vibration, in units of R."""
    x = HC_K * wavenumber / t
    return x * x * math.exp(x) / (math.exp(x) - 1) ** 2


def rotor_heat_capacity(b_cm, d_cm, t, parity):
    """Quantum rotor heat capacity in units of R, for even or odd J only."""
    z = e1 = e2 = 0.0
    for j in range(parity, 16, 2):
        eps = HC_K * (b_cm * j * (j + 1) - d_cm * (j * (j + 1)) ** 2) / t
        w = (2 * j + 1) * math.exp(-eps)
        z += w; e1 += w * eps; e2 += w * eps * eps
    return e2 / z - (e1 / z) ** 2


T0 = 298.15
FREEDOM = {"He": 3, "Ar": 3, "H2": 5, "N2": 5, "O2": 5, "CO": 5, "Cl2": 5, "CO2": 5}
LIMITS = {"Cl2": "vibration is already active at room temperature",
          "CO2": "bending vibrations are active at room temperature"}
VIB = DATA["vibration_cm1"]
for gas, cp in DATA["Cp_298K_J_per_molK"].items():
    f = FREEDOM[gas]
    kind = "translation only" if f == 3 else "translation + rotation"
    b = row(G_HEAT, gas, "C_p at 298 K", f"equipartition, f = {f} ({kind})",
            "heat_capacity_ratio; rotation: rot_share", (f / 2 + 1) * R_GAS, cp, "J/(mol·K)",
            "limit" if gas in LIMITS else "ok", 0.02, UNC["Cp"], LIMITS.get(gas, ""))
    if gas == "H2":
        b0_h2, dist = DATA["H2_B0_cm1"], DATA["H2_centrifugal_D_cm1"]

        def h2_cp(d_cm):
            rot = (0.25 * rotor_heat_capacity(b0_h2, d_cm, T0, 0)
                   + 0.75 * rotor_heat_capacity(b0_h2, d_cm, T0, 1))
            return (2.5 + rot + einstein(VIB[gas][0], T0)) * R_GAS

        q = row(G_HEAT, gas, "C_p at 298 K", "quantised rigid rotation (ortho + para)",
                "not formalised", h2_cp(0.0), cp, "J/(mol·K)", "ok", 5e-3, UNC["Cp"], base=b,
                fix="hydrogen is so light that its rotation is not fully excited at 298 K")
        row(G_HEAT, gas, "C_p at 298 K", "quantised rotation + centrifugal stretching",
            "rotor_spacing_distortion", h2_cp(dist), cp, "J/(mol·K)", "ok", 1e-3, UNC["Cp"],
            base=q, fix="the molecule stretches as it spins faster")
    elif gas in VIB:
        vib = sum(einstein(w, T0) for w in VIB[gas])
        row(G_HEAT, gas, "C_p at 298 K", "equipartition + quantised vibration", "not formalised",
            (f / 2 + 1 + vib) * R_GAS, cp, "J/(mol·K)", "ok", 1e-2, UNC["Cp"], base=b,
            fix="vibrations switch on gradually (quantised)")

T20 = 293.15
SOUND = DATA["sound_speed_20C_m_per_s"]
row(G_HEAT, "air", "speed of sound at 20 °C", "√(γRT/M), γ = 7/5", "gamma_values",
    math.sqrt(1.4 * R_GAS * T20 / 0.0289647), SOUND["air"], "m/s", "ok", 5e-3, UNC["sound_air"])
ideal_he = math.sqrt(5 / 3 * R_GAS * T20 / 0.004002602)
he_s = row(G_HEAT, "He", "speed of sound at 20 °C", "√(γRT/M), γ = 5/3", "gamma_values",
           ideal_he, SOUND["He"], "m/s", "ok", 5e-3, UNC["sound_He"])
virial = DATA["He_acoustic_virial_cm3_per_mol"] * 1e-6
row(G_HEAT, "He", "speed of sound at 20 °C", "ideal gas + acoustic virial coefficient",
    "not formalised", ideal_he * math.sqrt(1 + virial * 101325 / (R_GAS * T20)), SOUND["He"], "m/s",
    "ok", 5e-4, UNC["sound_He"], base=he_s, fix="real helium atoms repel slightly at 1 atm")

# ---------------------------------------------------------------- numerical quantum mechanics
STABILITY = []            # (description, bool): qualitative conclusions that must hold
VARIATIONAL = []          # (description, upper value, lower value): upper >= lower must hold
if quantum is not None:
    bench = DATA["benchmark_hartree"]
    he_hf = quantum.atom_hf(2, 1, 1)
    he_s_fci = quantum.helium_fci_s()
    he_hyl = quantum.hylleraas_helium(order=8, alpha=quantum.Fraction(2))
    li, li_ion = quantum.atom_hf(3, 2, 1), quantum.atom_hf(3, 1, 1)
    be, be_ion = quantum.atom_hf(4, 2, 2), quantum.atom_hf(4, 2, 1)
    r_e, e_min, curvature = quantum.h2_curve()
    e_h = quantum.hydrogen_atom_energy()

    he_ev = HARTREE_EV * recoil("4He", 2)
    a = row(G_NUM, "He", "total ground-state energy", "Hartree–Fock (average field)",
            "rayleigh_ritz_bound", he_hf * he_ev, he_total, "eV", "ok", 0.02, u_n, base=he_var,
            fix="each electron feels the exact average field of the other")
    b = row(G_NUM, "He", "total ground-state energy", "full CI, spherical orbitals only",
            "rayleigh_ritz_bound", he_s_fci * he_ev, he_total, "eV", "ok", 0.01, u_n, base=a,
            fix="electrons avoid each other radially (in/out correlation)")
    row(G_NUM, "He", "total ground-state energy", "Hylleraas (r₁₂ in the wavefunction)",
        "rayleigh_ritz_bound", he_hyl * he_ev, he_total, "eV", "ok", 5e-5, u_n, base=b,
        fix="electrons avoid each other in every direction; remaining error is relativity + QED",
        note="non-relativistic; the remaining 0.002 % is relativistic and QED")
    li_hf_row = row(G_NUM, "Li", "first ionization energy", "Hartree–Fock, E(Li⁺) − E(Li)",
        "rayleigh_ritz_bound", (li_ion - li) * HARTREE_EV * recoil("7Li", 3), ie["Li"], "eV",
        "ok", 0.02, u_n, base=li_b, fix="self-consistent screening and penetration",
        note="the missing 1 % is electron correlation")
    li_ci, li_ion_ci = quantum.atom_fci_sp(3, (2, 1), [(2, 1), (1, 1)])
    li_sp_row = row(G_NUM, "Li", "first ionization energy", "full CI with s and p orbitals",
        "rayleigh_ritz_bound", (li_ion_ci - li_ci) * HARTREE_EV * recoil("7Li", 3), ie["Li"], "eV",
        "ok", 3e-3, u_n, base=li_hf_row,
        fix="the outer electron polarises the inner pair (needs p orbitals)",
        )
    li_d, li_ion_d = quantum.atom_fci_sp(3, (2, 1), [(2, 1), (1, 1)], with_d=True)
    row(G_NUM, "Li", "first ionization energy", "full CI with s, p and d orbitals",
        "rayleigh_ritz_bound", (li_ion_d - li_d) * HARTREE_EV * recoil("7Li", 3), ie["Li"], "eV",
        "ok", 1.5e-3, u_n, base=li_sp_row,
        fix="tighter p and one d shell: finer polarization of the inner pair",
        note="the remaining 0.1 % needs still larger orbital sets")
    be_hf_row = row(G_NUM, "Be", "first ionization energy", "Hartree–Fock, E(Be⁺) − E(Be)",
                    "rayleigh_ritz_bound", (be_ion - be) * HARTREE_EV * recoil("9Be", 4), ie["Be"],
                    "eV", "limit", 0.05, u_n,
                    "average-field theory misses the strong 2s–2p correlation in beryllium")
    be_ci, be_ion_ci = quantum.atom_fci_sp(4, (2, 2), [(2, 2), (2, 1)])
    be_sp_row = row(G_NUM, "Be", "first ionization energy", "full CI with s and p orbitals",
        "rayleigh_ritz_bound", (be_ion_ci - be_ci) * HARTREE_EV * recoil("9Be", 4), ie["Be"], "eV",
        "ok", 1e-2, u_n, base=be_hf_row,
        fix="the two outer electrons spend part of the time in 2p instead of 2s",
        )
    be_d = quantum.beryllium_d_correction()
    be_comp_row = row(G_NUM, "Be", "first ionization energy", "+ d orbitals for the two outer electrons",
        "not formalised", (be_ion_ci - be_ci + be_d) * HARTREE_EV * recoil("9Be", 4), ie["Be"], "eV",
        "ok", 4e-3, u_n, base=be_sp_row,
        fix="d orbitals let the outer pair avoid each other at an angle",
        note="composite: the d-orbital change is computed with the 1s² core frozen and added; "
             "the remaining 0.25 % is mostly correlation between core and outer electrons")
    pre = PRECOMPUTED.get("Be_full_ci_spd")
    if pre:
        be_full_row = row(G_NUM, "Be", "first ionization energy",
            "full CI, all four electrons, s, p and d ("
            + f"{pre['determinants']:,}".replace(",", " ") + " determinants)",
            "rayleigh_ritz_bound",
            (pre["E_BePlus_hartree"] - pre["E_Be_hartree"]) * HARTREE_EV * recoil("9Be", 4), ie["Be"],
            "eV", "ok", 3e-3, u_n, base=be_comp_row,
            fix="core and outer electrons correlated together, no frozen core",
            note="precomputed with analyzer/be_full_ci.py (needs about 15 GB of memory); "
                 "the remaining 0.2 % needs f orbitals and larger sets")
    pre_f = PRECOMPUTED.get("Be_full_ci_spdf")
    if pre and pre_f:
        row(G_NUM, "Be", "first ionization energy",
            "full CI, all four electrons, s, p, d and f ("
            + f"{pre_f['determinants']:,}".replace(",", " ") + " determinants)",
            "rayleigh_ritz_bound",
            (pre_f["E_BePlus_hartree"] - pre_f["E_Be_hartree"]) * HARTREE_EV * recoil("9Be", 4),
            ie["Be"], "eV", "ok", 2.5e-3, u_n, base=be_full_row,
            fix="f orbitals: a finer angular pattern for the outer pair",
            note="precomputed with analyzer/be_full_ci.py (27 minutes, about 20 GB); "
                 "the remaining 0.15 % needs larger orbital sets of every kind")

    # H-: a proton with two electrons. Bound only because the electrons avoid each other.
    hm_hf = quantum.atom_hf(1, 1, 1)
    hm_s = quantum.two_electron_fci_s(1)
    hm_hyl = quantum.hylleraas_helium(Z=1, order=8, alpha=quantum.Fraction(7, 10))
    ea, h_ev, u_ea = DATA["H_electron_affinity_eV"], HARTREE_EV * recoil("1H", 1), UNC["electron_affinity"]
    a = row(G_NUM, "H⁻", "electron affinity of H", "Hartree–Fock (average field)",
            "rayleigh_ritz_bound", (-0.5 - hm_hf) * h_ev, ea, "eV", "limit", 0.05, u_ea,
            "negative: the average field predicts that H⁻ does not exist")
    b = row(G_NUM, "H⁻", "electron affinity of H", "full CI, spherical orbitals only",
            "rayleigh_ritz_bound", (-0.5 - hm_s) * h_ev, ea, "eV", "limit", 0.05, u_ea,
            "bound, but only half of the binding", base=a,
            fix="electrons avoid each other radially: the ion becomes stable")
    row(G_NUM, "H⁻", "electron affinity of H", "Hylleraas (r₁₂ in the wavefunction)",
        "rayleigh_ritz_bound", (-0.5 - hm_hyl) * h_ev, ea, "eV", "ok", 2e-3, u_ea, base=b,
        fix="electrons avoid each other in every direction",
        note="non-relativistic; nuclear motion included only through the reduced mass")

    mu_h2 = mu("H2")
    r_ang = r_e * BOHR_A
    row(G_NUM, "H₂", "bond length", "full CI, potential-curve minimum", "rayleigh_ritz_bound",
        r_ang, DATA["H2_bond_length_angstrom"], "Å", "ok", 5e-3, UNC["H2_bond"])
    row(G_NUM, "H₂", "binding energy D_e", "full CI, 2·E(H) − E(H₂)", "rayleigh_ritz_bound",
        (2 * e_h - e_min) * HARTREE_EV, DATA["H2_dissociation_De_eV"], "eV", "ok", 0.03,
        UNC["H2_De"], "basis lacks high angular functions, so ~1 % of the binding is missing")
    row(G_NUM, "H₂", "vibration wavenumber ω_e", "curvature of the computed potential",
        "not formalised", math.sqrt(curvature / (mu_h2 / ME_U)) * HARTREE_CM, VIB["H2"][0],
        "cm⁻¹", "ok", 0.01, UNC["H2_vibration"])
    row(G_NUM, "H₂", "rotational constant B", "rigid rotor at the computed bond length",
        "reduced_mass_inertia", rotational_constant(mu_h2, r_ang), BE["H2"], "cm⁻¹", "ok", 0.01,
        u_b, "fully independent prediction: no measured bond length is used")

    for label, got, key, tol in (("He, Hartree–Fock", he_hf, "He_HF", 2e-5),
                                 ("Li⁺, Hartree–Fock", li_ion, "LiPlus_HF", 2e-5),
                                 ("Be, Hartree–Fock", be, "Be_HF", 2e-5),
                                 ("He, Hylleraas", he_hyl, "He_exact", 1e-6),
                                 ("H⁻, Hartree–Fock", hm_hf, "HMinus_HF", 2e-5),
                                 ("H⁻, Hylleraas", hm_hyl, "HMinus_exact", 1e-5),
                                 ("H₂, full CI at minimum", e_min, "H2_exact", 3e-3)):
        row(G_VAL, label, "total energy", "this code vs. published benchmark",
            "rayleigh_ritz_bound", got, bench[key], "hartree", "ok", tol, 1e-7)
    VARIATIONAL = [("He: HF ≥ s-only CI", he_hf, he_s_fci),
                   ("He: s-only CI ≥ Hylleraas", he_s_fci, he_hyl),
                   ("He: Hylleraas ≥ exact", he_hyl, bench["He_exact"]),
                   ("H₂: full CI ≥ exact", e_min, bench["H2_exact"]),
                   ("Li: HF ≥ full CI (s, p)", li, li_ci),
                   ("Li: full CI ≥ exact", li_ci, bench["Li_exact"]),
                   ("Li: more orbitals lower the energy (s,p ≥ s,p,d)", li_ci, li_d),
                   ("Li: full CI (s, p, d) ≥ exact", li_d, bench["Li_exact"]),
                   ("Li⁺: full CI ≥ exact", li_ion_ci, bench["LiPlus_exact"]),
                   ("Be: HF ≥ full CI (s, p)", be, be_ci),
                   ("Be: full CI ≥ exact", be_ci, bench["Be_exact"]),
                   *([("Be: more orbitals lower the energy (s,p ≥ s,p,d)", be_ci, pre["E_Be_hartree"]),
                      ("Be: full CI (s, p, d) ≥ exact", pre["E_Be_hartree"], bench["Be_exact"])]
                     if pre else []),
                   *([("Be: more orbitals lower the energy (s,p,d ≥ s,p,d,f)",
                       pre["E_Be_hartree"], pre_f["E_Be_hartree"]),
                      ("Be: full CI (s, p, d, f) ≥ exact", pre_f["E_Be_hartree"], bench["Be_exact"])]
                     if pre and pre_f else []),
                   ("H⁻: HF ≥ s-only CI", hm_hf, hm_s),
                   ("H⁻: s-only CI ≥ Hylleraas", hm_s, hm_hyl),
                   ("H⁻: Hylleraas ≥ exact", hm_hyl, bench["HMinus_exact"])]
    STABILITY = [("H⁻ unbound in Hartree–Fock", hm_hf > -0.5),
                 ("H⁻ bound with correlation", hm_hyl < -0.5),
                 ("general CI solver reproduces the two-electron solver (He)",
                  abs(quantum.helium_fci_s_general() - he_s_fci) < 1e-8)]


# ---------------------------------------------------------------- chemical bonds
COULOMB = 14.3996454            # e²/(4πε₀) in eV·Å
DEBYE_PER_E_ANGSTROM = 4.80320471
NACL = DATA["NaCl"]
CURVE = {}                      # filled below; used for the figure


def nacl_model(alpha):
    """Ionic model of NaCl: point charges + exponential repulsion + polarization.

    E(r) = -K/r - K alpha/(2 r^4) + A exp(-r/rho), relative to free ions.
    A and rho are fixed by the measured bond length and vibration frequency.
    Returns (rho, A, energy function relative to neutral atoms)."""
    r = NACL["r_e_angstrom"]
    mu_kg = reduced(MASS["23Na"], MASS["35Cl"]) * U
    k = mu_kg * (2 * math.pi * C * 100 * NACL["omega_e_cm1"]) ** 2 / E_CH * 1e-20   # eV/Å²
    slope = COULOMB / r ** 2 + 2 * COULOMB * alpha / r ** 5
    rho = slope / (k + 2 * COULOMB / r ** 3 + 10 * COULOMB * alpha / r ** 6)
    a_rep = rho * slope * math.exp(r / rho)
    gap = NACL["Na_ionization_eV"] - NACL["Cl_electron_affinity_eV"]

    def energy(x):
        return -COULOMB / x - COULOMB * alpha / (2 * x ** 4) + a_rep * math.exp(-x / rho) + gap
    return rho, a_rep, energy


r_nacl = NACL["r_e_angstrom"]
gap_nacl = NACL["Na_ionization_eV"] - NACL["Cl_electron_affinity_eV"]
de_nacl = NACL["D0_eV"] + NACL["omega_e_cm1"] / 2 / 8065.543937
alpha_sum = NACL["alpha_Na_plus_A3"] + NACL["alpha_Cl_minus_A3"]
n1 = row(G_BOND, "NaCl", "binding energy D_e", "two point charges ±e at the bond length",
         "not formalised", COULOMB / r_nacl - gap_nacl, de_nacl, "eV", "limit", 0.05, UNC["NaCl_De"],
         "no repulsion and no polarization; the two omissions partly cancel")
n2 = row(G_BOND, "NaCl", "binding energy D_e", "point charges + short-range repulsion",
         "not formalised", -nacl_model(0.0)[2](r_nacl), de_nacl, "eV", "limit", 0.05,
         UNC["NaCl_De"], "repulsion range fitted to the measured vibration frequency", base=n1,
         fix="the electron clouds resist overlapping (this alone makes it worse)")
row(G_BOND, "NaCl", "binding energy D_e", "+ polarization of the ions", "not formalised",
    -nacl_model(alpha_sum)[2](r_nacl), de_nacl, "eV", "ok", 0.06, UNC["NaCl_De"], base=n2,
    fix="each ion's cloud is pulled towards the other ion",
    note="uses free-ion polarizabilities; van der Waals attraction is left out")
dip0 = r_nacl * DEBYE_PER_E_ANGSTROM
a1, a2 = NACL["alpha_Na_plus_A3"], NACL["alpha_Cl_minus_A3"]
induced = (r_nacl ** 4 * (a1 + a2) + 4 * r_nacl * a1 * a2) / (r_nacl ** 6 - 4 * a1 * a2)
n4 = row(G_BOND, "NaCl", "electric dipole moment", "two point charges ±e at the bond length",
         "not formalised", dip0, NACL["dipole_debye"], "D", "limit", 0.05, UNC["NaCl_dipole"],
         "a full electron is not cleanly on one side: the ions polarise each other")
row(G_BOND, "NaCl", "electric dipole moment", "+ polarization of the ions (Rittner)",
    "not formalised", (r_nacl - induced) * DEBYE_PER_E_ANGSTROM, NACL["dipole_debye"], "D", "ok",
    0.15, UNC["NaCl_dipole"], base=n4, fix="induced dipoles point against the main one",
    note="overcorrects: Cl⁻ is less polarizable when squeezed against Na⁺ than when free")

# NaCl with nothing borrowed from measurement: Hartree-Fock for all 28 electrons
# (precomputed by analyzer/nacl_hf.py). Everything below comes from the curve E(R).
NACL_HF = PRECOMPUTED.get("NaCl_hartree_fock")
NACL_CURVE = None
if NACL_HF and quantum is not None:
    np_ = quantum.np
    pts = sorted(NACL_HF["points"], key=lambda p: p["R_angstrom"])
    xs = np_.array([p["R_angstrom"] for p in pts])
    fit = np_.polyfit(xs - 2.4, [p["molecule"] for p in pts], 4)
    fine = np_.linspace(xs[0], xs[-1], 40001)
    r_hf = float(fine[np_.polyval(fit, fine - 2.4).argmin()])
    e_min = float(np_.polyval(fit, r_hf - 2.4))
    k_hf = float(np_.polyval(np_.polyder(fit, 2), r_hf - 2.4))               # hartree / Å²
    mu_kg = reduced(MASS["23Na"], MASS["35Cl"]) * U
    w_hf = math.sqrt(k_hf * HARTREE_EV * E_CH * 1e20 / mu_kg) / (2 * math.pi * C * 100)
    dip_fit = np_.polyfit(xs - 2.4, [abs(p["field_plus"] - p["field_minus"]) / (2 * NACL_HF["field_au"])
                                     * 2.541746473 for p in pts], 2)
    dip_hf = float(np_.polyval(dip_fit, r_hf - 2.4))
    dip_at_measured = float(np_.polyval(dip_fit, r_nacl - 2.4))
    ions = float(np_.polyval(np_.polyfit(xs - 2.4, [p["Na_plus"] + p["Cl_minus"] for p in pts], 2),
                             r_hf - 2.4))
    at = next(p for p in pts if "Na" in p)
    de_ions = (ions - e_min) * HARTREE_EV
    de_atoms = (at["Na"] + at["Cl"] - e_min) * HARTREE_EV
    ie_na, ea_cl = (at["Na_plus"] - at["Na"]) * HARTREE_EV, (at["Cl"] - at["Cl_minus"]) * HARTREE_EV
    NACL_CURVE = (pts, e_min, ions)
    HFR, HFL = NACL["hartree_fock_reference"], NACL["hartree_fock_limit_hartree"]
    hf = "Hartree–Fock, all 28 electrons, nothing measured"
    row(G_BOND, "NaCl", "bond length", hf, "rayleigh_ritz_bound", r_hf, r_nacl, "Å", "ok", 0.02, 1e-6,
        "precomputed; the minimum of the computed energy curve")
    row(G_BOND, "NaCl", "vibration wavenumber ω_e", hf, "rayleigh_ritz_bound", w_hf,
        NACL["omega_e_cm1"], "cm⁻¹", "ok", 0.04, 1e-5, "curvature of the computed energy curve")
    row(G_BOND, "NaCl", "electric dipole moment", hf + ", at its own bond length",
        "rayleigh_ritz_bound", dip_hf, NACL["dipole_debye"], "D", "ok", 0.08, UNC["NaCl_dipole"],
        "from the energy change in a weak electric field")
    row(G_BOND, "NaCl", "electric dipole moment", "Hartree–Fock, at the measured bond length",
        "rayleigh_ritz_bound", dip_at_measured, NACL["dipole_debye"], "D", "ok", 0.06,
        UNC["NaCl_dipole"], "shows how much of the dipole error is just the too-long bond")
    row(G_BOND, "NaCl", "energy to separate into ions Na⁺ + Cl⁻", hf, "rayleigh_ritz_bound",
        de_ions, de_nacl + gap_nacl, "eV", "ok", 0.05, UNC["NaCl_De"],
        "reference: measured binding energy + ionization energy − electron affinity")
    h1 = row(G_BOND, "NaCl", "binding energy D_e", hf, "rayleigh_ritz_bound", de_atoms, de_nacl,
             "eV", "limit", 0.05, UNC["NaCl_De"],
             "to neutral atoms: fails because Hartree–Fock gets the electron affinity of Cl wrong")
    CL_CI = PRECOMPUTED.get("Cl_affinity")
    if CL_CI:
        ea_ci = (CL_CI["E_Cl_hartree"] - CL_CI["E_Cl_minus_hartree"]) * HARTREE_EV
        h1 = row(G_BOND, "NaCl", "binding energy D_e", "Hartree–Fock + correlation in Cl and Cl⁻",
                 "rayleigh_ritz_bound", de_ions - ie_na + ea_ci, de_nacl, "eV", "limit", 0.05,
                 UNC["NaCl_De"], base=h1, fix="the outer electrons of Cl and Cl⁻ by full CI",
                 note="still nothing measured; closes over half the gap, limited by the electron affinity below")
    CL_SEL = PRECOMPUTED.get("Cl_affinity_selected")
    if CL_SEL:
        ea_sel = (CL_SEL["E_Cl_hartree"] - CL_SEL["E_Cl_minus_hartree"]) * HARTREE_EV
        h1 = row(G_BOND, "NaCl", "binding energy D_e", "Hartree–Fock + correlation in Cl and Cl⁻, 44 orbitals",
                 "rayleigh_ritz_bound", de_ions - ie_na + ea_sel, de_nacl, "eV", "ok", 0.05,
                 UNC["NaCl_De"], base=h1, fix="s, p, d and f correlating functions, selected determinants",
                 note="still nothing measured; the determinant list is not converged, see the electron affinity below")
    row(G_BOND, "NaCl", "binding energy D_e", "Hartree–Fock separation into ions + measured cost of the ions",
        "rayleigh_ritz_bound, coulomb_crossing", de_ions - gap_nacl, de_nacl, "eV", "ok", 0.07,
        UNC["NaCl_De"], base=h1, fix="measured ionization energy and electron affinity replace the Hartree–Fock ones",
        note="borrows two measured atomic numbers; nothing about the molecule is borrowed")
    row(G_BOND, "Na", "first ionization energy", "Hartree–Fock, E(Na⁺) − E(Na)", "rayleigh_ritz_bound",
        ie_na, NACL["Na_ionization_eV"], "eV", "ok", 0.05, UNC["ionization_neutral"],
        "the missing 4 % is electron correlation")
    row(G_BOND, "Cl", "electron affinity", "Hartree–Fock, E(Cl) − E(Cl⁻)", "rayleigh_ritz_bound",
        ea_cl, NACL["Cl_electron_affinity_eV"], "eV", "limit", 0.05, UNC["electron_affinity"],
        "the extra electron is held largely by correlation with the other seven outer electrons")
    if CL_CI:
        row(G_BOND, "Cl", "electron affinity",
            f"outer electrons by full CI, inner ten frozen ({CL_CI['active_orbitals']} orbitals)",
            "rayleigh_ritz_bound", ea_ci, NACL["Cl_electron_affinity_eV"], "eV", "limit", 0.05,
            UNC["electron_affinity"],
            "precomputed; over half the gap closed. A second d shell changed it by only 0.003 eV, "
            "so the rest lies elsewhere (f functions, more s and p functions)")
        VARIATIONAL += [("Cl⁻: Hartree–Fock ≥ valence full CI", CL_CI["E_Cl_minus_hartree_fock"],
                         CL_CI["E_Cl_minus_hartree"])]
    if CL_SEL:
        row(G_BOND, "Cl", "electron affinity",
            f"outer electrons by selected CI, inner ten frozen ({CL_SEL['active_orbitals']} orbitals, "
            f"{CL_SEL['determinants'] // 1000000} million determinants)",
            "rayleigh_ritz_bound, second_order_estimate_brackets", ea_sel, NACL["Cl_electron_affinity_eV"],
            "eV", "ok", 0.05, UNC["electron_affinity"],
            "precomputed; f functions and more s, p and d functions. Each energy is an upper bound, the "
            "difference is not; the value still rises slowly with the number of determinants")
        if CL_CI:
            VARIATIONAL += [("Cl⁻: 20-orbital full CI ≥ 44-orbital selected CI", CL_CI["E_Cl_minus_hartree"],
                             CL_SEL["E_Cl_minus_hartree"]),
                            ("Cl: 20-orbital full CI ≥ 44-orbital selected CI", CL_CI["E_Cl_hartree"],
                             CL_SEL["E_Cl_hartree"])]
    # the same numbers against other people's Hartree-Fock: tests the code, not the physics
    row(G_VAL, "NaCl", "bond length (Hartree–Fock)", "this code vs. large-basis Hartree–Fock",
        "rayleigh_ritz_bound", r_hf, HFR["r_e_angstrom"], "Å", "ok", 0.005, 1e-4)
    row(G_VAL, "NaCl", "vibration wavenumber (Hartree–Fock)", "this code vs. large-basis Hartree–Fock",
        "rayleigh_ritz_bound", w_hf, HFR["omega_e_cm1"], "cm⁻¹", "ok", 0.02, 3e-3)
    row(G_VAL, "NaCl", "dipole moment (Hartree–Fock)", "this code vs. large-basis Hartree–Fock",
        "rayleigh_ritz_bound", dip_hf, HFR["dipole_debye"], "D", "ok", 0.03, 1e-4)
    row(G_VAL, "NaCl", "total energy (Hartree–Fock)", "this code vs. large-basis Hartree–Fock",
        "rayleigh_ritz_bound", e_min, HFR["energy_hartree"], "Eh", "ok", 1e-4, 1e-9)
    for key, label in (("Na_plus", "Na⁺"), ("Cl_minus", "Cl⁻")):
        VARIATIONAL += [(f"{label}: this basis ≥ Hartree–Fock limit", min(p[key] for p in pts), HFL[key])]
    VARIATIONAL += [("NaCl: this basis ≥ large-basis Hartree–Fock energy", e_min, HFR["energy_hartree"])]
    STABILITY += [("NaCl Hartree–Fock is bound against separation into ions", de_ions > 0)]

if quantum is not None:
    e_atoms = 2 * e_h
    grid = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0]
    CURVE = {R: quantum.h2_point(R) for R in grid}
    for R, tol in ((1.0, 0.03), (2.0, 0.03), (3.0, 0.04), (4.0, 0.06)):
        exact = (-1.0 - DATA["H2_curve_benchmark_hartree"][f"{R:.1f}"]) * HARTREE_EV
        row(G_BOND, f"H₂ at {R * BOHR_A:.2f} Å", "energy below two free atoms", "full CI",
            "rayleigh_ritz_bound", (e_atoms - CURVE[R][0]) * HARTREE_EV, exact, "eV", "ok", tol, 1e-7,
            "reference: benchmark calculation, not a measurement")
    exact6 = (-1.0 - DATA["H2_curve_benchmark_hartree"]["6.0"]) * HARTREE_EV
    row(G_BOND, f"H₂ at {6 * BOHR_A:.2f} Å", "energy below two free atoms", "full CI",
        "rayleigh_ritz_bound", (e_atoms - CURVE[6.0][0]) * HARTREE_EV, exact6, "eV", "limit", 0.05,
        1e-7, "the weak long-range attraction needs more diffuse, directional orbitals")
    row(G_BOND, f"H₂ at {6 * BOHR_A:.2f} Å", "energy below two free atoms",
        "Hartree–Fock: both electrons forced to share one orbital", "rayleigh_ritz_bound",
        (e_atoms - CURVE[6.0][1]) * HARTREE_EV, exact6, "eV", "limit", 0.05, 1e-7,
        "a shared orbital puts both electrons on one atom half the time")
    for key, bench_t in DATA["H2_triplet_benchmark_hartree"].items():
        R = float(key)
        row(G_BOND, f"H₂, equal spins, at {R * BOHR_A:.2f} Å", "energy above two free atoms",
            "full CI, antisymmetric pair function", "rayleigh_ritz_bound",
            (CURVE[R][2] - e_atoms) * HARTREE_EV, (bench_t + 1.0) * HARTREE_EV, "eV",
            "ok" if R < 3.5 else "limit", 0.04 if R < 3.5 else 0.05, 1e-5,
            "reference: benchmark calculation, not a measurement" if R < 3.5 else
            "the weak long-range repulsion needs more diffuse, directional orbitals")
    # below about 0.6 Å other equal-spin states come close in energy, so start at 1.2 bohr
    rs_t = [R for R in sorted(CURVE) if 1.2 <= R <= 6.0]
    STABILITY += [("H₂ with equal spins repels at every distance checked (0.64 to 3.2 Å)",
                   all(CURVE[R][2] > e_atoms for R in rs_t)),
                  ("H₂ with equal spins: energy falls steadily as the atoms separate",
                   all(CURVE[a][2] > CURVE[b][2] for a, b in zip(rs_t, rs_t[1:]))),
                  ("H₂ with equal spins ends as two free atoms",
                   abs(CURVE[10.0][2] - e_atoms) * HARTREE_EV < 5e-3),
                  ("at the bond length, equal spins cost more than 10 eV",
                   (CURVE[1.4][2] - CURVE[1.4][0]) * HARTREE_EV > 10.0)]
    STABILITY += [("H₂ full CI separates into two free atoms",
                   abs(CURVE[10.0][0] - e_atoms) * HARTREE_EV < 1e-3),
                  ("H₂ Hartree–Fock fails to separate (several eV too high)",
                   (CURVE[10.0][1] - e_atoms) * HARTREE_EV > 5.0),
                  ("H₂ curve has its minimum at the bond length",
                   min(CURVE, key=lambda R: CURVE[R][0]) == 1.4)]


# ---------------------------------------------------------------- magnetism
# A magnetic field makes the whole electron cloud circulate. The induced current
# opposes the field: at the nucleus by the fraction (α²/3)·Σ⟨a₀/r⟩ (Lamb shielding),
# and as a whole by the susceptibility −(N_A α² a₀³/6)·Σ⟨r²/a₀²⟩ (Langevin).
MAG = DATA["magnetism"]
SHIELD_PPM = ALPHA ** 2 / 3 * 1e6
CHI_UNIT = 6.02214076e23 * ALPHA ** 2 * (BOHR_A * 1e-8) ** 3 / 6 * 1e6     # 10⁻⁶ cm³/mol per a₀²
row(G_MAG, "H", "field shielded at the nucleus", "Lamb formula, α²/3", "not formalised",
    SHIELD_PPM, MAG["H_shielding_ppm"], "ppm", "ok", 2e-3, UNC["shielding_theory"],
    "reference: calculation with relativistic and recoil terms, not a measurement")
m1 = row(G_MAG, "He", "field shielded at the nucleus", "two independent Bohr electrons",
         "not formalised", SHIELD_PPM * 2 * 2, MAG["He3_shielding_ppm"], "ppm", "limit", 0.05,
         UNC["shielding_theory"], "ignores electron–electron repulsion")
m2 = row(G_MAG, "He", "field shielded at the nucleus", "one screening parameter, ζ = Z − 5/16",
         "helium_variational", SHIELD_PPM * 2 * (2 - 5 / 16), MAG["He3_shielding_ppm"], "ppm", "ok",
         3e-3, UNC["shielding_theory"], base=m1, fix="average screening by the other electron")
chi_ref = MAG["He_susceptibility_measured_1e6_cm3_per_mol"]
chi_unc = MAG["He_susceptibility_measured_uncertainty"] / abs(chi_ref)
c1 = row(G_MAG, "He", "diamagnetic susceptibility", "two independent Bohr electrons",
         "not formalised", -CHI_UNIT * 2 * 3 / 2 ** 2, chi_ref, "10⁻⁶ cm³/mol", "limit", 0.05, chi_unc,
         "ignores electron–electron repulsion: the cloud comes out too small")
c2 = row(G_MAG, "He", "diamagnetic susceptibility", "one screening parameter, ζ = Z − 5/16",
         "helium_variational", -CHI_UNIT * 2 * 3 / (2 - 5 / 16) ** 2, chi_ref, "10⁻⁶ cm³/mol",
         "limit", 0.05, chi_unc, "a single exponential falls off too fast at large distance",
         base=c1, fix="average screening by the other electron")
if quantum is not None:
    _, inv_r, r_sq = quantum.hylleraas_helium(order=8, alpha=quantum.Fraction(2), expect=True)
    row(G_MAG, "He", "field shielded at the nucleus", "Hylleraas wavefunction, Σ⟨1/r⟩",
        "not formalised", SHIELD_PPM * inv_r, MAG["He3_shielding_ppm"], "ppm", "ok", 1e-3,
        UNC["shielding_theory"], base=m2, fix="the correlated electron cloud",
        note="non-relativistic; reference is a calculation with relativistic terms")
    chi_he = -CHI_UNIT * r_sq / recoil("4He", 2) ** 2
    row(G_MAG, "He", "diamagnetic susceptibility", "Hylleraas wavefunction, Σ⟨r²⟩",
        "not formalised", chi_he, chi_ref, "10⁻⁶ cm³/mol", "limit", 0.05, chi_unc, base=c2,
        fix="the correlated electron cloud",
        note="the measurement (1960) is the weak link here: see the next row")
    row(G_MAG, "He", "diamagnetic susceptibility", "Hylleraas vs. best published calculation",
        "not formalised", chi_he, MAG["He_susceptibility_theory_1e6_cm3_per_mol"], "10⁻⁶ cm³/mol",
        "ok", 1e-3, 1e-4)


# ---------------------------------------------------------------- van der Waals
# Two neutral atoms attract because each one's fluctuating electron cloud
# polarises the other. At long range the energy is -C6 / r^6.
DISP = DATA["helium_dispersion"]
HARTREE_K = 315775.02480407
alpha_meas = (DISP["molar_polarizability_cm3_per_mol"] * 3 / (4 * math.pi * 6.02214076e23)
              / (BOHR_A * 1e-8) ** 3)
u_alpha = DISP["molar_polarizability_uncertainty"] / DISP["molar_polarizability_cm3_per_mol"]
v1 = row(G_VDW, "He", "polarizability", "one screening parameter, ζ = Z − 5/16",
         "helium_variational", 9 / (2 - 5 / 16) ** 4, alpha_meas, "a₀³", "limit", 0.05, u_alpha,
         "a single exponential is too stiff: the cloud is easier to distort than that")
PAIR = None
HAVE_BEST = False
N_A = 6.02214076e23
if quantum is not None:
    disp = quantum.helium_dispersion()
    alpha_he, c6_he, f_sum = disp["alpha1"], disp["C6"], disp["f_sum"]
    row(G_VDW, "He", "polarizability", "full CI, sum over all excited states", "not formalised",
        alpha_he, alpha_meas, "a₀³", "ok", 0.005, u_alpha, base=v1,
        fix="the field mixes in excited states; correlation included",
        note="reference: measured dielectric constant of helium gas")
    row(G_VDW, "He", "quadrupole polarizability", "full CI, sum over all excited states",
        "not formalised", disp["alpha2"], DISP["quadrupole_polarizability_au"], "a₀⁵", "ok", 0.01,
        1e-9, "reference: benchmark calculation, not a measurement")
    row(G_VDW, "He", "octupole polarizability", "full CI, sum over all excited states",
        "not formalised", disp["alpha3"], DISP["octupole_polarizability_au"], "a₀⁷", "ok", 0.01,
        1e-8, "reference: benchmark calculation, not a measurement")
    i_he = ie["He"] / HARTREE_EV
    v3 = row(G_VDW, "He–He", "van der Waals coefficient C₆", "London formula, ¾ α² I",
             "not formalised", 0.75 * alpha_he ** 2 * i_he, DISP["C6_au"], "a.u.", "limit", 0.05,
             1e-6, "treats the atom as a single oscillator at the ionization energy")
    row(G_VDW, "He–He", "van der Waals coefficient C₆", "Casimir–Polder sum over all excited states",
        "not formalised", c6_he, DISP["C6_au"], "a.u.", "ok", 0.005, 1e-6, base=v3,
        fix="every excited state contributes with its own frequency",
        note="reference: benchmark calculation, not a measurement")
    row(G_VDW, "He–He", "van der Waals coefficient C₈", "Casimir–Polder sum, dipole with quadrupole",
        "not formalised", disp["C8"], DISP["C8_au"], "a.u.", "ok", 0.01, 1e-9,
        "reference: benchmark calculation, not a measurement")
    row(G_VDW, "He–He", "van der Waals coefficient C₁₀",
        "Casimir–Polder sum, dipole with octupole and quadrupole with quadrupole",
        "not formalised", disp["C10"], DISP["C10_au"], "a.u.", "ok", 0.01, 1e-8,
        "reference: benchmark calculation, not a measurement")

    # The whole curve: Hartree-Fock repulsion plus damped dispersion.
    PAIR = quantum.HeliumPair(disp)
    r_m = DISP["well_position_bohr"]
    where = lambda R: f"He–He at {R * BOHR_A:.2f} Å"
    for R in (4.0, 5.6):
        row(G_VDW, where(R), "Hartree–Fock repulsion", "Hartree–Fock for the pair, s and p functions",
            "not formalised", float(quantum.helium_pair_hf(R)) * 1e6 if R not in PAIR.R else
            float(PAIR.hf[list(PAIR.R).index(R)]) * 1e6,
            DISP["hartree_fock_limit_microhartree"][f"{R:.1f}"], "μEh", "ok", 0.005,
            0.005 / DISP["hartree_fock_limit_microhartree"][f"{R:.1f}"],
            "reference: Hartree–Fock solved on a grid, free of basis-set error")
    row(G_VDW, where(r_m), "interaction energy at the minimum", "Hartree–Fock alone",
        "not formalised", float(PAIR.repulsion(r_m)) * HARTREE_K, -DISP["well_depth_K"], "K",
        "limit", 0.05, 4e-4,
        "wrong sign: Hartree–Fock has no fluctuations, so two helium atoms only repel")
    row(G_VDW, where(r_m), "interaction energy at the minimum", "−C₆/r⁶ alone",
        "not formalised", -c6_he / r_m ** 6 * HARTREE_K, -DISP["well_depth_K"], "K", "limit", 0.05,
        4e-4, "attraction only: at this distance the clouds also overlap and repel")
    depth6, _, _ = PAIR.well(terms=(6,))
    depth, r_min, sigma = PAIR.well()
    depth_x, _, _ = PAIR.well(extrapolate=True)
    w6 = row(G_VDW, "He–He", "well depth", "Hartree–Fock repulsion + damped C₆ term",
             "not formalised", depth6 * HARTREE_K, DISP["well_depth_K"], "K", "limit", 0.05, 4e-4,
             "dipole fluctuations alone give little more than half the well")
    wd = row(G_VDW, "He–He", "well depth", "Hartree–Fock repulsion + damped C₆, C₈, C₁₀ terms",
             "not formalised", depth * HARTREE_K, DISP["well_depth_K"], "K", "ok", 0.04, 4e-4, base=w6,
             fix="quadrupole and octupole fluctuations (C₈, C₁₀)",
             note="nothing fitted to the reference; partly a cancellation, see the next rows")
    wx = row(G_VDW, "He–He", "well depth", "same, plus C₁₂ to C₁₆ estimated by extrapolation",
             "not formalised", depth_x * HARTREE_K, DISP["well_depth_K"], "K", "limit", 0.05, 4e-4,
             "more attraction makes it worse: correlation inside each atom also adds repulsion, "
             "which this model leaves out", base=wd, fix="the next three attraction terms, estimated")
    row(G_VDW, "He–He", "position of the minimum", "Hartree–Fock repulsion + damped C₆, C₈, C₁₀ terms",
        "not formalised", r_min * BOHR_A, r_m * BOHR_A, "Å", "ok", 0.01, 2e-5)
    row(G_VDW, "He–He", "distance where the energy is zero",
        "Hartree–Fock repulsion + damped C₆, C₈, C₁₀ terms", "not formalised", sigma * BOHR_A,
        DISP["zero_crossing_bohr"] * BOHR_A, "Å", "ok", 0.01, 1e-4,
        "reference interpolated from the tabulated benchmark points")
    for R, tol, note in ((3.0, 0.03, "on the repulsive wall"),
                         (4.0, 0.08, "the weakest region: the model lacks the extra repulsion "
                                     "that correlation inside each atom gives"),
                         (6.0, 0.02, ""), (7.0, 0.01, ""), (9.0, 0.005, ""),
                         (12.0, 0.005, "here only the C₆ term matters")):
        row(G_VDW, where(R), "interaction energy", "Hartree–Fock repulsion + damped C₆, C₈, C₁₀ terms",
            "not formalised", float(PAIR(R)) * HARTREE_K, DISP["potential_BO_K"][f"{R:.1f}"], "K",
            "ok", tol, 1e-5, note or "reference: benchmark calculation, not a measurement")

    # What the curve predicts for helium gas, and for the pair as a molecule.
    m_he = MASS["4He"] / ME_U
    t_virial = 273.16
    b_cl, b_q = (x * N_A * (BOHR_A * 1e-8) ** 3
                 for x in PAIR.second_virial(t_virial / HARTREE_K, m_he))
    u_b = DISP["second_virial_uncertainty"] / DISP["second_virial_273.16K_cm3_per_mol"]
    b1 = row(G_VDW, "He gas, 273.16 K", "second virial coefficient",
             "computed curve, atoms as classical particles", "not formalised", b_cl,
             DISP["second_virial_273.16K_cm3_per_mol"], "cm³/mol", "limit", 0.03, u_b,
             "helium atoms are light enough that their wave nature shows even at 0 °C")
    row(G_VDW, "He gas, 273.16 K", "second virial coefficient",
        "computed curve + first quantum correction", "not formalised", b_cl + b_q,
        DISP["second_virial_273.16K_cm3_per_mol"], "cm³/mol", "ok", 0.03, u_b, base=b1,
        fix="the atoms are waves: they cannot sit still at the wall",
        note="a measurement; the remaining 2 % is the too-soft wall of the model")
    bound = PAIR.bound_state(m_he)
    if bound is not None:
        e_dimer = -bound[0] * HARTREE_EV * 1e9
        row(G_VDW, "He₂ molecule", "binding energy", "lowest level in the computed curve",
            "not formalised", e_dimer, DISP["dimer_binding_neV"], "neV", "limit", 0.2,
            DISP["dimer_binding_uncertainty_neV"] / DISP["dimer_binding_neV"],
            f"bound, as measured, but the binding is about 1/6000 of the well depth, so a "
            f"{abs(depth * HARTREE_K / DISP['well_depth_K'] - 1) * 100:.0f} % error in the well "
            "changes it severalfold")
        STABILITY += [("He₂ is bound in the computed curve, by less than 1/1000 of the well depth",
                       0 < -bound[0] < depth / 1000)]
    # The repulsion Hartree-Fock misses, from a four-electron full CI (precomputed).
    he2 = PRECOMPUTED.get("He2_full_ci")
    if he2:
        pts = [p for p in he2["points"] if p["R"] <= 7.0]
        PAIR.add_correlation([p["R"] for p in pts],
                             [p["full_ci"] - p["dispersion"] - p["hartree_fock"] for p in pts])
        full = dict(correlated=True, extrapolate=True)
        label = "full-CI repulsion + damped C₆ to C₁₆ terms"
        at_min = next(p for p in pts if p["R"] == 5.6)
        STABILITY += [("He–He: correlation adds repulsion at every distance computed",
                       all(p["full_ci"] - p["dispersion"] - p["hartree_fock"] > 0 for p in pts)),
                      ("He–He four-electron full CI is itself bound at 2.96 Å", at_min["full_ci"] < 0)]
        VARIATIONAL += [("He–He at 2.96 Å: Hartree–Fock ≥ full CI interaction (same basis)",
                         at_min["hartree_fock"], at_min["full_ci"])]
        row(G_VDW, where(5.6), "interaction energy", "four-electron full CI alone, s and p functions",
            "not formalised", at_min["full_ci"] * HARTREE_K, DISP["potential_BO_K"]["5.6"], "K",
            "limit", 0.05, 2e-5,
            f"precomputed, {at_min['orbitals']} orbitals: bound, but a basis this small holds less "
            "than half of the attraction")
        depth_c, r_min_c, _ = PAIR.well(**full)
        wc = row(G_VDW, "He–He", "well depth", label, "not formalised", depth_c * HARTREE_K,
            DISP["well_depth_K"], "K", "ok", 0.05, 4e-4, base=wx,
            fix="repulsion from correlation, taken from a four-electron full CI",
            note="no cancellation left, but no closer: the damped series is now the weak part")
        row(G_VDW, "He–He", "position of the minimum", label, "not formalised", r_min_c * BOHR_A,
            r_m * BOHR_A, "Å", "ok", 0.01, 2e-5)
        row(G_VDW, where(4.0), "interaction energy", label, "not formalised",
            float(PAIR(4.0, **full)) * HARTREE_K, DISP["potential_BO_K"]["4.0"], "K", "ok", 0.08, 1e-5,
            "now too repulsive on the wall instead of too soft")
        b_c = sum(PAIR.second_virial(t_virial / HARTREE_K, m_he, **full)) * N_A * (BOHR_A * 1e-8) ** 3
        row(G_VDW, "He gas, 273.16 K", "second virial coefficient",
            "full-CI repulsion + damped C₆ to C₁₆ terms, with quantum correction", "not formalised",
            b_c, DISP["second_virial_273.16K_cm3_per_mol"], "cm³/mol", "ok", 0.03, u_b)
    # Best curve: repulsion from the larger full CI, attraction computed directly.
    he2_d, he2_disp = PRECOMPUTED.get("He2_full_ci_d"), PRECOMPUTED.get("He2_dispersion")
    if he2_d and he2_disp:
        series_min = float(PAIR.dispersion(5.6, extrapolate=True)) * HARTREE_K
        direct_min = next(p for p in he2_disp["points"] if p["R"] == 5.6)["dispersion"] * HARTREE_K
        d1 = row(G_VDW, where(5.6), "attraction (dispersion energy)", "damped series, C₆ to C₁₆",
                 "not formalised", series_min, DISP["dispersion_at_5.6_bohr_K"], "K", "limit", 0.02,
                 1e-4, "the assumed damping switches the attraction off too early")
        row(G_VDW, where(5.6), "attraction (dispersion energy)",
            "computed directly with overlapping clouds, large basis", "not formalised", direct_min,
            DISP["dispersion_at_5.6_bohr_K"], "K", "ok", 0.01, 1e-4, base=d1,
            fix="no series and no damping function: every excited state of both atoms, full Coulomb coupling",
            note=f"precomputed, {he2_disp['points'][0]['functions']} functions; reference quoted "
                 "from a secondary source")
        pts = sorted((p for p in he2_d["points"] if p["R"] <= 7.0), key=lambda p: p["R"])
        PAIR.add_correlation([p["R"] for p in pts],
                             [p["full_ci"] - p["dispersion"] - p["hartree_fock"] for p in pts])
        PAIR.add_direct_dispersion([p["R"] for p in he2_disp["points"]],
                                   [p["dispersion"] for p in he2_disp["points"]])
        HAVE_BEST = True
        best = "full-CI repulsion + directly computed attraction"
        depth_b, r_min_b, sigma_b = PAIR.well(best=True)
        row(G_VDW, "He–He", "well depth", best, "not formalised", depth_b * HARTREE_K,
            DISP["well_depth_K"], "K", "ok", 0.03, 4e-4, base=wc,
            fix="attraction computed directly; repulsion from a larger full CI (with d functions)",
            note="the repulsion is still about 0.3 K short at the minimum: its basis is the limit")
        row(G_VDW, "He–He", "position of the minimum", best, "not formalised", r_min_b * BOHR_A,
            r_m * BOHR_A, "Å", "ok", 0.005, 2e-5)
        row(G_VDW, "He–He", "distance where the energy is zero", best, "not formalised",
            sigma_b * BOHR_A, DISP["zero_crossing_bohr"] * BOHR_A, "Å", "ok", 0.005, 1e-4,
            "reference interpolated from the tabulated benchmark points")
        for R, tol in ((3.0, 0.03), (4.0, 0.02), (6.0, 0.02), (7.0, 0.02), (9.0, 0.01), (12.0, 0.005)):
            row(G_VDW, where(R), "interaction energy", best, "not formalised",
                float(PAIR.best(R)) * HARTREE_K, DISP["potential_BO_K"][f"{R:.1f}"], "K", "ok", tol,
                1e-5, "reference: benchmark calculation, not a measurement")
        b_b = sum(PAIR.second_virial(t_virial / HARTREE_K, m_he, best=True)) * N_A * (BOHR_A * 1e-8) ** 3
        row(G_VDW, "He gas, 273.16 K", "second virial coefficient", best + ", with quantum correction",
            "not formalised", b_b, DISP["second_virial_273.16K_cm3_per_mol"], "cm³/mol", "ok", 0.01,
            u_b, "a measurement")
        bound_b = PAIR.bound_state(m_he, best=True)
        if bound_b is not None:
            row(G_VDW, "He₂ molecule", "binding energy", "lowest level in the best curve",
                "not formalised", -bound_b[0] * HARTREE_EV * 1e9, DISP["dimer_binding_neV"], "neV",
                "limit", 0.2, DISP["dimer_binding_uncertainty_neV"] / DISP["dimer_binding_neV"],
                "still far off: this number magnifies the remaining error in the well many times")
    STABILITY += [("He oscillator strengths sum to the number of electrons (within 1 %)",
                   abs(f_sum - 2.0) < 0.02),
                  ("He–He Hartree–Fock repels at every distance computed (1.6 to 3.7 Å)",
                   bool((PAIR.hf > 0).all())),
                  ("He–He Hartree–Fock repulsion falls steadily with distance",
                   bool((PAIR.hf[1:] < PAIR.hf[:-1]).all()))]


def bond_figure():
    """SVG: H₂ curve (shared electrons) and NaCl ionic model (transferred electron)."""
    w, hgt, top, bottom = 430, 300, 40, 44
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 {hgt + 20}' "
             "font-family='Helvetica,Arial,sans-serif' font-size='13'>",
             f"<rect width='900' height='{hgt + 20}' fill='#ffffff'/>"]

    def panel(x0, title, xmax, ymin, ymax, xticks, yticks):
        left = x0 + 52

        def px(x):
            return left + (w - 70) * x / xmax

        def py(y):
            return top + (hgt - top - bottom) * (ymax - y) / (ymax - ymin)
        parts.append(f"<text x='{left}' y='22' font-size='15' font-weight='bold' fill='#1f2933'>{title}</text>")
        for t in xticks:
            parts.append(f"<line x1='{px(t):.1f}' y1='{py(ymin):.1f}' x2='{px(t):.1f}' y2='{py(ymin) + 5:.1f}' stroke='#616e7c'/>"
                         f"<text x='{px(t):.1f}' y='{py(ymin) + 19:.1f}' text-anchor='middle' fill='#616e7c'>{t:g}</text>")
        for t in yticks:
            parts.append(f"<line x1='{left}' y1='{py(t):.1f}' x2='{px(xmax):.1f}' y2='{py(t):.1f}' stroke='#e4e7eb'/>"
                         f"<text x='{left - 6}' y='{py(t) + 4:.1f}' text-anchor='end' fill='#616e7c'>{t:g}</text>")
        parts.append(f"<line x1='{left}' y1='{py(0):.1f}' x2='{px(xmax):.1f}' y2='{py(0):.1f}' stroke='#616e7c'/>")
        parts.append(f"<text x='{px(xmax / 2):.1f}' y='{hgt + 12}' text-anchor='middle' fill='#616e7c'>distance between the nuclei (Å)</text>")
        parts.append(f"<text transform='translate({x0 + 13},{py((ymin + ymax) / 2):.1f}) rotate(-90)' text-anchor='middle' fill='#616e7c'>energy (eV)</text>")
        return px, py

    def path(px, py, pts, color, dash=""):
        d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
        extra = f" stroke-dasharray='{dash}'" if dash else ""
        parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.2'{extra}/>")

    px, py = panel(0, "H₂: two shared electrons, by spin", 5.5, -6, 8, [1, 2, 3, 4, 5], [-4, -2, 0, 2, 4, 6])
    if CURVE:
        e_atoms = 2 * e_h
        rs = sorted(CURVE)
        path(px, py, [(R * BOHR_A, (CURVE[R][1] - e_atoms) * HARTREE_EV) for R in rs], "#dc2626", "6 4")
        path(px, py, [(R * BOHR_A, (CURVE[R][0] - e_atoms) * HARTREE_EV) for R in rs], "#2563eb")
        path(px, py, [(R * BOHR_A, (CURVE[R][2] - e_atoms) * HARTREE_EV) for R in rs
                      if R >= 1.2 and (CURVE[R][2] - e_atoms) * HARTREE_EV <= 8], "#059669")
        for key, val in DATA["H2_triplet_benchmark_hartree"].items():
            parts.append(f"<circle cx='{px(float(key) * BOHR_A):.1f}' cy='{py((val + 1) * HARTREE_EV):.1f}' r='3.5' fill='#1f2933'/>")
        parts.append(f"<text x='{px(2.9):.1f}' y='{py(0.9):.1f}' fill='#059669'>equal spins: always repels</text>")
        for key, val in DATA["H2_curve_benchmark_hartree"].items():
            parts.append(f"<circle cx='{px(float(key) * BOHR_A):.1f}' cy='{py((val + 1) * HARTREE_EV):.1f}' r='3.5' fill='#1f2933'/>")
        parts.append(f"<text x='{px(1.2):.1f}' y='{py(7.0):.1f}' fill='#dc2626'>forced sharing (Hartree–Fock)</text>")
        parts.append(f"<text x='{px(2.3):.1f}' y='{py(-2.6):.1f}' fill='#2563eb'>opposite spins: a bond</text>")
        parts.append(f"<text x='{px(2.3):.1f}' y='{py(-4.0):.1f}' fill='#1f2933'>● benchmark values</text>")

    px, py = panel(450, "NaCl: one transferred electron", 12, -6, 3, [2, 4, 6, 8, 10, 12], [-4, -2, 0, 2])
    _, _, energy = nacl_model(alpha_sum)
    xs = [1.9 + 0.1 * i for i in range(102)]
    path(px, py, [(x, energy(x)) for x in xs], "#2563eb")
    parts.append(f"<line x1='{px(0):.1f}' y1='{py(gap_nacl):.1f}' x2='{px(12):.1f}' y2='{py(gap_nacl):.1f}' stroke='#059669' stroke-dasharray='6 4' stroke-width='1.6'/>")
    cross = COULOMB / gap_nacl
    parts.append(f"<circle cx='{px(cross):.1f}' cy='{py(0):.1f}' r='4' fill='#dc2626'/>")
    parts.append(f"<text x='{px(5.2):.1f}' y='{py(gap_nacl) - 6:.1f}' fill='#059669'>free ions Na⁺ + Cl⁻</text>")
    parts.append(f"<text x='{px(8.3):.1f}' y='{py(0) + 17:.1f}' fill='#dc2626'>crossing at {cross:.1f} Å</text>")
    parts.append(f"<text x='{px(3.6):.1f}' y='{py(-3.6):.1f}' fill='#2563eb'>ionic model: −e²/r</text>")
    parts.append(f"<text x='{px(0.3):.1f}' y='{py(0) - 6:.1f}' fill='#616e7c'>neutral atoms</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def attach_theorems():
    """Name the Lean theorem behind each row that rests on a proved principle
    (Movement/Forces.lean). The numbers themselves are computed, not proved."""
    rules = [
        (lambda r: "London" in r["model"], "london_formula"),
        (lambda r: "Casimir–Polder" in r["model"], "casimir_polder_partial_fraction, casimir_polder_value"),
        (lambda r: "polarizability" in r["quantity"] and "full CI" in r["model"], "two_level_lowering"),
        (lambda r: "−C₆/r⁶ alone" in r["model"], "dispersion_nonpos"),
        (lambda r: "damped" in r["model"], "tang_toennies_bounds, dispersion_nonpos"),
        (lambda r: "directly" in r["model"] or "best curve" in r["model"],
         "dispersion_nonpos, two_level_lowering"),
        (lambda r: "four-electron full CI" in r["model"], "rayleigh_ritz_bound"),
        (lambda r: r["system"] == "NaCl" and r["model"].startswith("two point charges")
         and "binding" in r["quantity"], "coulomb_crossing"),
        (lambda r: r["system"] == "NaCl" and "short-range repulsion" in r["model"], "born_mayer_well"),
        (lambda r: r["system"] == "NaCl" and "polarization" in r["model"] and "binding" in r["quantity"],
         "born_mayer_well, rittner_induced_dipoles"),
        (lambda r: r["system"] == "NaCl" and "Rittner" in r["model"], "rittner_induced_dipoles"),
        (lambda r: "quantised vibration" in r["model"], "heat_capacity_ratio, einstein_function_le_one"),
    ]
    for r in ROWS:
        if "equal spins" in r["system"] and "antisymmetric" not in r["lean"]:
            r["lean"] = (r["lean"] + ", " if r["lean"] != "not formalised" else "") + "antisymmetric_vanishes_at_contact"
        if r["lean"] != "not formalised":
            continue
        for test, name in rules:
            if test(r):
                r["lean"] = name
                break


attach_theorems()


def helium_figure():
    """SVG: the helium-helium curve, and the pieces it is made of."""
    w, hgt, top, bottom = 430, 300, 40, 44
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 900 {hgt + 20}' "
             "font-family='Helvetica,Arial,sans-serif' font-size='13'>",
             f"<rect width='900' height='{hgt + 20}' fill='#ffffff'/>"]
    xmin, xmax = 2.0, 6.5

    def panel(x0, title, ymin, ymax, yticks):
        left = x0 + 52

        def px(x):
            return left + (w - 70) * (x - xmin) / (xmax - xmin)

        def py(y):
            return top + (hgt - top - bottom) * (ymax - y) / (ymax - ymin)
        parts.append(f"<text x='{left}' y='22' font-size='15' font-weight='bold' fill='#1f2933'>{title}</text>")
        for t in (2, 3, 4, 5, 6):
            parts.append(f"<line x1='{px(t):.1f}' y1='{py(ymin):.1f}' x2='{px(t):.1f}' y2='{py(ymin) + 5:.1f}' stroke='#616e7c'/>"
                         f"<text x='{px(t):.1f}' y='{py(ymin) + 19:.1f}' text-anchor='middle' fill='#616e7c'>{t:g}</text>")
        for t in yticks:
            parts.append(f"<line x1='{left}' y1='{py(t):.1f}' x2='{px(xmax):.1f}' y2='{py(t):.1f}' stroke='#e4e7eb'/>"
                         f"<text x='{left - 6}' y='{py(t) + 4:.1f}' text-anchor='end' fill='#616e7c'>{t:g}</text>")
        parts.append(f"<line x1='{left}' y1='{py(0):.1f}' x2='{px(xmax):.1f}' y2='{py(0):.1f}' stroke='#616e7c'/>")
        parts.append(f"<text x='{px((xmin + xmax) / 2):.1f}' y='{hgt + 12}' text-anchor='middle' fill='#616e7c'>distance between the nuclei (Å)</text>")
        parts.append(f"<text transform='translate({x0 + 13},{py((ymin + ymax) / 2):.1f}) rotate(-90)' text-anchor='middle' fill='#616e7c'>energy (kelvin)</text>")

        def path(pts, color, dash=""):
            pts = [(x, y) for x, y in pts if ymin <= y <= ymax]
            d = " ".join(f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(pts))
            extra = f" stroke-dasharray='{dash}'" if dash else ""
            parts.append(f"<path d='{d}' fill='none' stroke='{color}' stroke-width='2.2'{extra}/>")

        def text(x, y, label, color):
            parts.append(f"<text x='{px(x):.1f}' y='{py(y):.1f}' fill='{color}'>{label}</text>")
        return px, py, path, text

    if PAIR is None:
        return "\n".join(parts + ["</svg>"])
    rs = [xmin / BOHR_A + i * (xmax - xmin) / BOHR_A / 400 for i in range(401)]
    xs = [R * BOHR_A for R in rs]
    kelvin = lambda values: [float(v) * HARTREE_K for v in values]

    px, py, path, text = panel(0, "He–He: the whole curve", -15, 30, [-10, 0, 10, 20, 30])
    path(zip(xs, kelvin(PAIR.repulsion(rs))), "#dc2626", "6 4")
    path(zip(xs, kelvin(PAIR(rs, terms=(6,)))), "#059669")
    path(zip(xs, kelvin(PAIR.best(rs) if HAVE_BEST else PAIR(rs))), "#2563eb")
    for key, val in DISP["potential_BO_K"].items():
        if xmin <= float(key) * BOHR_A <= xmax and -15 <= val <= 30:
            parts.append(f"<circle cx='{px(float(key) * BOHR_A):.1f}' cy='{py(val):.1f}' r='3.5' fill='#1f2933'/>")
    text(3.25, 24, "Hartree–Fock: only repulsion", "#dc2626")
    text(3.75, 12, "+ C₆ term: half the well", "#059669")
    text(3.75, -11.5, "best curve: full CI + direct attraction" if HAVE_BEST else "+ C₈ and C₁₀: the model", "#2563eb")
    text(4.9, 4.5, "● benchmark values", "#1f2933")

    px, py, path, text = panel(450, "The pieces: a small difference of large terms", -30, 45,
                               [-30, -15, 0, 15, 30, 45])
    path(zip(xs, kelvin(PAIR.repulsion(rs))), "#dc2626", "6 4")
    path(zip(xs, kelvin(PAIR(rs, terms=(6,)) - PAIR.repulsion(rs))), "#059669")
    path(zip(xs, kelvin(PAIR(rs, terms=(8, 10)) - PAIR.repulsion(rs))), "#d97706")
    path(zip(xs, kelvin(PAIR(rs))), "#2563eb")
    text(3.05, 36, "repulsion of the overlapping clouds", "#dc2626")
    text(3.9, -19, "dipole attraction, −C₆/r⁶", "#059669")
    text(3.9, -26, "C₈ and C₁₀ terms", "#d97706")
    text(4.4, 8, "sum: the well, 11 K deep", "#2563eb")
    parts.append("</svg>")
    return "\n".join(parts)


# ---------------------------------------------------------------- reporting
def verdict(r):
    inside = abs(r["err"]) <= r["tol"]
    if r["expect"] == "ok":
        return ("matches", True) if inside else ("UNEXPECTED MISS", False)
    return ("fails, as expected", True) if not inside else ("UNEXPECTED MATCH", False)


def fmt(x):
    return f"{x:.7g}" if abs(x) < 1e5 else f"{x:.1f}"


def pct(e):
    a = abs(e) * 100
    if a < 0.001:
        return f"{e * 100:+.5f} %"
    return f"{e * 100:+.4f} %" if a < 0.1 else f"{e * 100:+.2f} %" if a < 10 else f"{e * 100:+.0f} %"


def times(r):
    """Error as a multiple of the reference uncertainty."""
    x = abs(r["err"]) / r["unc"]
    return f"{x:.1f}×" if x < 10 else f"{x:,.0f}×".replace(",", " ")


def groups():
    seen = []
    for r in ROWS:
        if r["group"] not in seen:
            seen.append(r["group"])
    return seen


def chains():
    """Follow each simple model through its refinements."""
    children = {}
    for i, r in enumerate(ROWS):
        if r["base"] is not None:
            children.setdefault(r["base"], []).append(i)
    out = []
    for i, r in enumerate(ROWS):
        if r["base"] is None and i in children:
            chain, cur = [i], i
            while cur in children:
                cur = children[cur][0]
                chain.append(cur)
            out.append(chain)
    return out


def reading(r):
    x = abs(r["err"]) / r["unc"]
    if x <= 3:
        return "within the reference uncertainty"
    if x <= 30:
        return "close to the reference uncertainty"
    return "model error, far above the reference uncertainty"


INTRO = ("Each formula's derivation is proved in Lean where a theorem is named; the comparison with "
         "reference data is what tests the model. Rows marked *limit* are cases where the model is "
         "known to be outside its domain, kept to show where it breaks. The last column gives the "
         "error as a multiple of the reference value's own uncertainty.")
CAVEATS = [
    "Ionization energies of one-electron ions in the NIST database are high-precision QED "
    "calculations, not direct measurements; experiment confirms them where it can reach.",
    "NIST-JANAF heat capacities are calculated from measured spectroscopic constants, not measured "
    "in a calorimeter. Agreement with the vibration-corrected rows therefore partly compares two "
    "versions of the same theory.",
    "Helium's diamagnetic susceptibility is the one case here where the measurement is the likely "
    "weak link: the only gas-phase value (1960, ±3.5 %) lies 7 % from both this code and the best "
    "published calculation, which agree with each other to better than 0.001 %. The cause is not known.",
    "The helium–helium curve comes in three versions. Hartree–Fock repulsion plus a damped series to "
    "C₁₀: 2 % too deep, partly by cancellation. Full-CI repulsion plus the damped series to C₁₆: 3 % "
    "too shallow. Full-CI repulsion (with d functions) plus directly computed dispersion: 1.4 % too "
    "deep, with the repulsion about 0.3 K short at the minimum. The full CI and the direct dispersion "
    "are precomputed on a larger machine. The reference curve is itself a calculation; only the virial "
    "coefficient and the binding energy of He₂ are measurements. Component references at the minimum "
    "(dispersion 22.395 K) are quoted from a secondary source.",
    "NaCl from first principles is unrestricted Hartree–Fock in a basis about 26 mEh above the "
    "large-basis energy, most of it in the inner shells. The chlorine electron affinity with "
    "correlation freezes the inner ten electrons; the 44-orbital value comes from a selected list of determinants and is converged in neither the orbitals nor the list.",
    "The NaCl ionic model takes the bond length and vibration frequency from measurement and uses "
    "free-ion polarizabilities, which are not well defined for Cl⁻ inside a molecule.",
    "The uncertainty column uses one representative relative uncertainty per class of data "
    "(see relative_uncertainty in reference_data.json), not a separate figure for every value.",
    "The numerical methods are non-relativistic with fixed nuclei, use small sets of s, p, d and f orbitals, and are "
    "not formalised in Lean. "
    "Lean covers only the principle that makes their energies upper bounds.",
]


def best_rows():
    """One row per quantity: the best model here. Returns (row, matched, number of models tried)."""
    order, by_key = [], {}
    for r in ROWS:
        key = (r["system"], r["quantity"])
        if key not in by_key:
            order.append(key)
        by_key.setdefault(key, []).append(r)
    out = []
    for key in order:
        rows = by_key[key]
        good = [r for r in rows if r["expect"] == "ok"]
        pick = min(good or rows, key=lambda r: abs(r["err"]))
        out.append((pick, bool(good), len(rows)))
    return out


BEST_INTRO = ("One line per quantity, showing only the best model in this analyzer. The sections "
              "below list every model tried, including the simple ones that fail.")


def markdown():
    out = ["# Analyzer report", "",
           "Generated by `python3 analyzer/analyze.py --write`. Do not edit by hand.", "", INTRO, "",
           "## Best model for each quantity", "", BEST_INTRO, "",
           "| System | Quantity | Best model | Predicted | Reference | Error | Models tried |",
           "|---|---|---|---|---|---|---|"]
    for r, matched, count in best_rows():
        flag = "" if matched else " (**no model here matches**)"
        out.append(f"| {r['system']} | {r['quantity']} | {r['model']}{flag} | {fmt(r['predicted'])} {r['unit']} | "
                   f"{fmt(r['measured'])} {r['unit']} | {pct(r['err'])} | {count} |")
    out.append("")
    for g in groups():
        out += [f"## {g}", "",
                "| System | Quantity | Model | Lean theorem | Predicted | Reference | Error | Verdict | Error ÷ uncertainty |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in (x for x in ROWS if x["group"] == g):
            note = f" ({r['note']})" if r["note"] else ""
            out.append(f"| {r['system']} | {r['quantity']} | {r['model']} | `{r['lean']}` | "
                       f"{fmt(r['predicted'])} {r['unit']} | {fmt(r['measured'])} {r['unit']} | "
                       f"{pct(r['err'])} | {verdict(r)[0]}{note} | {times(r)} |")
        out.append("")
    out += ["## Where do the deviations come from?", "",
            "Each line follows one quantity from the simplest model to the best one here.", "",
            "| System | Quantity | Step | Error | What the step adds |", "|---|---|---|---|---|"]
    for chain in chains():
        for n, i in enumerate(chain):
            r = ROWS[i]
            out.append(f"| {r['system'] if n == 0 else ''} | {r['quantity'] if n == 0 else ''} | "
                       f"{r['model']} | {pct(r['err'])} | {r['fix'] or 'starting point'} |")
        last = ROWS[chain[-1]]
        out.append(f"| | | **remaining** | {times(last)} the reference uncertainty | {reading(last)} |")
    out += ["", "## Caveats", ""] + [f"- {c}" for c in CAVEATS] + [""]
    ok = sum(1 for r in ROWS if r["expect"] == "ok")
    out += [f"**{len(ROWS)} comparisons: {ok} inside the model's domain and matching, "
            f"{len(ROWS) - ok} documented limits.**", ""]
    return "\n".join(out)


def html():
    css = """:root{--bg:#fbfbfa;--card:#fff;--ink:#1f2933;--muted:#616e7c;--line:#e4e7eb;--a:#2563eb;--b:#dc2626;--c:#059669;--code:#f3f4f6}
@media (prefers-color-scheme: dark){:root{--bg:#0f1115;--card:#171a21;--ink:#e5e7eb;--muted:#9aa5b1;--line:#2a2f3a;--a:#60a5fa;--b:#f87171;--c:#34d399;--code:#1f232c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:32px 16px 80px}h1{font-size:clamp(26px,5vw,36px);margin:8px 0}h2{margin:44px 0 8px;font-size:22px}
.lede{color:var(--muted);font-size:18px;max-width:820px}.wrap{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:12px}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:900px}th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600}td.n{font-variant-numeric:tabular-nums;white-space:nowrap}code{background:var(--code);padding:1px 5px;border-radius:4px;font-size:12px}
.ok{color:var(--c);font-weight:600}.lim{color:var(--b);font-weight:600}.note{color:var(--muted);font-size:13px}a{color:var(--a)}
tr.sum td{background:var(--code);font-weight:600}li{margin:6px 0;max-width:820px}"""
    out = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
           "<meta name='viewport' content='width=device-width, initial-scale=1'>",
           "<title>Atom and Molecule Analyzer</title>", f"<style>{css}</style></head><body><main>",
           "<p><a href='./'>← back to the explainer</a></p>",
           "<h1>Atom and molecule analyzer</h1>", f"<p class='lede'>{INTRO.replace('*limit*', '<em>limit</em>')}</p>"]
    out.append(f"<h2>Best model for each quantity</h2><p class='lede'>{BEST_INTRO}</p><div class='wrap'><table>"
               "<tr><th>System</th><th>Quantity</th><th>Best model</th><th>Predicted</th><th>Reference</th>"
               "<th>Error</th><th>Models tried</th></tr>")
    shown, by_first = None, {}
    for r in ROWS:
        by_first.setdefault((r["system"], r["quantity"]), r["group"])
    for r, matched, count in best_rows():
        if by_first[(r["system"], r["quantity"])] != shown:
            shown = by_first[(r["system"], r["quantity"])]
            out.append(f"<tr class='sum'><td colspan='7'>{shown}</td></tr>")
        flag = "" if matched else "<div class='lim'>no model here matches</div>"
        out.append(f"<tr><td>{r['system']}</td><td>{r['quantity']}</td><td>{r['model']}{flag}</td>"
                   f"<td class='n'>{fmt(r['predicted'])} {r['unit']}</td><td class='n'>{fmt(r['measured'])} {r['unit']}</td>"
                   f"<td class='n'>{pct(r['err'])}</td><td class='n'>{count}</td></tr>")
    out.append("</table></div><h2>Every model tried</h2><p class='lede'>The full list, section by section.</p>")
    for g in groups():
        out.append(f"<h2>{g}</h2><div class='wrap'><table><tr><th>System</th><th>Quantity</th>"
                   "<th>Model</th><th>Lean theorem</th><th>Predicted</th><th>Reference</th>"
                   "<th>Error</th><th>Verdict</th><th>Error ÷ uncertainty</th></tr>")
        for r in (x for x in ROWS if x["group"] == g):
            cls = "ok" if r["expect"] == "ok" else "lim"
            note = f"<div class='note'>{r['note']}</div>" if r["note"] else ""
            out.append(f"<tr><td>{r['system']}</td><td>{r['quantity']}</td><td>{r['model']}</td>"
                       f"<td><code>{r['lean']}</code></td><td class='n'>{fmt(r['predicted'])} {r['unit']}</td>"
                       f"<td class='n'>{fmt(r['measured'])} {r['unit']}</td><td class='n'>{pct(r['err'])}</td>"
                       f"<td class='{cls}'>{verdict(r)[0]}{note}</td><td class='n'>{times(r)}</td></tr>")
        out.append("</table></div>")
    out.append("<h2>Where do the deviations come from?</h2><p class='lede'>Each block follows one "
               "quantity from the simplest model to the best one here.</p><div class='wrap'><table>"
               "<tr><th>System</th><th>Quantity</th><th>Step</th><th>Error</th><th>What the step adds</th></tr>")
    for chain in chains():
        for n, i in enumerate(chain):
            r = ROWS[i]
            out.append(f"<tr><td>{r['system'] if n == 0 else ''}</td><td>{r['quantity'] if n == 0 else ''}</td>"
                       f"<td>{r['model']}</td><td class='n'>{pct(r['err'])}</td>"
                       f"<td>{r['fix'] or 'starting point'}</td></tr>")
        last = ROWS[chain[-1]]
        out.append(f"<tr class='sum'><td></td><td></td><td>remaining</td><td class='n'>{times(last)} "
                   f"the reference uncertainty</td><td>{reading(last)}</td></tr>")
    out.append("</table></div><h2>Two kinds of bond</h2><p class='lede'>Pulling H₂ apart costs energy only "
               "over a short distance, then the curve levels off: the shared electrons go back to one atom each. "
               "NaCl follows the −e²/r attraction of two ions out to about 9.4 Å, where it becomes cheaper for the "
               "electron to jump back and leave two neutral atoms.</p>"
               "<p class='lede'>The bond exists only for opposite spins. With equal spins the two electrons can never be at "
               "the same point, so they cannot gather between the nuclei, and the atoms repel at every distance.</p>"
               "<div class='wrap' style='padding:8px;background:#fff'><img src='img/bond.svg' alt='Energy curves "
               "of H2 and NaCl' style='width:100%;min-width:640px'></div>"
               "<h2>Two helium atoms</h2><p class='lede'>Hartree–Fock gives only repulsion. The shallow well appears when the "
               "attraction between the fluctuating electron clouds is added.</p>"
               "<div class='wrap' style='padding:8px;background:#fff'><img src='img/helium.svg' alt='The helium-helium "
               "curve' style='width:100%;min-width:640px'></div>")
    out.append("</table></div><h2>Caveats</h2><ul>" + "".join(f"<li>{c}</li>" for c in CAVEATS) + "</ul>")
    out.append("<p class='note'>Generated by <code>analyzer/analyze.py</code>. Reference values: NIST "
               "Atomic Spectra Database, NIST Chemistry WebBook, NIST-JANAF tables, CODATA.</p></main></body></html>")
    return "\n".join(out)


def main():
    bad = [r for r in ROWS if not verdict(r)[1]]
    broken = [d for d, upper, lower in VARIATIONAL if upper < lower - 1e-9]
    broken += [d for d, holds in STABILITY if not holds]
    if quantum is None:
        print("numpy not available: numerical quantum mechanics rows skipped")
    if "--write" in sys.argv:
        (ROOT / "analyzer" / "REPORT.md").write_text(markdown())
        (ROOT / "docs" / "analyzer.html").write_text(html())
        for folder in ("img", "docs/img"):
            (ROOT / folder / "bond.svg").write_text(bond_figure())
            (ROOT / folder / "helium.svg").write_text(helium_figure())
    if "--check" not in sys.argv or bad:
        for r in ROWS:
            print(f"{verdict(r)[0]:19} {r['system'][:18]:18} {r['quantity'][:26]:26} {r['model'][:40]:40} "
                  f"{fmt(r['predicted']):>11} vs {fmt(r['measured']):>11} {pct(r['err']):>11} {times(r):>12}")
    for d in broken:
        print("VARIATIONAL BOUND VIOLATED:", d)
    print(f"\n{len(ROWS)} comparisons, {len(bad)} outside their expected band, "
          f"{len(VARIATIONAL) + len(STABILITY) - len(broken)}/{len(VARIATIONAL) + len(STABILITY)} "
          f"variational and stability checks hold")
    if "--check" in sys.argv and quantum is None:
        return 1
    return 1 if bad or broken else 0


if __name__ == "__main__":
    sys.exit(main())

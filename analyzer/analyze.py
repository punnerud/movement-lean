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
    row(G_NUM, "Li", "first ionization energy", "full CI with s and p orbitals",
        "rayleigh_ritz_bound", (li_ion_ci - li_ci) * HARTREE_EV * recoil("7Li", 3), ie["Li"], "eV",
        "ok", 3e-3, u_n, base=li_hf_row,
        fix="the outer electron polarises the inner pair (needs p orbitals)",
        note="the remaining 0.1 % needs d orbitals and beyond")
    be_hf_row = row(G_NUM, "Be", "first ionization energy", "Hartree–Fock, E(Be⁺) − E(Be)",
                    "rayleigh_ritz_bound", (be_ion - be) * HARTREE_EV * recoil("9Be", 4), ie["Be"],
                    "eV", "limit", 0.05, u_n,
                    "average-field theory misses the strong 2s–2p correlation in beryllium")
    be_ci, be_ion_ci = quantum.atom_fci_sp(4, (2, 2), [(2, 2), (2, 1)])
    row(G_NUM, "Be", "first ionization energy", "full CI with s and p orbitals",
        "rayleigh_ritz_bound", (be_ion_ci - be_ci) * HARTREE_EV * recoil("9Be", 4), ie["Be"], "eV",
        "ok", 1e-2, u_n, base=be_hf_row,
        fix="the two outer electrons spend part of the time in 2p instead of 2s",
        note="the remaining 0.5 % needs d orbitals and beyond")

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
                   ("Li⁺: full CI ≥ exact", li_ion_ci, bench["LiPlus_exact"]),
                   ("Be: HF ≥ full CI (s, p)", be, be_ci),
                   ("Be: full CI ≥ exact", be_ci, bench["Be_exact"]),
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
    "The NaCl ionic model takes the bond length and vibration frequency from measurement and uses "
    "free-ion polarizabilities, which are not well defined for Cl⁻ inside a molecule.",
    "The uncertainty column uses one representative relative uncertainty per class of data "
    "(see relative_uncertainty in reference_data.json), not a separate figure for every value.",
    "The numerical methods are non-relativistic with fixed nuclei, use s and p orbitals only, and are "
    "not formalised in Lean. "
    "Lean covers only the principle that makes their energies upper bounds.",
]


def markdown():
    out = ["# Analyzer report", "",
           "Generated by `python3 analyzer/analyze.py --write`. Do not edit by hand.", "", INTRO, ""]
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
               "of H2 and NaCl' style='width:100%;min-width:640px'></div>")
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

"""Numerical quantum mechanics for few-electron atoms and H2.

Everything here is non-relativistic, with fixed (infinitely heavy) nuclei, in
Hartree atomic units. Three methods, in increasing accuracy:

* Hartree-Fock (UHF) in s-type Gaussians: each electron feels the average field
  of the others.
* Full configuration interaction (FCI) for two electrons: exact within the
  chosen one-electron basis.
* Hylleraas expansion for helium: the electron-electron distance r12 appears
  explicitly in the wavefunction.

All three are variational: by the Rayleigh-Ritz principle (Lean:
`rayleigh_ritz_bound`) every energy computed here is an upper bound on the
exact energy of the model Hamiltonian.

Only numpy is required.
"""
import math
from fractions import Fraction
from itertools import product  # noqa: F401

import numpy as np

_erf = np.frompyfunc(math.erf, 1, 1)


def boys0(t):
    """F0(t) = (1/2) sqrt(pi/t) erf(sqrt t), with the t -> 0 limit."""
    t = np.asarray(t, dtype=float)
    safe = np.where(t < 1e-12, 1.0, t)
    val = 0.5 * np.sqrt(np.pi / safe) * _erf(np.sqrt(safe)).astype(float)
    return np.where(t < 1e-12, 1.0 - t / 3.0, val)


# --------------------------------------------------------------------------
# Integrals over s-type Gaussians exp(-a |r - A|^2)
# --------------------------------------------------------------------------
def integrals(exponents, centers, nuclei):
    """Overlap S, core Hamiltonian h, and two-electron integrals (ij|kl).

    exponents: (n,), centers: (n, 3), nuclei: list of (charge, position)."""
    a = np.asarray(exponents, float)
    A = np.asarray(centers, float).reshape(len(a), 3)
    p = a[:, None] + a[None, :]
    red = a[:, None] * a[None, :] / p
    d2 = ((A[:, None, :] - A[None, :, :]) ** 2).sum(-1)
    K = np.exp(-red * d2)
    P = (a[:, None, None] * A[:, None, :] + a[None, :, None] * A[None, :, :]) / p[:, :, None]
    S = (np.pi / p) ** 1.5 * K
    T = red * (3.0 - 2.0 * red * d2) * S
    V = np.zeros_like(S)
    for charge, pos in nuclei:
        r2 = ((P - np.asarray(pos, float)) ** 2).sum(-1)
        V -= charge * 2.0 * np.pi / p * K * boys0(p * r2)
    pq = p[:, :, None, None] + p[None, None, :, :]
    rho = p[:, :, None, None] * p[None, None, :, :] / pq
    R2 = ((P[:, :, None, None, :] - P[None, None, :, :, :]) ** 2).sum(-1)
    eri = (2.0 * np.pi ** 2.5 / (p[:, :, None, None] * p[None, None, :, :] * np.sqrt(pq))
           * K[:, :, None, None] * K[None, None, :, :] * boys0(rho * R2))
    return S, T + V, eri


def orthogonalizer(S, threshold=1e-9):
    """Canonical orthogonalization; drops near-linear-dependent combinations."""
    d = np.sqrt(np.diag(S))
    w, U = np.linalg.eigh(S / np.outer(d, d))
    keep = w > threshold
    return (U[:, keep] / np.sqrt(w[keep])) / d[:, None]


def uhf(S, h, eri, n_alpha, n_beta, e_nuc=0.0, iterations=400, tol=1e-11):
    """Unrestricted Hartree-Fock. Returns the total energy."""
    X = orthogonalizer(S)

    def density(F, n):
        _, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C[:, :n]
        return C @ C.T

    Da, Db = density(h, n_alpha), density(h, n_beta)
    energy = 0.0
    for it in range(iterations):
        J = np.einsum("ijkl,kl->ij", eri, Da + Db)
        Fa = h + J - np.einsum("ikjl,kl->ij", eri, Da)
        Fb = h + J - np.einsum("ikjl,kl->ij", eri, Db)
        new = 0.5 * np.sum((Da + Db) * h + Da * Fa + Db * Fb) + e_nuc
        Na, Nb = density(Fa, n_alpha), density(Fb, n_beta)
        mix = 0.5 if it < 30 else 1.0
        Da, Db = mix * Na + (1 - mix) * Da, mix * Nb + (1 - mix) * Db
        if abs(new - energy) < tol and mix == 1.0:
            return new
        energy = new
    return energy


def fci_two_electrons(S, h, eri, e_nuc=0.0):
    """Exact singlet ground state of two electrons in the given basis."""
    X = orthogonalizer(S, 1e-8)
    m = X.shape[1]
    ho = X.T @ h @ X
    g = np.einsum("pi,pqrs->iqrs", X, eri)
    g = np.einsum("qj,iqrs->ijrs", X, g)
    g = np.einsum("rk,ijrs->ijks", X, g)
    g = np.einsum("sl,ijks->ijkl", X, g)
    eye = np.eye(m)
    H = (np.einsum("ik,jl->ijkl", ho, eye) + np.einsum("ik,jl->ijkl", eye, ho)
         + g.transpose(0, 2, 1, 3))
    # restrict to spatially symmetric (singlet) pair functions
    pairs = [(i, j) for i in range(m) for j in range(i, m)]
    B = np.zeros((m * m, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        if i == j:
            B[i * m + j, col] = 1.0
        else:
            B[i * m + j, col] = B[j * m + i, col] = math.sqrt(0.5)
    Hs = B.T @ H.reshape(m * m, m * m) @ B
    return float(np.linalg.eigvalsh(Hs)[0]) + e_nuc


def even_tempered(low, high, count):
    ratio = (high / low) ** (1.0 / (count - 1))
    return [low * ratio ** k for k in range(count)]


# --------------------------------------------------------------------------
# Atoms (all occupied orbitals are s-type: He, Li, Be and their ions)
# --------------------------------------------------------------------------
ATOM_BASIS = {1: (0.002, 3000.0, 22), 2: (0.05, 6000.0, 16), 3: (0.012, 20000.0, 20), 4: (0.02, 40000.0, 20)}


def atom_integrals(Z):
    exps = even_tempered(*ATOM_BASIS[Z])
    return integrals(exps, np.zeros((len(exps), 3)), [(Z, (0, 0, 0))])


def atom_hf(Z, n_alpha, n_beta):
    return uhf(*atom_integrals(Z), n_alpha, n_beta)


def two_electron_fci_s(Z=2):
    return fci_two_electrons(*atom_integrals(Z))


def helium_fci_s():
    return two_electron_fci_s(2)


# --------------------------------------------------------------------------
# Helium: Hylleraas expansion  psi = exp(-a s) * sum c s^l t^(2m) u^n
#   s = r1 + r2, t = r1 - r2, u = r12
# --------------------------------------------------------------------------
def _mul(p, q):
    out = {}
    for (a1, b1, c1), x in p.items():
        for (a2, b2, c2), y in q.items():
            k = (a1 + a2, b1 + b2, c1 + c2)
            out[k] = out.get(k, 0) + x * y
    return out


def _add(*polys):
    out = {}
    for p in polys:
        for k, x in p.items():
            out[k] = out.get(k, 0) + x
    return out


def _scale(p, x):
    return {k: v * x for k, v in p.items()}


def _integral(poly, alpha):
    """Integral of poly * exp(-2 alpha s) over 0 <= t <= u <= s < infinity."""
    total = Fraction(0)
    for (a, b, c), x in poly.items():
        if x == 0:
            continue
        n = a + b + c + 3
        total += x * Fraction(math.factorial(n - 1), (b + 1) * (b + c + 2)) / (2 * alpha) ** n
    return total


def hylleraas_helium(Z=2, order=5, alpha=Fraction(9, 5)):
    """Ground-state energy of a two-electron atom with nuclear charge Z."""
    terms = [(l, m, n) for l, m, n in product(range(order + 1), repeat=3) if l + 2 * m + n <= order]
    s2_t2 = {(2, 0, 0): 1, (0, 2, 0): -1}                       # s^2 - t^2
    weight = _mul(s2_t2, {(0, 0, 1): 1})                        # u (s^2 - t^2)
    potential = _add({(1, 0, 1): -4 * Z}, s2_t2)                # -4 Z s u + s^2 - t^2
    a_term = _mul({(1, 0, 0): 1}, {(0, 0, 2): 1, (0, 2, 0): -1})  # s (u^2 - t^2)
    b_term = _mul({(0, 1, 0): 1}, {(2, 0, 0): 1, (0, 0, 2): -1})  # t (s^2 - u^2)

    def parts(l, m, n):
        f = {(l, 2 * m, n): Fraction(1)}
        fs = {(l, 2 * m, n): -alpha}
        if l:
            fs[(l - 1, 2 * m, n)] = Fraction(l)
        ft = {(l, 2 * m - 1, n): Fraction(2 * m)} if m else {}
        fu = {(l, 2 * m, n - 1): Fraction(n)} if n else {}
        return f, fs, ft, fu

    P = [parts(*t) for t in terms]
    size = len(terms)
    N = np.zeros((size, size))
    Hm = np.zeros((size, size))
    for i in range(size):
        f1, s1, t1, u1 = P[i]
        for j in range(i, size):
            f2, s2, t2, u2 = P[j]
            norm = _integral(_mul(weight, _mul(f1, f2)), alpha)
            grad = _add(_mul(s1, s2), _mul(t1, t2), _mul(u1, u2))
            cross = _add(_mul(u1, _add(_mul(a_term, s2), _mul(b_term, t2))),
                         _mul(u2, _add(_mul(a_term, s1), _mul(b_term, t1))))
            kinetic = _integral(_add(_mul(weight, grad), cross), alpha)
            pot = _integral(_mul(potential, _mul(f1, f2)), alpha)
            N[i, j] = N[j, i] = float(norm)
            Hm[i, j] = Hm[j, i] = float(kinetic + pot)
    X = orthogonalizer(N, 1e-13)
    return float(np.linalg.eigvalsh(X.T @ Hm @ X)[0])


# --------------------------------------------------------------------------
# H2 molecule: FCI in s-Gaussians, with off-axis "lobe" functions standing in
# for p-type polarization.
# --------------------------------------------------------------------------
H_S = [0.03, 0.09, 0.27, 0.8, 2.4, 7.5, 28.0, 150.0]
BOND_S = [0.35, 1.0, 2.5]
LOBE = [(0.45, 0.35), (1.4, 0.2)]          # (exponent, sideways offset in bohr)


def h2_basis(R):
    exps, cen = [], []
    for z in (-R / 2, R / 2):
        for a in H_S:
            exps.append(a); cen.append((0, 0, z))
        for a, d in LOBE:
            for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d)):
                exps.append(a); cen.append((dx, dy, z))
    for a in BOND_S:
        exps.append(a); cen.append((0, 0, 0))
    for a, d in LOBE[:1]:
        for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d)):
            exps.append(a); cen.append((dx, dy, 0))
    return exps, np.array(cen, float)


def h2_energy(R):
    exps, cen = h2_basis(R)
    nuclei = [(1, (0, 0, -R / 2)), (1, (0, 0, R / 2))]
    return fci_two_electrons(*integrals(exps, cen, nuclei), e_nuc=1.0 / R)


def hydrogen_atom_energy():
    """Energy of one H atom in the same atomic basis (exact value: -0.5)."""
    exps = list(H_S)
    cen = [(0, 0, 0)] * len(H_S)
    for a, d in LOBE:
        for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d)):
            exps.append(a); cen.append((dx, dy, 0))
    S, h, _ = integrals(exps, np.array(cen, float), [(1, (0, 0, 0))])
    X = orthogonalizer(S, 1e-8)
    return float(np.linalg.eigvalsh(X.T @ h @ X)[0])


def h2_curve(points=None):
    """Potential curve, its minimum, and the curvature there.

    Returns (R_e in bohr, E_min in hartree, k in hartree/bohr^2)."""
    points = points or [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    energies = [h2_energy(R) for R in points]
    coeff = np.polyfit(points, energies, 4)
    d1, d2 = np.polyder(coeff), np.polyder(coeff, 2)
    R = points[int(np.argmin(energies))]
    for _ in range(50):
        R -= np.polyval(d1, R) / np.polyval(d2, R)
    return float(R), float(np.polyval(coeff, R)), float(np.polyval(d2, R))


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("He  HF          ", atom_hf(2, 1, 1), " (HF limit -2.861680)")
    print("He  FCI, s only ", helium_fci_s(), " (s limit  -2.879029)")
    print("He  Hylleraas   ", hylleraas_helium(), " (exact    -2.903724)")
    print("Li  UHF         ", atom_hf(3, 2, 1), " (HF limit -7.432727)")
    print("Li+ HF          ", atom_hf(3, 1, 1), " (HF limit -7.236415)")
    print("Be  HF          ", atom_hf(4, 2, 2), " (HF limit -14.573023)")
    print("Be+ UHF         ", atom_hf(4, 2, 1), " (HF limit -14.277395)")
    print("H atom          ", hydrogen_atom_energy())
    print("H2 FCI R=1.4    ", h2_energy(1.4), " (exact -1.174476)")
    print("H2 curve        ", h2_curve())
    print(f"{time.time() - t0:.1f} s")

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

Only numpy is required; scipy, if installed, speeds up the integrals.
"""
import math
from fractions import Fraction
from itertools import product  # noqa: F401

import numpy as np

try:                                    # optional: much faster for large basis sets
    from scipy.special import erf as _fast_erf
except ImportError:
    _fast_erf = None
_erf = np.frompyfunc(math.erf, 1, 1)


def boys0(t):
    """F0(t) = (1/2) sqrt(pi/t) erf(sqrt t), with the t -> 0 limit."""
    t = np.asarray(t, dtype=float)
    safe = np.where(t < 1e-12, 1.0, t)
    root = np.sqrt(safe)
    val = 0.5 * np.sqrt(np.pi / safe) * (_fast_erf(root) if _fast_erf else _erf(root).astype(float))
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


def fci_two_electrons(S, h, eri, e_nuc=0.0, triplet=False):
    """Exact lowest state of two electrons in the given basis.

    Singlet (opposite spins): the spatial pair function is symmetric.
    Triplet (equal spins): it is antisymmetric, so the two electrons are never
    at the same point."""
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
    pairs = [(i, j) for i in range(m) for j in range(i + (1 if triplet else 0), m)]
    B = np.zeros((m * m, len(pairs)))
    for col, (i, j) in enumerate(pairs):
        if i == j:
            B[i * m + j, col] = 1.0
        else:
            B[i * m + j, col] = math.sqrt(0.5)
            B[j * m + i, col] = -math.sqrt(0.5) if triplet else math.sqrt(0.5)
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


def hylleraas_helium(Z=2, order=5, alpha=Fraction(9, 5), expect=False):
    """Ground-state energy of a two-electron atom with nuclear charge Z.

    With expect=True, returns (energy, <1/r1 + 1/r2>, <r1^2 + r2^2>) in atomic
    units, evaluated with the same wavefunction."""
    terms = [(l, m, n) for l, m, n in product(range(order + 1), repeat=3) if l + 2 * m + n <= order]
    s2_t2 = {(2, 0, 0): 1, (0, 2, 0): -1}                       # s^2 - t^2
    weight = _mul(s2_t2, {(0, 0, 1): 1})                        # u (s^2 - t^2)
    potential = _add({(1, 0, 1): -4 * Z}, s2_t2)                # -4 Z s u + s^2 - t^2
    a_term = _mul({(1, 0, 0): 1}, {(0, 0, 2): 1, (0, 2, 0): -1})  # s (u^2 - t^2)
    b_term = _mul({(0, 1, 0): 1}, {(2, 0, 0): 1, (0, 0, 2): -1})  # t (s^2 - u^2)
    inv_r = {(1, 0, 1): 4}                                      # (1/r1 + 1/r2) * weight
    r_sq = _mul({(2, 0, 0): Fraction(1, 2), (0, 2, 0): Fraction(1, 2)}, weight)   # (r1^2 + r2^2) * weight

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
    A1 = np.zeros((size, size))
    A2 = np.zeros((size, size))
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
            if expect:
                ff = _mul(f1, f2)
                A1[i, j] = A1[j, i] = float(_integral(_mul(inv_r, ff), alpha))
                A2[i, j] = A2[j, i] = float(_integral(_mul(r_sq, ff), alpha))
    X = orthogonalizer(N, 1e-13)
    if not expect:
        return float(np.linalg.eigvalsh(X.T @ Hm @ X)[0])
    w, V = np.linalg.eigh(X.T @ Hm @ X)
    c = X @ V[:, 0]
    norm = c @ N @ c
    return float(w[0]), float(c @ A1 @ c / norm), float(c @ A2 @ c / norm)


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


def h2_point(R):
    """(full CI singlet, Hartree-Fock, full CI triplet) energies of H2 at distance R.

    With one alpha and one beta electron started from the same density, the
    Hartree-Fock solution keeps both electrons in one shared orbital."""
    exps, cen = h2_basis(R)
    nuclei = [(1, (0, 0, -R / 2)), (1, (0, 0, R / 2))]
    S, h, eri = integrals(exps, cen, nuclei)
    return (fci_two_electrons(S, h, eri, e_nuc=1.0 / R), uhf(S, h, eri, 1, 1, e_nuc=1.0 / R),
            fci_two_electrons(S, h, eri, e_nuc=1.0 / R, triplet=True))


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


# --------------------------------------------------------------------------
# p-type orbitals and full CI for any number of electrons
# --------------------------------------------------------------------------
def contracted_integrals(exponents, centers, nuclei, W, dipole=False, two_electron=True):
    """S, h and (pq|rs) for functions that are fixed combinations (columns of W)
    of s-Gaussians. The two-electron integrals are built in slices, so the full
    primitive array is never stored."""
    a = np.asarray(exponents, float)
    A = np.asarray(centers, float).reshape(len(a), 3)
    n, nc = len(a), W.shape[1]
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
    pf, Kf, Pf = p.ravel(), K.ravel(), P.reshape(n * n, 3)
    sq = (Pf ** 2).sum(-1)
    if not two_electron:
        return W.T @ S @ W, W.T @ (T + V) @ W, None
    half = np.empty((n * n, nc, nc))
    step = max(1, 4_000_000 // (n * n))
    for lo in range(0, n * n, step):
        hi = min(n * n, lo + step)
        pq = pf[lo:hi, None] + pf[None, :]
        R2 = np.maximum(sq[lo:hi, None] + sq[None, :] - 2.0 * Pf[lo:hi] @ Pf.T, 0.0)
        block = (2.0 * np.pi ** 2.5 / (pf[lo:hi, None] * pf[None, :] * np.sqrt(pq))
                 * Kf[lo:hi, None] * Kf[None, :] * boys0(pf[lo:hi, None] * pf[None, :] / pq * R2))
        block = block.reshape(hi - lo, n, n)
        half[lo:hi] = np.einsum("xkl,kc,ld->xcd", block, W, W, optimize=True)
    half = half.reshape(n, n, nc, nc)
    eri = np.einsum("ijcd,ia,jb->abcd", half, W, W, optimize=True)
    if dipole:
        # A harmonic polynomial averaged over a round Gaussian equals its value at
        # the centre, so the multipole operators r^l P_l(cos theta) need only P.
        z, r2 = P[:, :, 2], (P ** 2).sum(-1)
        Q = {1: z, 2: 1.5 * z ** 2 - 0.5 * r2, 3: 2.5 * z ** 3 - 1.5 * z * r2}[int(dipole)]
        return W.T @ S @ W, W.T @ (T + V) @ W, eri, W.T @ (Q * S) @ W
    return W.T @ S @ W, W.T @ (T + V) @ W, eri


_STENCIL = {0: {0: 1.0},                       # finite-difference derivatives on a line
            1: {1: 0.5, -1: -0.5},
            2: {1: 1.0, 0: -2.0, -1: 1.0},
            3: {2: 0.5, 1: -1.0, -1: 1.0, -2: -0.5}}
# the seven f shapes as harmonic polynomials in (d/dx, d/dy, d/dz): {(a, b, c): coefficient}
_F_SHAPES = [{(1, 1, 1): 1.0},
             {(2, 0, 1): 1.0, (0, 2, 1): -1.0},
             {(3, 0, 0): 1.0, (1, 2, 0): -3.0},
             {(2, 1, 0): 3.0, (0, 3, 0): -1.0},
             {(0, 0, 3): 2.0, (2, 0, 1): -3.0, (0, 2, 1): -3.0},
             {(1, 0, 2): 4.0, (3, 0, 0): -1.0, (1, 2, 0): -1.0},
             {(0, 1, 2): 4.0, (2, 1, 0): -1.0, (0, 3, 0): -1.0}]


def _difference_lobes(shape):
    """Grid points and weights of a finite-difference version of a derivative polynomial."""
    out = {}
    for (a, b, c), coef in shape.items():
        for i, wi in _STENCIL[a].items():
            for j, wj in _STENCIL[b].items():
                for k, wk in _STENCIL[c].items():
                    out[(i, j, k)] = out.get((i, j, k), 0.0) + coef * wi * wj * wk
    return {pt: w for pt, w in out.items() if abs(w) > 1e-14}


def atom_raw_basis(Z, p_exponents, d_exponents=(), f_exponents=(), delta_p=0.1, delta_d=0.3,
                   delta_f=0.4, dipole=False):
    """s, p and d functions for an atom, all made from s-Gaussians.

    p: g(r - d e) - g(r + d e) along x, y, z (odd along the axis).
    d: second differences. xy-type: four lobes in a plane with signs + - - +;
    x2-y2: lobes on the x axis minus lobes on the y axis; z2: twice the z
    lobes minus the x and y lobes. Each has the angular shape of a d orbital,
    with a small admixture of higher (g-type) functions.
    Returns (S, h, eri, number of s functions) in this raw, non-orthogonal basis."""
    s_exps = even_tempered(*ATOM_BASIS[Z])
    ns = len(s_exps)
    exps, cen, cols = list(s_exps), [(0.0, 0.0, 0.0)] * ns, []
    for k in range(ns):
        cols.append({k: 1.0})

    def lobe(a, pos):
        exps.append(a); cen.append(tuple(pos))
        return len(exps) - 1

    for a in p_exponents:
        d = delta_p / math.sqrt(a)
        for axis in range(3):
            plus, minus = [0.0] * 3, [0.0] * 3
            plus[axis], minus[axis] = d, -d
            cols.append({lobe(a, plus): 1.0, lobe(a, minus): -1.0})
    for a in d_exponents:
        d = delta_d / math.sqrt(a)
        ax = []
        for axis in range(3):
            plus, minus = [0.0] * 3, [0.0] * 3
            plus[axis], minus[axis] = d, -d
            ax.append((lobe(a, plus), lobe(a, minus)))
        for u, v in ((0, 1), (0, 2), (1, 2)):
            col = {}
            for su in (1.0, -1.0):
                for sv in (1.0, -1.0):
                    pos = [0.0] * 3
                    pos[u], pos[v] = su * d, sv * d
                    col[lobe(a, pos)] = su * sv
            cols.append(col)
        cols.append({ax[0][0]: 1.0, ax[0][1]: 1.0, ax[1][0]: -1.0, ax[1][1]: -1.0})
        cols.append({ax[2][0]: 2.0, ax[2][1]: 2.0, ax[0][0]: -1.0, ax[0][1]: -1.0,
                     ax[1][0]: -1.0, ax[1][1]: -1.0})
    for a in f_exponents:
        # f: third differences of the Gaussian on a small grid of lobes (step d).
        # Each has the angular shape of an f orbital plus a little p, which is harmless.
        d = delta_f / math.sqrt(a)
        grid = {}
        for shape in _F_SHAPES:
            col = {}
            for pt, w in _difference_lobes(shape).items():
                if pt not in grid:
                    grid[pt] = lobe(a, [d * x for x in pt])
                col[grid[pt]] = w
            cols.append(col)
    W = np.zeros((len(exps), len(cols)))
    for c, col in enumerate(cols):
        for k, v in col.items():
            W[k, c] = v
    if dipole:
        return contracted_integrals(exps, np.array(cen), [(Z, (0, 0, 0))], W, dipole=dipole)
    S, h, eri = contracted_integrals(exps, np.array(cen), [(Z, (0, 0, 0))], W)
    return S, h, eri, ns


HE_RESPONSE_P = (0.04, 0.09, 0.2, 0.45, 1.0, 2.3, 5.5)
# (p, d, f) exponents used for the response to a field of each multipole order
HE_RESPONSE = {1: (HE_RESPONSE_P, (0.2, 0.7, 2.5), ()),
               2: ((0.2, 1.0, 4.0), (0.07, 0.2, 0.55, 1.5), ()),
               3: ((0.2, 1.0, 4.0), (), (0.1, 0.28, 0.8))}


def helium_multipole_states(order, p_exponents, d_exponents=(), f_exponents=()):
    """Excited states of helium reached by a field of the given multipole order.

    Full CI for the two electrons gives the ground state and every excited
    state the basis can hold. Returns (excitation energies w_n, squared
    transition moments |<n| sum_i r_i^l P_l(cos theta_i) |0>|^2)."""
    S, h, eri, Q = atom_raw_basis(2, p_exponents, d_exponents, f_exponents, dipole=order)
    X = orthogonalizer(S, 1e-8)
    m = X.shape[1]
    ho, qo, g = X.T @ h @ X, X.T @ Q @ X, _transform(eri, X)
    eye = np.eye(m)
    I, J = np.triu_indices(m)                  # symmetric (singlet) pair functions
    norm = np.where(I == J, 0.5, math.sqrt(0.5))
    a, b, c, d = I[:, None], J[:, None], I[None, :], J[None, :]

    def pair_matrix(one, two=None):
        def element(i, j, k, l):
            out = one[i, k] * eye[j, l] + eye[i, k] * one[j, l]
            return out if two is None else out + two[i, k, j, l]
        return ((element(a, b, c, d) + element(b, a, c, d) + element(a, b, d, c)
                 + element(b, a, d, c)) * norm[:, None] * norm[None, :])

    w, V = np.linalg.eigh(pair_matrix(ho, g))
    moment = (V.T @ (pair_matrix(qo) @ V[:, 0]))[1:]
    omega = w[1:] - w[0]
    keep = np.abs(moment) > 1e-10
    return omega[keep], moment[keep] ** 2


def helium_dispersion():
    """How a helium atom responds to an electric field, and what that does to
    a second helium atom far away.

    With excitation energies w_n and squared transition moments a_n for a
    field of multipole order l (dipole 1, quadrupole 2, octupole 3):

      polarizability   alpha_l = 2 sum_n a_n / w_n
      pair sums        S(l, l') = sum_mn a_m a'_n / (w_m + w'_n)
      van der Waals    C6 = 6 S(1,1),  C8 = 30 S(1,2),  C10 = 56 S(1,3) + 70 S(2,2)

    These are the Casimir-Polder formulas for two identical atoms: the
    fluctuating multipoles of one atom induce multipoles in the other, and
    the energy at distance R is -C6/R^6 - C8/R^8 - C10/R^10.
    Returns a dict in atomic units; "f_sum" is the dipole oscillator-strength
    sum, which should equal 2, the number of electrons."""
    st = {l: helium_multipole_states(l, *HE_RESPONSE[l]) for l in (1, 2, 3)}

    def pair_sum(l1, l2):
        (w1, a1), (w2, a2) = st[l1], st[l2]
        return float(np.sum(np.outer(a1, a2) / (w1[:, None] + w2[None, :])))

    out = {f"alpha{l}": float(2.0 * np.sum(st[l][1] / st[l][0])) for l in (1, 2, 3)}
    out["f_sum"] = float(2.0 * np.sum(st[1][0] * st[1][1]))
    out["C6"] = 6.0 * pair_sum(1, 1)
    out["C8"] = 30.0 * pair_sum(1, 2)
    out["C10"] = 56.0 * pair_sum(1, 3) + 70.0 * pair_sum(2, 2)
    return out


HE_PAIR_P = (0.15, 0.5, 1.5, 4.5)
HE_PAIR_R = (3.0, 3.5, 4.0, 4.5, 5.0, 5.6, 6.2, 7.0)


def helium_pair_hf(R, p_exponents=HE_PAIR_P, delta=0.1):
    """Hartree-Fock energy of two helium atoms at distance R minus that of two
    separate atoms (hartree). Each atom is computed in the basis of the whole
    pair, so that borrowing the neighbour's basis functions does not show up
    as a false attraction (the counterpoise correction)."""
    s_exps = even_tempered(*ATOM_BASIS[2])
    exps, cen, cols = [], [], []
    for zc in (0.0, R):
        for a in s_exps:
            exps.append(a); cen.append((0.0, 0.0, zc)); cols.append({len(exps) - 1: 1.0})
        for a in p_exponents:
            d = delta / math.sqrt(a)
            for axis in range(3):
                plus, minus = [0.0, 0.0, zc], [0.0, 0.0, zc]
                plus[axis] += d; minus[axis] -= d
                exps += [a, a]; cen += [tuple(plus), tuple(minus)]
                cols.append({len(exps) - 2: 1.0, len(exps) - 1: -1.0})
    W = np.zeros((len(exps), len(cols)))
    for c, col in enumerate(cols):
        for k, v in col.items():
            W[k, c] = v
    cen = np.array(cen)
    S, h_pair, eri = contracted_integrals(exps, cen, [(2, (0, 0, 0)), (2, (0, 0, R))], W)
    _, h_one, _ = contracted_integrals(exps, cen, [(2, (0, 0, 0))], W, two_electron=False)
    return uhf(S, h_pair, eri, 2, 2, 4.0 / R) - 2.0 * uhf(S, h_one, eri, 1, 1)


class HeliumPair:
    """Model for the whole helium-helium curve: Hartree-Fock repulsion plus
    damped dispersion.

      V(R) = V_HF(R) - sum_n f_n(b R) C_n / R^n,      n = 6, 8, 10

    V_HF is computed at a few distances and interpolated as exp(quadratic in R).
    The damping f_n(x) = 1 - exp(-x) sum_{k<=n} x^k / k! (Tang and Toennies)
    switches the dispersion off where the clouds overlap; its range parameter
    b(R) = -d ln V_HF / dR is taken from the repulsion itself, so nothing is
    fitted to the reference curve."""

    def __init__(self, coefficients, distances=HE_PAIR_R):
        self.R = np.array(distances, float)
        self.hf = np.array([helium_pair_hf(R) for R in self.R])
        self.fit = np.polyfit(self.R, np.log(self.hf), 2)
        self.C = {6: coefficients["C6"], 8: coefficients["C8"], 10: coefficients["C10"]}
        self.full_fit = None

    def add_correlation(self, distances, extra):
        """Extra repulsion from a four-electron full CI (see he2_full_ci.py):
        full CI interaction minus Hartree-Fock minus the dispersion that the
        full CI basis holds, at the given distances (hartree). The corrected
        repulsion is again interpolated as exp(quadratic in R)."""
        distances = np.asarray(distances, float)
        total = self.repulsion(distances) + np.asarray(extra, float)
        self.full_fit = np.polyfit(distances, np.log(total), 2)

    def add_direct_dispersion(self, distances, values):
        """Dispersion computed directly with overlapping clouds (see
        he2_dispersion.py) at the given distances (hartree). Stored as its
        ratio to the damped series, which is close to 1 and smooth; beyond the
        last computed distance the ratio relaxes to 1."""
        self.direct_R = np.asarray(distances, float)
        series = self(self.direct_R, extrapolate=True) - self.repulsion(self.direct_R)
        ratio = np.asarray(values, float) / series
        self.direct_fit = np.polyfit(self.direct_R, ratio, 4)
        self.direct_residual = float(np.abs(np.polyval(self.direct_fit, self.direct_R) - ratio).max())

    def dispersion(self, R, direct=False, **kw):
        R = np.asarray(R, float)
        series = self(R, **kw) - self.repulsion(R)
        if not direct:
            return series
        lo, hi = self.direct_R[0], self.direct_R[-1]
        ratio = np.polyval(self.direct_fit, np.clip(R, lo, hi))
        ratio = np.where(R > hi, 1.0 + (ratio - 1.0) * np.exp(-(R - hi)), ratio)
        return ratio * series

    def best(self, R):
        """Repulsion with correlation plus directly computed dispersion."""
        return self.repulsion(R, correlated=True) + self.dispersion(R, direct=True, extrapolate=True)

    def repulsion(self, R, correlated=False):
        fit = self.full_fit if correlated else self.fit
        return np.exp(np.polyval(fit, np.asarray(R, float)))

    def __call__(self, R, terms=(6, 8, 10), extrapolate=False, correlated=False):
        R = np.asarray(R, float)
        C = {n: self.C[n] for n in terms}
        if extrapolate:                      # Tang-Toennies guess for the next three terms
            for n in (12, 14, 16):
                C[n] = (C[n - 2] / C[n - 4]) ** 3 * C[n - 6]
        fit = self.full_fit if correlated else self.fit
        x = -(2.0 * fit[0] * R + fit[1]) * R
        v = self.repulsion(R, correlated)
        for n, c in C.items():
            series = sum(x ** k / math.factorial(k) for k in range(n + 1))
            v = v - (1.0 - np.exp(-x) * series) * c / R ** n
        return v

    def well(self, **kw):
        """(depth, position of the minimum, distance where V = 0), atomic units."""
        grid = np.linspace(4.0, 9.0, 50001)
        v = self.best(grid) if kw.pop("best", False) else self(grid, **kw)
        i = int(v.argmin())
        j = int(np.where(v < 0)[0][0])
        sigma = grid[j - 1] + (grid[j] - grid[j - 1]) * v[j - 1] / (v[j - 1] - v[j])
        return float(-v[i]), float(grid[i]), float(sigma)

    def second_virial(self, kT, mass, **kw):
        """Second virial coefficient in bohr^3 per atom pair: (classical part,
        first quantum correction). kT in hartree, mass of one atom in electron masses.

          B_cl = -2 pi  int (exp(-V/kT) - 1) r^2 dr
          B_q  = pi / (6 m kT^3)  int exp(-V/kT) V'(r)^2 r^2 dr"""
        r = np.linspace(2.0, 80.0, 780001)
        v = self.best(r) if kw.pop("best", False) else self(r, **kw)
        boltz = np.exp(-v / kT)
        dr = r[1] - r[0]
        classical = -2.0 * math.pi * (np.sum((boltz - 1.0) * r * r) * dr - 2.0 ** 3 / 3.0)
        quantum = math.pi / (6.0 * mass * kT ** 3) * np.sum(boltz * np.gradient(v, r) ** 2 * r * r) * dr
        return float(classical), float(quantum)

    def bound_state(self, mass, r_max=4000.0, step=0.02, **kw):
        """Lowest vibrational level of the pair (hartree) and its mean
        separation (bohr), from the radial Schroedinger equation on a grid.
        Needs scipy; returns None without it."""
        try:
            from scipy.linalg import eigh_tridiagonal
        except ImportError:
            return None
        mu = mass / 2.0
        r = np.arange(2.5, r_max, step)
        diag = 1.0 / (mu * step * step) + (self.best(r) if kw.pop("best", False) else self(r, **kw))
        off = -0.5 / (mu * step * step) * np.ones(len(r) - 1)
        e, vec = eigh_tridiagonal(diag, off, select="i", select_range=(0, 0))
        u2 = vec[:, 0] ** 2
        return float(e[0]), float(np.sum(u2 * r) / np.sum(u2))


def _average_hf(S, h, eri, ns, occupations):
    """Hartree-Fock with averaged occupations in the s functions.
    Returns (occupied orbital coefficients, density matrix), both in the raw basis."""
    n = S.shape[0]
    X = orthogonalizer(S[:ns, :ns])
    hs, es = h[:ns, :ns], eri[:ns, :ns, :ns, :ns]
    D = np.zeros((ns, ns))
    for it in range(200):
        F = hs + np.einsum("ijkl,kl->ij", es, D) - 0.5 * np.einsum("ikjl,kl->ij", es, D)
        _, C = np.linalg.eigh(X.T @ F @ X)
        C = X @ C[:, :len(occupations)]
        new = (C * np.asarray(occupations, float)) @ C.T
        if np.abs(new - D).max() < 1e-10:
            break
        D = 0.5 * (D + new) if it < 30 else new
    Cf = np.zeros((n, len(occupations))); Cf[:ns] = C
    Df = np.zeros((n, n)); Df[:ns, :ns] = D
    return Cf, Df


def _transform(eri, W):
    g = np.einsum("pi,pqrs->iqrs", W, eri)
    g = np.einsum("qj,iqrs->ijrs", W, g)
    g = np.einsum("rk,ijrs->ijks", W, g)
    return np.einsum("sl,ijks->ijkl", W, g)


def atom_sp_orbitals(Z, occupations, s_keep, p_exponents, d_exponents=(), f_exponents=()):
    """Orthonormal orbitals for an atom, with h and (pq|rs) in that basis.

    s space: the Hartree-Fock orbitals of the atom (occupation numbers given),
    plus the primitive s-Gaussians with indices `s_keep`; then all p and d
    functions from `atom_raw_basis`."""
    S, h, eri, ns = atom_raw_basis(Z, p_exponents, d_exponents, f_exponents)
    n = S.shape[0]
    C, D = _average_hf(S, h, eri, ns, occupations)
    columns = [C[:, k] for k in range(C.shape[1])]
    for k in list(s_keep) + list(range(ns, n)):
        e = np.zeros(n); e[k] = 1.0
        columns.append(e)
    W = np.array(columns).T
    W = W @ orthogonalizer(W.T @ S @ W, 1e-8)
    # canonical orbitals of the average Fock operator (good for the CI preconditioner)
    F = h + np.einsum("ijkl,kl->ij", eri, D) - 0.5 * np.einsum("ikjl,kl->ij", eri, D)
    _, U = np.linalg.eigh(W.T @ F @ W)
    W = W @ U
    return W.T @ h @ W, _transform(eri, W)


def frozen_core_valence(Z, p_exponents, d_exponents=(), f_exponents=()):
    """Two valence electrons outside a frozen 1s² core (for beryllium).

    The core is the Hartree-Fock 1s orbital of the neutral atom. The valence
    electrons move in the nuclear field plus the average field of the core,
    in all orbitals orthogonal to it. Returns (energy of the valence pair by
    full CI, energy of a single valence electron); their difference is the
    ionization energy in this model."""
    S, h, eri, ns = atom_raw_basis(Z, p_exponents, d_exponents, f_exponents)
    C, _ = _average_hf(S, h, eri, ns, (2, 2))
    core = C[:, :1]
    Dc = 2.0 * core @ core.T
    h_eff = h + np.einsum("ijkl,kl->ij", eri, Dc) - 0.5 * np.einsum("ikjl,kl->ij", eri, Dc)
    X = orthogonalizer(S, 1e-8)
    c = X.T @ S @ core                           # core in the orthonormal basis
    _, vecs = np.linalg.eigh(np.eye(X.shape[1]) - c @ c.T)
    W = X @ vecs[:, 1:]                          # everything orthogonal to the core
    ho, g = W.T @ h_eff @ W, _transform(eri, W)
    pair = fci_two_electrons(np.eye(ho.shape[0]), ho, g)
    return pair, float(np.linalg.eigvalsh(ho)[0])


def _strings(n, k):
    """All ways to put k electrons of one spin into n orbitals, and the table
    of single excitations <I| a+_p a_q |J> = sign."""
    from itertools import combinations
    strings = list(combinations(range(n), k))
    index = {s: i for i, s in enumerate(strings)}
    pq, I, J, sign = [], [], [], []
    for j, s in enumerate(strings):
        occupied = set(s)
        for q in s:
            rest = [x for x in s if x != q]
            for p in range(n):
                if p in occupied and p != q:
                    continue
                target = tuple(sorted(rest + [p]))
                lo, hi = min(p, q), max(p, q)
                between = sum(1 for x in rest if lo < x < hi)
                pq.append(p * n + q); I.append(index[target]); J.append(j)
                sign.append(-1.0 if between % 2 else 1.0)
    occ = np.zeros((len(strings), n))
    for i, s in enumerate(strings):
        occ[i, list(s)] = 1.0
    return occ, np.array(pq), np.array(I), np.array(J), np.array(sign)


def fci(h, eri, n_alpha, n_beta, e_nuc=0.0, tol=1e-7, max_iter=200, low_memory=False):
    """Full configuration interaction ground state in an orthonormal orbital basis.

    The wavefunction is a matrix C[alpha string, beta string]; the lowest
    eigenvalue is found by Davidson iteration, never forming the Hamiltonian.
    With low_memory=True (needs scipy) the work array of size
    orbitals^2 x determinants is avoided: the same-spin parts become two
    dense matrices over strings, and the opposite-spin part is applied one
    orbital pair at a time through sparse matrices."""
    n = h.shape[0]
    occ_a, pq_a, I_a, J_a, s_a = _strings(n, n_alpha)
    occ_b, pq_b, I_b, J_b, s_b = _strings(n, n_beta)
    Na, Nb = len(occ_a), len(occ_b)
    k = h - 0.5 * np.einsum("prrq->pq", eri)
    g2 = eri.reshape(n * n, n * n)

    order_a, order_b = np.argsort(pq_a, kind="stable"), np.argsort(pq_b, kind="stable")
    pq_a, I_a, J_a, s_a = pq_a[order_a], I_a[order_a], J_a[order_a], s_a[order_a]
    pq_b, I_b, J_b, s_b = pq_b[order_b], I_b[order_b], J_b[order_b], s_b[order_b]
    step = max(1, min(n * n, 200_000_000 // (Na * Nb)))    # rows of G held at once (about 1.6 GB)

    def sigma(C):
        D = np.zeros((n * n, Na, Nb))
        np.add.at(D, (pq_a, I_a), s_a[:, None] * C[J_a])
        np.add.at(D.transpose(0, 2, 1), (pq_b, I_b), s_b[:, None] * C.T[J_b])
        D2 = D.reshape(n * n, Na * Nb)
        out = (k.ravel() @ D2).reshape(Na, Nb)
        for lo in range(0, n * n, step):
            hi = min(n * n, lo + step)
            G = (0.5 * g2[lo:hi] @ D2).reshape(hi - lo, Na, Nb)
            a0, a1 = np.searchsorted(pq_a, [lo, hi])
            np.add.at(out, I_a[a0:a1], s_a[a0:a1, None] * G[pq_a[a0:a1] - lo, J_a[a0:a1]])
            b0, b1 = np.searchsorted(pq_b, [lo, hi])
            np.add.at(out.T, I_b[b0:b1], s_b[b0:b1, None] * G.transpose(0, 2, 1)[pq_b[b0:b1] - lo, J_b[b0:b1]])
        return out

    if low_memory:
        from scipy import sparse
        kf = k.ravel()

        def same_spin(pq_x, I_x, J_x, s_x, N):
            """Matrix over strings of  sum k_pq E_pq + 1/2 sum (pq|rs) E_pq E_rs."""
            Hs = np.zeros((N, N))
            np.add.at(Hs, (I_x, J_x), s_x * kf[pq_x])
            for pq in range(n * n):
                lo, hi = np.searchsorted(pq_x, [pq, pq + 1])
                if lo == hi:
                    continue
                M = sparse.csr_matrix((s_x * g2[pq, pq_x], (I_x, J_x)), shape=(N, N))
                Hs[I_x[lo:hi]] += 0.5 * s_x[lo:hi, None] * M[J_x[lo:hi]].toarray()
            return Hs

        H_a = same_spin(pq_a, I_a, J_a, s_a, Na)
        H_b = H_a if n_alpha == n_beta else same_spin(pq_b, I_b, J_b, s_b, Nb)

        def sigma(C):                                      # noqa: F811
            out = H_a @ C + C @ H_b.T
            for pq in range(n * n):
                lo, hi = np.searchsorted(pq_a, [pq, pq + 1])
                if lo == hi:
                    continue
                M = sparse.csr_matrix((s_b * g2[pq, pq_b], (I_b, J_b)), shape=(Nb, Nb))
                T = s_a[lo:hi, None] * C[J_a[lo:hi]]
                out[I_a[lo:hi]] += (M @ T.T).T
            return out

    restart = 8 if low_memory else 25
    hd = np.diag(h)
    Jm = np.einsum("iijj->ij", eri)
    Km = np.einsum("ijji->ij", eri)
    same = Jm - Km
    diag = ((occ_a @ hd)[:, None] + (occ_b @ hd)[None, :]
            + 0.5 * np.einsum("ip,pq,iq->i", occ_a, same, occ_a)[:, None]
            + 0.5 * np.einsum("ip,pq,iq->i", occ_b, same, occ_b)[None, :]
            + occ_a @ Jm @ occ_b.T)

    start = np.zeros((Na, Nb))
    start.flat[np.argmin(diag)] = 1.0
    vectors, sigmas = [start], [sigma(start)]
    energy = diag.min()
    for _ in range(max_iter):
        V = np.array([v.ravel() for v in vectors])
        Sg = np.array([s.ravel() for s in sigmas])
        w, y = np.linalg.eigh(V @ Sg.T)
        energy, y = w[0], y[:, 0]
        x = (y @ V).reshape(Na, Nb)
        residual = (y @ Sg).reshape(Na, Nb) - energy * x
        if np.linalg.norm(residual) < tol:
            break
        if len(vectors) >= restart:
            x /= np.linalg.norm(x)
            vectors, sigmas = [x], [sigma(x)]
            residual = sigmas[0] - energy * x
        denom = energy - diag
        denom[np.abs(denom) < 1e-3] = -1e-3
        new = residual / denom
        for v in vectors:
            new -= np.vdot(v, new) * v
        for v in vectors:
            new -= np.vdot(v, new) * v
        new /= np.linalg.norm(new)
        vectors.append(new); sigmas.append(sigma(new))
    else:
        raise RuntimeError("full CI did not converge")
    return float(energy) + e_nuc


def helium_fci_s_general():
    """Helium, s orbitals only, through the general solver (a self-test)."""
    S, h, eri = atom_integrals(2)
    X = orthogonalizer(S, 1e-8)
    _, U = np.linalg.eigh(X.T @ h @ X)
    X = X @ U
    g = np.einsum("pi,qj,rk,sl,pqrs->ijkl", X, X, X, X, eri, optimize=True)
    return fci(X.T @ h @ X, g, 1, 1)


S_KEEP = {3: (0, 1, 2, 3, 4, 5, 7, 9, 11), 4: (1, 2, 3, 4, 5, 7, 9)}
P_EXPONENTS = {3: (0.03, 0.1, 0.35, 1.2, 4.0), 4: (0.06, 0.2, 0.7, 2.5, 8.0)}
# larger sets used together with d orbitals
P_EXPONENTS_D = {3: (0.03, 0.1, 0.35, 1.2, 4.0, 12.0), 4: (0.03, 0.08, 0.2, 0.5, 1.2, 3.0, 8.0)}
D_EXPONENTS = {3: (0.6,), 4: (0.13, 0.35, 0.9)}


def atom_fci_sp(Z, neutral_occupations, electron_counts, with_d=False):
    """Full CI energies in one s+p basis for several charge states of an atom.

    electron_counts: list of (n_alpha, n_beta). The basis is built from the
    neutral atom and reused for the ions, so the comparison is balanced."""
    if with_d:
        h, eri = atom_sp_orbitals(Z, neutral_occupations, S_KEEP[Z], P_EXPONENTS_D[Z], D_EXPONENTS[Z])
    else:
        h, eri = atom_sp_orbitals(Z, neutral_occupations, S_KEEP[Z], P_EXPONENTS[Z])
    return [fci(h, eri, na, nb) for na, nb in electron_counts]


def beryllium_d_correction():
    """Change in the ionization energy of Be (hartree) when the two valence
    electrons get a larger p set and d orbitals, with the 1s² core frozen."""
    small = frozen_core_valence(4, P_EXPONENTS[4])
    large = frozen_core_valence(4, P_EXPONENTS_D[4], D_EXPONENTS[4])
    return (large[1] - large[0]) - (small[1] - small[0])

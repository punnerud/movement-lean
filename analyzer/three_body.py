#!/usr/bin/env python3
"""Two nuclei and one light particle, with nothing held fixed: H2+, and the muonic ions ppμ, ddμ, ttμ.

In a molecule the nuclei vibrate slowly and the electron goes round quickly, so the electron
is solved with the nuclei at rest and the nuclei then move on the resulting curve (Born and
Oppenheimer). Replace the electron by a muon, 207 times heavier, and the two motions come
close: the nuclei are only 9 to 27 times heavier than the light particle instead of 1836.
Here the same systems are computed three ways, in units of the light particle (its mass, its
Bohr radius, its hartree):

  1. the threshold: one nucleus with the light particle, the other far away (exact: -μ/2);
  2. nuclei at rest, then vibration on the curve (our own curve from Gaussian functions, our
     own radial equation); the same curve serves all four, only the nuclear mass differs;
  3. walkers with all three particles free to move (cikernel/src/walk.rs): no separation of
     the motions is assumed. The guess is built from step 2 (where the nuclei sit and how far
     they swing), not from any reference value.

    python3 analyzer/three_body.py [quick] [nolimit] [name ...]
"""
import sys
import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.linalg import eigh, eigh_tridiagonal
from scipy.special import erf

import diffusion_mc

HARTREE_EV = 27.211386245988
# masses in electron masses as used in the references, so that the comparison is like for like
M_MU, M_P, M_D, M_T = 206.768262, 1836.152701, 3670.483014, 5496.92158
# name: (light mass in electron masses, nuclear mass in light masses, published energy in light units, source)
SYSTEMS = {
    "H2+": (1.0, M_P, -0.5971390631239, "Taylor, Yan, Dalgarno and Babb 1998 (arXiv:physics/9811043), table 2"),
    "ttμ": (M_MU, M_T / M_MU, -0.54637422561382, "Frolov and Wardlaw (arXiv:1008.3010), table 3"),
    "ddμ": (M_MU, M_D / M_MU, -0.53111113540239, "Frolov and Wardlaw (arXiv:1008.3010), table 3"),
    "ppμ": (M_MU, M_P / M_MU, -0.49438682024893, "Frolov and Wardlaw (arXiv:1008.3010), table 3"),
}


def light_energy(R, n_exp=10, n_cen=9):
    """Lowest energy of one particle (mass 1, charge -1) around two fixed unit charges a distance
    R apart: s Gaussians on a row of centres between the charges. An upper bound."""
    exps = 0.03 * 3.0 ** np.arange(n_exp)
    a, z = np.repeat(exps, n_cen), np.tile(np.linspace(-R / 2, R / 2, n_cen), n_exp)
    A, B = a[:, None], a[None, :]
    p, mu, d2 = A + B, A * B / (A + B), (z[:, None] - z[None, :]) ** 2
    S = (np.pi / p) ** 1.5 * np.exp(-mu * d2)
    T = mu * (3 - 2 * mu * d2) * S
    P = (A * z[:, None] + B * z[None, :]) / p

    def boys(t):
        t = np.maximum(t, 1e-14)
        return np.where(t < 1e-10, 1.0, 0.5 * np.sqrt(np.pi / t) * erf(np.sqrt(t)))

    V = sum(-2 * np.pi / p * np.exp(-mu * d2) * boys(p * (P - c) ** 2) for c in (-R / 2, R / 2))
    w, U = np.linalg.eigh(S)
    X = U[:, w > 1e-9 * w.max()] / np.sqrt(w[w > 1e-9 * w.max()])
    return float(eigh(X.T @ (T + V) @ X, eigvals_only=True)[0])


def curve():
    """U(R) = light energy + repulsion of the nuclei, as a function for any R."""
    grid = np.concatenate([np.arange(0.2, 4.0, 0.1), np.arange(4.0, 12.01, 0.5)])
    spline = CubicSpline(grid, [light_energy(R) + 1 / R for R in grid])
    tail = (spline(12.0) + 0.5) * 12.0 ** 4                      # beyond the grid: -1/2 plus a 1/R^4 pull
    return lambda r: np.where(r < 0.2, 1 / r - 2.0, np.where(r > 12.0, -0.5 + tail / r ** 4, spline(np.clip(r, 0.2, 12.0)))), spline


def vibration(U, M, r_max=14.0, points=14000):
    """Lowest state of two nuclei of mass M on the curve U: energy, mean distance, spread."""
    mu = M / 2
    h = r_max / (points + 1)
    r = h * np.arange(1, points + 1)
    e, v = eigh_tridiagonal(1 / (mu * h * h) + U(r), np.full(points - 1, -0.5 / (mu * h * h)), select="i", select_range=(0, 0))
    w = v[:, 0] ** 2
    mean = float((w * r).sum())
    return float(e[0]), mean, float(np.sqrt((w * (r - mean) ** 2).sum()))


def guess(M, r0, sigma, kappa):
    """Pair functions (a, b, c) of exp((a r + c r^2) / (1 + b r)): the nuclei repel with the right
    slope at contact, and peak at r0 with the spread sigma. The light particle belongs to both
    nuclei, exp(-kappa r_1) + exp(-kappa r_2) (the `shared` factor of the engine); its pair
    functions only correct the slope at each nucleus and die out at a distance."""
    a, g = M / 2, 1 / (4 * sigma ** 2)
    k = a / (g * r0)
    y = (-3 + np.sqrt(1 + 4 * k)) / 2 if k > 2 else 0.0          # (1 + y)(2 + y) = k, y = b r0
    nuclei = (a, y / r0, -g * (1 + y)) if y > 0 else (2 * g * r0, 0.0, -g)
    m = M / (1 + M)
    light = (-m + kappa / (1 + np.exp(-kappa * r0)), 1.0, 0.0)
    return {(0, 1): nuclei, (0, 2): light, (1, 2): light}


def main():
    args = sys.argv[1:]
    quick = "quick" in args
    if "nolimit" in args:                                        # no limit on the reaction rate (see diffusion_mc.py steps)
        import mesh_ci
        mesh_ci.kernel().walk_options(0.0, 0)
    names = [a for a in args if a in SYSTEMS] or list(SYSTEMS)
    U, spline = curve()
    r_min = float(spline.derivative().roots()[0])
    k = float(spline(r_min, 2))
    print(f"our curve: lowest at R = {r_min:.4f} with U = {float(spline(r_min)):.6f} (known for H2+: 1.9972 and -0.602635), stiffness {k:.4f}", flush=True)
    for name in names:
        light, M, published, source = SYSTEMS[name]
        t0 = time.time()
        threshold = -0.5 * M / (1 + M)
        e_bo, r0, sigma = vibration(U, M)
        quantum = np.sqrt(k / (M / 2))
        print(f"{name}: nuclei {M:.4f} times the light particle; vibration quantum {quantum:.4f} against the orbit's 0.5: ratio {0.5 / quantum:.1f}", flush=True)
        print(f"  nuclei at rest, then vibration: {e_bo:.6f}   (mean distance {r0:.3f}, spread {sigma:.3f})   published {published:.6f}   "
              f"difference {1e3 * (e_bo - published):+.3f} milli-units = {100 * (e_bo - published) / (threshold - published):+.1f} % of the binding energy", flush=True)
        # the nuclei settle in about 1 / (vibration quantum): the warm-up of 20 time units is six of those, at least 100
        unit = max(0.3 / quantum, 5.0)
        kw = dict(walkers=3000, duration=40.0) if quick else dict(walkers=20000, duration=250.0)
        fars = (0.8, 0.9, 1.0, 1.1, 1.2)
        alone = [diffusion_mc.particles_energy((M, M, 1.0), (1, 1, -1), guess(M, r0, sigma, f), 0.02, 1.5, unit, walkers=3000, duration=5.0, seed=7,
                                               shared=(0, 1, 2, f))[2] for f in fars]
        far = fars[int(np.argmin(alone))]
        pairs = guess(M, r0, sigma, far)
        kw["shared"] = (0, 1, 2, far)
        taus = (0.02, 0.01, 0.005)
        runs = [diffusion_mc.particles_energy((M, M, 1.0), (1, 1, -1), pairs, tau, 1.5, unit, seed=k + 1, **kw) for k, tau in enumerate(taus)]
        e0, s0 = diffusion_mc.straight_line(taus, runs)
        to_ev = HARTREE_EV * light
        print(f"  the guess alone: {min(alone):.5f} (decay constant {far}); walkers, all three free: " +
              ", ".join(f"step {tau:g}: {e:.5f} ± {s:.5f}" for tau, (e, s, _) in zip(taus, runs)) + f"; step -> 0: {e0:.5f} ± {s0:.5f}", flush=True)
        print(f"  walkers - published: {1e3 * (e0 - published):+.3f} milli-units = {(e0 - published) / s0:+.1f} errors.   "
              f"Binding energy (threshold {threshold:.6f}): walkers {(threshold - e0) * to_ev:.4g} ± {s0 * to_ev:.2g} eV, "
              f"nuclei-at-rest picture {(-0.5 - e_bo) * to_ev:.4g} eV against its own threshold -1/2, {(threshold - e_bo) * to_ev:.4g} eV against the true one, "
              f"published {(threshold - published) * to_ev:.4g} eV   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

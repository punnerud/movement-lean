#!/usr/bin/env python3
"""The Schrödinger equation solved as diffusion with reaction.

In imaginary time the Schrödinger equation is  du/dt = D lap(u) - (V - E_T) u
with D = 1/2: walkers that diffuse (the random walk of Movement/Heat.lean)
and multiply or die at the rate E_T - V. The population is steady exactly
when E_T is the lowest energy (Lean: `eigenstate_reaction_diffusion`,
`steady_iff_shift_eq_energy`), so the energy is read off as the shift that
keeps the number of walkers constant. No basis functions, no determinants.

The plain version fails on atoms because V is infinite at a nucleus. The
standard cure is used here: a guess g for the wavefunction turns the equation
into one for f = u g, where the walkers also drift along grad(ln g) and the
rate becomes E_T - (H g)/g, which stays finite when g has the right cusps.
For two electrons of opposite spin the ground state has no nodes, so the
result does not depend on g: only the statistical noise does. The remaining
error is the finite time step, removed by repeating with smaller steps.
With more electrons the wavefunction changes sign. The walkers are then kept
on one side of the surfaces where g is zero, and the result is the lowest
energy with those surfaces: an upper bound that depends on g.

Reactions are differences of such energies: the barrier of H + H2 -> H2 + H
is E(H3 at the saddle) - E(H) - E(H2).

    python3 analyzer/diffusion_mc.py            # all species and reactions, long (hours on two threads)
    python3 analyzer/diffusion_mc.py H3 H3+     # only these species; results are kept in ci_cache/diffusion_mc.json
    python3 analyzer/diffusion_mc.py quick      # fewer steps
    python3 analyzer/diffusion_mc.py particles [quick]   # nothing held fixed: positronium, hydrogen with a moving proton, ...
    python3 analyzer/diffusion_mc.py charmonium [quick]  # a charm quark and antiquark with a Coulomb-like pull plus a constant one
    python3 analyzer/diffusion_mc.py python     # the walker loop in numpy instead of Rust (same algorithm, slower)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HARTREE_EV = 27.211386245981


class Guess:
    """g = det(up orbitals) det(down orbitals) exp(sum over pairs u(r_ij)).
    An orbital is sum over nuclei c_A exp(-zeta_A |r - R_A|); `up` and `down`
    list the coefficients c of each occupied orbital (default: one orbital
    with all c = 1, for one or two electrons). u(r) = a r / (1 + b r) with
    a = 1/2 for opposite spins and 1/4 for equal ones. `zeta` is one number
    or one per nucleus. The numpy `evaluate` covers the default case only."""

    def __init__(self, nuclei, zeta, b=0.4, up=None, down=None):
        self.up = np.array(up if up is not None else [[1.0] * len(nuclei)], float)
        self.down = np.array(down if down is not None else [[1.0] * len(nuclei)], float).reshape(-1, len(nuclei))
        self.Z = np.array([z for z, _ in nuclei], float)
        self.R = np.array([p for _, p in nuclei], float)
        self.zeta, self.b = np.broadcast_to(np.asarray(zeta, float), (len(nuclei),)).copy(), b
        self.e_nuc = sum(self.Z[i] * self.Z[j] / np.linalg.norm(self.R[i] - self.R[j])
                         for i in range(len(nuclei)) for j in range(i))

    def evaluate(self, x):
        """x: (walkers, electrons, 3). Returns (ln g, grad ln g, local energy (H g)/g)."""
        d = x[:, :, None, :] - self.R[None, None]                # walkers, electrons, nuclei, 3
        r = np.linalg.norm(d, axis=-1)
        e = np.exp(-self.zeta * r)
        phi = e.sum(-1)
        grad = (-self.zeta * e / r)[..., None] * d
        grad = grad.sum(2) / phi[..., None]                      # grad ln phi
        lap = (e * (self.zeta ** 2 - 2 * self.zeta / r)).sum(-1) / phi - (grad ** 2).sum(-1)
        ln = np.log(phi).sum(-1)
        V = -(self.Z / r).sum((1, 2)) + self.e_nuc
        if x.shape[1] == 2:
            s = x[:, 0] - x[:, 1]
            r12 = np.linalg.norm(s, axis=-1)
            q = 1.0 + self.b * r12
            u1 = 0.5 / q ** 2
            ln = ln + 0.5 * r12 / q
            pair = (u1 / r12)[:, None] * s
            grad = grad + np.stack([pair, -pair], axis=1)
            lap = lap + (-self.b / q ** 3 + 2 * u1 / r12)[:, None]
            V = V + 1.0 / r12
        return ln, grad, -0.5 * (lap + (grad ** 2).sum(-1)).sum(-1) + V


def shorten(drift, tau):
    """The drift of each electron, shortened where it is very large (see cikernel/src/walk.rs)."""
    v2 = (drift ** 2).sum(-1, keepdims=True) * tau
    return drift * np.where(v2 > 1e-12, (np.sqrt(1 + 2 * v2) - 1) / np.maximum(v2, 1e-300), 1.0)


def free_walk(walkers=20000, steps=200, tau=0.01, seed=1):
    """No guess, no potential: <x^2> must be 2 D t per direction (Lean: `einstein_relation`)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((walkers, 3))
    for _ in range(steps):
        x += rng.normal(0.0, np.sqrt(tau), x.shape)
    return float((x ** 2).mean()), 2 * 0.5 * steps * tau


def _blocks(e):
    """Mean and its statistical error. Successive steps are correlated, so the error is taken
    from the means of long blocks; the largest value over several block lengths is used, since
    blocks shorter than the correlation time give too small an error."""
    errors = []
    for count in (40, 20, 10, 5):
        blocks = e[:len(e) // count * count].reshape(count, -1).mean(1)
        errors.append(blocks.std(ddof=1) / np.sqrt(count))
    return float(e.mean()), float(max(errors))


def energy_rust(guess, electrons, tau, walkers=3000, duration=150.0, seed=1):
    """The same as `energy`, with the walker loop in the Rust kernel (cikernel/src/walk.rs)."""
    import ctypes
    import mesh_ci
    lib = mesh_ci.kernel()
    warm, steps = int(20.0 / tau), int(duration / tau)
    out, alone = np.zeros(steps), ctypes.c_double(0.0)
    Z, R = np.ascontiguousarray(guess.Z), np.ascontiguousarray(guess.R.ravel())
    up = np.ascontiguousarray(guess.up)
    down = np.ascontiguousarray(guess.down[:electrons - len(up)])
    if lib.walk_run(len(Z), mesh_ci._d(Z), mesh_ci._d(R), mesh_ci._d(guess.zeta), len(up), len(down), mesh_ci._d(up),
                    mesh_ci._d(down), guess.b, tau, walkers, warm, steps, seed, mesh_ci._d(out), ctypes.byref(alone)):
        raise ValueError("too many electrons for the walker loop")
    return (*_blocks(out), alone.value)


def energy(guess, electrons, tau, walkers=3000, duration=150.0, seed=1):
    """Lowest energy at time step tau, followed for `duration` atomic units of time after a
    warm-up of 20: (mean, statistical error, mean of the guess alone)."""
    warm, steps = int(20.0 / tau), int(duration / tau)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (walkers, electrons, 3)) + guess.R[rng.integers(len(guess.R), size=(walkers, electrons))]
    ln, drift, local = guess.evaluate(x)
    drift = shorten(drift, tau)
    shift = local.mean()
    samples, first = [], []
    for step in range(warm + steps):
        # diffusion with drift, accepted or rejected so that g^2 is sampled exactly at any step
        new = x + tau * drift + rng.normal(0.0, np.sqrt(tau), x.shape)
        ln2, drift2, local2 = guess.evaluate(new)
        drift2 = shorten(drift2, tau)
        forward = ((new - x - tau * drift) ** 2).sum((1, 2))
        back = ((x - new - tau * drift2) ** 2).sum((1, 2))
        accept = np.log(rng.random(walkers)) < 2 * (ln2 - ln) + (forward - back) / (2 * tau)
        limit = 0.2 * np.sqrt(electrons / tau)                   # see cikernel/src/walk.rs
        old_local = np.clip(local, shift - limit, shift + limit)
        x[accept], ln[accept], drift[accept], local[accept] = new[accept], ln2[accept], drift2[accept], local2[accept]
        if step < warm // 2:                                     # no reaction yet: this samples the guess alone
            if step >= warm // 4:
                first.append(local.mean())
            continue
        # reaction: multiply or die at the rate shift - local energy
        here = np.clip(local, shift - limit, shift + limit)
        weight = np.where(accept, np.exp(-tau * (0.5 * (here + old_local) - shift)), 1.0)   # no move, no reaction
        samples.append((np.sum(weight * here) / np.sum(weight), weight.mean()))
        copies = np.floor(weight * walkers / weight.sum() + rng.random(walkers)).astype(int)
        keep = np.repeat(np.arange(walkers), copies)
        if len(keep) != walkers:                                 # hold the population at its size
            keep = rng.choice(keep, walkers)
        x, ln, drift, local = x[keep], ln[keep], drift[keep], local[keep]
        shift += 0.1 * (samples[-1][0] - shift) if step < warm else 0.0
    return (*_blocks(np.array([s[0] for s in samples[warm - warm // 2:]])), float(np.mean(first)))


def particles_energy(masses, charges, pairs, tau, length=1.0, time_unit=1.0, walkers=3000, duration=150.0, seed=1,
                     linear=None, odd=None):
    """Lowest energy of particles with the given masses and charges, nothing held fixed. The
    force between a pair is charge_i charge_j / r plus linear[(i, j)] r. With odd=(i, j) the
    guess is multiplied by z_i - z_j: the lowest state odd under reflection (a P state). pairs[(i, j)] = (a, b, c) gives the guess
    exp(sum of (a r + c r^2) / (1 + b r)) (see cikernel/src/walk.rs). `duration` and the
    warm-up of 20 are in units of `time_unit`. Returns (mean, error, mean of the guess alone)."""
    import ctypes
    import mesh_ci
    lib = mesh_ci.kernel()
    n = len(masses)
    A, B, C = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
    for (i, j), (a, b, c) in pairs.items():
        A[i, j] = A[j, i] = a
        B[i, j] = B[j, i] = b
        C[i, j] = C[j, i] = c
    L = np.zeros((n, n))
    for (i, j), k in (linear or {}).items():
        L[i, j] = L[j, i] = k
    warm, steps = int(20.0 * time_unit / tau), int(duration * time_unit / tau)
    out, alone = np.zeros(steps), ctypes.c_double(0.0)
    m, q = np.array(masses, float), np.array(charges, float)
    if lib.walk_particles(n, mesh_ci._d(m), mesh_ci._d(q), mesh_ci._d(A), mesh_ci._d(B), mesh_ci._d(C), mesh_ci._d(L),
                          -1 if odd is None else odd[0] * n + odd[1], length, tau,
                          walkers, warm, steps, seed, mesh_ci._d(out), ctypes.byref(alone)):
        raise ValueError("two to eight particles")
    return (*_blocks(out), alone.value)


def straight_line(taus, runs):
    e, s = np.array([r[0] for r in runs]), np.array([r[1] for r in runs])
    A = np.vstack([np.ones(len(taus)), taus]).T / s[:, None]
    cov = np.linalg.inv(A.T @ A)
    fit = cov @ A.T @ (e / s)
    return float(fit[0]), float(np.sqrt(cov[0, 0]))


M_P, M_MU = 1836.15267343, 206.7682830          # proton and muon masses in electron masses (CODATA 2018)
MU_H, MU_MUH = M_P / (1 + M_P), M_MU * M_P / (M_MU + M_P)
# name: masses, charges, pair functions, length, time unit, reference, source
PARTICLES = {
    "positronium (e- e+)": ((1, 1), (-1, 1), {(0, 1): (-0.5, 0.5, -0.2)}, 2.0, 4.0, -0.25, "exact: -1/4"),
    "hydrogen, proton free to move": ((1, M_P), (-1, 1), {(0, 1): (-MU_H, 0.5, -0.45)}, 1.0, 1.0, -MU_H / 2,
                                      "exact: -reduced mass / 2"),
    "muonic hydrogen (mu- p)": ((M_MU, M_P), (-1, 1), {(0, 1): (-MU_MUH, 0.5 * MU_MUH, -0.45 * MU_MUH ** 2)},
                                1 / MU_MUH, 1 / MU_MUH, -MU_MUH / 2, "exact: -reduced mass / 2"),
    "positronium ion (e- e- e+)": ((1, 1, 1), (-1, -1, 1), {(0, 2): (-0.5, 0.3, -0.12), (1, 2): (-0.5, 0.3, -0.12),
                                                            (0, 1): (0.5, 0.3, 0.0)}, 3.0, 4.0, -0.262005070,
                                   "variational calculations; from memory, not checked"),
}


def particles_main(quick):
    for name, (masses, charges, pairs, length, time_unit, reference, source) in PARTICLES.items():
        t0 = time.time()
        taus = [f * time_unit for f in (0.02, 0.01, 0.005)]
        kw = dict(duration=30.0) if quick else dict(duration=600.0, walkers=20000)
        runs = [particles_energy(masses, charges, pairs, tau, length, time_unit, seed=k + 1, **kw) for k, tau in enumerate(taus)]
        e0, s0 = straight_line(taus, runs)
        print(f"{name}: the guess alone {runs[-1][2]:.6g}; " +
              ", ".join(f"step {tau:.3g}: {e:.6g} ± {s:.2g}" for tau, (e, s, _) in zip(taus, runs)) +
              f"; step -> 0: {e0:.6g} ± {s0:.2g}   reference {reference:.6g} ({source}), "
              f"difference {100 * (e0 - reference) / abs(reference):+.2f} % = {(e0 - reference) / s0:+.1f} errors   ({time.time() - t0:.0f} s)",
              flush=True)


def radial_levels(mu, potential, l=0, count=2, r_max=30.0, points=20000):
    """Lowest energies of two particles with reduced mass mu and angular momentum l, from the
    radial equation on a grid: the independent check of the walkers."""
    from scipy.linalg import eigh_tridiagonal
    h = r_max / (points + 1)
    r = h * np.arange(1, points + 1)
    diagonal = 1.0 / (mu * h * h) + potential(r) + l * (l + 1) / (2 * mu * r * r)
    return eigh_tridiagonal(diagonal, np.full(points - 1, -0.5 / (mu * h * h)), select="i", select_range=(0, count - 1))[0]


# The Cornell model of charmonium (Eichten et al.): V = -KAPPA / r + r / A^2, in GeV and 1/GeV.
# These are the single-channel parameters (kappa = 0.52, a = 2.34 / GeV, m_c = 1.84 GeV), as quoted in
# arXiv:1208.2186; the model's own table of levels has not been compared.
M_C, KAPPA, A_CORNELL = 1.84, 0.52, 2.34
# measured masses in GeV (entered from memory): spin-averaged 1S (eta_c, J/psi) and the centre of the 1P states
MEASURED = {"1S": (3 * 3.0969 + 2.9839) / 4, "1P": 3.5253}


def charmonium_main(quick):
    sigma, mu = 1.0 / A_CORNELL ** 2, M_C / 2
    V = lambda r: -KAPPA / r + sigma * r
    grid = {"1S": radial_levels(mu, V, 0)[0], "1P": radial_levels(mu, V, 1)[0]}
    check = radial_levels(1.0, lambda r: -1.0 / r, 0, r_max=60.0)[0], radial_levels(1.0, lambda r: -1.0 / r, 1, r_max=60.0)[0]
    print(f"radial grid on hydrogen: 1s {check[0]:.6f} (exact -0.5), 2p {check[1]:.6f} (exact -0.125)")
    q = np.sqrt(KAPPA)
    kw = dict(duration=30.0) if quick else dict(duration=300.0, walkers=20000)
    found = {}
    for state, odd, (b, c) in (("1S", None, (0.5, -0.5)), ("1P", (0, 1), (0.5, -0.4))):
        t0 = time.time()
        taus = [f * 4.0 for f in (0.02, 0.01, 0.005)]
        runs = [particles_energy((M_C, M_C), (q, -q), {(0, 1): (-KAPPA * mu, b, c)}, tau, 2.0, 4.0, seed=k + 1,
                                 linear={(0, 1): sigma}, odd=odd, **kw) for k, tau in enumerate(taus)]
        e0, s0 = straight_line(taus, runs)
        found[state] = (e0, s0)
        print(f"{state}: the guess alone {runs[-1][2]:.4f}; " + ", ".join(f"step {tau:g}: {e:.4f} ± {s:.4f}" for tau, (e, s, _) in zip(taus, runs)) +
              f"; step -> 0: {e0:.4f} ± {s0:.4f} GeV   radial grid {grid[state]:.4f}, difference {1e3 * (e0 - grid[state]):+.1f} MeV "
              f"= {(e0 - grid[state]) / s0:+.1f} errors   ({time.time() - t0:.0f} s)", flush=True)
    # the model's absolute masses need its additive constant, which is not entered here; the
    # difference between two states does not depend on it
    d, s = found["1P"][0] - found["1S"][0], np.hypot(found["1P"][1], found["1S"][1])
    print(f"1P - 1S: {1e3 * d:.0f} ± {1e3 * s:.0f} MeV   (radial grid {1e3 * (grid['1P'] - grid['1S']):.0f}, measured {1e3 * (MEASURED['1P'] - MEASURED['1S']):.0f})")


def extrapolated(guess, electrons, taus, run=energy_rust, **kw):
    """Energies at each time step and the straight-line value at step zero."""
    runs = [run(guess, electrons, tau, seed=k + 1, **kw) for k, tau in enumerate(taus)]
    e, s = np.array([r[0] for r in runs]), np.array([r[1] for r in runs])
    A = np.vstack([np.ones(len(taus)), taus]).T / s[:, None]
    cov = np.linalg.inv(A.T @ A)
    fit = cov @ A.T @ (e / s)
    return runs, float(fit[0]), float(np.sqrt(cov[0, 0]))


S3 = 1.65 * np.sqrt(3) / 2
# name: nuclei, electrons (up, down), zeta, orbitals up / down (None: one orbital, all coefficients 1), reference, source
CASES = {
    "H": ([(1, (0, 0, 0))], (1, 0), 0.9, None, None, -0.5, "exact"),
    "He": ([(2, (0, 0, 0))], (1, 1), 2.0, None, None, -2.903724, "Pekeris, non-relativistic"),
    "H2": ([(1, (0, 0, -0.7)), (1, (0, 0, 0.7))], (1, 1), 1.19, None, None, -1.174476, "Kolos and Wolniewicz, 1.4 bohr"),
    "H3+": ([(1, (0, 0, 0)), (1, (1.65, 0, 0)), (1, (0.825, S3, 0))], (1, 1), 1.35, None, None, -1.343836,
            "equilateral, side 1.65 bohr; from memory, not checked"),
    "HeH+": ([(2, (0, 0, 0)), (1, (0, 0, 1.4632))], (1, 1), (2.116, 1.18), [[1.0, 0.25]], [[1.0, 0.25]], -2.978706,
             "1.4632 bohr; from memory, not checked"),
    # three electrons: the wavefunction changes sign, the walkers stay on one side (upper bound)
    "H3": ([(1, (0, 0, -1.757)), (1, (0, 0, 0)), (1, (0, 0, 1.757))], (2, 1), 1.2, [[1.0, 1.0, 1.0], [1.0, 0.0, -1.0]],
           [[1.0, 1.0, 1.0]], -1.659162, "linear saddle of H + H2, 1.757 bohr; from memory (barrier 9.61 kcal/mol), not checked"),
}
# reactions: name, {species: count} (products positive), reference in hartree
REACTIONS = [("H2 -> 2 H (binding energy)", {"H": 2, "H2": -1}, 0.174476),
             ("H + H2 -> H3 (barrier of the exchange reaction)", {"H3": 1, "H": -1, "H2": -1}, 0.015314),
             ("H3+ -> H2 + H+ (proton leaves)", {"H2": 1, "H3+": -1}, 0.169360),
             ("HeH+ -> He + H+ (proton leaves)", {"He": 1, "HeH+": -1}, 0.074982)]
STORE = Path(__file__).resolve().parent / "ci_cache" / "diffusion_mc.json"


def main():
    args = sys.argv[1:]
    quick = "quick" in args
    if "particles" in args:
        return particles_main(quick)
    if "charmonium" in args:
        return charmonium_main(quick)
    run = energy if "python" in args else energy_rust
    names = [a for a in args if a in CASES] or list(CASES)
    kw = dict(duration=30.0) if quick else dict(duration=150.0) if run is energy else dict(duration=600.0, walkers=20000)
    got, want = free_walk()
    print(f"free walk: <x^2 + y^2 + z^2>/3 = {got:.4f}, 2 D t = {want:.4f}")
    taus = (0.02, 0.01, 0.005)
    results = json.loads(STORE.read_text()) if STORE.exists() and not quick else {}
    for name in names:
        nuclei, (n_up, n_dn), zeta, up, down, reference, source = CASES[name]
        if run is energy and (up is not None or n_up > 1):
            continue                                             # the numpy loop has the simple guess only
        t0 = time.time()
        runs, e0, s0 = extrapolated(Guess(nuclei, zeta, up=up, down=down), n_up + n_dn, taus, run=run, **kw)
        results[name] = (e0, s0)
        print(f"{name}: the guess alone {runs[-1][2]:.4f}; " +
              ", ".join(f"step {tau:g}: {e:.5f} ± {s:.5f}" for tau, (e, s, _) in zip(taus, runs)) +
              f"; step -> 0: {e0:.5f} ± {s0:.5f}   reference {reference:.6f} ({source}), "
              f"difference {1e3 * (e0 - reference):+.2f} mEh   ({time.time() - t0:.0f} s)", flush=True)
        if not quick:
            STORE.parent.mkdir(exist_ok=True)
            STORE.write_text(json.dumps(results, indent=1))
    for name, parts, reference in REACTIONS:
        if all(s in results for s in parts):
            d = sum(k * results[s][0] for s, k in parts.items())
            s = np.sqrt(sum((k * results[s][1]) ** 2 for s, k in parts.items()))
            print(f"{name}: {1e3 * d:.2f} ± {1e3 * s:.2f} mEh = {d * HARTREE_EV:.3f} ± {s * HARTREE_EV:.3f} eV   "
                  f"(reference {1e3 * reference:.2f} mEh)")


if __name__ == "__main__":
    main()

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

    python3 analyzer/diffusion_mc.py            # free walk, H, He, H2 (about ten minutes)
    python3 analyzer/diffusion_mc.py quick      # fewer steps
    python3 analyzer/diffusion_mc.py python     # the walker loop in numpy instead of Rust (same algorithm, slower)
"""
import sys
import time

import numpy as np

HARTREE_EV = 27.211386245981


class Guess:
    """g = prod_i phi(r_i) * exp(u(r_12)), phi(r) = sum over nuclei of
    exp(-zeta |r - R|), u(r) = r / (2 (1 + b r)). One or two electrons."""

    def __init__(self, nuclei, zeta, b=0.4):
        self.Z = np.array([z for z, _ in nuclei], float)
        self.R = np.array([p for _, p in nuclei], float)
        self.zeta, self.b = zeta, b
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


def free_walk(walkers=20000, steps=200, tau=0.01, seed=1):
    """No guess, no potential: <x^2> must be 2 D t per direction (Lean: `einstein_relation`)."""
    rng = np.random.default_rng(seed)
    x = np.zeros((walkers, 3))
    for _ in range(steps):
        x += rng.normal(0.0, np.sqrt(tau), x.shape)
    return float((x ** 2).mean()), 2 * 0.5 * steps * tau


def _blocks(e):
    blocks = e[:len(e) // 20 * 20].reshape(20, -1).mean(1)
    return float(blocks.mean()), float(blocks.std(ddof=1) / np.sqrt(20))


def energy_rust(guess, electrons, tau, walkers=3000, duration=150.0, seed=1):
    """The same as `energy`, with the walker loop in the Rust kernel (cikernel/src/walk.rs)."""
    import ctypes
    import mesh_ci
    lib = mesh_ci.kernel()
    warm, steps = int(20.0 / tau), int(duration / tau)
    out, alone = np.zeros(steps), ctypes.c_double(0.0)
    Z, R = np.ascontiguousarray(guess.Z), np.ascontiguousarray(guess.R.ravel())
    if lib.walk_run(len(Z), mesh_ci._d(Z), mesh_ci._d(R), electrons, guess.zeta, guess.b, tau, walkers, warm, steps,
                    seed, mesh_ci._d(out), ctypes.byref(alone)):
        raise ValueError("one or two electrons only")
    return (*_blocks(out), alone.value)


def energy(guess, electrons, tau, walkers=3000, duration=150.0, seed=1):
    """Lowest energy at time step tau, followed for `duration` atomic units of time after a
    warm-up of 20: (mean, statistical error, mean of the guess alone)."""
    warm, steps = int(20.0 / tau), int(duration / tau)
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, (walkers, electrons, 3)) + guess.R[rng.integers(len(guess.R), size=(walkers, electrons))]
    ln, drift, local = guess.evaluate(x)
    shift = local.mean()
    samples, first = [], []
    for step in range(warm + steps):
        # diffusion with drift, accepted or rejected so that g^2 is sampled exactly at any step
        new = x + tau * drift + rng.normal(0.0, np.sqrt(tau), x.shape)
        ln2, drift2, local2 = guess.evaluate(new)
        forward = ((new - x - tau * drift) ** 2).sum((1, 2))
        back = ((x - new - tau * drift2) ** 2).sum((1, 2))
        accept = np.log(rng.random(walkers)) < 2 * (ln2 - ln) + (forward - back) / (2 * tau)
        old_local = local.copy()
        x[accept], ln[accept], drift[accept], local[accept] = new[accept], ln2[accept], drift2[accept], local2[accept]
        if step < warm // 2:                                     # no reaction yet: this samples the guess alone
            if step >= warm // 4:
                first.append(local.mean())
            continue
        # reaction: multiply or die at the rate shift - local energy
        weight = np.exp(-tau * (0.5 * (local + old_local) - shift))
        samples.append((np.sum(weight * local) / np.sum(weight), weight.mean()))
        copies = np.floor(weight * walkers / weight.sum() + rng.random(walkers)).astype(int)
        keep = np.repeat(np.arange(walkers), copies)
        if len(keep) != walkers:                                 # hold the population at its size
            keep = rng.choice(keep, walkers)
        x, ln, drift, local = x[keep], ln[keep], drift[keep], local[keep]
        shift += 0.1 * (samples[-1][0] - shift) if step < warm else 0.0
    return (*_blocks(np.array([s[0] for s in samples[warm - warm // 2:]])), float(np.mean(first)))


def extrapolated(guess, electrons, taus, run=energy_rust, **kw):
    """Energies at each time step and the straight-line value at step zero."""
    runs = [run(guess, electrons, tau, seed=k + 1, **kw) for k, tau in enumerate(taus)]
    e, s = np.array([r[0] for r in runs]), np.array([r[1] for r in runs])
    A = np.vstack([np.ones(len(taus)), taus]).T / s[:, None]
    cov = np.linalg.inv(A.T @ A)
    fit = cov @ A.T @ (e / s)
    return runs, float(fit[0]), float(np.sqrt(cov[0, 0]))


CASES = [("H atom", [(1, (0, 0, 0))], 1, 0.9, -0.5, "exact"),
         ("He atom", [(2, (0, 0, 0))], 2, 2.0, -2.903724, "Pekeris, non-relativistic"),
         ("H2 at 1.4 bohr", [(1, (0, 0, -0.7)), (1, (0, 0, 0.7))], 2, 1.19, -1.174476, "Kolos and Wolniewicz")]


def main():
    quick = "quick" in sys.argv[1:]
    run = energy if "python" in sys.argv[1:] else energy_rust
    kw = dict(duration=30.0) if quick else dict(duration=150.0) if run is energy else dict(duration=600.0, walkers=20000)
    got, want = free_walk()
    print(f"free walk: <x^2 + y^2 + z^2>/3 = {got:.4f}, 2 D t = {want:.4f}")
    taus = (0.02, 0.01, 0.005)
    results = {}
    for name, nuclei, electrons, zeta, reference, source in CASES:
        t0 = time.time()
        runs, e0, s0 = extrapolated(Guess(nuclei, zeta), electrons, taus, run=run, **kw)
        results[name] = (e0, s0)
        print(f"{name}: the guess alone {runs[-1][2]:.4f}; " +
              ", ".join(f"step {tau:g}: {e:.5f} ± {s:.5f}" for tau, (e, s, _) in zip(taus, runs)) +
              f"; step -> 0: {e0:.5f} ± {s0:.5f}   reference {reference:.6f} ({source}), "
              f"difference {1e3 * (e0 - reference):+.2f} mEh   ({time.time() - t0:.0f} s)", flush=True)
    d, s = 2 * results["H atom"][0] - results["H2 at 1.4 bohr"][0], np.hypot(2 * results["H atom"][1], results["H2 at 1.4 bohr"][1])
    print(f"H2 binding energy: {d * HARTREE_EV:.4f} ± {s * HARTREE_EV:.4f} eV   (2 x 0.5 - reference: {(1.174476 - 1) * HARTREE_EV:.4f})")


if __name__ == "__main__":
    main()

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
    python3 analyzer/diffusion_mc.py improve    # improve the guesses of He, H2 and H3 from samples: simple and richer form,
                                                # simplex and iterated search, and what each buys (add polish: only the last variant)
    python3 analyzer/diffusion_mc.py improved [quick] [species...]   # long runs with improved guesses (hours)
    python3 analyzer/diffusion_mc.py steps [quick] [all]   # where the error in the time step comes from
    python3 analyzer/diffusion_mc.py gradient   # improve the guesses along the gradient, compared with the search
    python3 analyzer/diffusion_mc.py orbitals [long]   # H3 with richer orbitals: do the zero surfaces get better?
    python3 analyzer/diffusion_mc.py python     # the walker loop in numpy instead of Rust (same algorithm, slower)
"""
import ctypes
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
    or one per nucleus. `terms` adds powers of the scaled distances
    s(r) = r / (1 + scale r) to the exponent: a list of ((m, n, o), c) with c
    one number or one per nucleus, meaning c s(r_iA)^m for each electron when
    n = o = 0, c s(r_ij)^o for each pair when m = n = 0, and otherwise
    c (s(r_iA)^m s(r_jA)^n + s(r_iA)^n s(r_jA)^m) s(r_ij)^o for each pair and
    nucleus (see cikernel/src/walk.rs). `kinds` gives each centre's function:
    "s" (the default) or "x", "y", "z" for exp(-zeta r) times that coordinate
    (a p function). A centre with charge 0 only carries a function, so an atom
    gets a second exponent or a p function by an extra centre at its place.
    The numpy `evaluate` covers the default case only."""

    def __init__(self, nuclei, zeta, b=0.4, up=None, down=None, terms=(), scale=1.0, kinds=None):
        self.kinds = np.array(["sxyz".index(k) for k in kinds or "s" * len(nuclei)], np.int32)
        self.powers = np.array([t for t, _ in terms], np.int32).reshape(-1, 3)
        self.coeff = np.array([np.broadcast_to(np.asarray(c, float), (len(nuclei),)) for _, c in terms]).reshape(-1, len(nuclei))
        self.scale = scale
        self.up = np.array(up if up is not None else [[1.0] * len(nuclei)], float)
        self.down = np.array(down if down is not None else [[1.0] * len(nuclei)], float).reshape(-1, len(nuclei))
        self.Z = np.array([z for z, _ in nuclei], float)
        self.R = np.array([p for _, p in nuclei], float)
        self.zeta, self.b = np.broadcast_to(np.asarray(zeta, float), (len(nuclei),)).copy(), b
        self.e_nuc = sum(self.Z[i] * self.Z[j] / np.linalg.norm(self.R[i] - self.R[j])      # a centre with charge 0
                         for i in range(len(nuclei)) for j in range(i) if self.Z[i] * self.Z[j])   # only carries functions

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


def _c_guess(guess, electrons):
    """The guess as the leading arguments of the kernel's walk_* functions."""
    import mesh_ci
    guess._keep = (np.ascontiguousarray(guess.Z), np.ascontiguousarray(guess.R.ravel()), np.ascontiguousarray(guess.zeta),
                   np.ascontiguousarray(guess.up), np.ascontiguousarray(guess.down[:electrons - len(guess.up)]),
                   np.ascontiguousarray(guess.powers), np.ascontiguousarray(guess.coeff), np.ascontiguousarray(guess.kinds))
    Z, R, zeta, up, down, powers, coeff, kinds = guess._keep
    return (len(Z), mesh_ci._d(Z), mesh_ci._d(R), mesh_ci._d(zeta), len(up), len(down), mesh_ci._d(up), mesh_ci._d(down),
            float(guess.b), len(powers), powers.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), mesh_ci._d(coeff), float(guess.scale),
            kinds.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))


def sample(guess, electrons, walkers=4000, steps=600, tau=0.05, seed=1):
    """Positions distributed as g^2 (walkers x 3 electrons) and (H g)/g there: no reaction."""
    import mesh_ci
    x, local = np.zeros((walkers, 3 * electrons)), np.zeros(walkers)
    if mesh_ci.kernel().walk_sample(*_c_guess(guess, electrons), tau, walkers, steps, seed, mesh_ci._d(x), mesh_ci._d(local)):
        raise ValueError("too many electrons for the walker loop")
    return x, local


def local_energies(guess, electrons, x):
    """(H g)/g and ln |g| at the given positions."""
    import mesh_ci
    x = np.ascontiguousarray(x)
    local, ln = np.zeros(len(x)), np.zeros(len(x))
    mesh_ci.kernel().walk_local(*_c_guess(guess, electrons), len(x), mesh_ci._d(x), mesh_ci._d(local), mesh_ci._d(ln))
    return local, ln


def check_derivatives(guess, electrons, points=200, h=1e-4):
    """Largest difference between (H g)/g from the kernel and the same from differences of
    ln g (which uses none of the kernel's derivative formulas), over sample points."""
    x = sample(guess, electrons, points, steps=200)[0]
    local, ln = local_energies(guess, electrons, x)
    lap, square = np.zeros(points), np.zeros(points)
    for k in range(3 * electrons):
        step = np.zeros(3 * electrons)
        step[k] = h
        up, down = local_energies(guess, electrons, x + step)[1], local_energies(guess, electrons, x - step)[1]
        lap += (up - 2 * ln + down) / h ** 2
        square += ((up - down) / (2 * h)) ** 2
    r = x.reshape(points, electrons, 1, 3) - guess.R
    V = -(guess.Z / np.linalg.norm(r, axis=-1)).sum((1, 2)) + guess.e_nuc
    for i in range(electrons):
        for j in range(i):
            V += 1.0 / np.linalg.norm(x[:, 3 * i:3 * i + 3] - x[:, 3 * j:3 * j + 3], axis=1)
    return float(np.abs(-0.5 * (lap + square) + V - local).max())


def iterated(cost, start, steps, iterations=1500, starts=3, history=20, stall=150, seed=0):
    """Minimises cost(p) the way the MPEE route optimiser works (github.com/punnerud/mpee,
    crates/brooom): many cheap moves instead of a few careful ones.
      - a move changes one parameter (or, one time in four, all of them a little);
      - a move is kept if it is no worse than now, or no worse than `history` moves ago
        (late acceptance: the search can walk through a slightly worse stretch);
      - the size of the moves of each parameter grows when they succeed and shrinks when not;
      - after `stall` moves without a new best, the search is kicked away from the best point;
      - several starts: the first from `start` itself, the others from kicked copies of the best;
      - the best point ever seen is what is returned, so the result is never worse than `start`.
    Returns (best parameters, best cost, number of cost evaluations)."""
    rng = np.random.default_rng(seed)
    best, best_cost, count = np.array(start, float), cost(np.array(start, float)), 1
    steps = np.array(steps, float)
    for attempt in range(starts):
        size = steps.copy()
        p = best + (3 * size * rng.normal(size=len(best)) if attempt else 0)
        now = cost(p)
        past, since = [now] * history, 0
        for it in range(iterations):
            if since >= stall:
                p, size, since = best + 3 * steps * rng.normal(size=len(best)), steps.copy(), 0
                now = cost(p)
                past = [now] * history
            k = rng.integers(len(p)) if rng.random() < 0.75 else None
            q = p.copy()
            if k is None:
                q += 0.3 * size * rng.normal(size=len(p))
            else:
                q[k] += size[k] * rng.normal()
            new = cost(q)
            count += 1
            if k is not None:
                size[k] *= 1.3 if new < now else 0.85
            if new <= now or new <= past[it % history]:
                p, now = q, new
            past[it % history] = now
            since += 1
            if now < best_cost:
                best, best_cost, since = p.copy(), now, 0
    return best, best_cost, count


class Counted:
    """A cost function that counts its evaluations, remembers the best point and stops at a budget."""

    class Spent(Exception):
        pass

    def __init__(self, cost, budget):
        self.cost, self.budget, self.count, self.best, self.best_cost = cost, budget, 0, None, np.inf

    def __call__(self, p):
        if self.count >= self.budget:
            raise Counted.Spent
        self.count += 1
        c = self.cost(np.array(p, float))
        if c < self.best_cost:
            self.best, self.best_cost = np.array(p, float), c
        return c


def slope(cost, u, h=1e-4):
    """Gradient of cost at u by central differences. The cost here is a smooth function of the
    parameters on a fixed set of sample points, with no noise, so differences are accurate.
    (Automatic differentiation would give the same numbers at the price of about three
    evaluations instead of two per parameter.)"""
    g = np.zeros(len(u))
    for k in range(len(u)):
        up, down = u.copy(), u.copy()
        up[k] += h
        down[k] -= h
        g[k] = (cost(up) - cost(down)) / (2 * h)
    return g


def by_gradient(cost, start, positive=(), budget=3000, method="normalised", rate=0.03):
    """Minimises cost(p) along its gradient, within `budget` evaluations; returns the best point.
    Parameters listed in `positive` are searched on a logarithmic scale (a step is then a
    relative change, and they cannot change sign). Three ways to use the gradient:
      "descent":    a step straight down the gradient, its length found by halving;
      "normalised": each parameter's step is divided by the typical size of its own gradient
                    (the rule known as Adam), so a parameter with a huge gradient and one with a
                    tiny gradient both move at a sensible pace;
      "curvature":  the BFGS method, which learns from successive gradients how the parameters
                    are scaled and coupled."""
    positive = list(positive)
    to_u = lambda p: np.array([np.log(x) if k in positive else x for k, x in enumerate(p)])
    to_p = lambda u: np.array([np.exp(x) if k in positive else x for k, x in enumerate(u)])
    f = Counted(lambda u: cost(to_p(u)), budget)
    u = to_u(np.array(start, float))
    try:
        now = f(u)
        if method == "curvature":
            from scipy.optimize import minimize
            minimize(f, u, jac=lambda v: slope(f, v), method="BFGS", options={"gtol": 1e-9, "maxiter": 10 ** 6})
        elif method == "normalised":
            m, v = np.zeros(len(u)), np.zeros(len(u))
            steps = budget // (2 * len(u) + 1)
            for k in range(1, steps + 1):
                g = slope(f, u)
                m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
                u = u - rate * (1 - 0.9 * k / steps) * (m / (1 - 0.9 ** k)) / (np.sqrt(v / (1 - 0.999 ** k)) + 1e-12)
                f(u)
        else:
            t = 0.1
            while True:
                g = slope(f, u)
                t *= 4
                while True:
                    new = f(u - t * g)
                    if new < now or t < 1e-12:
                        break
                    t /= 2
                if new < now:
                    u, now = u - t * g, new
    except Counted.Spent:
        pass
    return to_p(f.best), f.count


def improve(make, start, electrons, rounds=3, walkers=4000, report=None, steps=None, budget=None, polish=False, minimiser=None):
    """Better parameters for a guess. make(parameters) returns a Guess. The exact wavefunction has
    (H g)/g equal to the energy everywhere, so the spread of (H g)/g over a fixed set of sample
    points measures how wrong g is, and is minimised: by the simplex method of Nelder and Mead,
    or, when `steps` gives a first move size for each parameter, by `iterated` (with `polish`:
    followed by the simplex method from the point it found). The points are
    samples of the current g^2 and are redrawn each round, each round starting from the result
    of the one before. Nothing here knows the true energy."""
    from scipy.optimize import minimize
    p = np.array(start, float)
    for k in range(rounds + 1):
        x, local = sample(make(p), electrons, walkers, seed=k + 1)
        if report:
            report(k, p, local.mean(), local.std() / np.sqrt(walkers), local.std())
        if k == rounds:
            break

        def spread(q):
            try:
                e = local_energies(make(q), electrons, x)[0]
            except (ValueError, FloatingPointError, AssertionError):
                return 1e9
            return float(np.var(e)) if np.all(np.isfinite(e)) else 1e9

        if minimiser is not None:
            p = minimiser(spread, p)
            continue
        simplex = lambda p, n: minimize(spread, p, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": n, "maxfev": n}).x
        if steps is None:
            p = simplex(p, budget or 400)
        elif polish:                                             # two thirds of the evaluations to the search, one third to the polish
            found = iterated(spread, p, steps, iterations=(budget or 4500) // 3, starts=2, seed=k)[0]
            polished = simplex(found, (budget or 4500) // 3)
            p = polished if spread(polished) <= spread(found) else found
        else:
            p = iterated(spread, p, steps, iterations=(budget or 4500) // 3, seed=k)[0]
    return p


H3_NUCLEI = [(1, (0, 0, -1.757)), (1, (0, 0, 0)), (1, (0, 0, 1.757))]
# name: electrons (up, down), make(parameters), starting parameters, what they are
IMPROVE = {
    "He": ((1, 1), lambda p: Guess([(2, (0, 0, 0)), (0, (0, 0, 0))], (p[0], p[1]), p[3], up=[[1.0, p[2]]], down=[[1.0, p[2]]]),
           (2.0, 1.0, 0.0, 0.4), "two exponents, weight of the second, b"),
    "H2": ((1, 1), lambda p: Guess([(1, (0, 0, -0.7)), (1, (0, 0, 0.7))], p[0], p[1]), (1.19, 0.4), "exponent, b"),
    "H3": ((2, 1), lambda p: Guess(H3_NUCLEI, (p[0], p[1], p[0]), p[2], up=[[1.0, p[3], 1.0], [1.0, 0.0, -1.0]], down=[[1.0, p[3], 1.0]]),
           (1.2, 1.2, 0.4, 1.0), "exponent outer, exponent middle, b, weight of the middle atom"),
}


# the extra terms of the richer guess: powers (m, n, o) of the scaled distances electron-nucleus,
# electron-nucleus, electron-electron (see Guess). No power is 1, so the slopes at contact stay.
TERMS = ((2, 0, 0), (3, 0, 0), (4, 0, 0), (0, 0, 2), (0, 0, 3), (0, 0, 4), (2, 2, 0), (2, 0, 2))


def richer(make, count):
    """make(parameters) with the extra TERMS: the parameters are followed by the scale of the
    distances and one coefficient per term (the same on every nucleus with a charge)."""
    def made(p):
        assert min(p[count], *np.atleast_1d(make(p[:count]).zeta), make(p[:count]).b) > 0.05
        g = make(p[:count])
        on = (g.Z > 0).astype(float)
        g.powers, g.coeff, g.scale = np.array(TERMS, np.int32), np.outer(p[count + 1:], on), p[count]
        return g
    return made


def h3_orbitals(p):
    """H3 on a line with richer orbitals: on each atom a second exponent and a p function along
    the line, mixed in as far as the symmetry of each orbital allows. The two orbitals of the
    equal-spin pair (one even, one odd under reflection) decide where the guess is zero.
    p = exponents (outer, middle), b, weight of the middle atom [as the simple H3 guess], then
    second exponents (outer, middle), p exponents (outer, middle), and the weights: even orbital
    second-outer, second-middle, p-outer; odd orbital second-outer, p-outer, p-middle."""
    zo, zm, b, w, z2o, z2m, zpo, zpm, a2, b2, pg, u2, pu, pb = p
    centres = H3_NUCLEI + [(0, r) for _, r in H3_NUCLEI] * 2
    even = [1.0, w, 1.0, a2, b2, a2, pg, 0.0, -pg]
    odd = [1.0, 0.0, -1.0, u2, 0.0, -u2, pu, pb, pu]
    return Guess(centres, (zo, zm, zo, z2o, z2m, z2o, zpo, zpm, zpo), b, up=[even, odd], down=[even], kinds="sssssszzz")


H3_ORBITAL_START = (0.8, 0.8, 1.5, 1.5) + (0.0,) * 6


def orbitals_main(long_run):
    """H3 with the simple and the richer orbitals, both with the extra TERMS, improved by the
    iterated search and the simplex; the richer one starts from the result of the simple one.
    Then walkers: only the zero surfaces can change the answer."""
    reference = CASES["H3"][5]
    simple, start = IMPROVE["H3"][1], IMPROVE["H3"][2]
    steps = lambda n: [0.05] * n
    found = {}
    for label, m, p0 in (("simple orbitals", richer(simple, 4), tuple(start) + (1.0,) + (0.0,) * len(TERMS)),
                         ("richer orbitals", richer(h3_orbitals, 14), None)):
        if p0 is None:                                           # warm start: the simple result, new parts switched off
            q = found["simple orbitals"]
            p0 = tuple(q[:4]) + H3_ORBITAL_START + tuple(q[4:])
        t0 = time.time()
        trace = []
        best = improve(m, p0, 3, report=lambda *a: trace.append(a), steps=steps(len(p0)), budget=300 * len(p0), polish=True)
        found[label] = best
        for row in (trace[0], trace[-1]):
            print(f"  H3, {label}, round {row[0]}: guess alone {row[2]:.4f} ± {row[3]:.4f}  spread of (H g)/g {row[4]:.3f}")
        print(f"    parameters {np.round(best, 3)}   ({time.time() - t0:.0f} s)", flush=True)
        t0 = time.time()
        kw = dict(duration=200.0, walkers=20000) if long_run else dict(duration=30.0)
        runs, e0, s0 = extrapolated(m(best), 3, (0.02, 0.01), **kw)
        print(f"    walkers: " + ", ".join(f"step {tau:g}: {e:.5f} ± {s:.5f}" for tau, (e, s, _) in zip((0.02, 0.01), runs)) +
              f"; step -> 0: {e0:.5f} ± {s0:.5f}   (reference {reference:.6f}, difference {1e3 * (e0 - reference):+.2f} mEh)   "
              f"({time.time() - t0:.0f} s)", flush=True)


def improve_main(only=None):
    """Three ways to improve each guess: the simple form, the richer form with the simplex method,
    and the richer form with the iterated search. All get the same number of evaluations."""
    taus = (0.02, 0.01, 0.005)
    for name, ((n_up, n_dn), make, start, what) in IMPROVE.items():
        reference = CASES[name][5]
        rich, rich_start = richer(make, len(start)), tuple(start) + (1.0,) + (0.0,) * len(TERMS)
        first = [0.05] * (len(start) + 1) + [0.05] * len(TERMS)
        for label, m, p0, steps, polish in ((f"simple form ({what}), simplex", make, start, None, False),
                                            (f"richer form (+ scale and {len(TERMS)} terms), simplex", rich, rich_start, None, False),
                                            ("richer form, iterated search", rich, rich_start, first, False),
                                            ("richer form, iterated search then simplex", rich, rich_start, first, True)):
            if only and only not in label:
                continue
            t0 = time.time()
            trace = []
            best = improve(m, p0, n_up + n_dn, report=lambda *a: trace.append(a), steps=steps, budget=4500, polish=polish)
            for row in (trace[0], trace[-1]):
                print(f"  {name}, {label}, round {row[0]}: guess alone {row[2]:.4f} ± {row[3]:.4f}  spread of (H g)/g {row[4]:.3f}")
            print(f"    parameters {np.round(best, 3)}   ({time.time() - t0:.0f} s)", flush=True)
            for which, p in (("before", p0), ("after", best)) if m is make else (("after", best),):
                runs, e0, s0 = extrapolated(m(np.array(p, float)), n_up + n_dn, taus, duration=30.0)
                print(f"    walkers with the guess {which}: {e0:.5f} ± {s0:.5f}   (reference {reference:.6f}, "
                      f"difference {1e3 * (e0 - reference):+.2f} mEh)", flush=True)


def energy_rust(guess, electrons, tau, walkers=3000, duration=150.0, seed=1):
    """The same as `energy`, with the walker loop in the Rust kernel (cikernel/src/walk.rs)."""
    import mesh_ci
    lib = mesh_ci.kernel()
    warm, steps = int(20.0 / tau), int(duration / tau)
    out, alone = np.zeros(steps), ctypes.c_double(0.0)
    if lib.walk_run(*_c_guess(guess, electrons), tau, walkers, warm, steps, seed, mesh_ci._d(out), ctypes.byref(alone)):
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
                     linear=None, odd=None, shared=None):
    """Lowest energy of particles with the given masses and charges, nothing held fixed. The
    force between a pair is charge_i charge_j / r plus linear[(i, j)] r. With odd=(i, j) the
    guess is multiplied by z_i - z_j: the lowest state odd under reflection (a P state). pairs[(i, j)] = (a, b, c) gives the guess
    exp(sum of (a r + c r^2) / (1 + b r)) (see cikernel/src/walk.rs). `duration` and the
    warm-up of 20 are in units of `time_unit`. With shared=(i, j, k, kappa) the guess is multiplied by
    exp(-kap r_ik) + exp(-kap r_jk): particle k belongs to i and to j alike; with shared=(i, j, k, kappa,
    extra, range) the constant follows their distance R, kap = kappa + extra exp(-R / range). Returns (mean, error, mean of the guess alone)."""
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
    if shared is None:
        lib.walk_shared(-1, 0.0, 0.0, 1.0)
    else:
        lib.walk_shared((shared[0] * 8 + shared[1]) * 8 + shared[2], *(tuple(shared[3:]) + (0.0, 1.0))[:3])
    try:
        failed = lib.walk_particles(n, mesh_ci._d(m), mesh_ci._d(q), mesh_ci._d(A), mesh_ci._d(B), mesh_ci._d(C), mesh_ci._d(L),
                                    -1 if odd is None else odd[0] * n + odd[1], length, tau,
                                    walkers, warm, steps, seed, mesh_ci._d(out), ctypes.byref(alone))
    finally:
        lib.walk_shared(-1, 0.0, 0.0, 1.0)
    if failed:
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
                                   "variational: -0.262005070233, Frolov"),
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
# measured masses in GeV (Particle Data Group values as tabulated on Wikipedia's Quarkonium page; the PDG table itself
# was not read): spin-averaged 1S (J/psi 3.0969, eta_c 2.9834) and the centre of the 1P states (chi_c0, chi_c1, chi_c2)
MEASURED = {"1S": (3 * 3.0969 + 2.9834) / 4, "1P": (3.41475 + 3 * 3.51066 + 5 * 3.55620) / 9}


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
            "equilateral, side 1.65 bohr: -1.34383562502, quoted in arXiv:1105.5435"),
    "HeH+": ([(2, (0, 0, 0)), (1, (0, 0, 1.4632))], (1, 1), (2.116, 1.18), [[1.0, 0.25]], [[1.0, 0.25]], -2.978708,
             "Pachucki 2012: -2.978708311 at 1.463283 bohr (arXiv:1203.3927); run at 1.4632"),
    # three electrons: the wavefunction changes sign, the walkers stay on one side (upper bound)
    "H3": ([(1, (0, 0, -1.757)), (1, (0, 0, 0)), (1, (0, 0, 1.757))], (2, 1), 1.2, [[1.0, 1.0, 1.0], [1.0, 0.0, -1.0]],
           [[1.0, 1.0, 1.0]], -1.659162, "linear saddle of H + H2, 1.757 bohr: E(H) + E(H2) + barrier 9.61 ± 0.01 kcal/mol (Diedrich and Anderson 1992)"),
}
# reactions: name, {species: count} (products positive), reference in hartree
REACTIONS = [("H2 -> 2 H (binding energy)", {"H": 2, "H2": -1}, 0.174476),
             ("H + H2 -> H3 (barrier of the exchange reaction)", {"H3": 1, "H": -1, "H2": -1}, 0.015314),
             ("H3+ -> H2 + H+ (proton leaves)", {"H2": 1, "H3+": -1}, 0.169360),
             ("HeH+ -> He + H+ (proton leaves)", {"He": 1, "HeH+": -1}, 0.074982)]
STORE = Path(__file__).resolve().parent / "ci_cache" / "diffusion_mc.json"


def step_main(args):
    """Where does the error in the time step come from? Hydrogen with a deliberately wrong guess
    (exponent 0.9, so the slope at the nucleus is wrong; the exact energy is -1/2) and H2 with
    its improved guess, each at five steps, and at step 0.01 with one thing changed at a time:
    no limit on the rate, reaction also on rejected moves, a tenth of the walkers."""
    import mesh_ci
    lib = mesh_ci.kernel()
    kw = dict(duration=30.0) if "quick" in args else dict(duration=600.0, walkers=20000)
    stored = json.loads(IMPROVED_STORE.read_text()) if IMPROVED_STORE.exists() else {}
    cases = [("H, exponent 0.9", Guess(CASES["H"][0], 0.9), 1, -0.5)]
    cases.append(("H, exponent 0.9 and 1.1 mixed (slope right at the nucleus)",
                  Guess([(1, (0, 0, 0)), (0, (0, 0, 0))], (0.9, 1.1), up=[[1.0, 1.0]]), 1, -0.5))
    if "H2" in stored:
        cases.append(("H2, improved guess", richer(IMPROVE["H2"][1], 2)(np.array(stored["H2"]["parameters"])), 2, CASES["H2"][5]))
    if "HeH+" in stored and "all" in args:
        make = richer(lambda p: Guess(CASES["HeH+"][0], (p[0], p[1]), p[2], up=[[1.0, p[3]]], down=[[1.0, p[3]]]), 4)
        cases.append(("HeH+, improved guess", make(np.array(stored["HeH+"]["parameters"])), 2, CASES["HeH+"][5]))
    lib.walk_options(0.2, 0)                                     # the limit as it was before this study
    for name, guess, electrons, exact in cases:
        print(f"{name}   (exact or published {exact:.6f})", flush=True)
        for k, tau in enumerate((0.04, 0.02, 0.01, 0.005, 0.0025)):
            e, s, _ = energy_rust(guess, electrons, tau, seed=k + 11, **kw)
            print(f"  step {tau:g}: {e:.5f} ± {s:.5f}   difference {1e3 * (e - exact):+.3f} mEh", flush=True)
        small = dict(kw, walkers=kw.get("walkers", 3000) // 10)
        for label, clamp, rejected, use in (("no limit on the rate", 0.0, 0, kw), ("reaction also on rejected moves", 0.2, 1, kw),
                                            ("a tenth of the walkers", 0.2, 0, small)):
            lib.walk_options(clamp, rejected)
            e, s, _ = energy_rust(guess, electrons, 0.01, seed=31, **use)
            lib.walk_options(0.2, 0)                             # the study is of the old limit, 0.2
            print(f"  step 0.01, {label}: {e:.5f} ± {s:.5f}   difference {1e3 * (e - exact):+.3f} mEh", flush=True)


def improved_guess(name, report=None):
    """The guess of a species in CASES with the extra TERMS (for H3 also the richer orbitals),
    its parameters found by the iterated search and the simplex. Returns (make, parameters)."""
    nuclei = CASES[name][0]
    simple = {"H": (lambda p: Guess(nuclei, p[0], 0.4), (0.9,)),
              "He": IMPROVE["He"][1:3], "H2": IMPROVE["H2"][1:3], "H3": IMPROVE["H3"][1:3],
              "H3+": (lambda p: Guess(nuclei, p[0], p[1]), (1.35, 0.4)),
              "HeH+": (lambda p: Guess(nuclei, (p[0], p[1]), p[2], up=[[1.0, p[3]]], down=[[1.0, p[3]]]), (2.116, 1.18, 0.4, 0.25))}[name]
    electrons = sum(CASES[name][1])
    stages = [(richer(simple[0], len(simple[1])), tuple(simple[1]) + (1.0,) + (0.0,) * len(TERMS))]
    if name == "H3":                                             # then the richer orbitals, started from the simple result
        stages.append((richer(h3_orbitals, 14), None))
    best = None
    for make, start in stages:
        if start is None:
            start = tuple(best[:4]) + H3_ORBITAL_START + tuple(best[4:])
        best = improve(make, start, electrons, report=report, steps=[0.05] * len(start), budget=300 * len(start), polish=True)
    return make, best


def unlimited_main(names):
    """The long runs again with the stored improved guesses, but with no limit on the reaction
    rate: the study of the time step (`steps`) found that the limit raises the energy."""
    import mesh_ci
    mesh_ci.kernel().walk_options(0.0, 0)
    stored = json.loads(IMPROVED_STORE.read_text())
    store = STORE.with_name("diffusion_mc_unlimited.json")
    results = json.loads(store.read_text()) if store.exists() else {}
    taus = (0.02, 0.01, 0.005)
    for name in names or list(CASES):
        t0 = time.time()
        nuclei = CASES[name][0]
        simple = {"H": (lambda p: Guess(nuclei, p[0], 0.4), 1), "He": (IMPROVE["He"][1], len(IMPROVE["He"][2])),
                  "H2": (IMPROVE["H2"][1], len(IMPROVE["H2"][2])), "H3": (h3_orbitals, 14),
                  "H3+": (lambda p: Guess(nuclei, p[0], p[1]), 2),
                  "HeH+": (lambda p: Guess(nuclei, (p[0], p[1]), p[2], up=[[1.0, p[3]]], down=[[1.0, p[3]]]), 4)}[name]
        guess = richer(*simple)(np.array(stored[name]["parameters"]))
        runs, e0, s0 = extrapolated(guess, sum(CASES[name][1]), taus, duration=600.0, walkers=20000)
        reference = CASES[name][5]
        results[name] = {"energy": e0, "error": s0, "steps": [list(r[:2]) for r in runs]}
        store.write_text(json.dumps(results, indent=1))
        print(f"{name}, no limit on the rate: " + ", ".join(f"step {tau:g}: {e:.5f} ± {s:.5f}" for tau, (e, s, _) in zip(taus, runs)) +
              f"; step -> 0: {e0:.5f} ± {s0:.5f}   published {reference:.6f}, difference {1e3 * (e0 - reference):+.2f} mEh = "
              f"{(e0 - reference) / s0:+.1f} errors   (with the limit: {stored[name]['energy']:.5f} ± {stored[name]['error']:.5f})   ({time.time() - t0:.0f} s)", flush=True)


def gradient_main():
    """The richer guesses of He, H2 and H3 improved four ways with the same number of evaluations:
    search then simplex (no gradients), and the three uses of the gradient in `by_gradient`.
    What is printed is the spread on fresh sample points, which the optimiser has not seen."""
    simple_h3 = richer(IMPROVE["H3"][1], 4)
    cases = {"He": (2, richer(IMPROVE["He"][1], 4), tuple(IMPROVE["He"][2]) + (1.0,) + (0.0,) * len(TERMS), (0, 1, 3, 4)),
             "H2": (2, richer(IMPROVE["H2"][1], 2), tuple(IMPROVE["H2"][2]) + (1.0,) + (0.0,) * len(TERMS), (0, 1, 2)),
             "H3, richer orbitals": (3, richer(h3_orbitals, 14), None, (0, 1, 2, 4, 5, 6, 7, 14))}
    for name, (electrons, make, start, positive) in cases.items():
        if start is None:                                        # as in `orbitals`: from the improved simple guess
            q = improve(simple_h3, tuple(IMPROVE["H3"][2]) + (1.0,) + (0.0,) * len(TERMS), 3, steps=[0.05] * 13, budget=3900, polish=True)
            start = tuple(q[:4]) + H3_ORBITAL_START + tuple(q[4:])
        budget = 300 * len(start)
        print(f"{name}: {len(start)} parameters, {budget} evaluations per round, three rounds", flush=True)
        ways = [("search then simplex (no gradient)", dict(steps=[0.05] * len(start), budget=budget, polish=True))]
        for method, label in (("descent", "gradient, plain descent"), ("normalised", "gradient, each parameter normalised"),
                              ("curvature", "gradient with curvature (BFGS)")):
            for log in (False, True):
                ways.append((label + (", positive ones on a logarithmic scale" if log else ""),
                             dict(minimiser=lambda f, p, method=method, log=log: by_gradient(f, p, positive if log else (), budget, method)[0])))
        for label, kw in ways:
            t0 = time.time()
            trace = []
            improve(make, start, electrons, report=lambda *a: trace.append(a), **kw)
            print(f"  {label}: spread of (H g)/g " + " -> ".join(f"{row[4]:.3f}" for row in trace) +
                  f"   guess alone {trace[-1][2]:.4f} ± {trace[-1][3]:.4f}   ({time.time() - t0:.0f} s)", flush=True)


IMPROVED_STORE = STORE.with_name("diffusion_mc_improved.json")


def improved_main(names, quick):
    """Long runs of the species with improved guesses. Nothing here uses a reference value: the
    parameters come from the spread of (H g)/g, the energies from the walkers."""
    kw = dict(duration=30.0) if quick else dict(duration=600.0, walkers=20000)
    taus = (0.02, 0.01, 0.005)
    results = json.loads(IMPROVED_STORE.read_text()) if IMPROVED_STORE.exists() and not quick else {}
    for name in names or list(CASES):
        t0 = time.time()
        trace = []
        make, best = improved_guess(name, report=lambda *a: trace.append(a))
        runs, e0, s0 = extrapolated(make(best), sum(CASES[name][1]), taus, **kw)
        reference, source = CASES[name][5:]
        results[name] = {"energy": e0, "error": s0, "steps": [list(r[:2]) for r in runs], "parameters": [float(x) for x in best],
                         "spread": trace[-1][4], "guess_alone": trace[-1][2]}
        print(f"{name}: spread of (H g)/g {trace[0][4]:.3f} -> {trace[-1][4]:.3f}, the guess alone {trace[-1][2]:.4f}; " +
              ", ".join(f"step {tau:g}: {e:.5f} ± {s:.5f}" for tau, (e, s, _) in zip(taus, runs)) +
              f"; step -> 0: {e0:.5f} ± {s0:.5f}   published {reference:.6f} ({source}), "
              f"difference {1e3 * (e0 - reference):+.2f} mEh = {(e0 - reference) / s0:+.1f} errors   ({time.time() - t0:.0f} s)", flush=True)
        if not quick:
            IMPROVED_STORE.parent.mkdir(exist_ok=True)
            IMPROVED_STORE.write_text(json.dumps(results, indent=1))
    for name, parts, reference in REACTIONS:
        if all(k in results for k in parts):
            d = sum(k * results[n]["energy"] for n, k in parts.items())
            s = np.sqrt(sum((k * results[n]["error"]) ** 2 for n, k in parts.items()))
            print(f"{name}: {1e3 * d:.2f} ± {1e3 * s:.2f} mEh = {d * HARTREE_EV:.3f} ± {s * HARTREE_EV:.3f} eV   "
                  f"(published {1e3 * reference:.2f} mEh)")


def main():
    args = sys.argv[1:]
    quick = "quick" in args
    if "particles" in args:
        return particles_main(quick)
    if "improve" in args:
        return improve_main("then simplex" if "polish" in args else None)
    if "steps" in args:
        return step_main(args)
    if "unlimited" in args:
        return unlimited_main([a for a in args if a in CASES])
    if "gradient" in args:
        return gradient_main()
    if "improved" in args:
        return improved_main([a for a in args if a in CASES], "quick" in args)
    if "orbitals" in args:
        return orbitals_main("long" in args)
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

#!/usr/bin/env python3
"""Selected CI on a free list of determinants: a mesh that is fine only where
the wavefunction needs it.

The earlier solvers work on a grid: every chosen arrangement of the spin-up
electrons paired with every chosen arrangement of the spin-down electrons.
Here any determinant can be in the list on its own, so an important spin-up
arrangement can have thousands of partners and an unimportant one a handful.

The loop is the one a route planner uses: solve with what is known, ask which
missing pieces the current answer pulls on hardest, add those, and solve again
starting from the previous answer.

    grow:   candidates ranked by (sum_i H_ji c_i)^2 / (E - H_jj)
    solve:  Davidson iteration, the Hamiltonian applied on the fly

The inner loops are in Rust (analyzer/cikernel, built with cargo on first use)
and called through ctypes.
"""
import ctypes
import subprocess
import sys
from pathlib import Path

import numpy as np

KERNEL = Path(__file__).resolve().parent / "cikernel"
_lib = None


def kernel():
    """The compiled kernel, built on first use (needs cargo)."""
    global _lib
    if _lib is None:
        name = "libcikernel.dylib" if sys.platform == "darwin" else "libcikernel.so"
        path = KERNEL / "target" / "release" / name
        source = max(p.stat().st_mtime for p in [KERNEL / "Cargo.toml", *(KERNEL / "src").glob("*.rs")])
        if not path.exists() or path.stat().st_mtime < source:
            subprocess.run(["cargo", "build", "--release", "--target-dir", "target"], cwd=KERNEL, check=True)
        lib = ctypes.CDLL(str(path))
        ptr, u64, dbl, size = ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_double), ctypes.c_size_t
        for w in ("", "_wide"):                             # 64-bit and 128-bit strings
            f = lambda name: getattr(lib, name + w)
            f("ci_new").restype = ptr
            f("ci_new").argtypes = [size, dbl, dbl, size, u64, u64]
            f("ci_free").argtypes = [ptr]
            f("ci_diag").argtypes = [ptr, dbl]
            f("ci_sigma").argtypes = [ptr, dbl, dbl]
            f("ci_rdm1").argtypes = [ptr, dbl, dbl]
            f("ci_plan").restype = size
            f("ci_plan").argtypes = [ptr, size, dbl]
            f("ci_pt2").restype = size
            f("ci_pt2").argtypes = [ptr, dbl, ctypes.c_double, ctypes.c_double, ctypes.c_uint64, ctypes.c_uint64, dbl]
            f("ci_candidates").restype = size
            f("ci_candidates").argtypes = [ptr, dbl, ctypes.c_double, ctypes.c_double, size, u64, u64, dbl, dbl,
                                           ctypes.POINTER(size)]
        lib.walk_run.restype = ctypes.c_int
        guess = [size, dbl, dbl, dbl, size, size, dbl, dbl, ctypes.c_double, size, ctypes.POINTER(ctypes.c_int32), dbl, ctypes.c_double]
        lib.walk_run.argtypes = guess + [ctypes.c_double, size, size, size, ctypes.c_uint64, dbl, dbl]
        lib.walk_sample.restype = lib.walk_local.restype = ctypes.c_int
        lib.walk_sample.argtypes = guess + [ctypes.c_double, size, size, ctypes.c_uint64, dbl, dbl]
        lib.walk_local.argtypes = guess + [size, dbl, dbl, dbl]
        lib.walk_particles.restype = ctypes.c_int
        lib.walk_particles.argtypes = [size, dbl, dbl, dbl, dbl, dbl, dbl, ctypes.c_int64] + [ctypes.c_double] * 2 + [size, size, size, ctypes.c_uint64, dbl, dbl]
        _lib = lib
    return _lib


def _d(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _u(x):
    return x.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _memory(label, size):
    """With CI_MEMORY_LOG set: print the memory in use now and the peak so far, after each phase."""
    import os
    if not os.environ.get("CI_MEMORY_LOG"):
        return
    import resource
    now = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout or 0) / 1e6
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    print(f"      memory after {label:10s} ({size} determinants): now {now:.2f} GB, peak {peak:.2f} GB", flush=True)


def strings(values, wide):
    """Bit strings (Python integers) as an array: one 64-bit word each, or
    with wide=True two words each (lowest first), for up to 128 orbitals."""
    mask = (1 << 64) - 1
    if wide:
        return np.array([[v & mask, v >> 64] for v in values], dtype=np.uint64).reshape(len(values), 2)
    return np.array(values, dtype=np.uint64)


def electrons(string):
    """Number of set bits in one string of either width."""
    return sum(bin(int(w)).count("1") for w in np.atleast_1d(string))


class Mesh:
    """A list of determinants with the Hamiltonian acting on it. `alpha` and
    `beta` are arrays of 64-bit strings, or of shape (determinants, 2) for
    128-bit strings (needed above 64 orbitals)."""

    def __init__(self, h, eri, alpha, beta):
        self.lib = kernel()
        self.n = h.shape[0]
        self.wide = np.ndim(alpha) == 2
        if self.n > (128 if self.wide else 64):
            raise ValueError("at most 64 orbitals with 64-bit strings and 128 with wide ones")
        self.h = np.ascontiguousarray(h, dtype=np.float64)
        self.g = np.ascontiguousarray(eri, dtype=np.float64)
        self.alpha = np.ascontiguousarray(alpha, dtype=np.uint64)
        self.beta = np.ascontiguousarray(beta, dtype=np.uint64)
        self.handle = self._f("ci_new")(self.n, _d(self.h), _d(self.g), len(self.alpha), _u(self.alpha), _u(self.beta))
        self.diag = np.zeros(len(self.alpha))
        self._f("ci_diag")(self.handle, _d(self.diag))
        self.cached, self.work_saved = 0, 0.0

    def _f(self, name):
        # a method, not a lambda stored on the object: a lambda that refers to `self` makes a
        # reference cycle, and then `del mesh` frees nothing until Python's cycle collector
        # happens to run, so the index and plan of earlier steps (gigabytes) stay alive
        return getattr(self.lib, name + ("_wide" if self.wide else ""))

    def plan(self, budget_bytes=2_000_000_000):
        """Measure, for every determinant, what it costs to find its partners,
        and remember the partner lists where that pays, within the memory
        budget. Returns (determinants cached, share of the search work removed)."""
        saved = ctypes.c_double(0.0)
        self.cached = self._f("ci_plan")(self.handle, int(budget_bytes), ctypes.byref(saved))
        self.work_saved = saved.value
        return self.cached, self.work_saved

    def __del__(self):
        if getattr(self, "handle", None):
            self._f("ci_free")(self.handle)
            self.handle = None

    def sigma(self, c):
        out = np.empty_like(c)
        self._f("ci_sigma")(self.handle, _d(c), _d(out))
        return out

    def density(self, c):
        """One-particle density matrix <c| a+_p a_q |c>, both spins added."""
        out = np.zeros((self.n, self.n))
        self._f("ci_rdm1")(self.handle, _d(np.ascontiguousarray(c, dtype=np.float64)), _d(out))
        return out

    def candidates(self, c, energy, eps, count):
        """Up to `count` determinants outside the list, best first, with the
        energy each is estimated to bring, and the estimate summed over all
        that were found."""
        shape = (count, 2) if self.wide else count
        a, b, gain = np.zeros(shape, np.uint64), np.zeros(shape, np.uint64), np.zeros(count)
        total, written = ctypes.c_double(0.0), ctypes.c_size_t(0)
        found = self._f("ci_candidates")(self.handle, _d(c), energy, eps, count, _u(a), _u(b), _d(gain),
                                       ctypes.byref(total), ctypes.byref(written))
        k = written.value
        return a[:k], b[:k], gain[:k], total.value, found

    def rest(self, c, energy, eps, parts=1, report=None):
        """Second-order estimate of the energy of everything outside the list
        that couples to `c` more strongly than `eps`, computed in `parts`
        pieces to bound the memory. Returns (estimate, determinants looked at).
        Lowering eps makes the estimate more complete; it is not a bound."""
        total, found = 0.0, 0
        for part in range(parts):
            piece = ctypes.c_double(0.0)
            found += self._f("ci_pt2")(self.handle, _d(c), energy, eps, parts, part, ctypes.byref(piece))
            total += piece.value
            if report:
                report(part + 1, parts, total, found)
        return total, found

    def solve(self, start=None, tol=1e-6, max_vectors=8, max_iter=400):
        """Lowest energy and its vector (Davidson)."""
        if start is None:
            start = np.zeros(len(self.alpha))
            start[np.argmin(self.diag)] = 1.0
        x = start / np.linalg.norm(start)
        vectors, sigmas = [x], [self.sigma(x)]
        for _ in range(max_iter):
            m = len(vectors)
            G = np.array([[vectors[i] @ sigmas[j] for j in range(m)] for i in range(m)])
            w, y = np.linalg.eigh(0.5 * (G + G.T))
            energy, y = w[0], y[:, 0]
            x = sum(c * v for c, v in zip(y, vectors))
            residual = sum(c * s for c, s in zip(y, sigmas)) - energy * x
            if np.linalg.norm(residual) < tol:
                return float(energy), x
            if m >= max_vectors:
                x /= np.linalg.norm(x)
                vectors, sigmas = [x], [residual + energy * x]
            denom = energy - self.diag
            denom[np.abs(denom) < 1e-3] = -1e-3
            new = residual / denom
            for _twice in range(2):
                for v in vectors:
                    new -= (v @ new) * v
            new /= np.linalg.norm(new)
            vectors.append(new); sigmas.append(self.sigma(new))
        raise RuntimeError("mesh CI did not converge")


def rest_table(mesh, c, energy, thresholds=(1e-4, 3e-5, 1e-5, 3e-6, 1e-6), memory=3e9, report=None):
    """The second-order estimate of what the list leaves out, for a series of
    ever lower thresholds, so that its convergence can be seen. The work is
    split into as many pieces as the memory allows. Returns a list of
    (threshold, estimate, determinants looked at)."""
    table, found = [], 0
    for eps in thresholds:
        expected = max(found, len(c)) * 8                  # found grows by a few times per step
        parts = max(1, int(np.ceil(expected * 10 * 24 / memory)))
        total, found = mesh.rest(c, energy, eps, parts)
        table.append((eps, total, found))
        if report:
            report(eps, total, found, parts)
    return table


def save_state(path, alpha, beta, c, history, mesh=None, energy=None):
    """Determinants in the order they were added, coefficients in single
    precision. A starting point for a later run, not a result.

    With `mesh` and `energy` (64-bit strings only) the compact form is
    written instead, about a third of the size. It needs the Hamiltonian to
    be read back, so it is used where the file is keyed by the integrals:
    - The list is cut into a first sixteenth and blocks that double, and the strings are
      sorted inside each block and stored as differences.
    - The coefficients of a block are predicted from the blocks before it,
      c_j = (sum_i H_ji c_i) / (E - H_jj), and only what the prediction
      misses is stored, on a logarithmic scale with a threshold. Reading
      repeats the prediction and adds the stored remainder: a transform
      whose loss is kept next to it, so that it can be undone."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if mesh is None or np.ndim(alpha) != 1:
        np.savez_compressed(path, alpha=alpha, beta=beta, c=c.astype(np.float32), history=np.array(history, dtype=float))
        return
    import quantum
    size = len(c)
    bounds = _blocks(size)
    order = np.concatenate([lo + np.lexsort((beta[lo:hi], alpha[lo:hi])) for lo, hi in zip([0] + bounds[:-1], bounds)])
    keep = 1e-5 * np.abs(c).max()
    codes, known = np.zeros(size, dtype=np.int16), np.zeros(size)
    known[:bounds[0]] = c[:bounds[0]].astype(np.float32)
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        guess = _predict(mesh, known, energy, lo, hi)
        codes[lo:hi] = quantum._log_code(c[lo:hi] - guess, keep)
        known[lo:hi] = guess + quantum._log_decode(codes[lo:hi])
    np.savez_compressed(path, packed=2, size=size, energy=energy, history=np.array(history, dtype=float),
                        alpha_steps=np.diff(alpha[order], prepend=np.uint64(0)),
                        beta_steps=np.diff(beta[order], prepend=np.uint64(0)),
                        head=c[order[:bounds[0]]].astype(np.float32), codes=codes[order])


def _blocks(size):
    """Ends of the blocks: the first sixteenth of the list (stored as it is),
    then blocks that double. Each block costs one application of the
    Hamiltonian to write and one to read."""
    ends, end = [], max(256, size // 16)
    while end < size:
        ends.append(end)
        end *= 2
    return ends + [size]


def _predict(mesh, known, energy, lo, hi):
    """First-order estimate of the coefficients lo..hi from those before lo."""
    source = known.copy()
    source[lo:] = 0.0
    return mesh.sigma(source)[lo:hi] / np.minimum(energy - mesh.diag[lo:hi], -1e-6)


def _unpack(mesh, packed):
    """Coefficients of a compact state, for the first len(mesh.alpha) determinants."""
    import quantum
    size = len(mesh.alpha)
    bounds = [b for b in _blocks(int(packed["size"])) if b < size] + [size]
    c = np.zeros(size)
    c[:bounds[0]] = packed["head"][:bounds[0]]
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        c[lo:hi] = _predict(mesh, c, float(packed["energy"]), lo, hi) + quantum._log_decode(packed["codes"][lo:hi])
    return c


def exchange(h, eri, mesh, c, energy, eps, swap=0.25, plan_bytes=2_000_000_000):
    """One round of exchange at fixed size. Leaving determinant i out of the
    list costs about c_i^2 (H_ii - E); taking determinant j in gains about
    (sum_i H_ji c_i)^2 / (H_jj - E). The cheapest inside are paired with the
    best outside and exchanged for as long as the gain exceeds the cost, at
    most the fraction `swap` of the list. Both numbers are second-order
    estimates, so the new list is solved and only its energy counts.
    Returns (mesh, c, energy, rest, exchanged) or None if nothing is worth exchanging."""
    size = len(c)
    most = max(1, int(swap * size))
    new_a, new_b, gain, rest, found = mesh.candidates(c, energy, eps, most)
    for _ in range(12):
        # a gain g comes from couplings of size about sqrt(g); the search must follow
        # couplings well below that of the weakest gain considered, or the ranking is noise
        needed = 0.1 * np.sqrt(-gain[-1]) if len(gain) == most else eps * 0.3
        if eps <= needed or eps < 1e-9:
            break
        eps = min(needed, eps * 0.3)
        new_a, new_b, gain, rest, found = mesh.candidates(c, energy, eps, most)
    cost = c ** 2 * (mesh.diag - energy)
    cheapest = np.argsort(cost)[:len(gain)]
    worth = -gain > cost[cheapest]                         # gain is negative, best first; cost rises
    m = int(np.argmin(worth)) if not worth.all() else len(worth)
    if m == 0:
        return None
    keep = np.ones(size, bool)
    keep[cheapest[:m]] = False
    alpha = np.concatenate([mesh.alpha[keep], new_a[:m]])
    beta = np.concatenate([mesh.beta[keep], new_b[:m]])
    start = np.concatenate([c[keep], np.zeros(m)])
    new = Mesh(h, eri, alpha, beta)
    if plan_bytes:
        new.plan(plan_bytes)
    new_energy, new_c = new.solve(start=start)
    if new_energy >= energy:
        return None
    return new, new_c, new_energy, rest, m


def natural_orbitals(h, eri, n_alpha, n_beta, pilot=20000, rounds=1, keep=0.0, report=None, wide=None):
    """Orbitals learned from a small run. A pilot list of `pilot` determinants
    is solved, the one-particle density matrix of its wavefunction is
    diagonalised, and the orbitals are rotated to its eigenvectors (the
    natural orbitals), most occupied first. In these orbitals a list of a
    given size reaches a lower energy. Repeating (`rounds`) learns from the
    improved wavefunction. Orbitals occupied by less than `keep` electrons are
    dropped. Returns (h, eri, rotation, occupations) in the new orbitals."""
    import quantum
    U_total = np.eye(h.shape[0])
    occupation = None
    for _ in range(rounds):
        mesh, c = _solve_list(h, eri, n_alpha, n_beta, pilot, wide)
        gamma = mesh.density(c)
        del mesh
        occupation, U = np.linalg.eigh(0.5 * (gamma + gamma.T))
        order = np.argsort(-occupation)
        occupation, U = occupation[order], U[:, order]
        if report:
            report(occupation)
        h, eri = U.T @ h @ U, quantum._transform(eri, U)
        U_total = U_total @ U
    if keep > 0.0:
        m = max(int(np.sum(occupation >= keep)), n_alpha + 1)
        h, eri, U_total, occupation = h[:m, :m], np.ascontiguousarray(eri[:m, :m, :m, :m]), U_total[:, :m], occupation[:m]
    return h, eri, U_total, occupation


def _solve_list(h, eri, n_alpha, n_beta, size, wide=None):
    """Grow a list to `size` determinants and return (mesh, coefficients)."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "pilot.npz"
        mesh_ci(h, eri, n_alpha, n_beta, max_determinants=size, store=path, wide=wide, plan_bytes=0)
        with np.load(path) as f:
            alpha, beta, c = f["alpha"], f["beta"], f["c"].astype(float)
    mesh = Mesh(h, eri, alpha, beta)
    _, c = mesh.solve(start=c)
    return mesh, c


def mesh_ci(h, eri, n_alpha, n_beta, e_nuc=0.0, max_determinants=1_000_000, growth=2.0, eps=1e-4, tol=0.0,
            report=None, store=None, plan_bytes=2_000_000_000, wide=None, refine=0, swap=0.25, nearest=False):
    """Grow a list of determinants from the reference until it holds
    `max_determinants`, or until a step lowers the energy by less than `tol`,
    or until nothing more is found (then the result is full CI).

    With refine > 0 the full list is then improved without growing it, in at
    most `refine` rounds: the determinants that carry least are exchanged for
    better ones from outside (at most the fraction `swap` per round), and the
    problem is solved again. See `exchange`.

    With store="content" and nearest=True, a problem that has not been solved
    before starts from the list of the most similar one that has (same
    numbers of orbitals and electrons, closest one-electron energies and
    Coulomb integrals), instead of from a single determinant. The list is a
    starting point only: it is solved again in the new integrals.

    Returns (energy, estimate of what is missing, history). Each history
    entry is (determinants, energy, estimate); the energy is an upper bound,
    the estimate (second-order perturbation theory over the candidates that
    were looked at) is not."""
    wide = h.shape[0] > 64 if wide is None else wide      # 128-bit strings; forced on only for testing
    alpha, beta = strings([(1 << n_alpha) - 1], wide), strings([(1 << n_beta) - 1], wide)
    compact = isinstance(store, str) and store == "content" and not wide
    if isinstance(store, str) and store == "content":      # keyed by the integrals: the same problem finds its list again
        import content_cache
        store = content_cache.path("state:1", np.ascontiguousarray(h), np.ascontiguousarray(eri), n_alpha, n_beta,
                                   suffix=".npz") if content_cache.ENABLED else None
    c, history = np.ones(1), []
    packed = None                                          # a compact state whose coefficients are still to be decoded
    unchanged = False                                      # the list is exactly the one in the compact file
    source = store
    if compact and store is not None:
        import content_cache
        shape_key = f"{h.shape[0]}_{n_alpha}_{n_beta}"
        descriptor = np.concatenate([np.diag(h), np.einsum("iijj->ij", eri).ravel()])
        if nearest and not Path(store).exists():
            found = content_cache.nearest("state", shape_key, descriptor)
            if found:
                source = found[0]
    if source and Path(source).exists():
        with np.load(source) as f:
            if "packed" in f and not wide:
                k = min(int(f["size"]), max_determinants)
                a, b = np.cumsum(f["alpha_steps"], dtype=np.uint64)[:k], np.cumsum(f["beta_steps"], dtype=np.uint64)[:k]
                if k and electrons(a[0]) == n_alpha and electrons(b[0]) == n_beta:
                    alpha, beta, c = a, b, None
                    unchanged = k == int(f["size"]) and source == store
                    packed = {name: f[name] for name in ("size", "energy", "head", "codes")}
            elif "packed" not in f:
                k = min(len(f["alpha"]), max_determinants)
                if (k and (f["alpha"].ndim == 2) == wide and electrons(f["alpha"][0]) == n_alpha
                        and electrons(f["beta"][0]) == n_beta):
                    alpha, beta, c = f["alpha"][:k].copy(), f["beta"][:k].copy(), f["c"][:k].astype(float)
    while True:
        mesh = Mesh(h, eri, alpha, beta)
        _memory("index", len(alpha))
        if plan_bytes:
            mesh.plan(plan_bytes)
            _memory("plan", len(alpha))
        if packed is not None:
            c, packed = _unpack(mesh, packed), None
        energy, c = mesh.solve(start=c)
        _memory("solve", len(alpha))
        room = min(max(int(len(alpha) * (growth - 1.0)), 200), max_determinants - len(alpha))
        rest, step_eps = 0.0, eps
        new_a = new_b = strings([], wide)
        for _ in range(12):                                # look further out until enough candidates turn up
            new_a, new_b, gain, rest, found = mesh.candidates(c, energy, step_eps, max(room, 1))
            _memory(f"search {step_eps:.0e}"[:10], f"{found} candidates, {len(alpha)}")
            if found >= 3 * room or step_eps < 1e-9:
                break
            step_eps *= 0.3
        history.append((len(alpha), energy + e_nuc, rest))
        if store and not unchanged:                        # a plain file per step, in case the run is cut short
            save_state(store, alpha, beta, c, history)
        if report:
            report(*history[-1])
        if room <= 0 or len(new_a) == 0 or (len(history) > 1 and history[-2][1] - history[-1][1] < tol):
            if room <= 0:
                for _ in range(refine):
                    better = exchange(h, eri, mesh, c, energy, step_eps, swap, plan_bytes)
                    if better is None or energy - better[2] <= tol:
                        break
                    mesh, c, energy, rest, swapped = better
                    unchanged = False
                    alpha, beta = mesh.alpha, mesh.beta
                    history.append((len(alpha), energy + e_nuc, rest))
                    if store:
                        save_state(store, alpha, beta, c, history)
                    if report:
                        report(*history[-1])
            break
        alpha = np.concatenate([alpha, new_a[:room]])
        beta = np.concatenate([beta, new_b[:room]])
        unchanged = False
        c = np.concatenate([c, np.zeros(min(room, len(new_a)))])
        del mesh
    if store and compact and not unchanged:
        save_state(store, alpha, beta, c, history, mesh, energy)
    if store and compact:
        content_cache.describe(store, shape_key, descriptor)
    return energy + e_nuc, rest, history

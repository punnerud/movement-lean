#!/usr/bin/env python3
"""The zeros of Riemann's zeta function, computed here and compared with the eigenvalues of
self-adjoint operators (the idea of Hilbert and Polya: if the zeros are 1/2 + i E with E the
eigenvalues of a self-adjoint operator, they all lie on the line; Lean: `symmetric_eigenvalue_real`,
`zero_on_line_of_symmetric`).

Nothing is taken from tables. zeta(s) comes from its defining sum with the Euler-Maclaurin
correction, the zeros on the line from sign changes of Z(t) = exp(i theta(t)) zeta(1/2 + i t),
which is real. Three tests:
  1. the number of zeros on the line below T against the number of zeros in the whole strip,
     counted independently from the change of the argument of zeta around the strip: if they
     agree, every zero below T is on the line (a numerical check, not a proof);
  2. the gaps between neighbouring zeros against the gaps between eigenvalues of random
     self-adjoint matrices built here: complex Hermitian, real symmetric, and no operator at all
     (independent points);
  3. the same for the eigenvalues of each kind, so the yardstick itself is measured, not quoted.

    python3 analyzer/zeta_zeros.py [T]      # default T = 5000: about 4500 zeros, a few minutes
"""
import sys
import time

import numpy as np

BERNOULLI = (1 / 6, -1 / 30, 1 / 42, -1 / 30, 5 / 66, -691 / 2730)     # B2 ... B12


def zeta(s):
    """zeta(s) for an array of complex s with real part > 0: the sum up to N, the integral of the
    rest, and the Euler-Maclaurin corrections. N is taken well above |s|."""
    s = np.asarray(s, complex)
    N = int(1.2 * np.abs(s).max()) + 30
    n = np.arange(1, N)
    total = np.zeros(s.shape, complex)
    for start in range(0, s.size, 2000):                         # in pieces, to bound the memory
        part = s.ravel()[start:start + 2000]
        total.ravel()[start:start + 2000] = np.exp(-part[:, None] * np.log(n)[None, :]).sum(1)
    tail = N ** (1 - s) / (s - 1) + 0.5 * N ** (-s)
    term = s * N ** (-s - 1)                                     # s (s+1) ... (s+2k-2) N^(-s-2k+1)
    factorial = 2.0
    for k, b in enumerate(BERNOULLI, start=1):
        tail += b / factorial * term
        term = term * (s + 2 * k - 1) * (s + 2 * k) / N ** 2
        factorial *= (2 * k + 1) * (2 * k + 2)
    return total + tail


def theta(t):
    """The phase that makes Z real: Im ln Gamma(1/4 + i t / 2) - (t / 2) ln pi, from Stirling's series."""
    return t / 2 * np.log(t / (2 * np.pi)) - t / 2 - np.pi / 8 + 1 / (48 * t) + 7 / (5760 * t ** 3) + 31 / (80640 * t ** 5)


def Z(t):
    return (np.exp(1j * theta(t)) * zeta(0.5 + 1j * t)).real


def zeros_on_line(T, start=10.0):
    """Zeros of Z between `start` and T: sign changes on a fine grid, then bisection."""
    spacing = 2 * np.pi / np.log(T / (2 * np.pi))
    t = np.arange(start, T, spacing / 40)
    z = Z(t)
    k = np.nonzero(np.sign(z[1:]) != np.sign(z[:-1]))[0]
    lo, hi, flo = t[k], t[k + 1], z[k]
    for _ in range(46):
        mid = 0.5 * (lo + hi)
        fm = Z(mid)
        left = np.sign(fm) == np.sign(flo)
        lo, hi, flo = np.where(left, mid, lo), np.where(left, hi, mid), np.where(left, fm, flo)
    return 0.5 * (lo + hi)


def zeros_in_strip(T):
    """Number of zeros with 0 < Im < T anywhere in the strip 0 < Re < 1, from the change of the
    argument: N(T) = 1 + theta(T) / pi + arg zeta(1/2 + i T) / pi, with the argument followed
    continuously from 2 + i T, where it is small (|zeta - 1| < 0.65 on the line Re = 2)."""
    x = np.linspace(2.0, 0.5, 6001)
    values = zeta(x + 1j * T)
    arg = np.unwrap(np.angle(values))
    return 1 + theta(T) / np.pi + arg[-1] / np.pi, float(np.abs(np.diff(arg)).max())


def gap_ratio(levels):
    """Mean of min(gap, next gap) / max(gap, next gap): needs no knowledge of the density."""
    gaps = np.diff(np.sort(levels))
    r = np.minimum(gaps[1:], gaps[:-1]) / np.maximum(gaps[1:], gaps[:-1])
    return r


def matrix_ratios(kind, size=600, count=60, seed=1):
    """Gap ratios of the eigenvalues (middle half) of random self-adjoint matrices."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        if kind == "none":
            levels = np.sort(rng.random(size))
        else:
            a = rng.normal(size=(size, size))
            if kind == "complex":
                a = a + 1j * rng.normal(size=(size, size))
            levels = np.linalg.eigvalsh(a + a.conj().T)
        out.append(gap_ratio(levels[size // 4: 3 * size // 4]))
    return np.concatenate(out)


def distance(a, b):
    """Largest difference between the two cumulative distributions (Kolmogorov-Smirnov)."""
    grid = np.sort(np.concatenate([a, b]))
    return float(np.abs(np.searchsorted(np.sort(a), grid, "right") / len(a) - np.searchsorted(np.sort(b), grid, "right") / len(b)).max())


def main():
    T = float(sys.argv[1]) if len(sys.argv) > 1 else 5000.0
    t0 = time.time()
    print(f"zeta(2) = {zeta(np.array([2.0]))[0].real:.12f}   (pi^2/6 = {np.pi ** 2 / 6:.12f})")
    zeros = zeros_on_line(T)
    print(f"{len(zeros)} zeros on the line below {T:g}; the first five: " + ", ".join(f"{z:.6f}" for z in zeros[:5]) +
          f"   ({time.time() - t0:.0f} s)")
    print("  widely published values of the first three: 14.134725, 21.022040, 25.010858")
    residual = np.abs(zeta(0.5 + 1j * zeros)).max()
    print(f"  largest |zeta| at the zeros found: {residual:.1e}")
    # T must not be a zero: move it to the middle of the last gap
    top = 0.5 * (zeros[-1] + zeros[-2])
    strip, jump = zeros_in_strip(top)
    on_line = int((zeros < top).sum())
    print(f"zeros in the whole strip below {top:.3f}, from the argument of zeta: {strip:.4f}   on the line: {on_line}   "
          f"(largest step of the argument along the path {jump:.4f} rad)")
    print("  " + ("every zero below this height is on the line" if abs(strip - on_line) < 0.05 else "THE COUNTS DIFFER"))
    r = gap_ratio(zeros)
    error = r.std() / np.sqrt(len(r))
    print(f"gap ratio of the zeros: mean {r.mean():.4f} ± {error:.4f}  ({len(r)} ratios)")
    for kind, label in (("complex", "complex Hermitian matrices (no symmetry under time reversal)"),
                        ("real", "real symmetric matrices (symmetric under time reversal, like our molecules)"),
                        ("none", "independent points (no operator)")):
        m = matrix_ratios(kind)
        print(f"  {label}: mean {m.mean():.4f} ± {m.std() / np.sqrt(len(m)):.4f}; distance between the distributions {distance(r, m):.3f}")
    for lo, hi in ((0, len(r) // 2), (len(r) // 2, len(r))):
        print(f"  zeros {lo + 1} to {hi}: mean {r[lo:hi].mean():.4f} ± {r[lo:hi].std() / np.sqrt(hi - lo):.4f}")
    print(f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()

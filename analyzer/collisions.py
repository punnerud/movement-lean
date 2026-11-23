#!/usr/bin/env python3
"""Two kinds of balls on a line that only collide: does the energy end up equally shared?

Lean proves that in a collision between balls with uncorrelated velocities the energy flows
from the kind with more energy to the kind with less, at the rate 4 m1 m2 / (m1 + m2)^2 times
the difference (`mean_energy_flow`, `energy_flows_hot_to_cold`). Here that is tried on numbers:
random pairs, one ball of each kind, collide elastically, again and again.

  1. the energy gained per collision at the start, against the formula;
  2. the mean energies at the end: equal (one temperature), whatever the masses;
  3. the approach to equality: the difference is multiplied by 1 - 2k per round (Lean:
     `energy_gap_factor`), so equal masses (k = 1) swap their energies for ever and never share;
  4. the shape of the velocity distributions at the end (a Gaussian has kurtosis 3).

    python3 analyzer/collisions.py
"""
import numpy as np


def run(m1, m2, e1, e2, balls=200000, rounds=40, seed=1):
    rng = np.random.default_rng(seed)
    # start far from a Gaussian: every ball has the same speed, with a random sign
    u1 = np.sqrt(2 * e1 / m1) * rng.choice([-1.0, 1.0], balls)
    u2 = np.sqrt(2 * e2 / m2) * rng.choice([-1.0, 1.0], balls)
    k = 4 * m1 * m2 / (m1 + m2) ** 2
    history = []
    for _ in range(rounds):
        partner = rng.permutation(balls)                         # each ball of kind 1 meets one of kind 2
        a, b = u1, u2[partner]
        before = 0.5 * m1 * (a ** 2).mean()
        cross = (a * b).mean()
        new_a = ((m1 - m2) * a + 2 * m2 * b) / (m1 + m2)
        new_b = ((m2 - m1) * b + 2 * m1 * a) / (m1 + m2)
        history.append((before, 0.5 * m2 * (b ** 2).mean(), 0.5 * m1 * (new_a ** 2).mean() - before, cross))
        u1 = new_a
        u2 = np.empty(balls)
        u2[partner] = new_b
    kurtosis = lambda u: float((u ** 4).mean() / (u ** 2).mean() ** 2)
    return k, history, 0.5 * m1 * (u1 ** 2).mean(), 0.5 * m2 * (u2 ** 2).mean(), kurtosis(u1), kurtosis(u2)


def main():
    for m1, m2, e1, e2 in ((1.0, 1.0, 3.0, 1.0), (1.0, 4.0, 3.0, 1.0), (1.0, 40.0, 3.0, 1.0), (4.0, 131.0, 1.0, 5.0)):
        k, history, f1, f2, k1, k2 = run(m1, m2, e1, e2)
        first = history[0]
        print(f"masses {m1:g} and {m2:g}, mean energies {e1:g} and {e2:g} at the start (transfer fraction k = {k:.4f})")
        print(f"  energy gained by kind 1 in the first round: {first[2]:+.4f}   formula k (E2 - E1): {k * (first[1] - first[0]):+.4f}"
              f"   (mean of u1 u2: {first[3]:+.4f})")
        gaps = [h[1] - h[0] for h in history[:6]]
        print("  difference E2 - E1, round by round: " + ", ".join(f"{g:+.4f}" for g in gaps) +
              f"   ratio per round {gaps[1] / gaps[0]:.4f}, formula 1 - 2k = {1 - 2 * k:.4f}")
        print(f"  at the end: {f1:.4f} and {f2:.4f}   (equal sharing would be {(e1 + e2) / 2:.4f})   "
              f"kurtosis {k1:.3f} and {k2:.3f} (Gaussian: 3; at the start: 1)")


if __name__ == "__main__":
    main()

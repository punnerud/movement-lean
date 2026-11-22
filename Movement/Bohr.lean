import Movement.Quantum

/-!
# 6. Fixed electron orbits: the Bohr model

Classically, an electron can circle a proton at **any** radius: the Coulomb
force `K/r²` (with `K = k e²`) supplies exactly the centripetal force
`m v²/r`, and a closer orbit is simply a faster one (`v² = K / (m r)`).

Add one quantum rule, angular momentum `m v r = n ħ`, and only discrete
orbits survive, with fixed speeds, radii and energies.

This is the semi-classical Bohr model. It gives the right hydrogen energies,
but real electrons do not follow orbits; the full theory is the Schrödinger
equation of section 5.
-/

namespace Movement

/-- Classical balance alone: closer orbits are faster, `v² = K / (m r)`,
and every radius is allowed. -/
theorem orbit_speed {m K v r : ℝ} (hm : 0 < m) (hr : 0 < r)
    (force : m * v ^ 2 / r = K / r ^ 2) : v ^ 2 = K / (m * r) := by
  rw [div_eq_div_iff hr.ne' (pow_ne_zero 2 hr.ne')] at force
  rw [eq_div_iff (by positivity)]
  exact mul_left_cancel₀ hr.ne' (by linear_combination force)

/-- **Bohr orbits.** Force balance plus quantised angular momentum `N = n ħ`
fix the speed, the radius and the energy:
`v = K/N`, `r = N²/(m K)`, `E = −m K² / (2 N²)`. -/
theorem bohr_orbit {m K N v r : ℝ} (hm : 0 < m) (hK : 0 < K) (hN : 0 < N) (hr : 0 < r)
    (force : m * v ^ 2 / r = K / r ^ 2) (quant : m * v * r = N) :
    v = K / N ∧ r = N ^ 2 / (m * K) ∧
      m * v ^ 2 / 2 - K / r = -(m * K ^ 2) / (2 * N ^ 2) := by
  have h1 : m * v ^ 2 * r = K := by
    rw [div_eq_div_iff hr.ne' (pow_ne_zero 2 hr.ne')] at force
    exact mul_left_cancel₀ hr.ne' (by linear_combination force)
  have hv : v = K / N := by
    rw [eq_div_iff hN.ne']; linear_combination h1 - v * quant
  have hr' : r = N ^ 2 / (m * K) := by
    rw [eq_div_iff (by positivity)]
    linear_combination (-(r * m)) * h1 + (m * v * r + N) * quant
  refine ⟨hv, hr', ?_⟩
  rw [hv, hr']
  field_simp
  ring

/-- Energy levels scale as `1/n²`: `E_n = E₁ / n²` (for hydrogen,
`E₁ = −13.6 eV`). -/
theorem bohr_levels {m K ħ : ℝ} (hħ : ħ ≠ 0) (n : ℕ) (hn : n ≠ 0) :
    -(m * K ^ 2) / (2 * (n * ħ) ^ 2) = (-(m * K ^ 2) / (2 * ħ ^ 2)) / n ^ 2 := by
  have : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  field_simp

/-- The orbital speed is `c·α/n`, where `α = K/(ħ c) ≈ 1/137` is the
fine-structure constant: about 0.7 % of the speed of light in the ground state. -/
theorem bohr_speed_fraction {K ħ c : ℝ} (hħ : ħ ≠ 0) (hc : c ≠ 0) (n : ℕ) (hn : n ≠ 0) :
    K / (n * ħ) = c * (K / (ħ * c)) / n := by
  have : (n : ℝ) ≠ 0 := by exact_mod_cast hn
  field_simp

/-! ### Elliptical orbits (Sommerfeld)

Bohr's circles are a special case. Sommerfeld allowed ellipses, with one
quantum number for the angular momentum and one for the in-and-out motion. -/

/-- **An ellipse's energy depends only on its length.** At the nearest
point `rp` and the farthest point `ra` the electron moves purely sideways,
so there `E = L²/(2 m r²) − K/r`. These two equations alone give
`E = −K/(rp + ra)`: the energy depends on the major axis `rp + ra` and not
on how flattened the ellipse is. They also fix the angular momentum. -/
theorem ellipse_energy_and_momentum {m K L E rp ra : ℝ} (hm : 0 < m) (hp : 0 < rp)
    (ha : 0 < ra) (hne : rp ≠ ra)
    (h1 : E = L ^ 2 / (2 * m * rp ^ 2) - K / rp)
    (h2 : E = L ^ 2 / (2 * m * ra ^ 2) - K / ra) :
    E = -K / (rp + ra) ∧ L ^ 2 = 2 * m * K * rp * ra / (rp + ra) := by
  have e1 : 2 * m * rp ^ 2 * E = L ^ 2 - 2 * m * K * rp := by
    rw [h1]; field_simp
  have e2 : 2 * m * ra ^ 2 * E = L ^ 2 - 2 * m * K * ra := by
    rw [h2]; field_simp
  have key : E * (rp + ra) = -K := by
    have h : (2 * m * (rp - ra)) * (E * (rp + ra) + K) = 0 := by linear_combination e1 - e2
    rcases mul_eq_zero.mp h with h | h
    · exact absurd h (mul_ne_zero (by positivity) (sub_ne_zero.mpr hne))
    · linarith
  have hsum : 0 < rp + ra := by linarith
  refine ⟨by rw [eq_div_iff hsum.ne']; exact key, ?_⟩
  rw [eq_div_iff hsum.ne']
  linear_combination (-(rp + ra)) * e1 + (2 * m * rp ^ 2) * key

/-- **Sommerfeld's quantised ellipses.** An ellipse with half-length `a`
and eccentricity `ecc` has `E = −K/(2a)` and `L² = m K a (1 − ecc²)`.
Quantise the angular momentum, `L = k ħ`, and the in-and-out motion,
`√(m K a) − L = nᵣ ħ` (the radial action integral, taken as given; `s`
stands for the square root). Then with `n = nᵣ + k`:

* the energy is Bohr's `−m K²/(2 n² ħ²)`, whatever the shape;
* the half-length is Bohr's radius for level `n`;
* the shape is fixed by `1 − ecc² = (k/n)²`.

So level `n` holds one circle (`k = n`) and `n − 1` ellipses, all with the
same energy. -/
theorem sommerfeld_orbit {m K ħ a s L E ecc : ℝ} {k nr : ℕ} (hm : 0 < m) (hK : 0 < K)
    (hħ : 0 < ħ) (ha : 0 < a) (hk : k ≠ 0)
    (hs : s ^ 2 = m * K * a) (energy : E = -K / (2 * a))
    (shape : L ^ 2 = m * K * a * (1 - ecc ^ 2))
    (angular : L = k * ħ) (radial : s - L = nr * ħ) :
    E = -(m * K ^ 2) / (2 * ((nr + k) * ħ) ^ 2) ∧ a = ((nr + k) * ħ) ^ 2 / (m * K) ∧
      1 - ecc ^ 2 = (k / (nr + k)) ^ 2 := by
  have hkpos : (0 : ℝ) < k := by exact_mod_cast Nat.pos_of_ne_zero hk
  have hn : (0 : ℝ) < nr + k := by
    have : (0 : ℝ) ≤ nr := Nat.cast_nonneg _
    linarith
  have hs' : s = (nr + k) * ħ := by rw [angular] at radial; linarith
  have hX : 0 < ((nr + k) * ħ) ^ 2 := pow_pos (mul_pos hn hħ) 2
  have haeq : m * K * a = ((nr + k) * ħ) ^ 2 := by rw [← hs, hs']
  have hmK : 0 < m * K := mul_pos hm hK
  refine ⟨?_, ?_, ?_⟩
  · rw [energy, div_eq_div_iff (by positivity) (mul_pos two_pos hX).ne']
    linear_combination (2 * K) * haeq
  · rw [eq_div_iff hmK.ne']
    linear_combination haeq
  · have h := shape
    rw [angular, haeq] at h
    have h0 : ħ ^ 2 * ((1 - ecc ^ 2) * (nr + k) ^ 2 - k ^ 2) = 0 := by linear_combination -h
    rcases mul_eq_zero.mp h0 with h' | h'
    · exact absurd h' (pow_ne_zero 2 hħ.ne')
    · rw [div_pow, eq_div_iff (pow_ne_zero 2 hn.ne')]
      linarith

/-- **How flat the ellipses are.** The short half-axis `b` of an ellipse
satisfies `b² = a² (1 − ecc²)`, so for a Sommerfeld orbit `b/a = k/n`:
the circle has `k = n`, and the smallest `k` gives the most stretched orbit. -/
theorem sommerfeld_axis_ratio {a b k n : ℝ} (ha : 0 < a) (hb : 0 < b) (hk : 0 < k) (hn : 0 < n)
    (h : b ^ 2 = a ^ 2 * (k / n) ^ 2) : b / a = k / n := by
  have h4 : (b - a * (k / n)) * (b + a * (k / n)) = 0 := by linear_combination h
  rcases mul_eq_zero.mp h4 with h5 | h5
  · rw [div_eq_iff ha.ne']
    linear_combination h5
  · have : 0 < b + a * (k / n) := by positivity
    linarith

end Movement

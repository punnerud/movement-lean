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

end Movement

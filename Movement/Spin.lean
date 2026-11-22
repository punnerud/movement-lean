import Movement.Bohr

/-!
# 7. Is spin a circulation at the speed of light?

**This section tests an interpretation, not established physics.**

Idea: the electron's spin `ħ/2` is the angular momentum of something
circulating, and the circulation speed `u` combines with the forward speed `v`
so that the total is always `c` (a helix).

What the algebra says:

* `spinning_sphere_speed`: a spinning ball of classical-electron-radius size
  would need a surface speed of `(5/4)·c/α ≈ 171 c`. Ruled out.
* `zitter_radius`: circulation at exactly `c` forces the radius `ħ/2mc` and the
  angular frequency `2mc²/ħ`, the *Zitterbewegung* scale of the Dirac equation.
* `helix_speed`: `v² + u² = c²` gives `u = c·√(1 − v²/c²)`, the time-dilation
  factor.
* `classical_g_factor`, `no_radius_gives_both`: **the model fails.** Any point
  charge on a circle has magnetic moment `μ = (e/2m)·L`, so spin `ħ/2` gives
  half a Bohr magneton. The measured value is one Bohr magneton (g ≈ 2).
  No radius can produce both.
-/

namespace Movement

/-! ### A spinning ball is ruled out -/

/-- A solid ball (`L = ⅖ m r v`) with the classical electron radius
`r = K/(m c²)` and spin `ħ/2` needs surface speed `v = (5/4)·c/α`,
where `α = K/(ħ c)`. -/
theorem spinning_sphere_speed {m c ħ K r v : ℝ} (hm : 0 < m) (hc : 0 < c) (hK : 0 < K)
    (hħ : 0 < ħ) (hr : r = K / (m * c ^ 2)) (hL : 2 / 5 * m * r * v = ħ / 2) :
    v = 5 / 4 * (c / (K / (ħ * c))) := by
  subst hr
  have h1 : 2 / 5 * m * (K / (m * c ^ 2)) * v = 2 / 5 * K * v / c ^ 2 := by field_simp
  rw [h1, div_eq_iff (by positivity)] at hL
  have h2 : 5 / 4 * (c / (K / (ħ * c))) = 5 / 4 * ħ * c ^ 2 / K := by field_simp
  rw [h2, eq_div_iff hK.ne']
  linear_combination 5 / 2 * hL

/-- Since `α < 1` (in fact `α ≈ 1/137`), that speed exceeds `c`. -/
theorem spinning_sphere_exceeds_c {c α : ℝ} (hc : 0 < c) (hα : 0 < α) (hα1 : α < 1) :
    c < 5 / 4 * (c / α) := by
  have h : c < c / α := by rw [lt_div_iff₀ hα]; nlinarith
  nlinarith [div_pos hc hα]

/-! ### Circulation at exactly `c` -/

/-- A point mass circulating at speed `c` with angular momentum `ħ/2` must
have radius `ħ/(2mc)` (half the reduced Compton wavelength) and angular
frequency `2mc²/ħ`. -/
theorem zitter_radius {m c ħ r : ℝ} (hm : 0 < m) (hc : 0 < c) (hħ : 0 < ħ)
    (h : m * c * r = ħ / 2) :
    r = ħ / (2 * m * c) ∧ c / r = 2 * m * c ^ 2 / ħ := by
  have hr : r = ħ / (2 * m * c) := by
    rw [eq_div_iff (by positivity)]; linear_combination 2 * h
  refine ⟨hr, ?_⟩
  rw [hr]; field_simp

/-- **Total speed `c`.** If forward speed `v` and circulation speed `u` are
perpendicular and combine to `c`, then `u = c·√(1 − v²/c²)`: the circulation
slows by exactly the relativistic time-dilation factor. -/
theorem helix_speed {c v u : ℝ} (hc : 0 < c) (hu : 0 ≤ u) (h : v ^ 2 + u ^ 2 = c ^ 2) :
    u = c * √(1 - v ^ 2 / c ^ 2) := by
  have e : 1 - v ^ 2 / c ^ 2 = (u / c) ^ 2 := by
    field_simp; linarith
  rw [e, Real.sqrt_sq (div_nonneg hu hc.le)]
  field_simp

/-- In hydrogen's ground state (`v = α c`) the circulation speed would be
`c·√(1 − α²) ≈ 0.99997 c`. -/
theorem hydrogen_circulation {c α u : ℝ} (hc : 0 < c) (hu : 0 ≤ u)
    (h : (α * c) ^ 2 + u ^ 2 = c ^ 2) : u = c * √(1 - α ^ 2) := by
  have := helix_speed hc hu h
  rw [this]; congr 2; field_simp

/-! ### The magnetic moment test -/

/-- Magnetic moment of a circulating point charge: current `e·v/(2πr)` times
enclosed area `πr²` is `e v r / 2`. -/
theorem current_loop_moment {e v r : ℝ} (hr : r ≠ 0) :
    e * v / (2 * Real.pi * r) * (Real.pi * r ^ 2) = e * v * r / 2 := by
  have := Real.pi_ne_zero
  field_simp

/-- **Classical g-factor is 1.** For any point charge on a circle, whatever
its speed and radius, `μ = (e/2m)·L`. -/
theorem classical_g_factor {m e r v L μ : ℝ} (hm : m ≠ 0)
    (hL : L = m * r * v) (hμ : μ = e * v * r / 2) : μ = e / (2 * m) * L := by
  subst hL hμ; field_simp

/-- So circulation at `c` with spin `ħ/2` gives **half** a Bohr magneton
(`μ_B = eħ/2m`). The measured electron moment is 1.00116 `μ_B`. -/
theorem zitter_moment {m e c ħ r : ℝ} (hm : 0 < m) (hc : 0 < c)
    (h : m * c * r = ħ / 2) : e * c * r / 2 = 1 / 2 * (e * ħ / (2 * m)) := by
  have hr : r = ħ / (2 * m * c) := by
    rw [eq_div_iff (by positivity)]; linear_combination 2 * h
  rw [hr]; field_simp

/-- **No-go.** No radius gives a point charge circulating at `c` both the
measured spin `ħ/2` and a magnetic moment of one Bohr magneton. -/
theorem no_radius_gives_both {m e c ħ : ℝ} (hm : 0 < m) (he : 0 < e) (hħ : 0 < ħ) (r : ℝ) :
    ¬ (m * c * r = ħ / 2 ∧ e * c * r / 2 = e * ħ / (2 * m)) := by
  rintro ⟨h1, h2⟩
  rw [div_eq_div_iff two_ne_zero (by positivity)] at h2
  have h3 : e * (m * c * r) = e * ħ := by linear_combination h2 / 2
  have h4 : m * c * r = ħ := mul_left_cancel₀ he.ne' h3
  linarith

/-- The same no-go at **any** speed: spin `ħ/2` and moment `μ_B` are
incompatible for a circulating point charge. -/
theorem no_orbit_gives_both {m e ħ : ℝ} (hm : 0 < m) (he : 0 < e) (hħ : 0 < ħ) (r v : ℝ) :
    ¬ (m * r * v = ħ / 2 ∧ e * v * r / 2 = e * ħ / (2 * m)) := by
  rintro ⟨h1, h2⟩
  have h := classical_g_factor (e := e) hm.ne' h1.symm rfl
  rw [h2] at h
  have hX : 0 < e * ħ / (2 * m) := by positivity
  have e2 : e / (2 * m) * (ħ / 2) = e * ħ / (2 * m) / 2 := by ring
  linarith

end Movement

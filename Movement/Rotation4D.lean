import Movement.Spin

/-!
# 8. Can every part of a spinning thing move at the same speed?

* `rigid_rotation_has_pole`: a rigid rotation in 3D always leaves two points
  of the sphere at rest (the poles), so the surface speed cannot be equal
  everywhere.
* `ring_speed`: on a ring, every point has the same speed `|ω|·ρ`.
* `isoclinic_velocity`, `isoclinic_speed`, `isoclinic_stays_on_sphere`: in
  four dimensions (`ℂ² = ℝ⁴`) the rotation `z ↦ e^{iωt} z` moves **every**
  point of the unit sphere at the same speed `|ω|`. There are no poles.
* `spinor_full_turn`, `spinor_double_turn`: a spin-½ state picks up the phase
  `e^{iθ/2}`, so a 360° turn gives `−1` and only 720° gives `+1`.

The mathematics is standard. Reading it as a literal motion of the electron is
an interpretation, as in section 7.
-/

namespace Movement

open Complex

/-! ### Three dimensions: there is always a pole -/

/-- A rigid rotation with angular velocity `w = (wx, wy, wz) ≠ 0` gives the
point `p` the velocity `w × p`. The point `p = w/|w|` lies on the unit sphere
and has velocity zero. -/
theorem rigid_rotation_has_pole {wx wy wz : ℝ} (hw : wx ^ 2 + wy ^ 2 + wz ^ 2 ≠ 0) :
    ∃ px py pz : ℝ, px ^ 2 + py ^ 2 + pz ^ 2 = 1 ∧
      wy * pz - wz * py = 0 ∧ wz * px - wx * pz = 0 ∧ wx * py - wy * px = 0 := by
  have hpos : 0 < wx ^ 2 + wy ^ 2 + wz ^ 2 :=
    lt_of_le_of_ne (by positivity) (Ne.symm hw)
  set s := √(wx ^ 2 + wy ^ 2 + wz ^ 2) with hs
  have hs0 : s ≠ 0 := (Real.sqrt_pos.mpr hpos).ne'
  have hss : s ^ 2 = wx ^ 2 + wy ^ 2 + wz ^ 2 := Real.sq_sqrt hpos.le
  refine ⟨wx / s, wy / s, wz / s, ?_, ?_, ?_, ?_⟩
  · rw [div_pow, div_pow, div_pow, ← add_div, ← add_div, hss, div_self hw]
  · ring
  · ring
  · ring

/-- The speed of a point in a rotation about the z-axis is `|ω|` times its
distance from the axis: fastest at the equator, zero on the axis. -/
theorem rotation_speed_sq (ω x y : ℝ) :
    (-(ω * y)) ^ 2 + (ω * x) ^ 2 = ω ^ 2 * (x ^ 2 + y ^ 2) := by
  ring

/-- **A ring works.** Every point at distance `ρ` from the axis has the same
speed, `ω²ρ²` squared. -/
theorem ring_speed {ω ρ x y : ℝ} (h : x ^ 2 + y ^ 2 = ρ ^ 2) :
    (-(ω * y)) ^ 2 + (ω * x) ^ 2 = (ω * ρ) ^ 2 := by
  rw [rotation_speed_sq, h]; ring

/-! ### Four dimensions: equal speed everywhere -/

/-- Velocity of a point under the rotation `z ↦ e^{iωt}·z`. -/
theorem isoclinic_velocity (ω : ℝ) (z : ℂ) (t : ℝ) :
    HasDerivAt (fun t : ℝ => cexp (I * ω * t) * z) (I * ω * (cexp (I * ω * t) * z)) t := by
  have h : HasDerivAt (fun t : ℝ => I * ω * (t : ℂ)) (I * ω) t := by
    simpa using (hasDerivAt_id t).ofReal_comp.const_mul (I * ω)
  have h1 : HasDerivAt (fun t : ℝ => cexp (I * ω * (t : ℂ)) * z) _ t := h.cexp.mul_const z
  exact h1.congr_deriv (by ring)

theorem norm_phase (ω t : ℝ) : ‖cexp (I * ω * t)‖ = 1 := by
  rw [Complex.norm_exp]
  have : (I * ω * t).re = 0 := by simp [mul_assoc, ← ofReal_mul]
  rw [this, Real.exp_zero]

/-- **Isoclinic rotation.** Under `(z₁, z₂) ↦ e^{iωt}(z₁, z₂)` in `ℂ² = ℝ⁴`,
every point of the unit 3-sphere moves with the same speed `|ω|`. -/
theorem isoclinic_speed (ω t : ℝ) {z₁ z₂ : ℂ} (h : ‖z₁‖ ^ 2 + ‖z₂‖ ^ 2 = 1) :
    ‖I * ω * (cexp (I * ω * t) * z₁)‖ ^ 2 + ‖I * ω * (cexp (I * ω * t) * z₂)‖ ^ 2 = ω ^ 2 := by
  simp only [norm_mul, norm_I, norm_phase, one_mul, Complex.norm_real, Real.norm_eq_abs, mul_pow,
    sq_abs]
  rw [← mul_add, h, mul_one]

/-- The rotation keeps every point on the unit sphere. -/
theorem isoclinic_stays_on_sphere (ω t : ℝ) {z₁ z₂ : ℂ} (h : ‖z₁‖ ^ 2 + ‖z₂‖ ^ 2 = 1) :
    ‖cexp (I * ω * t) * z₁‖ ^ 2 + ‖cexp (I * ω * t) * z₂‖ ^ 2 = 1 := by
  simp only [norm_mul, norm_phase, one_mul]
  exact h

/-! ### Spin-½ and the 720° rule -/

/-- A rotation by angle `θ` multiplies a spin-½ state by `e^{iθ/2}`.
After one full turn (`θ = 2π`) the state has changed sign. -/
theorem spinor_full_turn : cexp (I * (2 * Real.pi : ℝ) / 2) = -1 := by
  have : I * ((2 * Real.pi : ℝ) : ℂ) / 2 = Real.pi * I := by push_cast; ring
  rw [this, Complex.exp_pi_mul_I]

/-- Only after two full turns (`θ = 4π`) is it back where it started. -/
theorem spinor_double_turn : cexp (I * (4 * Real.pi : ℝ) / 2) = 1 := by
  have : I * ((4 * Real.pi : ℝ) : ℂ) / 2 = 2 * Real.pi * I := by push_cast; ring
  rw [this, Complex.exp_two_pi_mul_I]

end Movement

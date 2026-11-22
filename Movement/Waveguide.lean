import Movement.Rotation4D

/-!
# 9. Waveguides and spacetime: slower than `c`, yet always `c`

**Waveguide.** Light between two mirrors zigzags at speed `c`. Its speed
*along* the guide is lower, and the dispersion relation
`ω² = ω_c² + c²k²` has exactly the form of the relativistic
energy–momentum relation `E² = (mc²)² + (pc)²`, with the cutoff frequency
`ω_c` playing the role of mass.

**Spacetime.** In special relativity the speed through space `v` and the
"speed through time" `c·dτ/dt` always combine to `c`.

Both are established physics. They are the precise versions of the
"total speed is always `c`" picture from sections 7 and 8.
-/

namespace Movement

/-! ### Waveguide -/

section Waveguide

variable {c ω ωc k : ℝ}

/-- In a waveguide `c·k < ω` whenever there is a cutoff. -/
theorem ck_lt_omega (hc : 0 < c) (hω : 0 < ω) (hωc : ωc ≠ 0) (hk : 0 ≤ k)
    (hdisp : ω ^ 2 = ωc ^ 2 + c ^ 2 * k ^ 2) : c * k < ω := by
  have hpos : 0 < ωc ^ 2 := by positivity
  by_contra h
  rw [not_lt] at h
  have h1 : 0 ≤ c * k := mul_nonneg hc.le hk
  nlinarith [mul_nonneg (sub_nonneg.mpr h) (add_nonneg h1 hω.le)]

/-- Group velocity (`c²k/ω`, the speed of a pulse) times phase velocity
(`ω/k`, the speed of the crests) is always `c²`. -/
theorem group_times_phase (hω : ω ≠ 0) (hk : k ≠ 0) :
    (c ^ 2 * k / ω) * (ω / k) = c ^ 2 := by
  field_simp

/-- **Zigzag at `c`.** The forward (group) speed `c²k/ω` and the sideways
speed `c·ω_c/ω` combine to exactly `c`. -/
theorem waveguide_zigzag (hω : 0 < ω) (hdisp : ω ^ 2 = ωc ^ 2 + c ^ 2 * k ^ 2) :
    (c ^ 2 * k / ω) ^ 2 + (c * ωc / ω) ^ 2 = c ^ 2 := by
  rw [div_pow, div_pow, ← add_div, div_eq_iff (by positivity)]
  linear_combination (-(c ^ 2)) * hdisp

/-- A pulse in the guide travels slower than `c`. -/
theorem group_lt_c (hc : 0 < c) (hω : 0 < ω) (hωc : ωc ≠ 0) (hk : 0 ≤ k)
    (hdisp : ω ^ 2 = ωc ^ 2 + c ^ 2 * k ^ 2) : c ^ 2 * k / ω < c := by
  have h := ck_lt_omega hc hω hωc hk hdisp
  rw [div_lt_iff₀ hω]
  nlinarith

/-- The crests travel faster than `c` (they carry no signal). -/
theorem phase_gt_c (hc : 0 < c) (hω : 0 < ω) (hωc : ωc ≠ 0) (hk : 0 < k)
    (hdisp : ω ^ 2 = ωc ^ 2 + c ^ 2 * k ^ 2) : c < ω / k := by
  rw [lt_div_iff₀ hk]
  exact ck_lt_omega hc hω hωc hk.le hdisp

/-- Without a cutoff (free space) the pulse travels at exactly `c`. -/
theorem no_cutoff_speed (hc : 0 < c) (hk : 0 < k) : c ^ 2 * k / (c * k) = c := by
  field_simp

/-- The group velocity really is the derivative `dω/dk` of the dispersion
relation `ω(k) = √(ω_c² + c²k²)`. -/
theorem group_velocity_deriv (h : 0 < ωc ^ 2 + c ^ 2 * k ^ 2) :
    HasDerivAt (fun k : ℝ => √(ωc ^ 2 + c ^ 2 * k ^ 2))
      (c ^ 2 * k / √(ωc ^ 2 + c ^ 2 * k ^ 2)) k := by
  have h1 : HasDerivAt (fun k : ℝ => ωc ^ 2 + c ^ 2 * k ^ 2) (c ^ 2 * (2 * k)) k := by
    have := ((hasDerivAt_pow 2 k).const_mul (c ^ 2)).const_add (ωc ^ 2)
    simpa using this
  have hs : √(ωc ^ 2 + c ^ 2 * k ^ 2) ≠ 0 := (Real.sqrt_pos.mpr h).ne'
  refine (h1.sqrt h.ne').congr_deriv ?_
  field_simp

/-- **Cutoff acts as mass.** With `E = ħω`, `p = ħk` and `m = ħω_c/c²`, the
waveguide dispersion relation is `E² = (mc²)² + (pc)²`. -/
theorem waveguide_mass {ħ : ℝ} (hc : c ≠ 0) (hdisp : ω ^ 2 = ωc ^ 2 + c ^ 2 * k ^ 2) :
    (ħ * ω) ^ 2 = (ħ * ωc / c ^ 2 * c ^ 2) ^ 2 + (ħ * k * c) ^ 2 := by
  have : ħ * ωc / c ^ 2 * c ^ 2 = ħ * ωc := by field_simp
  rw [this]
  linear_combination ħ ^ 2 * hdisp

end Waveguide

/-! ### Spacetime -/

section Spacetime

variable {c v m : ℝ}

/-- **Everything moves at `c` through spacetime.** Speed through space `v`
and speed through time `c·dτ/dt = c·√(1 − v²/c²)` combine to `c`. -/
theorem spacetime_speed (hc : 0 < c) (hv : v ^ 2 ≤ c ^ 2) :
    v ^ 2 + (c * √(1 - v ^ 2 / c ^ 2)) ^ 2 = c ^ 2 := by
  have h0 : 0 ≤ 1 - v ^ 2 / c ^ 2 := sub_nonneg.mpr ((div_le_one (by positivity)).mpr hv)
  rw [mul_pow, Real.sq_sqrt h0]
  field_simp
  ring

/-- The Lorentz factor squared: `γ² = c²/(c² − v²)`. -/
theorem gamma_sq (hc : 0 < c) (hv : v ^ 2 < c ^ 2) :
    (1 / √(1 - v ^ 2 / c ^ 2)) ^ 2 = c ^ 2 / (c ^ 2 - v ^ 2) := by
  have h0 : 0 < 1 - v ^ 2 / c ^ 2 := sub_pos.mpr ((div_lt_one (by positivity)).mpr hv)
  have hne : c ^ 2 - v ^ 2 ≠ 0 := (sub_pos.mpr hv).ne'
  rw [div_pow, one_pow, Real.sq_sqrt h0.le]
  field_simp

/-- **Four-velocity has length `c`.** `(γc)² − (γv)² = c²`. -/
theorem four_velocity_norm (hc : 0 < c) (hv : v ^ 2 < c ^ 2) :
    (1 / √(1 - v ^ 2 / c ^ 2) * c) ^ 2 - (1 / √(1 - v ^ 2 / c ^ 2) * v) ^ 2 = c ^ 2 := by
  have hne : c ^ 2 - v ^ 2 ≠ 0 := (sub_pos.mpr hv).ne'
  rw [mul_pow, mul_pow, gamma_sq hc hv]
  field_simp

/-- **Energy–momentum relation.** With `E = γmc²` and `p = γmv`:
`E² = (mc²)² + (pc)²`, the same form as the waveguide. -/
theorem energy_momentum (hc : 0 < c) (hv : v ^ 2 < c ^ 2) :
    (1 / √(1 - v ^ 2 / c ^ 2) * m * c ^ 2) ^ 2 =
      (m * c ^ 2) ^ 2 + (1 / √(1 - v ^ 2 / c ^ 2) * m * v * c) ^ 2 := by
  have h := four_velocity_norm hc hv
  linear_combination (m ^ 2 * c ^ 2) * h

/-- A particle's velocity is `pc²/E`, just as the waveguide's group velocity
is `c²k/ω`. -/
theorem particle_velocity {γ : ℝ} (hγ : γ ≠ 0) (hm : m ≠ 0) (hc : c ≠ 0) :
    (γ * m * v) * c ^ 2 / (γ * m * c ^ 2) = v := by
  field_simp

/-- The cutoff frequency matching an electron's mass is the Compton frequency
`mc²/ħ`; the Zitterbewegung frequency of section 7 is exactly twice that. -/
theorem zitter_is_twice_compton {ħ : ℝ} : 2 * m * c ^ 2 / ħ = 2 * (m * c ^ 2 / ħ) := by
  ring

end Spacetime

end Movement

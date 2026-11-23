import Movement.Spectrum

/-!
# From collisions to heat: what was missing between sections 2 and 10

* a collision carries energy from the hotter to the colder side, so equal
  mean energies (one temperature) is the only state collisions leave alone;
* a sudden hit on one atom of a molecule puts a fixed share into vibration;
* vibration frequencies of isotopes, the ground state of the oscillator,
  and the lower bound that completes Einstein's function;
* the simple mean-free-path picture of transport, and the decay of a
  magnetic field inside a superconductor.
-/

namespace Movement

/-! ### Collisions carry energy from hot to cold -/

/-- **Energy gained in one collision.** Ball 1 (mass `m₁`, velocity `u₁`)
hits ball 2 elastically. Its energy changes by the transfer fraction
`4m₁m₂/(m₁+m₂)²` times the difference of the two energies, plus a cross term
in `u₁u₂`. -/
theorem collision_energy_flow {m₁ m₂ u₁ u₂ : ℝ} (hm : m₁ + m₂ ≠ 0) :
    m₁ / 2 * (((m₁ - m₂) * u₁ + 2 * m₂ * u₂) / (m₁ + m₂)) ^ 2 - m₁ / 2 * u₁ ^ 2 =
      4 * m₁ * m₂ / (m₁ + m₂) ^ 2 *
        (m₂ / 2 * u₂ ^ 2 - m₁ / 2 * u₁ ^ 2 + (m₁ - m₂) / 2 * (u₁ * u₂)) := by
  field_simp
  ring

/-- **Many collisions.** If the two velocities are uncorrelated (the sum of
`u₁u₂` over the collisions is zero), the total energy gained by the balls of
kind 1 is the transfer fraction times the difference of the total energies. -/
theorem mean_energy_flow {ι : Type*} (s : Finset ι) {m₁ m₂ : ℝ} (u₁ u₂ : ι → ℝ)
    (hm : m₁ + m₂ ≠ 0) (hcross : ∑ i ∈ s, u₁ i * u₂ i = 0) :
    ∑ i ∈ s, (m₁ / 2 * (((m₁ - m₂) * u₁ i + 2 * m₂ * u₂ i) / (m₁ + m₂)) ^ 2 - m₁ / 2 * u₁ i ^ 2) =
      4 * m₁ * m₂ / (m₁ + m₂) ^ 2 *
        (∑ i ∈ s, m₂ / 2 * u₂ i ^ 2 - ∑ i ∈ s, m₁ / 2 * u₁ i ^ 2) := by
  have h : ∀ i ∈ s, m₁ / 2 * (((m₁ - m₂) * u₁ i + 2 * m₂ * u₂ i) / (m₁ + m₂)) ^ 2 - m₁ / 2 * u₁ i ^ 2 =
      4 * m₁ * m₂ / (m₁ + m₂) ^ 2 * (m₂ / 2 * u₂ i ^ 2) - 4 * m₁ * m₂ / (m₁ + m₂) ^ 2 * (m₁ / 2 * u₁ i ^ 2)
        + 4 * m₁ * m₂ / (m₁ + m₂) ^ 2 * ((m₁ - m₂) / 2) * (u₁ i * u₂ i) := by
    intro i _
    rw [collision_energy_flow hm]
    ring
  rw [Finset.sum_congr rfl h, Finset.sum_add_distrib, Finset.sum_sub_distrib]
  simp only [← Finset.mul_sum, hcross]
  ring

/-- **Heat flows from hot to cold.** With uncorrelated velocities, the balls
of kind 1 gain energy exactly when their total energy is the smaller one.
Nothing flows only when the mean energies are equal: that is what a common
temperature means, and it is why every way of moving gets the same share. -/
theorem energy_flows_hot_to_cold {ι : Type*} (s : Finset ι) {m₁ m₂ : ℝ} (u₁ u₂ : ι → ℝ)
    (h1 : 0 < m₁) (h2 : 0 < m₂) (hcross : ∑ i ∈ s, u₁ i * u₂ i = 0) :
    0 < ∑ i ∈ s, (m₁ / 2 * (((m₁ - m₂) * u₁ i + 2 * m₂ * u₂ i) / (m₁ + m₂)) ^ 2 - m₁ / 2 * u₁ i ^ 2) ↔
      ∑ i ∈ s, m₁ / 2 * u₁ i ^ 2 < ∑ i ∈ s, m₂ / 2 * u₂ i ^ 2 := by
  have hc : 0 < 4 * m₁ * m₂ / (m₁ + m₂) ^ 2 := by positivity
  rw [mean_energy_flow s u₁ u₂ (add_pos h1 h2).ne' hcross, mul_pos_iff_of_pos_left hc, sub_pos]

/-- **How fast the sharing goes.** Each round of collisions multiplies the
difference of the two mean energies by `1 − 2k`, with `k` the transfer
fraction. That factor is below 1 in size unless the masses are equal: equal
balls on a line swap their energies for ever and never come to share them. -/
theorem energy_gap_factor {m₁ m₂ : ℝ} (h1 : 0 < m₁) (h2 : 0 < m₂) :
    -1 ≤ 1 - 2 * (4 * m₁ * m₂ / (m₁ + m₂) ^ 2) ∧ 1 - 2 * (4 * m₁ * m₂ / (m₁ + m₂) ^ 2) < 1 ∧
      (1 - 2 * (4 * m₁ * m₂ / (m₁ + m₂) ^ 2) = -1 ↔ m₁ = m₂) := by
  have hs : 0 < (m₁ + m₂) ^ 2 := by positivity
  have hk : 4 * m₁ * m₂ / (m₁ + m₂) ^ 2 ≤ 1 := by
    rw [div_le_one hs]
    nlinarith [sq_nonneg (m₁ - m₂)]
  have hk0 : 0 < 4 * m₁ * m₂ / (m₁ + m₂) ^ 2 := by positivity
  refine ⟨by linarith, by linarith, ?_⟩
  constructor
  · intro h
    have h3 : 4 * m₁ * m₂ / (m₁ + m₂) ^ 2 = 1 := by linarith
    rw [div_eq_one_iff_eq hs.ne'] at h3
    have h4 : (m₁ - m₂) ^ 2 = 0 := by nlinarith
    have h5 : m₁ - m₂ = 0 := (pow_eq_zero_iff two_ne_zero).mp h4
    linarith
  · intro h
    subst h
    have hm : m₁ ≠ 0 := h1.ne'
    field_simp
    ring

/-- **A sudden hit on one atom of a molecule.** Atom A (mass `mA`) of a
molecule A–B is knocked to velocity `u` while B is still at rest. The energy
splits into motion of the whole molecule and motion of the atoms against
each other (vibration, with the reduced mass). -/
theorem sudden_hit_split {mA mB u : ℝ} (h : mA + mB ≠ 0) :
    mA / 2 * u ^ 2 =
      (mA + mB) / 2 * (mA * u / (mA + mB)) ^ 2 + mA * mB / (mA + mB) / 2 * u ^ 2 := by
  field_simp

/-- The share that goes into vibration is `mB / (mA + mB)`: the companion of
`rot_share`. Classically a vibration therefore takes part in every exchange. -/
theorem vib_share {mA mB u : ℝ} (hA : 0 < mA) (hB : 0 < mB) (hu : u ≠ 0) :
    (mA * mB / (mA + mB) / 2 * u ^ 2) / (mA / 2 * u ^ 2) = mB / (mA + mB) := by
  have h : mA + mB ≠ 0 := (add_pos hA hB).ne'
  have hA' : mA ≠ 0 := hA.ne'
  field_simp

/-! ### Vibrations -/

/-- **Isotopes vibrate at frequencies that go as one over the square root of
the reduced mass**, the spring `k` being the same (it is made by the
electrons). -/
theorem isotope_vibration_frequency {k μ₁ μ₂ : ℝ} (hk : 0 < k) (h1 : 0 < μ₁) (h2 : 0 < μ₂) :
    Real.sqrt (k / μ₂) / Real.sqrt (k / μ₁) = Real.sqrt (μ₁ / μ₂) := by
  rw [← Real.sqrt_div (div_nonneg hk.le h2.le)]
  congr 1
  field_simp

/-- Einstein's function is positive: a vibration always holds some heat.
With `einstein_function_le_one` it lies strictly between nothing and the
classical share. -/
theorem einstein_function_pos {x : ℝ} (hx : 0 < x) :
    0 < x ^ 2 * Real.exp x / (Real.exp x - 1) ^ 2 := by
  have h := Real.add_one_lt_exp (ne_of_gt hx)
  have h2 : 0 < Real.exp x - 1 := by linarith
  positivity

/-- First derivative of the Gaussian `exp(−x²/2)`. -/
theorem oscillator_ground_dx (x : ℝ) :
    HasDerivAt (fun x : ℝ => Real.exp (-(1 / 2) * x ^ 2)) (-x * Real.exp (-(1 / 2) * x ^ 2)) x := by
  have h := ((hasDerivAt_pow 2 x).const_mul (-(1 / 2 : ℝ))).exp
  refine h.congr_deriv ?_
  norm_num
  ring

/-- Second derivative of the Gaussian. -/
theorem oscillator_ground_dxx (x : ℝ) :
    HasDerivAt (fun x : ℝ => -x * Real.exp (-(1 / 2) * x ^ 2))
      ((x ^ 2 - 1) * Real.exp (-(1 / 2) * x ^ 2)) x := by
  have h := ((hasDerivAt_id x).neg).mul (oscillator_ground_dx x)
  convert h using 1
  · rfl
  · simp only [Pi.neg_apply, id_eq]
    ring

/-- **The oscillator's ground state and zero-point energy.** With
`ψ = exp(−x²/2)` and `ψ'' = (x² − 1)ψ` (the two theorems above),
`−½ψ'' + ½x²ψ = ½ψ`: the lowest energy is one half quantum, not zero. -/
theorem oscillator_ground_state (x : ℝ) :
    -(1 / 2) * ((x ^ 2 - 1) * Real.exp (-(1 / 2) * x ^ 2)) + 1 / 2 * x ^ 2 * Real.exp (-(1 / 2) * x ^ 2) =
      1 / 2 * Real.exp (-(1 / 2) * x ^ 2) := by
  ring

/-! ### Transport in the mean-free-path picture -/

/-- **The simplest transport theory.** Molecules (number density `n`, mass
`m`, speed `v`, free path `l`, heat capacity `c` each) carry momentum and
energy the same way, so viscosity `η = ⅓ n m v l` and heat conduction
`κ = ⅓ n v l c` have the ratio `κ/η = c/m`. Measured: 2.5 times that. -/
theorem simple_transport_ratio {n m v l c : ℝ} (hn : n ≠ 0) (hm : m ≠ 0) (hv : v ≠ 0) (hl : l ≠ 0) :
    (1 / 3 * n * v * l * c) / (1 / 3 * n * m * v * l) = c / m := by
  field_simp

/-! ### A magnetic field inside a superconductor -/

/-- First derivative of an exponentially decaying field. -/
theorem london_decay_dx (B₀ lam x : ℝ) :
    HasDerivAt (fun x : ℝ => B₀ * Real.exp (-x / lam)) (-(1 / lam) * (B₀ * Real.exp (-x / lam))) x := by
  have h : HasDerivAt (fun x : ℝ => -x / lam) (-(1 / lam)) x := by
    have h0 := ((hasDerivAt_id x).neg).div_const lam
    convert h0 using 1
    · rfl
    · ring
  convert (h.exp).const_mul B₀ using 1
  ring

/-- **London's equation.** `B = B₀ exp(−x/λ)` satisfies `B'' = B/λ²`: the
field dies out over the depth `λ`. -/
theorem london_decay (B₀ lam x : ℝ) :
    HasDerivAt (fun x : ℝ => -(1 / lam) * (B₀ * Real.exp (-x / lam)))
      (1 / lam ^ 2 * (B₀ * Real.exp (-x / lam))) x := by
  convert (london_decay_dx B₀ lam x).const_mul (-(1 / lam)) using 1
  ring

/-- **Why an atom shields only a little.** With the penetration depth
`λ² = m/(μ₀ n e²)`, the Bohr radius `a₀ = ħ/(m c α)` and the fine-structure
constant `α = μ₀ e² c/(4πħ)`, an electron cloud of density `n` has
`(a₀/λ)² = 4π α² · n a₀³`. For one electron per `a₀³` the atom is `1/(α√(4π))`,
about 39, times thinner than the depth, and the shielding is of order `α²`. -/
theorem london_depth_atomic {m c ħ μ₀ e n α a₀ lam : ℝ} (hm : 0 < m) (hc : 0 < c) (hħ : 0 < ħ)
    (hμ : 0 < μ₀) (he : 0 < e) (hn : 0 < n)
    (hα : α = μ₀ * e ^ 2 * c / (4 * Real.pi * ħ)) (ha : a₀ = ħ / (m * c * α))
    (hl : lam ^ 2 = m / (μ₀ * n * e ^ 2)) :
    a₀ ^ 2 / lam ^ 2 = 4 * Real.pi * α ^ 2 * (n * a₀ ^ 3) := by
  have hπ := Real.pi_pos
  rw [hl, ha, hα]
  field_simp

end Movement

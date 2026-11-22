import Movement.Waveguide

/-!
# 10. Formulas behind the atom and molecule analyzer

The script `analyzer/analyze.py` evaluates the formulas below with measured
constants and compares the results with measured data. This file proves that
each formula follows from the model it claims to come from.

Lean checks the derivations. Whether the models match nature is decided by the
comparison with measurements, and for several systems they do not.
-/

namespace Movement

/-! ### One-electron atoms -/

/-- **Nuclear charge `Z`.** Replacing the force constant `K₀` by `Z·K₀` in
`bohr_orbit` multiplies every energy level by `Z²`. -/
theorem hydrogenlike_energy (m K₀ N Z : ℝ) :
    -(m * (Z * K₀) ^ 2) / (2 * N ^ 2) = Z ^ 2 * (-(m * K₀ ^ 2) / (2 * N ^ 2)) := by
  ring

/-- **Rydberg formula.** With levels `Eₙ = −R/n²`, the photon emitted in a
jump from `n₂` to `n₁` has energy `R·(1/n₁² − 1/n₂²)`. -/
theorem rydberg_formula {R : ℝ} {n₁ n₂ : ℕ} (h1 : n₁ ≠ 0) (h2 : n₂ ≠ 0) :
    -R / (n₂ : ℝ) ^ 2 - -R / (n₁ : ℝ) ^ 2 = R * (1 / (n₁ : ℝ) ^ 2 - 1 / (n₂ : ℝ) ^ 2) := by
  have : (n₁ : ℝ) ≠ 0 := by exact_mod_cast h1
  have : (n₂ : ℝ) ≠ 0 := by exact_mod_cast h2
  field_simp
  ring

/-- The Dirac ground-state binding energy `mc²·(1 − √(1 − x))`, `x = (Zα)²`,
is never smaller than the Bohr value `mc²·x/2`: relativity binds tighter. -/
theorem dirac_binding_ge_bohr {x : ℝ} (hx1 : x ≤ 1) :
    x / 2 ≤ 1 - √(1 - x) := by
  have h : √(1 - x) ≤ 1 - x / 2 := by
    rw [Real.sqrt_le_iff]
    constructor
    · linarith
    · nlinarith [sq_nonneg x]
  linarith

/-! ### Two-electron atoms -/

/-- **Helium, variational estimate.** With one screening parameter `ζ` the
energy (in units of 2 Rydberg) is `ζ² − 2Zζ + (5/8)ζ`. Its minimum is
`−(Z − 5/16)²`, reached at `ζ = Z − 5/16`. The energy expression itself comes
from integrals that are not formalised here. -/
theorem helium_variational (Z ζ : ℝ) :
    -(Z - 5 / 16) ^ 2 ≤ ζ ^ 2 - 2 * Z * ζ + 5 / 8 * ζ ∧
      (Z - 5 / 16) ^ 2 - 2 * Z * (Z - 5 / 16) + 5 / 8 * (Z - 5 / 16) = -(Z - 5 / 16) ^ 2 := by
  constructor
  · nlinarith [sq_nonneg (ζ - (Z - 5 / 16))]
  · ring

/-! ### Rotating molecules -/

/-- **Reduced mass.** Two masses a distance `r` apart, rotating about their
common centre of mass, have moment of inertia `μ r²` with
`μ = m₁m₂/(m₁+m₂)`. -/
theorem reduced_mass_inertia {m₁ m₂ r₁ r₂ r : ℝ} (hm : 0 < m₁ + m₂)
    (hcm : m₁ * r₁ = m₂ * r₂) (hr : r₁ + r₂ = r) :
    m₁ * r₁ ^ 2 + m₂ * r₂ ^ 2 = m₁ * m₂ / (m₁ + m₂) * r ^ 2 := by
  subst hr
  rw [eq_comm, div_mul_eq_mul_div, div_eq_iff hm.ne']
  have e1 : m₁ * r₁ - m₂ * r₂ = 0 := by linarith
  linear_combination (m₂ * r₂ - m₁ * r₁) * e1

/-- **Rigid rotor.** With levels `B·J(J+1)`, neighbouring levels are
`2B(J+1)` apart, so the spectral lines are evenly spaced by `2B`. -/
theorem rotor_spacing (B : ℝ) (J : ℕ) :
    B * ((J : ℝ) + 1) * ((J : ℝ) + 2) - B * J * ((J : ℝ) + 1) = 2 * B * ((J : ℝ) + 1) := by
  ring

/-- **Isotope rule.** At the same bond length the rotational constant is
inversely proportional to the reduced mass. -/
theorem isotope_rotational_constant {C μ₁ μ₂ r : ℝ} (hC : C ≠ 0) (h1 : μ₁ ≠ 0) (h2 : μ₂ ≠ 0)
    (hr : r ≠ 0) : C / (μ₂ * r ^ 2) = C / (μ₁ * r ^ 2) * (μ₁ / μ₂) := by
  field_simp

/-- **Rotor that stretches.** With levels `B·J(J+1) − D·J²(J+1)²` (the
molecule stretches as it spins), neighbouring levels are
`2B(J+1) − 4D(J+1)³` apart. -/
theorem rotor_spacing_distortion (B D : ℝ) (J : ℕ) :
    (B * ((J : ℝ) + 1) * ((J : ℝ) + 2) - D * (((J : ℝ) + 1) * ((J : ℝ) + 2)) ^ 2) -
        (B * J * ((J : ℝ) + 1) - D * ((J : ℝ) * ((J : ℝ) + 1)) ^ 2) =
      2 * B * ((J : ℝ) + 1) - 4 * D * ((J : ℝ) + 1) ^ 3 := by
  ring

/-! ### Why numerical energies are upper bounds -/

/-- **Rayleigh–Ritz bound.** Write any trial state in the basis of true
energy states, with amplitudes `c i` and energies `E i`. Its energy
`Σ Eᵢcᵢ²` is at least the ground-state energy `E₀` times its norm `Σ cᵢ²`.
So every energy found by minimising over trial states (Hartree–Fock,
configuration interaction, Hylleraas) lies above the true one. -/
theorem rayleigh_ritz_bound {ι : Type*} (s : Finset ι) (E c : ι → ℝ) (E₀ : ℝ)
    (h : ∀ i ∈ s, E₀ ≤ E i) :
    E₀ * ∑ i ∈ s, c i ^ 2 ≤ ∑ i ∈ s, E i * c i ^ 2 := by
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun i hi => mul_le_mul_of_nonneg_right (h i hi) (sq_nonneg _)

/-! ### Heat capacity from counting motions -/

/-- **Equipartition.** A gas whose molecules have `f` ways to move
(3 for an atom, 5 for a rotating diatomic molecule) has
`C_v = f/2·R`, `C_p = C_v + R` and ratio `γ = (f + 2)/f`. -/
theorem heat_capacity_ratio {f R : ℝ} (hf : f ≠ 0) (hR : R ≠ 0) :
    (f / 2 * R + R) / (f / 2 * R) = (f + 2) / f := by
  field_simp

/-- A monatomic gas has `γ = 5/3`; a rotating diatomic gas has `γ = 7/5`. -/
theorem gamma_values : ((3 : ℝ) + 2) / 3 = 5 / 3 ∧ ((5 : ℝ) + 2) / 5 = 7 / 5 := by
  constructor <;> norm_num

end Movement

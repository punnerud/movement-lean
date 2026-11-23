import Mathlib

/-!
# 11. Forces between atoms: what can be proved

The analyzer computes bonds and the weak attraction between neutral atoms
numerically. Numbers from a computer run are not theorems, but the
principles the numbers rest on are. This file proves those.
-/

namespace Movement

/-! ### Equal spins: the Pauli hole -/

/-- **Two electrons with equal spin are never at the same point.** Their
pair function changes sign when the two are swapped, so it is zero where
they coincide. This is why H₂ with equal spins has no bond: the electrons
cannot gather between the nuclei. -/
theorem antisymmetric_vanishes_at_contact {α : Type*} (ψ : α → α → ℝ)
    (h : ∀ x y, ψ x y = -ψ y x) (x : α) : ψ x x = 0 := by
  have := h x x
  linarith

/-! ### Why neutral atoms attract -/

/-- **Mixing in an excited state lowers the energy.** Two states with
energies `E₀ < E₁` coupled by `c` have a lowest level `lam`, the lower root
of `(E₀ - lam)(E₁ - lam) = c²`. It lies below `E₀`, by at most `c²/(E₁ - E₀)`,
which is the second-order perturbation estimate. -/
theorem two_level_lowering {E₀ E₁ c lam : ℝ} (hgap : E₀ < E₁)
    (h : (E₀ - lam) * (E₁ - lam) = c ^ 2) (hl : lam < E₁) :
    lam ≤ E₀ ∧ E₀ - c ^ 2 / (E₁ - E₀) ≤ lam := by
  have h1 : 0 < E₁ - lam := by linarith
  have h0 : 0 ≤ E₀ - lam := by
    by_contra hneg
    have := mul_neg_of_neg_of_pos (not_le.mp hneg) h1
    nlinarith [sq_nonneg c]
  refine ⟨by linarith, ?_⟩
  have hb : E₀ - lam ≤ c ^ 2 / (E₁ - E₀) := by
    rw [le_div_iff₀ (by linarith)]
    nlinarith [mul_le_mul_of_nonneg_left (show E₁ - E₀ ≤ E₁ - lam by linarith) h0]
  linarith

/-- **Dispersion always attracts.** The second-order energy
`-Σ cᵢ²/gapᵢ` over excited states with positive excitation energies is
never positive, whatever the couplings are. -/
theorem dispersion_nonpos {ι : Type*} (s : Finset ι) (c gap : ι → ℝ)
    (h : ∀ i ∈ s, 0 < gap i) : -∑ i ∈ s, c i ^ 2 / gap i ≤ 0 := by
  have : 0 ≤ ∑ i ∈ s, c i ^ 2 / gap i :=
    Finset.sum_nonneg fun i hi => div_nonneg (sq_nonneg _) (h i hi).le
  linarith

/-- **Casimir–Polder, the algebra.** The product of two response functions
`1/(a² + x²)` and `1/(b² + x²)` splits into a difference of single ones.
Integrating each over all imaginary frequencies `x` gives `π/(2a)` and
`π/(2b)`. -/
theorem casimir_polder_partial_fraction {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b)
    (x : ℝ) :
    1 / ((a ^ 2 + x ^ 2) * (b ^ 2 + x ^ 2))
      = 1 / (b ^ 2 - a ^ 2) * (1 / (a ^ 2 + x ^ 2) - 1 / (b ^ 2 + x ^ 2)) := by
  have h1 : a ^ 2 + x ^ 2 ≠ 0 := by positivity
  have h2 : b ^ 2 + x ^ 2 ≠ 0 := by positivity
  have h3 : b ^ 2 - a ^ 2 ≠ 0 := by
    have : b ^ 2 - a ^ 2 = (b - a) * (b + a) := by ring
    rw [this]
    exact mul_ne_zero (sub_ne_zero.mpr (Ne.symm hab)) (by positivity)
  field_simp
  ring

/-- **Casimir–Polder, the result.** With the two integrals inserted, the
combination collapses to `π / (2ab(a + b))`: each pair of excited states
contributes in proportion to `1/(a + b)`, the sum of the two excitation
energies. This is where the formula `C₆ = 6 Σ aₘaₙ/(wₘ + wₙ)` comes from. -/
theorem casimir_polder_value {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
    1 / (b ^ 2 - a ^ 2) * (Real.pi / (2 * a) - Real.pi / (2 * b))
      = Real.pi / (2 * a * b * (a + b)) := by
  have h3 : b ^ 2 - a ^ 2 ≠ 0 := by
    have : b ^ 2 - a ^ 2 = (b - a) * (b + a) := by ring
    rw [this]
    exact mul_ne_zero (sub_ne_zero.mpr (Ne.symm hab)) (by positivity)
  have h4 : b - a ≠ 0 := sub_ne_zero.mpr (Ne.symm hab)
  have h5 : a + b ≠ 0 := by positivity
  rw [div_mul_eq_mul_div, one_mul, div_eq_div_iff h3 (by positivity)]
  field_simp
  ring

/-- **London's formula.** If an atom had a single excited state, with
squared transition moment `a` and excitation energy `w`, its polarizability
would be `α = 2a/w` and the pair sum `6·a·a/(w + w)` equals `¾ α² w`. -/
theorem london_formula {a w : ℝ} (hw : 0 < w) :
    6 * (a * a / (w + w)) = 3 / 4 * (2 * a / w) ^ 2 * w := by
  field_simp
  ring

/-- **The damping factor stays between 0 and 1.** The Tang–Toennies factor
`1 - e^(-x) Σ_{k<n} x^k/k!` only ever weakens an attraction term; it never
reverses or amplifies it. -/
theorem tang_toennies_bounds {x : ℝ} (hx : 0 ≤ x) (n : ℕ) :
    0 ≤ 1 - Real.exp (-x) * ∑ k ∈ Finset.range n, x ^ k / (k.factorial : ℝ) ∧
      1 - Real.exp (-x) * ∑ k ∈ Finset.range n, x ^ k / (k.factorial : ℝ) ≤ 1 := by
  have hs := Real.sum_le_exp_of_nonneg hx n
  have hpos : 0 ≤ ∑ k ∈ Finset.range n, x ^ k / (k.factorial : ℝ) :=
    Finset.sum_nonneg fun k _ => by positivity
  have he : 0 < Real.exp (-x) := Real.exp_pos _
  have hle := mul_le_mul_of_nonneg_left hs he.le
  rw [← Real.exp_add, neg_add_cancel, Real.exp_zero] at hle
  exact ⟨by linarith, by nlinarith [mul_nonneg he.le hpos]⟩

/-! ### What a truncated calculation leaves out -/

/-- **The second-order estimate brackets the energy.** A state of energy
`E₀` is coupled by `c j` to outside states of energies `E j`, which are not
coupled to each other. The lowest level `lam` then satisfies
`lam = E₀ - Σ c j² / (E j - lam)`. It lies between `E₀` and the estimate
`E₀ - Σ c j² / (E j - E₀)`: leaving the outside states out gives too high an
energy, and the second-order estimate over-corrects. This is the formula the
analyzer uses to estimate what a selected set of determinants leaves out. -/
theorem second_order_estimate_brackets {ι : Type*} (s : Finset ι) (c E : ι → ℝ) {E₀ lam : ℝ}
    (hgap : ∀ j ∈ s, E₀ < E j) (hlow : ∀ j ∈ s, lam < E j)
    (hsec : lam = E₀ - ∑ j ∈ s, c j ^ 2 / (E j - lam)) :
    E₀ - ∑ j ∈ s, c j ^ 2 / (E j - E₀) ≤ lam ∧ lam ≤ E₀ := by
  have hpos : 0 ≤ ∑ j ∈ s, c j ^ 2 / (E j - lam) :=
    Finset.sum_nonneg fun j hj => div_nonneg (sq_nonneg _) (by linarith [hlow j hj])
  have hle : lam ≤ E₀ := by linarith
  have hsum : ∑ j ∈ s, c j ^ 2 / (E j - lam) ≤ ∑ j ∈ s, c j ^ 2 / (E j - E₀) :=
    Finset.sum_le_sum fun j hj =>
      div_le_div_of_nonneg_left (sq_nonneg _) (by linarith [hgap j hj]) (by linarith)
  exact ⟨by linarith, hle⟩

/-! ### The ionic bond -/

/-- **Where the electron jumps back.** Two ions attract with energy `K/r`.
Making the ions from neutral atoms costs `gap` (ionization energy minus
electron affinity). The two balance at exactly `r = K/gap`; further out,
neutral atoms are cheaper. -/
theorem coulomb_crossing {K gap r : ℝ} (hg : 0 < gap) (hr : 0 < r) :
    K / r = gap ↔ r = K / gap := by
  rw [div_eq_iff hr.ne', eq_div_iff hg.ne']
  constructor <;> intro h <;> linarith [mul_comm gap r]

/-- **Depth of an ionic well.** For `V(r) = -K/r + A e^(-r/ρ)` the force
vanishes where `A e^(-r/ρ)/ρ = K/r²`. At that distance the energy is the
Coulomb energy reduced by the fraction `ρ/r` (Born–Mayer). -/
theorem born_mayer_well {K A ρ r : ℝ} (hr : 0 < r) (hρ : 0 < ρ)
    (heq : A * Real.exp (-r / ρ) / ρ = K / r ^ 2) :
    -K / r + A * Real.exp (-r / ρ) = -(K / r) * (1 - ρ / r) := by
  have h2 : r ^ 2 ≠ 0 := pow_ne_zero 2 hr.ne'
  rw [div_eq_div_iff hρ.ne' h2] at heq
  have h : A * Real.exp (-r / ρ) = K * ρ / r ^ 2 := by
    rw [eq_div_iff h2]
    linarith
  rw [h]
  field_simp
  ring

/-- **Two ions polarise each other.** Each ion's induced dipole is its
polarizability times the field of the other ion's charge and induced
dipole. Solving the two equations gives the total induced dipole used in
the Rittner model. -/
theorem rittner_induced_dipoles {q r a₁ a₂ μ₁ μ₂ : ℝ} (hr : 0 < r)
    (hd : r ^ 6 - 4 * a₁ * a₂ ≠ 0)
    (h1 : μ₁ = a₁ * (q / r ^ 2 + 2 * μ₂ / r ^ 3))
    (h2 : μ₂ = a₂ * (q / r ^ 2 + 2 * μ₁ / r ^ 3)) :
    μ₁ + μ₂ = q * (r ^ 4 * (a₁ + a₂) + 4 * r * a₁ * a₂) / (r ^ 6 - 4 * a₁ * a₂) := by
  have hr' : r ≠ 0 := hr.ne'
  have e1 : μ₁ * r ^ 3 = a₁ * (q * r + 2 * μ₂) := by
    have h : μ₁ * r ^ 3 = a₁ * (q / r ^ 2 + 2 * μ₂ / r ^ 3) * r ^ 3 := by rw [← h1]
    rw [h]
    field_simp
  have e2 : μ₂ * r ^ 3 = a₂ * (q * r + 2 * μ₁) := by
    have h : μ₂ * r ^ 3 = a₂ * (q / r ^ 2 + 2 * μ₁ / r ^ 3) * r ^ 3 := by rw [← h2]
    rw [h]
    field_simp
  rw [eq_div_iff hd]
  linear_combination (r ^ 3 + 2 * a₂) * e1 + (r ^ 3 + 2 * a₁) * e2

/-! ### Heat capacity of a vibration -/

/-- **A quantised vibration holds at most the classical share of heat.**
Einstein's function `x² eˣ/(eˣ - 1)²`, with `x = hν/kT`, never exceeds 1:
a vibration contributes at most `R` to the heat capacity, and less when the
quantum of vibration is large compared with `kT`. -/
theorem einstein_function_le_one {x : ℝ} (hx : 0 < x) :
    x ^ 2 * Real.exp x / (Real.exp x - 1) ^ 2 ≤ 1 := by
  have hpos : 0 < Real.exp x - 1 := by
    have := Real.add_one_lt_exp (ne_of_gt hx)
    linarith
  rw [div_le_one (by positivity)]
  set y := Real.exp (x / 2) with hy
  have hypos : 0 < y := Real.exp_pos _
  have hexp : Real.exp x = y ^ 2 := by
    rw [hy, ← Real.exp_nat_mul]
    congr 1
    push_cast
    ring
  have hs : x / 2 ≤ Real.sinh (x / 2) := Real.self_le_sinh_iff.mpr (by linarith)
  rw [Real.sinh_eq, Real.exp_neg, ← hy] at hs
  have hxy : x * y ≤ y ^ 2 - 1 := by
    have h' : x ≤ y - y⁻¹ := by linarith
    calc x * y ≤ (y - y⁻¹) * y := mul_le_mul_of_nonneg_right h' hypos.le
      _ = y ^ 2 - 1 := by field_simp
  have hsq := mul_self_le_mul_self (mul_nonneg hx.le hypos.le) hxy
  rw [hexp]
  nlinarith [hsq]

end Movement

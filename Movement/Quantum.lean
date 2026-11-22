import Movement.Heat

/-!
# 5. From diffusion to quantum mechanics

The free Schrödinger equation `iħ ∂ₜψ = −(ħ²/2m) ∂ₓₓψ` is the heat equation
`∂ₜu = D ∂ₓₓu` with an **imaginary diffusion coefficient** `D = iħ/2m`
(equivalently: in imaginary time, a "Wick rotation").

* `mode_solves`: Fourier modes solve the diffusion equation for any complex `c`.
* `schrodinger_mode`: with `c = iħ/2m` that is the Schrödinger equation.
* `norm_heat_mode` / `norm_schrodinger_mode`: heat modes decay, quantum modes
  keep norm 1. Diffusion forgets; quantum evolution only rotates phase.
* `wick_rotation`, `heat_to_schrodinger`: any solution of the heat equation
  that extends to complex time gives a solution of the Schrödinger equation.
-/

namespace Movement

open Complex

/-- A Fourier mode `exp(−c k² t) · exp(i k x)` with complex diffusion
coefficient `c`. -/
noncomputable def mode (c : ℂ) (k t x : ℝ) : ℂ :=
  cexp (-(c * k ^ 2 * t)) * cexp (I * k * x)

theorem mode_dt (c : ℂ) (k t x : ℝ) :
    HasDerivAt (fun t : ℝ => mode c k t x) (-(c * k ^ 2) * mode c k t x) t := by
  have h0 : HasDerivAt (fun t : ℝ => c * k ^ 2 * (t : ℂ)) (c * k ^ 2) t := by
    simpa using (hasDerivAt_id t).ofReal_comp.const_mul (c * k ^ 2)
  have h : HasDerivAt (fun t : ℝ => -(c * k ^ 2 * (t : ℂ))) (-(c * k ^ 2)) t := h0.neg
  have h1 : HasDerivAt (fun t : ℝ => cexp (-(c * k ^ 2 * (t : ℂ))) * cexp (I * k * x)) _ t :=
    h.cexp.mul_const (cexp (I * k * x))
  show HasDerivAt (fun t : ℝ => cexp (-(c * k ^ 2 * (t : ℂ))) * cexp (I * k * x)) _ t
  refine h1.congr_deriv ?_
  unfold mode; ring

theorem mode_dx (c : ℂ) (k t x : ℝ) :
    HasDerivAt (fun x : ℝ => mode c k t x) (I * k * mode c k t x) x := by
  have h : HasDerivAt (fun x : ℝ => I * k * (x : ℂ)) (I * k) x := by
    have := (hasDerivAt_id x).ofReal_comp.const_mul (I * k)
    simpa using this
  have h1 : HasDerivAt (fun x : ℝ => cexp (-(c * k ^ 2 * t)) * cexp (I * k * (x : ℂ))) _ x :=
    h.cexp.const_mul (cexp (-(c * k ^ 2 * t)))
  show HasDerivAt (fun x : ℝ => cexp (-(c * k ^ 2 * t)) * cexp (I * k * (x : ℂ))) _ x
  refine h1.congr_deriv ?_
  unfold mode; ring

theorem mode_dxx (c : ℂ) (k t x : ℝ) :
    HasDerivAt (fun x : ℝ => I * k * mode c k t x) (-(k : ℂ) ^ 2 * mode c k t x) x := by
  have := (mode_dx c k t x).const_mul (I * k)
  convert this using 1
  linear_combination (-(k : ℂ) ^ 2 * mode c k t x) * I_sq

/-- Every Fourier mode solves `∂ₜu = c · ∂ₓₓu`, for real **or complex** `c`. -/
theorem mode_solves (c : ℂ) (k t x : ℝ) :
    -(c * k ^ 2) * mode c k t x = c * (-(k : ℂ) ^ 2 * mode c k t x) := by
  ring

/-- **Schrödinger equation** for a free particle: with `c = iħ/2m`, the mode
satisfies `iħ ∂ₜψ = −(ħ²/2m) ∂ₓₓψ`. -/
theorem schrodinger_mode {ħ m : ℝ} (hm : m ≠ 0) (k t x : ℝ) :
    let c : ℂ := I * ħ / (2 * m)
    I * ħ * (-(c * k ^ 2) * mode c k t x) =
      -((ħ : ℂ) ^ 2 / (2 * m)) * (-(k : ℂ) ^ 2 * mode c k t x) := by
  intro c
  have hm' : (m : ℂ) ≠ 0 := by exact_mod_cast hm
  simp only [c]
  field_simp
  linear_combination (-((ħ : ℂ) ^ 2 * k ^ 2 * mode (I * ħ / (2 * m)) k t x)) * I_sq

/-- **Diffusion forgets.** A heat mode decays: its size is `exp(−D k² t)`. -/
theorem norm_heat_mode (D k t x : ℝ) :
    ‖mode D k t x‖ = Real.exp (-(D * k ^ 2 * t)) := by
  simp only [mode, norm_mul, Complex.norm_exp]
  have h1 : (-((D : ℂ) * k ^ 2 * t)).re = -(D * k ^ 2 * t) := by
    simp [← ofReal_pow, ← ofReal_mul]
  have h2 : (I * k * x).re = 0 := by simp
  rw [h1, h2, Real.exp_zero, mul_one]

/-- **Quantum evolution remembers.** A Schrödinger mode never decays: its
norm is exactly 1 at all times (only the phase rotates). -/
theorem norm_schrodinger_mode (κ k t x : ℝ) :
    ‖mode (I * κ) k t x‖ = 1 := by
  simp only [mode, norm_mul, Complex.norm_exp]
  have h1 : (-(I * κ * k ^ 2 * t)).re = 0 := by
    simp [← ofReal_pow, ← ofReal_mul, mul_assoc]
  have h2 : (I * k * x).re = 0 := by simp
  rw [h1, h2, Real.exp_zero, mul_one]

/-- **Wick rotation** (chain rule): substituting `s = c·t` for complex `c`. -/
theorem wick_rotation {φ : ℂ → ℂ} {φ' : ℂ} (c : ℂ) (t : ℝ)
    (h : HasDerivAt φ φ' (c * t)) :
    HasDerivAt (fun t : ℝ => φ (c * t)) (c * φ') t := by
  have h1 : HasDerivAt (fun s : ℂ => c * s) c (t : ℂ) := by
    simpa using (hasDerivAt_id (t : ℂ)).const_mul c
  have h2 : HasDerivAt (fun s : ℂ => φ (c * s)) (φ' * c) (t : ℂ) := h.comp (t : ℂ) h1
  have h3 : HasDerivAt (fun t : ℝ => φ (c * (t : ℂ))) (φ' * c) t := h2.comp_ofReal
  exact h3.congr_deriv (mul_comm _ _)

/-- **Heat equation in imaginary time is the Schrödinger equation.**
If `φ(s, x)` solves `∂ₛφ = D ∂ₓₓφ` for complex time `s`, then
`ψ(t, x) = φ(i t, x)` solves `∂ₜψ = i D ∂ₓₓψ`. -/
theorem heat_to_schrodinger {φ φs φxx : ℂ → ℝ → ℂ} {D : ℂ}
    (hs : ∀ s x, HasDerivAt (fun s => φ s x) (φs s x) s)
    (heat : ∀ s x, φs s x = D * φxx s x) (t x : ℝ) :
    HasDerivAt (fun t : ℝ => φ (I * t) x) (I * D * φxx (I * t) x) t := by
  have := wick_rotation I t (hs (I * t) x)
  rw [heat] at this
  convert this using 1
  ring

/-- With `D = ħ/2m` the rotated equation is `iħ ∂ₜψ = −(ħ²/2m) ∂ₓₓψ`. -/
theorem heat_to_schrodinger_units {ħ m : ℝ} (hm : m ≠ 0) (ψxx : ℂ) :
    I * ħ * (I * (ħ / (2 * m) : ℂ) * ψxx) = -((ħ : ℂ) ^ 2 / (2 * m)) * ψxx := by
  have hm' : (m : ℂ) ≠ 0 := by exact_mod_cast hm
  field_simp
  linear_combination ((ħ : ℂ) ^ 2 * ψxx) * I_sq

end Movement

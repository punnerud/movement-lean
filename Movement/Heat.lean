import Movement.Basic

/-!
# 4. From the random walk to the heat equation

* `master_equation`: the number of paths ending at `x` obeys
  `N(n+1, x) = N(n, x-1) + N(n, x+1)`.
* `discrete_heat`: the probabilities therefore satisfy a discrete heat equation.
* `diffusive_scaling`: with grid spacing `h` and time step `τ` this is a finite
  difference heat equation with diffusion coefficient `D = h² / 2τ`.
* `second_difference_tendsto`: the discrete Laplacian converges to `f''`.
* `heatKernel_solves`: the Gaussian `exp(-x²/4Dt)/√t` solves `∂ₜu = D ∂ₓₓu`.
-/

namespace Movement

open Filter Topology

/-! ### The master equation -/

section Master

theorem count_step (L : List ℤ) (x : ℤ) :
    (step L).count x = L.count (x - 1) + L.count (x + 1) := by
  induction L with
  | nil => rfl
  | cons a L ih =>
    simp only [step, List.flatMap_cons] at ih ⊢
    rw [List.count_append, ih]
    simp only [List.count_cons, List.count_nil, beq_iff_eq]
    split_ifs <;> omega

/-- Number of ±1 paths of length `n` that end at `x`. -/
def paths (n : ℕ) (x : ℤ) : ℕ := (positions n).count x

/-- **Master equation.** To be at `x` after `n+1` steps you were at `x-1`
or `x+1` after `n` steps. -/
theorem master_equation (n : ℕ) (x : ℤ) :
    paths (n + 1) x = paths n (x - 1) + paths n (x + 1) :=
  count_step _ _

/-- Probability of being at `x` after `n` steps. -/
noncomputable def prob (n : ℕ) (x : ℤ) : ℝ := paths n x / 2 ^ n

theorem prob_step (n : ℕ) (x : ℤ) :
    prob (n + 1) x = (prob n (x - 1) + prob n (x + 1)) / 2 := by
  simp only [prob, master_equation, Nat.cast_add, pow_succ]
  field_simp

/-- **Discrete heat equation.** The change in probability over one time step
is half the discrete Laplacian (second difference) in space. -/
theorem discrete_heat (n : ℕ) (x : ℤ) :
    prob (n + 1) x - prob n x =
      (1 / 2) * (prob n (x + 1) - 2 * prob n x + prob n (x - 1)) := by
  rw [prob_step]; ring

end Master

/-! ### Diffusive scaling: `D = h² / 2τ` -/

section Scaling

/-- If a field on a grid (spacing `h`, time step `τ`) is updated by averaging
its two neighbours, it satisfies the finite-difference heat equation
`Δₜu/τ = D · Δₓ²u/h²` with `D = h²/(2τ)`. -/
theorem diffusive_scaling {u : ℝ → ℝ → ℝ} {h τ t x : ℝ} (hh : h ≠ 0) (hτ : τ ≠ 0)
    (hstep : u (t + τ) x = (u t (x - h) + u t (x + h)) / 2) :
    (u (t + τ) x - u t x) / τ =
      h ^ 2 / (2 * τ) * ((u t (x + h) - 2 * u t x + u t (x - h)) / h ^ 2) := by
  rw [hstep]; field_simp; ring

/-- **Einstein relation.** `n` steps of length `h`, one every `τ` seconds:
the mean squared displacement `n·h²` equals `2Dt` with `t = nτ`. -/
theorem einstein_relation {h τ : ℝ} (n : ℕ) (hτ : τ ≠ 0) :
    (n : ℝ) * h ^ 2 = 2 * (h ^ 2 / (2 * τ)) * (n * τ) := by
  field_simp

end Scaling

/-! ### Continuum limit of the discrete Laplacian -/

section Limit

/-- The second difference quotient converges to the second derivative:
`(f(x+h) − 2f(x) + f(x−h)) / h² → f''(x)` as `h → 0`. -/
theorem second_difference_tendsto {f f' : ℝ → ℝ} {f'' x : ℝ}
    (hf : ∀ y, HasDerivAt f (f' y) y) (hf' : HasDerivAt f' f'' x) :
    Tendsto (fun h => (f (x + h) - 2 * f x + f (x - h)) / h ^ 2) (𝓝[≠] 0) (𝓝 f'') := by
  -- numerator and its derivative
  have hN : ∀ h, HasDerivAt (fun h => f (x + h) - 2 * f x + f (x - h))
      (f' (x + h) - f' (x - h)) h := by
    intro h
    have h1 := (hf (x + h)).comp_const_add x h
    have h2 := (hf (x - h)).comp_const_sub x h
    have := (h1.sub_const (2 * f x)).add h2
    exact this
  have hD : ∀ h : ℝ, HasDerivAt (fun h : ℝ => h ^ 2) (2 * h) h := by
    intro h; simpa using hasDerivAt_pow 2 h
  -- derivative of the numerator's derivative at 0 is 2 f''
  have hG : HasDerivAt (fun h => f' (x + h) - f' (x - h)) (2 * f'') 0 := by
    have h1 : HasDerivAt (fun h => f' (x + h)) f'' 0 :=
      HasDerivAt.comp_const_add x 0 (by simpa using hf')
    have h2 : HasDerivAt (fun h => f' (x - h)) (-f'') 0 :=
      HasDerivAt.comp_const_sub x 0 (by simpa using hf')
    convert h1.sub h2 using 1; ring
  have hlim : Tendsto (fun h => (f' (x + h) - f' (x - h)) / (2 * h)) (𝓝[≠] 0) (𝓝 f'') := by
    have := hG.tendsto_slope_zero
    have h2 : Tendsto (fun t : ℝ => (1 / 2 : ℝ) * (t⁻¹ • ((f' (x + (0 + t)) - f' (x - (0 + t)))
        - (f' (x + 0) - f' (x - 0))))) (𝓝[≠] 0) (𝓝 ((1 / 2) * (2 * f''))) :=
      this.const_mul _
    have e : (1 / 2 : ℝ) * (2 * f'') = f'' := by ring
    rw [e] at h2
    refine h2.congr fun t => ?_
    simp only [zero_add, add_zero, sub_zero, sub_self, smul_eq_mul]
    ring
  refine HasDerivAt.lhopital_zero_nhdsNE (f' := fun h => f' (x + h) - f' (x - h))
    (g' := fun h => 2 * h) (Eventually.of_forall hN) (Eventually.of_forall hD) ?_ ?_ ?_ hlim
  · filter_upwards [self_mem_nhdsWithin] with h hh
    exact mul_ne_zero two_ne_zero hh
  · have : Tendsto (fun h => f (x + h) - 2 * f x + f (x - h)) (𝓝 0) (𝓝 0) := by
      have := (hN 0).continuousAt.tendsto
      simpa [two_mul] using this
    exact this.mono_left nhdsWithin_le_nhds
  · have : Tendsto (fun h : ℝ => h ^ 2) (𝓝 0) (𝓝 0) := by
      have := (hD 0).continuousAt.tendsto
      simpa using this
    exact this.mono_left nhdsWithin_le_nhds

end Limit

/-! ### The heat kernel solves the heat equation -/

section Kernel

/-- The Gaussian heat kernel (up to a constant factor): the shape a random
walk's distribution takes in the continuum limit. Its width grows like `√t`. -/
noncomputable def heatKernel (D t x : ℝ) : ℝ := Real.exp (-x ^ 2 / (4 * D * t)) / √t

theorem heatKernel_dt {D t : ℝ} (hD : D ≠ 0) (ht : 0 < t) (x : ℝ) :
    HasDerivAt (fun t => heatKernel D t x)
      (heatKernel D t x * (x ^ 2 / (4 * D * t ^ 2) - 1 / (2 * t))) t := by
  -- write t = s² so that √t = s
  obtain ⟨s, hs0, rfl⟩ : ∃ s : ℝ, 0 < s ∧ t = s ^ 2 :=
    ⟨√t, Real.sqrt_pos.mpr ht, (Real.sq_sqrt ht.le).symm⟩
  have hsq : √(s ^ 2) = s := Real.sqrt_sq hs0.le
  have hs : √(s ^ 2) ≠ 0 := by rw [hsq]; exact hs0.ne'
  have h4 : 4 * D * s ^ 2 ≠ 0 := by positivity
  have h0 : HasDerivAt (fun t : ℝ => -x ^ 2 / (4 * D * t))
      ((0 * (4 * D * s ^ 2) - -x ^ 2 * (4 * D * 1)) / (4 * D * s ^ 2) ^ 2) (s ^ 2) :=
    (hasDerivAt_const (s ^ 2) (-x ^ 2)).div ((hasDerivAt_id (s ^ 2)).const_mul (4 * D)) h4
  have h1 : HasDerivAt (fun t : ℝ => Real.exp (-x ^ 2 / (4 * D * t)) / √t) _ (s ^ 2) :=
    h0.exp.div (Real.hasDerivAt_sqrt (pow_pos hs0 2).ne') hs
  show HasDerivAt (fun t : ℝ => Real.exp (-x ^ 2 / (4 * D * t)) / √t) _ (s ^ 2)
  refine h1.congr_deriv ?_
  simp only [heatKernel, hsq]
  field_simp
  ring

theorem heatKernel_dx (D t x : ℝ) :
    HasDerivAt (fun x => heatKernel D t x) (heatKernel D t x * (-x / (2 * D * t))) x := by
  have hin : HasDerivAt (fun x : ℝ => -x ^ 2 / (4 * D * t)) (-x / (2 * D * t)) x := by
    have := ((hasDerivAt_pow 2 x).neg).div_const (4 * D * t)
    convert this using 1
    simp; ring
  have h1 : HasDerivAt (fun x : ℝ => Real.exp (-x ^ 2 / (4 * D * t)) / √t) _ x :=
    hin.exp.div_const (√t)
  show HasDerivAt (fun x : ℝ => Real.exp (-x ^ 2 / (4 * D * t)) / √t) _ x
  refine h1.congr_deriv ?_
  simp only [heatKernel]
  ring

theorem heatKernel_dxx (D t x : ℝ) :
    HasDerivAt (fun x => heatKernel D t x * (-x / (2 * D * t)))
      (heatKernel D t x * (x ^ 2 / (4 * D ^ 2 * t ^ 2) - 1 / (2 * D * t))) x := by
  have h1 := heatKernel_dx D t x
  have h2 : HasDerivAt (fun x : ℝ => -x / (2 * D * t)) (-1 / (2 * D * t)) x :=
    ((hasDerivAt_id x).neg).div_const _
  convert h1.mul h2 using 1
  by_cases h : D * t = 0
  · rcases mul_eq_zero.mp h with rfl | rfl <;> simp
  · have hD : D ≠ 0 := left_ne_zero_of_mul h
    have ht : t ≠ 0 := right_ne_zero_of_mul h
    field_simp; ring

/-- **The heat equation.** `∂ₜu = D · ∂ₓₓu` for the Gaussian kernel: the
time derivative (from `heatKernel_dt`) equals `D` times the second space
derivative (from `heatKernel_dxx`). -/
theorem heatKernel_solves {D t : ℝ} (hD : D ≠ 0) (ht : 0 < t) (x : ℝ) :
    heatKernel D t x * (x ^ 2 / (4 * D * t ^ 2) - 1 / (2 * t)) =
      D * (heatKernel D t x * (x ^ 2 / (4 * D ^ 2 * t ^ 2) - 1 / (2 * D * t))) := by
  have := ht.ne'
  field_simp

end Kernel

end Movement

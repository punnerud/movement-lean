import Mathlib

/-!
# Motion, collisions and diffusion

1. **Collision course (CBDR).** Relative position `r t = r₀ + t • v`.
   Collision ⇔ the relative velocity points straight at you (constant bearing)
   and the range decreases.
2. **Elastic collisions in 1D.** Conservation of momentum and energy gives the
   collision formulas; equal masses swap velocities (Newton's cradle); energy
   transfer is maximal exactly when the masses are equal.
2b. **Collisions with rotation.** A body that can spin behaves like a lighter
   body with effective mass `M = m₂ I / (I + m₂ b²)`.
3. **Diffusion.** Random walk with ±1 steps: after `n` steps the mean squared
   displacement is `n`, i.e. spread ∝ √n (not ∝ n).
-/

namespace Movement

/-! ## 1. Collision course -/

section CBDR

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- Position of the other object relative to you at time `t`. -/
def rel (r₀ v : V) (t : ℝ) : V := r₀ + t • v

/-- Constant bearing: the relative velocity is `-c • r₀` with `c > 0`.
Then the object stays in the same direction (a positive multiple of `r₀`),
the range shrinks linearly, and you collide at `t = 1/c`. -/
theorem cbdr_collision (r₀ : V) {c : ℝ} (hc : 0 < c) :
    (∀ t, 0 ≤ t → t < 1 / c →
        ∃ k : ℝ, 0 < k ∧ rel r₀ (-(c • r₀)) t = k • r₀) ∧
    (∀ s t, 0 ≤ s → s < t → t ≤ 1 / c →
        ‖rel r₀ (-(c • r₀)) t‖ ≤ ‖rel r₀ (-(c • r₀)) s‖) ∧
    rel r₀ (-(c • r₀)) (1 / c) = 0 := by
  have key : ∀ t, rel r₀ (-(c • r₀)) t = (1 - t * c) • r₀ := by
    intro t
    simp only [rel]; module
  refine ⟨?_, ?_, ?_⟩
  · intro t _ ht
    refine ⟨1 - t * c, ?_, key t⟩
    have : t * c < 1 := by rwa [lt_div_iff₀ hc] at ht
    linarith
  · intro s t hs hst ht
    have htc : t * c ≤ 1 := by rwa [le_div_iff₀ hc] at ht
    rw [key, key, norm_smul, norm_smul]
    apply mul_le_mul_of_nonneg_right _ (norm_nonneg _)
    have h1 : 0 ≤ 1 - t * c := by linarith
    have h2 : s * c ≤ t * c := by nlinarith
    rw [Real.norm_of_nonneg h1, Real.norm_of_nonneg (by linarith)]
    linarith
  · rw [key]; field_simp; simp

/-- Converse: if you do collide at `t₀ > 0`, the relative velocity pointed
straight at you, and before impact the object stayed in the same direction
with strictly decreasing range. Collision ⇒ constant bearing + closing range. -/
theorem collision_implies_cbdr {r₀ v : V} (hr₀ : r₀ ≠ 0) {t₀ : ℝ} (ht₀ : 0 < t₀)
    (hcol : rel r₀ v t₀ = 0) :
    v = -((1 / t₀) • r₀) ∧
    (∀ t, 0 ≤ t → t < t₀ → ∃ k : ℝ, 0 < k ∧ rel r₀ v t = k • r₀) ∧
    (∀ s t, 0 ≤ s → s < t → t ≤ t₀ → ‖rel r₀ v t‖ < ‖rel r₀ v s‖) := by
  have hv : v = -((1 / t₀) • r₀) := by
    have h : t₀ • v = -r₀ := eq_neg_of_add_eq_zero_right hcol
    have : v = (1 / t₀) • (t₀ • v) := by
      rw [smul_smul, one_div, inv_mul_cancel₀ ht₀.ne', one_smul]
    rw [this, h, smul_neg]
  have key : ∀ t, rel r₀ v t = (1 - t / t₀) • r₀ := by
    intro t
    simp only [rel, hv]; module
  refine ⟨hv, ?_, ?_⟩
  · intro t _ ht
    refine ⟨1 - t / t₀, ?_, key t⟩
    have : t / t₀ < 1 := (div_lt_one ht₀).mpr ht
    linarith
  · intro s t hs hst ht
    rw [key, key, norm_smul, norm_smul]
    apply mul_lt_mul_of_pos_right _ (norm_pos_iff.mpr hr₀)
    have h1 : t / t₀ ≤ 1 := (div_le_one ht₀).mpr ht
    have h2 : s / t₀ < t / t₀ := div_lt_div_of_pos_right hst ht₀
    rw [Real.norm_of_nonneg (by linarith), Real.norm_of_nonneg (by linarith)]
    linarith

/-- If the relative velocity is **not** parallel to the line of sight (the
bearing drifts), you never collide: you pass each other. -/
theorem no_collision_of_not_parallel {r₀ v : V} (hr₀ : r₀ ≠ 0)
    (hpar : ∀ c : ℝ, v ≠ c • r₀) (t : ℝ) : rel r₀ v t ≠ 0 := by
  intro h
  have ht : t ≠ 0 := by
    rintro rfl; simp [rel] at h; exact hr₀ h
  apply hpar (-(1 / t))
  have h' : t • v = -r₀ := eq_neg_of_add_eq_zero_right h
  calc v = (1 / t) • (t • v) := by rw [smul_smul, one_div, inv_mul_cancel₀ ht, one_smul]
    _ = -(1 / t) • r₀ := by rw [h', smul_neg, neg_smul]

/-- If the object is completely still relative to you (same speed and heading),
the range is constant: you travel in formation and never collide. -/
theorem same_velocity_keeps_distance (r₀ : V) (t : ℝ) : rel r₀ 0 t = r₀ := by
  simp [rel]

end CBDR

/-! ## 2. Elastic collisions in one dimension -/

section Elastic

/-- Conservation of momentum and kinetic energy in a collision between
masses `m₁, m₂` with velocities `u₁, u₂` before and `v₁, v₂` after. -/
structure ElasticCollision (m₁ m₂ u₁ u₂ v₁ v₂ : ℝ) : Prop where
  pos₁ : 0 < m₁
  pos₂ : 0 < m₂
  momentum : m₁ * u₁ + m₂ * u₂ = m₁ * v₁ + m₂ * v₂
  energy : m₁ * u₁ ^ 2 + m₂ * u₂ ^ 2 = m₁ * v₁ ^ 2 + m₂ * v₂ ^ 2
  /-- they actually hit (otherwise `v = u` is a trivial solution) -/
  hit : v₁ ≠ u₁

/-- The collision formulas follow from the conservation laws alone. -/
theorem elastic_formulas {m₁ m₂ u₁ u₂ v₁ v₂ : ℝ}
    (h : ElasticCollision m₁ m₂ u₁ u₂ v₁ v₂) :
    v₁ = ((m₁ - m₂) * u₁ + 2 * m₂ * u₂) / (m₁ + m₂) ∧
    v₂ = ((m₂ - m₁) * u₂ + 2 * m₁ * u₁) / (m₁ + m₂) := by
  obtain ⟨h1, h2, hp, he, hhit⟩ := h
  -- m₁(u₁ - v₁) = m₂(v₂ - u₂)  and  m₁(u₁² - v₁²) = m₂(v₂² - u₂²)
  -- ⇒ u₁ + v₁ = u₂ + v₂  (relative velocity reverses)
  have hrel : u₁ + v₁ = u₂ + v₂ := by
    have hne : m₁ * (u₁ - v₁) ≠ 0 := mul_ne_zero h1.ne' (sub_ne_zero.mpr (Ne.symm hhit))
    have e1 : m₁ * (u₁ - v₁) * (u₁ + v₁) = m₁ * (u₁ - v₁) * (u₂ + v₂) := by
      have : m₁ * (u₁ - v₁) = m₂ * (v₂ - u₂) := by linarith
      calc m₁ * (u₁ - v₁) * (u₁ + v₁) = m₁ * u₁ ^ 2 - m₁ * v₁ ^ 2 := by ring
        _ = m₂ * v₂ ^ 2 - m₂ * u₂ ^ 2 := by linarith
        _ = m₂ * (v₂ - u₂) * (u₂ + v₂) := by ring
        _ = m₁ * (u₁ - v₁) * (u₂ + v₂) := by rw [this]
    exact mul_left_cancel₀ hne e1
  have hs : m₁ + m₂ ≠ 0 := by linarith
  constructor
  · rw [eq_div_iff hs]; nlinarith
  · rw [eq_div_iff hs]; nlinarith

/-- **Newton's cradle:** equal masses swap velocities. In a head-on hit on a
ball at rest, the first stops dead and the second takes all the motion. -/
theorem equal_masses_swap {m u₁ u₂ v₁ v₂ : ℝ}
    (h : ElasticCollision m m u₁ u₂ v₁ v₂) : v₁ = u₂ ∧ v₂ = u₁ := by
  have hm := h.pos₁
  obtain ⟨e1, e2⟩ := elastic_formulas h
  constructor
  · rw [e1]; field_simp; ring
  · rw [e2]; field_simp; ring

/-- Fraction of kinetic energy transferred to a ball at rest (`u₂ = 0`). -/
theorem energy_transfer_fraction {m₁ m₂ u₁ v₁ v₂ : ℝ} (hu : u₁ ≠ 0)
    (h : ElasticCollision m₁ m₂ u₁ 0 v₁ v₂) :
    (m₂ * v₂ ^ 2) / (m₁ * u₁ ^ 2) = 4 * m₁ * m₂ / (m₁ + m₂) ^ 2 := by
  have h1 := h.pos₁; have h2 := h.pos₂
  obtain ⟨-, e2⟩ := elastic_formulas h
  rw [e2]
  have : m₁ + m₂ ≠ 0 := by linarith
  field_simp
  ring

/-- The transfer fraction never exceeds 1, and it **equals 1 exactly when the
masses are equal**. Equal masses exchange energy most efficiently. -/
theorem transfer_le_one {m₁ m₂ : ℝ} (h1 : 0 < m₁) (h2 : 0 < m₂) :
    4 * m₁ * m₂ / (m₁ + m₂) ^ 2 ≤ 1 ∧
    (4 * m₁ * m₂ / (m₁ + m₂) ^ 2 = 1 ↔ m₁ = m₂) := by
  have hs : 0 < (m₁ + m₂) ^ 2 := by positivity
  constructor
  · rw [div_le_one hs]; nlinarith [sq_nonneg (m₁ - m₂)]
  · rw [div_eq_one_iff_eq hs.ne']
    constructor
    · intro h
      have : (m₁ - m₂) ^ 2 = 0 := by nlinarith
      linarith [pow_eq_zero_iff (n := 2) (two_ne_zero) |>.mp this]
    · rintro rfl; ring

end Elastic

/-! ## 2b. Collisions with rotation (molecules with a moment of inertia)

Particle 1 (point mass `m₁`, velocity `u₁ → v₁`) hits body 2 (mass `m₂`,
moment of inertia `I`, at rest) with lever arm `b` from its centre of mass.
After impact body 2 has velocity `w` and angular velocity `ω`. Conserved:
linear momentum, angular momentum about body 2's centre of mass, and kinetic
energy (translation + rotation).
-/

section Rotation

structure RotCollision (m₁ m₂ I b u₁ v₁ w ω : ℝ) : Prop where
  pos₁ : 0 < m₁
  pos₂ : 0 < m₂
  posI : 0 < I
  momentum : m₁ * u₁ = m₁ * v₁ + m₂ * w
  angular : b * (m₁ * u₁) = b * (m₁ * v₁) + I * ω
  energy : m₁ * u₁ ^ 2 = m₁ * v₁ ^ 2 + m₂ * w ^ 2 + I * ω ^ 2
  hit : v₁ ≠ u₁

/-- Effective mass for an off-centre hit: the body "feels" lighter because part
of the impulse goes into spinning it. -/
noncomputable def effMass (m₂ I b : ℝ) : ℝ := m₂ * I / (I + m₂ * b ^ 2)

theorem effMass_pos {m₂ I b : ℝ} (h2 : 0 < m₂) (hI : 0 < I) : 0 < effMass m₂ I b := by
  unfold effMass; positivity

/-- The effective mass never exceeds the real mass, and equals it exactly for a
central hit (`b = 0`). -/
theorem effMass_le {m₂ I b : ℝ} (h2 : 0 < m₂) (hI : 0 < I) :
    effMass m₂ I b ≤ m₂ ∧ (effMass m₂ I b = m₂ ↔ b = 0) := by
  have hS : 0 < I + m₂ * b ^ 2 := by positivity
  unfold effMass
  constructor
  · rw [div_le_iff₀ hS]; nlinarith [mul_nonneg (mul_nonneg h2.le h2.le) (sq_nonneg b)]
  · rw [div_eq_iff hS.ne']
    constructor
    · intro h
      have : m₂ * m₂ * b ^ 2 = 0 := by linarith
      have hb : b ^ 2 = 0 := by
        rcases mul_eq_zero.mp this with h' | h'
        · nlinarith
        · exact h'
      exact pow_eq_zero_iff two_ne_zero |>.mp hb
    · rintro rfl; ring

/-- Key relation: `m₂ I (u₁ + v₁) = (I + m₂ b²) m₁ (u₁ − v₁)`. -/
theorem rot_key {m₁ m₂ I b u₁ v₁ w ω : ℝ} (h : RotCollision m₁ m₂ I b u₁ v₁ w ω) :
    m₂ * I * (u₁ + v₁) = (I + m₂ * b ^ 2) * (m₁ * (u₁ - v₁)) := by
  have hJw : m₂ * w = m₁ * (u₁ - v₁) := by linear_combination -h.momentum
  have hJω : I * ω = b * (m₁ * (u₁ - v₁)) := by linear_combination -h.angular
  have e : m₂ * I * (m₁ * (u₁ ^ 2 - v₁ ^ 2)) = (I + m₂ * b ^ 2) * (m₁ * (u₁ - v₁)) ^ 2 := by
    have : m₂ * I * (m₁ * (u₁ ^ 2 - v₁ ^ 2)) = I * (m₂ * w) ^ 2 + m₂ * (I * ω) ^ 2 := by
      linear_combination (m₂ * I) * h.energy
    rw [this, hJw, hJω]; ring
  have hne : m₁ * (u₁ - v₁) ≠ 0 :=
    mul_ne_zero h.pos₁.ne' (sub_ne_zero.mpr (Ne.symm h.hit))
  exact mul_left_cancel₀ hne (by linear_combination e)

/-- Collision formula with rotation: the same as an ordinary collision, but
with effective mass `M = m₂ I / (I + m₂ b²)` in place of `m₂`. -/
theorem rot_formula {m₁ m₂ I b u₁ v₁ w ω : ℝ} (h : RotCollision m₁ m₂ I b u₁ v₁ w ω) :
    v₁ = (m₁ - effMass m₂ I b) / (m₁ + effMass m₂ I b) * u₁ := by
  have k := rot_key h
  have h1 := h.pos₁; have h2 := h.pos₂; have hI := h.posI
  have hS : 0 < I + m₂ * b ^ 2 := by positivity
  have hD : 0 < m₁ * (I + m₂ * b ^ 2) + m₂ * I := by positivity
  unfold effMass
  rw [eq_comm, div_mul_eq_mul_div, div_eq_iff (by positivity)]
  field_simp
  linear_combination -k

/-- Fraction of energy transferred: `4 m₁ M / (m₁ + M)²`. It is maximal (= 1)
when `m₁ = M`, i.e. when the mass matches the **effective** mass. -/
theorem rot_energy_transfer {m₁ m₂ I b u₁ v₁ w ω : ℝ} (hu : u₁ ≠ 0)
    (h : RotCollision m₁ m₂ I b u₁ v₁ w ω) :
    (m₂ * w ^ 2 + I * ω ^ 2) / (m₁ * u₁ ^ 2) =
      4 * m₁ * effMass m₂ I b / (m₁ + effMass m₂ I b) ^ 2 := by
  have h1 := h.pos₁
  have hM := effMass_pos (b := b) h.pos₂ h.posI
  have hE : m₂ * w ^ 2 + I * ω ^ 2 = m₁ * u₁ ^ 2 - m₁ * v₁ ^ 2 := by
    linear_combination -h.energy
  rw [hE, rot_formula h]
  have : m₁ + effMass m₂ I b ≠ 0 := by linarith
  field_simp
  ring

/-- How much of the transferred energy ends up as **rotation**:
`m₂ b² / (I + m₂ b²)`. Zero for a central hit, growing with the lever arm. -/
theorem rot_share {m₁ m₂ I b u₁ v₁ w ω : ℝ} (h : RotCollision m₁ m₂ I b u₁ v₁ w ω) :
    I * ω ^ 2 / (m₂ * w ^ 2 + I * ω ^ 2) = m₂ * b ^ 2 / (I + m₂ * b ^ 2) := by
  have h1 := h.pos₁; have h2 := h.pos₂; have hI := h.posI
  set J := m₁ * (u₁ - v₁) with hJ
  have hJ0 : J ≠ 0 := mul_ne_zero h1.ne' (sub_ne_zero.mpr (Ne.symm h.hit))
  have hw : w = J / m₂ := by
    rw [eq_div_iff h2.ne']; linear_combination -h.momentum
  have hω : ω = b * J / I := by
    rw [eq_div_iff hI.ne']; linear_combination -h.angular
  rw [hw, hω]
  have hS : 0 < I + m₂ * b ^ 2 := by positivity
  rw [div_eq_div_iff (by positivity) hS.ne']
  field_simp

end Rotation

/-! ## 3. Diffusion as a random walk -/

section RandomWalk

/-- One ±1 step from every position. -/
def step (L : List ℤ) : List ℤ := L.flatMap fun x => [x + 1, x - 1]

/-- All possible end positions after `n` steps of ±1, one per path
(so the list has `2ⁿ` entries, with multiplicity). -/
def positions : ℕ → List ℤ
  | 0 => [0]
  | n + 1 => step (positions n)

theorem step_map_sum (L : List ℤ) (g : ℤ → ℤ) :
    ((step L).map g).sum = (L.map fun x => g (x + 1) + g (x - 1)).sum := by
  induction L with
  | nil => rfl
  | cons a L ih =>
    simp only [step, List.flatMap_cons] at ih ⊢
    simp [ih]; ring

theorem length_step (L : List ℤ) : (step L).length = 2 * L.length := by
  induction L with
  | nil => rfl
  | cons a L ih =>
    simp only [step, List.flatMap_cons] at ih ⊢
    simp [ih]; ring

theorem sum_map_affine (L : List ℤ) (f : ℤ → ℤ) (a c : ℤ) :
    (L.map fun x => a * f x + c).sum = a * (L.map f).sum + c * L.length := by
  induction L with
  | nil => simp
  | cons x L ih => simp [ih]; ring

theorem length_positions (n : ℕ) : (positions n).length = 2 ^ n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [positions, length_step, ih, pow_succ, mul_comm]

/-- On average you stay put: the sum over all paths is 0 (symmetry). -/
theorem sum_positions (n : ℕ) : ((positions n).map id).sum = 0 := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [positions, step_map_sum]
    rw [show (fun x : ℤ => id (x + 1) + id (x - 1)) = fun x => 2 * id x + 0 by
      funext x; simp; ring, sum_map_affine, ih]
    simp

/-- **Diffusion law:** the sum of squared displacement over all `2ⁿ` paths
is `n · 2ⁿ`, i.e. the mean squared displacement is exactly `n`. -/
theorem sum_sq_positions (n : ℕ) :
    ((positions n).map fun x => x ^ 2).sum = (n : ℤ) * 2 ^ n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    rw [positions, step_map_sum]
    rw [show (fun x : ℤ => (x + 1) ^ 2 + (x - 1) ^ 2) = fun x => 2 * x ^ 2 + 2 by
      funext x; ring]
    rw [sum_map_affine, ih, length_positions]
    push_cast; ring

/-- Mean squared displacement = `n`. The spread (RMS distance) is therefore
`√n`: diffusive, not ballistic (`n`). -/
theorem mean_sq_displacement (n : ℕ) :
    ((((positions n).map fun x => x ^ 2).sum : ℤ) : ℚ) / (positions n).length = n := by
  rw [sum_sq_positions, length_positions]
  push_cast
  field_simp

end RandomWalk

end Movement

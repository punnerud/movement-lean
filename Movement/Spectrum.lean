import Movement.Forces

/-!
# Eigenvalues of a self-adjoint operator are real

The energies computed in the analyzer are eigenvalues of a self-adjoint
operator, and so are real. The same lemma is the whole content of the idea of
Hilbert and Pólya about the zeros of Riemann's zeta function: *if* the zeros
are `1/2 + i E` with `E` running over the eigenvalues of some self-adjoint
operator, then every zero has real part `1/2`. Nothing here says such an
operator exists; nobody has found one.
-/

namespace Movement

open ComplexConjugate

/-- **A self-adjoint operator has real eigenvalues.** If `T` is symmetric,
`⟨T x, y⟩ = ⟨x, T y⟩`, and `T v = μ v` for some `v ≠ 0`, then `μ` equals its
own complex conjugate. -/
theorem symmetric_eigenvalue_real {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    {T : E →ₗ[ℂ] E} (hT : T.IsSymmetric) {μ : ℂ} {v : E} (hv : v ≠ 0) (h : T v = μ • v) :
    conj μ = μ := by
  have h1 : inner ℂ (T v) v = inner ℂ v (T v) := hT v v
  rw [h, inner_smul_left, inner_smul_right] at h1
  have hne : (inner ℂ v v : ℂ) ≠ 0 := inner_self_ne_zero.mpr hv
  exact mul_right_cancel₀ hne h1

/-- A number that equals its conjugate has no imaginary part. -/
theorem im_eq_zero_of_conj_eq {μ : ℂ} (h : conj μ = μ) : μ.im = 0 := by
  have h2 := congrArg Complex.im h
  rw [Complex.conj_im] at h2
  linarith

/-- **Hilbert–Pólya, the easy half.** If `s = 1/2 + i μ` and `μ` is an
eigenvalue of a symmetric operator, then `s` has real part `1/2`. -/
theorem zero_on_line_of_symmetric {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    {T : E →ₗ[ℂ] E} (hT : T.IsSymmetric) {μ s : ℂ} {v : E} (hv : v ≠ 0) (h : T v = μ • v)
    (hs : s = 1 / 2 + Complex.I * μ) : s.re = 1 / 2 := by
  have him := im_eq_zero_of_conj_eq (symmetric_eigenvalue_real hT hv h)
  rw [hs]
  simp [him]

/-- **Eigenvectors of different eigenvalues are orthogonal.** -/
theorem symmetric_eigenvectors_orthogonal {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℂ E]
    {T : E →ₗ[ℂ] E} (hT : T.IsSymmetric) {μ ν : ℂ} {v w : E} (hv : v ≠ 0)
    (h1 : T v = μ • v) (h2 : T w = ν • w) (hne : μ ≠ ν) : inner ℂ v w = 0 := by
  have hμ := symmetric_eigenvalue_real hT hv h1
  have h3 : inner ℂ (T v) w = inner ℂ v (T w) := hT v w
  rw [h1, h2, inner_smul_left, inner_smul_right, hμ] at h3
  have h4 : (μ - ν) * inner ℂ v w = 0 := by rw [sub_mul, h3]; ring
  rcases mul_eq_zero.mp h4 with h5 | h5
  · exact absurd (sub_eq_zero.mp h5) hne
  · exact h5

end Movement

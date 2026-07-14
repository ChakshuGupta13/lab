import Mathlib

/-!
# Sequel meta-theorem: rationality of the transfer-matrix generating function (`lem:ratGF`)

Standalone formalisation of the transfer-matrix generating-function identity of
the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:ratGF` is the first ingredient of the period-`1` (polynomiality)
mechanism: the column-by-column transfer-matrix construction yields a generating
function that is **rational** in `z`.

> *Statement.* $\sum_n c_{m,n}(x)\,z^n = \mathbf u(x)^\top (I - z T_m(x))^{-1}
> \mathbf v(x)$, where $c_{m,n}(x) = \sum_h x^{\#\text{extrema}(h)}$ and
> $\mathbf u, \mathbf v$ encode the boundary columns. Hence
> $\sum_n E_d(m,n)\,z^n = [x^d](\cdots)$ is rational in $z$.

The substance is the abstract identity: for any square matrix `T` over a
commutative ring `R` and vectors `u, v` of compatible size, the scalar sequence
`c n := u ⬝ᵥ T^n *ᵥ v` has a generating function that is rational in `z`, with
denominator divisible by the *reverse* characteristic polynomial of `T`. The
transfer-matrix specifics (admissible column pairs, extrema-counting weight in
the auxiliary variable `x`) are how the lemma is *applied* in the paper; the
identity itself is matrix-algebraic and is what is formalised here.

This module proves the abstract identity, working over an arbitrary commutative
ring `R` and a finite, decidable-eq index type `ι`:

* `charpoly_pow_sum_zero` : the matrix-level Cayley-Hamilton consequence — for
  every `n`, `∑_k T.charpoly.coeff k • T^(k+n) = 0`. The Cayley-Hamilton
  identity `χ_T(T) = 0` multiplied through by `T^n`;
* `transfer_recurrence` (scalar form of `lem:ratGF`) : taking the dot product of
  the matrix-level sum with `u` and `v` yields the **linear recurrence**
  `∑_k T.charpoly.coeff k * (u ⬝ᵥ T^(k+n) *ᵥ v) = 0`, the recurrence of order
  `T.charpoly.natDegree` satisfied by the scalar sequence `c n = u ⬝ᵥ T^n *ᵥ v`;
* `transfer_GF_rational` (`lem:ratGF`, **complete**) : the explicit **rational
  generating function** identity at the formal-power-series level — for every
  `n ≥ T.charpoly.natDegree`, the coefficient of `z^n` in the product
  `revQ * C` is zero, where `C(z) = ∑_n c(n) z^n` is the generating function of
  the sequence and `revQ(z) = ∑_{j=0}^d T.charpoly.coeff(d-j) z^j` is the
  reverse characteristic polynomial. Equivalently, `revQ * C` is a polynomial
  of degree strictly less than `d`, which is exactly the rational-`z`
  conclusion of the paper.

`charpoly_pow_sum_zero` and `transfer_recurrence` are the recurrence content;
`transfer_GF_rational` is the rational-generating-function content. The paper's
statement uses the inverse form `(I - zT)^{-1}`; the equivalent denominator
form `revQ(z) · C(z) ∈ R[z]` used here is the standard rational-function
restatement and avoids power-series matrix inversion. The two are equivalent
because `revQ(0) = T.charpoly.coeff d = 1` is a unit in `R[[z]]`, so dividing by
`revQ` is well-defined; the inverse form is then `C = (u^⊤ · \text{adj}(I-zT) ·
v) / \det(I-zT)`, with both numerator and denominator polynomial in `z`.

Scope: this module proves the **abstract** rational-GF identity for any
matrix/vectors over a commutative ring. The application to the sequel paper's
column transfer matrix `T_m(x)` over `ℤ[x]` is the standard instantiation (set
`R = ℤ[x]`, `ι = admissible column pairs`, `T = T_m(x)`); no Lean adapter is
needed beyond the abstract identity. For the paper's application `ι` is
nonempty (admissible column pairs of `P_m` for `m ≥ 2`); on an empty index
type the theorem holds vacuously (`charpoly = 1`, `natDegree = 0`,
`transferGF = 0`) but carries no content.

This module is the first step of the transfer-matrix chain
(`lem:ratGF → lem:poles → lem:quotient`). The companion lemmas `lem:poles`
(reciprocal-eigenvalue poles) and `lem:quotient` (peripheral spectrum is `{1}`
on the colour-rotation quotient) are not formalised in this module.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.transfer_GF_rational`.
-/

namespace OrigamiCone.Sequel

open Matrix PowerSeries

/-- **Cayley-Hamilton, applied form.** For a square matrix `T` over a commutative
ring `R`, the characteristic polynomial sum `∑_k χ_T.coeff(k) • T^(k+n) = 0` for
every `n ≥ 0`. This is `χ_T(T) = 0` multiplied through by `T^n` on the right and
expanded with `Polynomial.aeval_eq_sum_range`. -/
theorem charpoly_pow_sum_zero {R : Type*} [CommRing R] {ι : Type*}
    [Fintype ι] [DecidableEq ι] (T : Matrix ι ι R) (n : ℕ) :
    ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • T ^ (k + n) = 0 := by
  have hCH : (Polynomial.aeval T) T.charpoly = 0 := Matrix.aeval_self_charpoly T
  have hExpand : (Polynomial.aeval T) T.charpoly
      = ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff k • T ^ k := Polynomial.aeval_eq_sum_range T
  rw [hExpand] at hCH
  calc ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff k • T ^ (k + n)
      = ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff k • (T ^ k * T ^ n) := by
        refine Finset.sum_congr rfl ?_
        intro k _; rw [pow_add]
    _ = (∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff k • T ^ k) * T ^ n := by
        rw [Finset.sum_mul]
        refine Finset.sum_congr rfl ?_
        intro k _; exact (Matrix.smul_mul _ _ _).symm
    _ = 0 * T ^ n := by rw [hCH]
    _ = 0 := Matrix.zero_mul _

/-- **Transfer recurrence** (scalar form of `lem:ratGF`). The sequence
`c(n) = u ⬝ᵥ T^n *ᵥ v` satisfies the linear recurrence given by the
characteristic polynomial of `T`: for every `n`,
`∑_k χ_T.coeff(k) * c(k + n) = 0`. Equivalent restatement of
`charpoly_pow_sum_zero` after applying `u` on the left and `v` on the right. -/
theorem transfer_recurrence {R : Type*} [CommRing R] {ι : Type*}
    [Fintype ι] [DecidableEq ι] (T : Matrix ι ι R) (u v : ι → R) (n : ℕ) :
    ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k * (u ⬝ᵥ ((T ^ (k + n)) *ᵥ v)) = 0 := by
  have hMat := charpoly_pow_sum_zero T n
  have hDot : u ⬝ᵥ ((∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
                       T.charpoly.coeff k • T ^ (k + n)) *ᵥ v) = 0 := by
    rw [hMat, Matrix.zero_mulVec, dotProduct_zero]
  rw [Matrix.sum_mulVec, dotProduct_sum] at hDot
  convert hDot using 1
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- The **reverse characteristic polynomial** of `T`, viewed as a formal power
series: `revCharpoly T = ∑_{j=0}^d T.charpoly.coeff(d-j) · z^j` where
`d = T.charpoly.natDegree`. Its constant term is `T.charpoly.coeff d = 1` (since
`T.charpoly` is monic). -/
noncomputable def revCharpoly {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]
    (T : Matrix ι ι R) : PowerSeries R :=
  PowerSeries.mk fun j =>
    if j ≤ T.charpoly.natDegree then T.charpoly.coeff (T.charpoly.natDegree - j) else 0

/-- The **generating function** of the transfer sequence: `transferGF u v T =
∑_n (u ⋝ᵥ T^n *ᵥ v) · z^n` in `R[[z]]`. -/
noncomputable def transferGF {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]
    (u v : ι → R) (T : Matrix ι ι R) : PowerSeries R :=
  PowerSeries.mk fun n => u ⬝ᵥ ((T ^ n) *ᵥ v)

/-- **Rational generating function** (`lem:ratGF`, complete). The product
`revCharpoly · transferGF` has zero coefficient at every `n ≥ T.charpoly.natDegree`,
so `revCharpoly · transferGF` is a polynomial of degree strictly less than
`T.charpoly.natDegree`. Equivalently, `transferGF = P / revCharpoly` as formal
power series for some polynomial `P` of bounded degree — the rational form. -/
theorem transfer_GF_rational {R : Type*} [CommRing R] {ι : Type*}
    [Fintype ι] [DecidableEq ι] (T : Matrix ι ι R) (u v : ι → R)
    (n : ℕ) (hn : T.charpoly.natDegree ≤ n) :
    (PowerSeries.coeff (R := R) n) (revCharpoly T * transferGF u v T) = 0 := by
  set d := T.charpoly.natDegree
  rw [PowerSeries.coeff_mul]
  -- Convert antidiagonal sum to a range sum.
  have hAnti : ∑ p ∈ Finset.antidiagonal n,
      (PowerSeries.coeff (R := R) p.1) (revCharpoly T) *
        (PowerSeries.coeff (R := R) p.2) (transferGF u v T)
      = ∑ j ∈ Finset.range (n + 1),
        (PowerSeries.coeff (R := R) j) (revCharpoly T) *
          (PowerSeries.coeff (R := R) (n - j)) (transferGF u v T) := by
    refine Finset.sum_nbij' (i := fun p => p.1) (j := fun k => (k, n - k)) ?_ ?_ ?_ ?_ ?_
    · intro ⟨a, b⟩ h
      simp only [Finset.mem_antidiagonal] at h
      simp only [Finset.mem_range]; omega
    · intro k h
      simp only [Finset.mem_range] at h
      simp only [Finset.mem_antidiagonal]; omega
    · intro ⟨a, b⟩ h
      simp only [Finset.mem_antidiagonal] at h
      show ((a, n - a) : ℕ × ℕ) = (a, b); congr; omega
    · intro k _; rfl
    · intro ⟨a, b⟩ h
      simp only [Finset.mem_antidiagonal] at h
      show (PowerSeries.coeff (R := R) a) (revCharpoly T) *
        (PowerSeries.coeff (R := R) b) (transferGF u v T)
        = (PowerSeries.coeff (R := R) a) (revCharpoly T) *
          (PowerSeries.coeff (R := R) (n - a)) (transferGF u v T)
      congr 1
      have : b = n - a := by omega
      rw [this]
  rw [hAnti]
  -- Unfold revCharpoly and transferGF.
  have hRev : ∀ j, (PowerSeries.coeff (R := R) j) (revCharpoly T)
      = if j ≤ d then T.charpoly.coeff (d - j) else 0 := by
    intro j; unfold revCharpoly; exact PowerSeries.coeff_mk j _
  have hC : ∀ i, (PowerSeries.coeff (R := R) i) (transferGF u v T)
      = u ⬝ᵥ ((T ^ i) *ᵥ v) := by
    intro i; unfold transferGF; exact PowerSeries.coeff_mk i _
  simp only [hRev, hC]
  -- Truncate range (n+1) to range (d+1) since the revQ-coefficient vanishes for j > d.
  have hTrunc : ∑ j ∈ Finset.range (n + 1),
      (if j ≤ d then T.charpoly.coeff (d - j) else 0) * (u ⬝ᵥ ((T ^ (n - j)) *ᵥ v))
      = ∑ j ∈ Finset.range (d + 1),
        T.charpoly.coeff (d - j) * (u ⬝ᵥ ((T ^ (n - j)) *ᵥ v)) := by
    rw [show n + 1 = (d + 1) + (n - d) from by omega, Finset.sum_range_add]
    have hHi : ∑ j ∈ Finset.range (n - d),
        (if (d + 1) + j ≤ d then T.charpoly.coeff (d - ((d + 1) + j)) else 0) *
          (u ⬝ᵥ ((T ^ (n - ((d + 1) + j))) *ᵥ v)) = 0 := by
      refine Finset.sum_eq_zero ?_
      intro j _
      have : ¬ (d + 1) + j ≤ d := by omega
      simp [this]
    rw [hHi, add_zero]
    refine Finset.sum_congr rfl ?_
    intro j hj
    simp only [Finset.mem_range] at hj
    have : j ≤ d := by omega
    simp [this]
  rw [hTrunc]
  -- Reindex k = d - j (so j = d - k); the sum matches transfer_recurrence at m = n - d.
  have hReindex : ∑ j ∈ Finset.range (d + 1),
      T.charpoly.coeff (d - j) * (u ⬝ᵥ ((T ^ (n - j)) *ᵥ v))
      = ∑ k ∈ Finset.range (d + 1),
        T.charpoly.coeff k * (u ⬝ᵥ ((T ^ (k + (n - d))) *ᵥ v)) := by
    refine Finset.sum_nbij' (i := fun j => d - j) (j := fun k => d - k) ?_ ?_ ?_ ?_ ?_
    · intro j h
      simp only [Finset.mem_range] at h
      simp only [Finset.mem_range]; omega
    · intro k h
      simp only [Finset.mem_range] at h
      simp only [Finset.mem_range]; omega
    · intro j h
      simp only [Finset.mem_range] at h
      show d - (d - j) = j; omega
    · intro k h
      simp only [Finset.mem_range] at h
      show d - (d - k) = k; omega
    · intro j hj
      simp only [Finset.mem_range] at hj
      have hpow : n - j = (d - j) + (n - d) := by omega
      rw [hpow]
  rw [hReindex]
  exact transfer_recurrence T u v (n - d)

end OrigamiCone.Sequel

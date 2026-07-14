import Mathlib

/-!
# Sequel: polynomial fit from vanishing forward differences

**Purpose.**  Task E building block: given a sequence `f : ℕ → ℚ` whose
`(k+1)`-fold forward difference `Δ_[1]^[k+1] f` vanishes identically, `f`
agrees on all of ℕ with a polynomial of natDegree ≤ k.  Explicit witness:
`newtonPoly f k`, Newton's series `∑ i, (Δ_[1]^[i] f 0) · (n choose i)` cast
as a polynomial in `n`.

**Role in Task E.**  The sequel paper's `thm:poly` (via `lem:quotient`) reduces
`E_d(m, ·)` to a scalar transfer sequence satisfying a linear recurrence with
characteristic polynomial `T_0^{trivial}.charpoly = X^{k_0} (X-1)^{k_1}`
(spectrum `{0, 1}` on the ρ-quotient of the extremum-free matrix).  On the tail
`n ≥ k_0`, that recurrence collapses to an `(X-1)^{k_1}`-recurrence, which is
exactly the `Δ_[1]^{k_1}`-vanishing hypothesis this module consumes.  The
present module supplies the LAST step of the paper's chain: recurrence ↦
polynomial witness.

The paper's `lem:quotient` still separately establishes the spectrum-collapse
step (Task E.b, future session), which involves passing from the abstract
`T_m(x).charpoly^{d+1}`-recurrence (already proved in
`SequelPolesArbDPoly.RseqMat_sandwich_polypow_recurrence`) to the trivial-block
`(X-1)^{k+1}`-recurrence via `SequelTransferInst.T0_quotient_action`.

## Design choices

* **Two entry points.**  The core theorem
  `exists_polynomial_of_fwdDiff_pow_eq_zero` consumes the forward-difference
  form `Δ_[1]^[k+1] f = 0`.  A companion
  `exists_polynomial_of_polyCoeff_recurrence` consumes the
  polynomial-coefficient form
  `∀ n, ∑ i ∈ range (k+2), ((X-1)^{k+1}).coeff i * f(n+i) = 0` — the natural
  output of `SequelPolesArbDPoly.RseqMat_sandwich_polypow_recurrence` after the
  spectrum-collapse step.  A small bridge (`fwdDiff_iter_eq_polyCoeffSum`)
  identifies the two.

* **Explicit witness via `descPochhammer`.**  `choosePolyℚ i :=
  (descPochhammer ℚ i) / (i.factorial : ℚ)` is the polynomial of degree `i`
  evaluating to `n.choose i` at natural inputs.  Newton's series
  `∑ i ∈ range (k+1), (Δ_[1]^[i] f 0) · choosePolyℚ i` is then a polynomial of
  natDegree ≤ k, and matches `f` on ℕ via Mathlib's Gregory-Newton formula
  `shift_eq_sum_fwdDiff_iter`.

## Theorems

* `choosePolyℚ (i : ℕ) : Polynomial ℚ` — polynomial representative of `n.choose i`.
* `choosePolyℚ_natDegree_le (i : ℕ) : (choosePolyℚ i).natDegree ≤ i`.
* `choosePolyℚ_eval_nat (n i : ℕ) : (choosePolyℚ i).eval (n : ℚ) = (n.choose i : ℚ)`.
* `newtonPoly (f : ℕ → ℚ) (k : ℕ) : Polynomial ℚ` — Newton's series polynomial.
* `newtonPoly_natDegree_le (f : ℕ → ℚ) (k : ℕ) : (newtonPoly f k).natDegree ≤ k`.
* **`exists_polynomial_of_fwdDiff_pow_eq_zero`** (Task E core primitive):
  Given `f : ℕ → ℚ`, `k : ℕ`, and `Δ_[1]^[k+1] f = 0`, exists a polynomial
  `p : Polynomial ℚ` with `p.natDegree ≤ k` and `∀ n : ℕ, f n = p.eval (n : ℚ)`.
* `coeff_X_sub_one_pow (k i : ℕ) : ((X - 1)^k).coeff i = (-1)^(k-i) * (k.choose i : ℚ)`.
* `fwdDiff_iter_eq_polyCoeffSum` — bridge: `Δ_[1]^[k] f n = ∑ i, ((X-1)^k).coeff i · f(n+i)`.
* `fwdDiff_iter_eq_zero_of_polyRecurrence` — poly-coefficient recurrence ⟹ Δ-vanishing.
* **`exists_polynomial_of_polyCoeff_recurrence`** (Task E poly-coefficient
  entry point): given the poly-coefficient recurrence in `(X-1)^{k+1}`, exists
  a polynomial witness of natDegree ≤ k.
* `polyRecurrence_shift_of_X_pow_mul` — tail-shift: `X^a · q`-recurrence on `f`
  ⟹ `q`-recurrence on `n ↦ f (n + a)`.
* **`tail_polyFit_of_X_pow_mul_X_sub_one_pow`** (Task E.γ.a composed entry
  point): given `X^a · (X-1)^{k+1}`-recurrence on `f`, exists a polynomial
  witness of natDegree ≤ k agreeing with `f` on `{m ≥ a}`.
* `sandwich_charpoly_recurrence` — Cayley-Hamilton on the sandwich:
  the sequence `n ↦ u ⬝ᵥ T^n *ᵥ v` satisfies the `T.charpoly`-recurrence.
* **`eventually_polynomial_of_charpoly_factored`** (Task E.γ.b abstract entry
  point): given `T : Matrix ι ι ℚ` with `T.charpoly = X^a · (X-1)^b` and
  `b ≥ 1`, exists a polynomial `p` of natDegree ≤ b - 1 with
  `u ⬝ᵥ T^m *ᵥ v = p.eval (m : ℚ)` for every `m ≥ a`.  Direct composition of
  `sandwich_charpoly_recurrence` with `tail_polyFit_of_X_pow_mul_X_sub_one_pow`.
* `charpoly_diagonal_binary` — a diagonal `ℚ`-matrix with entries in `{0, 1}`
  has charpoly `X^{|zeros|} · (X - 1)^{|ones|}`.
* **`eventually_polynomial_of_diagonal_binary`** (Task E.γ.b concrete bridge):
  composed corollary — a diagonal `{0, 1}`-matrix has eventually-polynomial
  sandwich `u ⬝ᵥ T^m *ᵥ v` on `{m ≥ #zeros}` with natDegree ≤ `#ones - 1`.

## Substrate

Mathlib's `Mathlib.Algebra.Group.ForwardDiff` (Gregory-Newton formula
`shift_eq_sum_fwdDiff_iter`) and `Mathlib.RingTheory.Polynomial.Pochhammer`
(`descPochhammer` as the polynomial lift of the descending factorial).  No
Sequel-specific dependency; this is a pure Mathlib-based primitive.

## Discipline note

This module does NOT import any other Sequel module (per the cross-collision
avoidance discipline in `origami-flipgraph-sequel-lean-map.md`).  It is a
standalone building block, importable alongside any Sequel envelope module.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
Check with
`#print axioms OrigamiCone.Sequel.exists_polynomial_of_fwdDiff_pow_eq_zero`.
-/

namespace OrigamiCone.Sequel

open Polynomial Finset fwdDiff

/-- Polynomial representative of `n ↦ n.choose i : ℕ → ℚ`.  Concretely
`(descPochhammer ℚ i) / (i.factorial : ℚ)` (with the factorial multiplied as
`C (i.factorial : ℚ)⁻¹`).  Evaluates to `n.choose i` at `n : ℕ` and has
`natDegree ≤ i`. -/
noncomputable def choosePolyℚ (i : ℕ) : Polynomial ℚ :=
  Polynomial.C ((i.factorial : ℚ)⁻¹) * descPochhammer ℚ i

@[simp]
lemma choosePolyℚ_zero : choosePolyℚ 0 = 1 := by
  unfold choosePolyℚ
  simp [descPochhammer_zero]

/-- `choosePolyℚ i` has natDegree ≤ i. -/
lemma choosePolyℚ_natDegree_le (i : ℕ) : (choosePolyℚ i).natDegree ≤ i := by
  unfold choosePolyℚ
  by_cases h_fact : ((i.factorial : ℚ)⁻¹) = 0
  · simp [h_fact]
  · rw [natDegree_C_mul h_fact, descPochhammer_natDegree]

/-- `choosePolyℚ i` evaluated at `(n : ℚ)` is `(n.choose i : ℚ)`. -/
lemma choosePolyℚ_eval_nat (n i : ℕ) :
    (choosePolyℚ i).eval (n : ℚ) = (n.choose i : ℚ) := by
  unfold choosePolyℚ
  rw [eval_mul, eval_C, descPochhammer_eval_eq_descFactorial ℚ n i]
  -- Goal: (i.factorial : ℚ)⁻¹ * (n.descFactorial i : ℚ) = (n.choose i : ℚ)
  have h_desc : (n.descFactorial i : ℚ) = (i.factorial : ℚ) * (n.choose i : ℚ) := by
    have := Nat.descFactorial_eq_factorial_mul_choose n i
    exact_mod_cast this
  rw [h_desc]
  have hfact_pos : (0 : ℚ) < i.factorial := by exact_mod_cast Nat.factorial_pos i
  field_simp

/-- `Δ_[1]` iterated on the zero function is zero. -/
private lemma iter_fwdDiff_zero (m : ℕ) : (Δ_[(1 : ℕ)])^[m] (0 : ℕ → ℚ) = 0 := by
  induction m with
  | zero => rfl
  | succ m IH =>
    rw [Function.iterate_succ_apply', IH]
    show (Δ_[(1 : ℕ)] (fun _ : ℕ => (0 : ℚ))) = 0
    exact fwdDiff_const (M := ℕ) (G := ℚ) (h := 1) 0

/-- **Newton's series polynomial** for `f : ℕ → ℚ`, truncated at degree `k`:
`∑ i ∈ range (k+1), (Δ_[1]^[i] f 0) · choosePolyℚ i`. -/
noncomputable def newtonPoly (f : ℕ → ℚ) (k : ℕ) : Polynomial ℚ :=
  ∑ i ∈ Finset.range (k + 1),
    Polynomial.C ((Δ_[(1 : ℕ)])^[i] f 0) * choosePolyℚ i

/-- `newtonPoly f k` has natDegree ≤ k. -/
lemma newtonPoly_natDegree_le (f : ℕ → ℚ) (k : ℕ) :
    (newtonPoly f k).natDegree ≤ k := by
  unfold newtonPoly
  apply Polynomial.natDegree_sum_le_of_forall_le
  intro i hi
  rw [Finset.mem_range] at hi
  by_cases hΔ : (Δ_[(1 : ℕ)])^[i] f 0 = 0
  · simp [hΔ]
  · rw [natDegree_C_mul hΔ]
    exact (choosePolyℚ_natDegree_le i).trans (Nat.lt_succ_iff.mp hi)

/-- Evaluation of `newtonPoly f k` at `(n : ℚ)` is the Gregory-Newton
binomial sum. -/
lemma newtonPoly_eval_nat (f : ℕ → ℚ) (k n : ℕ) :
    (newtonPoly f k).eval (n : ℚ) =
      ∑ i ∈ Finset.range (k + 1),
        (n.choose i : ℚ) * (Δ_[(1 : ℕ)])^[i] f 0 := by
  unfold newtonPoly
  rw [eval_finset_sum]
  apply Finset.sum_congr rfl
  intro i _
  rw [eval_mul, eval_C, choosePolyℚ_eval_nat]
  ring

/-- **Main theorem** (Task E core primitive).  Given `f : ℕ → ℚ` whose
`(k+1)`-fold forward difference vanishes identically, `f` agrees with a
polynomial of natDegree ≤ k on all of ℕ.  Explicit witness: `newtonPoly f k`.

The proof combines Mathlib's Gregory-Newton formula `shift_eq_sum_fwdDiff_iter`
with two case-splits handling `k ≤ n` (extra summands at higher `i` vanish
because `Δ_[1]^[i] f 0 = 0`) and `n < k` (extra summands at higher `i` vanish
because `n.choose i = 0`).

For the transfer-matrix bridge (Task E.b), a caller with a
`(X-1)^{k+1}`-coefficient recurrence
`∀ n, ∑ i ∈ range (k+2), ((X-1)^{k+1}).coeff i * f (n+i) = 0` translates that
to `Δ_[1]^[k+1] f = 0` via a straightforward bridge lemma (deferred). -/
theorem exists_polynomial_of_fwdDiff_pow_eq_zero
    (f : ℕ → ℚ) (k : ℕ) (hf : (Δ_[(1 : ℕ)])^[k + 1] f = 0) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ k ∧
      ∀ n : ℕ, f n = p.eval (n : ℚ) := by
  refine ⟨newtonPoly f k, newtonPoly_natDegree_le f k, ?_⟩
  intro n
  -- Gregory-Newton: f n = ∑ i ∈ range (n + 1), (n.choose i : ℚ) * Δ_[1]^[i] f 0
  have hGN : f n = ∑ i ∈ Finset.range (n + 1),
      (n.choose i : ℚ) * (Δ_[(1 : ℕ)])^[i] f 0 := by
    have h := shift_eq_sum_fwdDiff_iter (h := (1 : ℕ)) f n 0
    simp only [zero_add, nsmul_eq_mul, mul_one] at h
    calc f n = ∑ i ∈ range (n + 1), (n.choose i) • (Δ_[(1 : ℕ)])^[i] f 0 := h
      _ = ∑ i ∈ range (n + 1), (n.choose i : ℚ) * (Δ_[(1 : ℕ)])^[i] f 0 := by
          refine Finset.sum_congr rfl ?_
          intro i _
          rw [nsmul_eq_mul]
  rw [hGN, newtonPoly_eval_nat]
  -- Now compare two sums over range (n+1) and range (k+1) respectively.
  by_cases hkn : k + 1 ≤ n + 1
  · -- Case k ≤ n: split range (n+1) = range (k+1) ∪ Ico (k+1) (n+1);
    -- the Ico part vanishes because Δ_[1]^[i] f = 0 for i ≥ k+1.
    rw [← Finset.sum_range_add_sum_Ico _ hkn]
    have hzero : ∀ i ∈ Finset.Ico (k + 1) (n + 1),
        (n.choose i : ℚ) * (Δ_[(1 : ℕ)])^[i] f 0 = 0 := by
      intro i hi
      rw [Finset.mem_Ico] at hi
      obtain ⟨hi1, _⟩ := hi
      have hi_eq : i = (i - (k + 1)) + (k + 1) := by omega
      have hΔ_zero : (Δ_[(1 : ℕ)])^[i] f = 0 := by
        rw [hi_eq, Function.iterate_add_apply, hf]
        exact iter_fwdDiff_zero (i - (k + 1))
      rw [show (Δ_[(1 : ℕ)])^[i] f 0 = (0 : ℕ → ℚ) 0 by rw [hΔ_zero]]
      simp
    rw [Finset.sum_eq_zero hzero, add_zero]
  · -- Case n < k: split range (k+1) = range (n+1) ∪ Ico (n+1) (k+1);
    -- the Ico part vanishes because n.choose i = 0 for i > n.
    push_neg at hkn
    have hnk : n + 1 ≤ k + 1 := by omega
    rw [← Finset.sum_range_add_sum_Ico _ hnk]
    have hzero : ∀ i ∈ Finset.Ico (n + 1) (k + 1),
        (n.choose i : ℚ) * (Δ_[(1 : ℕ)])^[i] f 0 = 0 := by
      intro i hi
      rw [Finset.mem_Ico] at hi
      obtain ⟨hi1, _⟩ := hi
      have h_choose_zero : n.choose i = 0 := Nat.choose_eq_zero_of_lt hi1
      simp [h_choose_zero]
    rw [Finset.sum_eq_zero hzero, add_zero]

/-! ## Poly-coefficient bridge (Task E.β)

Callers with a linear recurrence in polynomial-coefficient form (the natural
output of `SequelPolesArbDPoly.RseqMat_sandwich_polypow_recurrence` after the
spectrum-collapse step of Task E.γ) can plug directly into this API without
manually translating to the forward-difference form. -/

/-- Coefficient of `X^i` in `(X - 1)^k : Polynomial ℚ`.  Direct corollary of
Mathlib's `Polynomial.coeff_X_add_C_pow` at `r = -1`. -/
lemma coeff_X_sub_one_pow (k i : ℕ) :
    ((X - 1 : Polynomial ℚ)^k).coeff i = ((-1 : ℚ)^(k - i)) * (k.choose i : ℚ) := by
  have h_rewrite : (X - 1 : Polynomial ℚ) = X + Polynomial.C (-1) := by
    rw [Polynomial.C_neg, Polynomial.C_1]; ring
  rw [h_rewrite, Polynomial.coeff_X_add_C_pow]

/-- **Bridge**: the `k`-fold forward difference agrees with the
`(X - 1)^k`-coefficient sum.  Combines Mathlib's `fwdDiff_iter_eq_sum_shift`
with the coefficient formula `coeff_X_sub_one_pow`. -/
lemma fwdDiff_iter_eq_polyCoeffSum (f : ℕ → ℚ) (k n : ℕ) :
    (Δ_[(1 : ℕ)])^[k] f n =
      ∑ i ∈ Finset.range (k + 1),
        ((X - 1 : Polynomial ℚ)^k).coeff i * f (n + i) := by
  rw [fwdDiff_iter_eq_sum_shift]
  apply Finset.sum_congr rfl
  intro i _
  rw [coeff_X_sub_one_pow]
  have h1 : (i : ℕ) • (1 : ℕ) = i := by simp
  rw [h1, zsmul_eq_mul]
  push_cast
  ring

/-- Poly-coefficient recurrence at all `n` ⟹ `Δ_[1]^[k] f = 0`. -/
lemma fwdDiff_iter_eq_zero_of_polyRecurrence (f : ℕ → ℚ) (k : ℕ)
    (hf : ∀ n, ∑ i ∈ Finset.range (k + 1),
      ((X - 1 : Polynomial ℚ)^k).coeff i * f (n + i) = 0) :
    (Δ_[(1 : ℕ)])^[k] f = 0 := by
  funext n
  show (Δ_[(1 : ℕ)])^[k] f n = 0
  rw [fwdDiff_iter_eq_polyCoeffSum, hf]

/-- **Main theorem (poly-coefficient form)** — Task E entry point for callers
producing an `(X-1)^{k+1}`-coefficient linear recurrence.  Given `f : ℕ → ℚ`
satisfying `∀ n, ∑ i ∈ range (k+2), ((X-1)^{k+1}).coeff i * f(n+i) = 0`, exists
a polynomial `p : Polynomial ℚ` with `natDegree p ≤ k` and
`∀ n, f n = p.eval (n : ℚ)`.

Direct corollary of `exists_polynomial_of_fwdDiff_pow_eq_zero` +
`fwdDiff_iter_eq_zero_of_polyRecurrence`. -/
theorem exists_polynomial_of_polyCoeff_recurrence
    (f : ℕ → ℚ) (k : ℕ)
    (hf : ∀ n, ∑ i ∈ Finset.range (k + 2),
      ((X - 1 : Polynomial ℚ)^(k + 1)).coeff i * f (n + i) = 0) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ k ∧
      ∀ n : ℕ, f n = p.eval (n : ℚ) := by
  apply exists_polynomial_of_fwdDiff_pow_eq_zero f k
  exact fwdDiff_iter_eq_zero_of_polyRecurrence f (k + 1) hf

/-! ## Tail-shift primitive (Task E.γ.a)

The spectrum-collapse step of Task E.γ (deferred) will produce a recurrence
whose characteristic polynomial factors as `X^a · (X-1)^{k+1}`, corresponding
to eigenvalue 0 with multiplicity `a` and eigenvalue 1 with multiplicity `k+1`
in the T₀-quotient block.  The `X^a` factor contributes an initial-transient
regime; the polynomial fit governs the tail `n ≥ a`.  This section supplies:

1. `polyRecurrence_shift_of_X_pow_mul` — a `X^a · q`-recurrence on `f`
   transfers to a `q`-recurrence on `n ↦ f (n + a)`.
2. `tail_polyFit_of_X_pow_mul_X_sub_one_pow` — direct entry point: given a
   `X^a · (X-1)^{k+1}`-recurrence on `f`, exists a polynomial `p` of
   natDegree ≤ k with `f m = p.eval (m : ℚ)` for every `m ≥ a`. -/

/-- **Tail-shift**: if `f : ℕ → ℚ` satisfies an `X^a · q`-coefficient
recurrence, then the tail `n ↦ f (n + a)` satisfies the `q`-coefficient
recurrence.  Equivalently: `f` obeys the `q`-recurrence starting from index
`a`. -/
lemma polyRecurrence_shift_of_X_pow_mul
    (f : ℕ → ℚ) (a : ℕ) (q : Polynomial ℚ)
    (hf : ∀ n, ∑ i ∈ Finset.range ((X^a * q : Polynomial ℚ).natDegree + 1),
             (X^a * q : Polynomial ℚ).coeff i * f (n + i) = 0) :
    ∀ n, ∑ i ∈ Finset.range (q.natDegree + 1),
             q.coeff i * f (n + a + i) = 0 := by
  intro n
  by_cases hq : q = 0
  · simp [hq]
  have hnd : (X^a * q : Polynomial ℚ).natDegree = q.natDegree + a := by
    exact Polynomial.natDegree_X_pow_mul (n := a) hq
  have h_orig := hf n
  rw [hnd] at h_orig
  have hsplit : q.natDegree + a + 1 = a + (q.natDegree + 1) := by ring
  rw [hsplit] at h_orig
  have h_range_split : Finset.range (a + (q.natDegree + 1))
      = Finset.range a ∪ Finset.Ico a (a + (q.natDegree + 1)) := by
    ext i
    simp only [Finset.mem_range, Finset.mem_union, Finset.mem_Ico]
    omega
  rw [h_range_split, Finset.sum_union] at h_orig
  · -- First (range a) sum is zero because coeff (X^a * q) i = 0 for i < a.
    have h_low_zero : ∀ i ∈ Finset.range a,
        (X^a * q : Polynomial ℚ).coeff i * f (n + i) = 0 := by
      intro i hi
      rw [Finset.mem_range] at hi
      rw [Polynomial.coeff_X_pow_mul']
      rw [if_neg (by omega)]
      ring
    rw [Finset.sum_eq_zero h_low_zero, zero_add] at h_orig
    rw [Finset.sum_Ico_eq_sum_range] at h_orig
    have hsub : a + (q.natDegree + 1) - a = q.natDegree + 1 := by omega
    rw [hsub] at h_orig
    -- Reindex: (X^a * q).coeff (a + i) = q.coeff i (via coeff_X_pow_mul').
    convert h_orig using 1
    apply Finset.sum_congr rfl
    intro i _
    rw [Polynomial.coeff_X_pow_mul' (n := a)]
    rw [if_pos (Nat.le_add_right a i)]
    have h_diff : a + i - a = i := by omega
    rw [h_diff]
    have h_arg : n + a + i = n + (a + i) := by ring
    rw [h_arg]
  · -- disjoint
    rw [Finset.disjoint_left]
    intro i hi1 hi2
    rw [Finset.mem_range] at hi1
    rw [Finset.mem_Ico] at hi2
    omega

/-- **Composed tail-fit** (Task E.γ.a entry point).  Given a sequence
`f : ℕ → ℚ` satisfying an `X^a · (X-1)^{k+1}`-coefficient recurrence, there
exists a polynomial `p : Polynomial ℚ` with `natDegree p ≤ k` such that
`f m = p.eval (m : ℚ)` for every `m ≥ a`.

The proof composes `polyRecurrence_shift_of_X_pow_mul` (to strip the `X^a`
factor and shift indexing) with `exists_polynomial_of_polyCoeff_recurrence`
(the Task E.β poly-coefficient entry point).  The polynomial witness is then
shifted back via `p'.comp (X - C a)`, whose natDegree bound follows from
`Polynomial.natDegree_comp_le` and `natDegree_X_sub_C`. -/
theorem tail_polyFit_of_X_pow_mul_X_sub_one_pow
    (f : ℕ → ℚ) (a k : ℕ)
    (hf : ∀ n, ∑ i ∈ Finset.range ((X^a * (X - 1)^(k + 1) : Polynomial ℚ).natDegree + 1),
             (X^a * (X - 1)^(k + 1) : Polynomial ℚ).coeff i * f (n + i) = 0) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ k ∧
      ∀ m : ℕ, a ≤ m → f m = p.eval (m : ℚ) := by
  -- Step 1: shift out X^a, giving (X-1)^{k+1}-recurrence on the tail.
  have h_shift := polyRecurrence_shift_of_X_pow_mul f a ((X - 1)^(k + 1)) hf
  set g : ℕ → ℚ := fun n => f (n + a) with hg_def
  -- Rewrite the shift result as a recurrence on g.
  have h_shift_g : ∀ n, ∑ i ∈ Finset.range (((X - 1 : Polynomial ℚ)^(k+1)).natDegree + 1),
      ((X - 1 : Polynomial ℚ)^(k+1)).coeff i * g (n + i) = 0 := by
    intro n
    have hh := h_shift n
    convert hh using 1
    apply Finset.sum_congr rfl
    intro i _
    show ((X - 1 : Polynomial ℚ)^(k+1)).coeff i * f (n + i + a) =
         ((X - 1 : Polynomial ℚ)^(k+1)).coeff i * f (n + a + i)
    congr 1
    congr 1
    ring
  -- (X - 1)^(k+1) has natDegree k + 1, so range = range (k+2).
  have h_nd : ((X - 1 : Polynomial ℚ)^(k+1)).natDegree + 1 = k + 2 := by
    rw [Polynomial.natDegree_pow]
    rw [show (X - 1 : Polynomial ℚ) = X - Polynomial.C 1 by simp]
    rw [Polynomial.natDegree_X_sub_C]
    ring
  rw [h_nd] at h_shift_g
  -- Step 2: apply the Task E.β entry point.
  obtain ⟨p', hp'_deg, hp'_eval⟩ :=
    exists_polynomial_of_polyCoeff_recurrence g k h_shift_g
  -- Step 3: shift the polynomial back: p := p'.comp (X - C a).
  refine ⟨p'.comp (Polynomial.X - Polynomial.C (a : ℚ)), ?_, ?_⟩
  · calc (p'.comp (Polynomial.X - Polynomial.C (a : ℚ))).natDegree
        ≤ p'.natDegree * (Polynomial.X - Polynomial.C (a : ℚ)).natDegree :=
          Polynomial.natDegree_comp_le
      _ ≤ k * 1 := by
          gcongr
          rw [Polynomial.natDegree_X_sub_C]
      _ = k := Nat.mul_one k
  · intro m hm
    have hg_val : f m = g (m - a) := by
      show f m = f ((m - a) + a)
      congr 1
      omega
    rw [hg_val, hp'_eval (m - a)]
    have hcast : ((m - a : ℕ) : ℚ) = (m : ℚ) - (a : ℚ) := by
      rw [Nat.cast_sub hm]
    rw [hcast, Polynomial.eval_comp]
    simp

/-! ## Cayley-Hamilton sandwich (Task E.γ.b abstract fragment)

The paper's `lem:quotient` (§8) concludes eventually-polynomial from a
spectrum-collapse argument: on the ρ-invariant block of `T_0 := T_m(0)`, the
spectrum reduces to `{0, 1}`, so the charpoly factors as `X^a · (X-1)^b`.
This section supplies the abstract d=0 piece of that chain: given a ℚ-matrix
`T` whose charpoly is `X^a · (X-1)^b`, the sandwich `n ↦ u ⬝ᵥ T^n *ᵥ v` is
eventually polynomial on `{m ≥ a}`.

The remaining Task E.γ.b work (deferred, own session) is the concrete
spectral computation showing `T_0^{trivial}.charpoly = X^{k_0} · (X-1)^{k_1}`
for the sequel's specific T_m matrix at x=0.  Once that arrives, this section
takes it the rest of the way to a polynomial witness. -/

section CayleyHamiltonSandwich

open Matrix

variable {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Cayley-Hamilton sandwich recurrence.**  For any square matrix `T` over
`ℚ` with boundary vectors `u v : ι → ℚ`, the sandwich sequence
`n ↦ u ⬝ᵥ T^n *ᵥ v` satisfies the `T.charpoly`-coefficient recurrence.

Proved by expanding `Polynomial.aeval T T.charpoly = 0` (Cayley-Hamilton,
via `Matrix.aeval_self_charpoly`), multiplying by `T^n`, and sandwiching. -/
lemma sandwich_charpoly_recurrence
    (T : Matrix ι ι ℚ) (u v : ι → ℚ) (n : ℕ) :
    ∑ i ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff i * (u ⬝ᵥ (T^(n + i)) *ᵥ v) = 0 := by
  have hCH : (Polynomial.aeval T) T.charpoly = 0 := Matrix.aeval_self_charpoly _
  have hExpand : (Polynomial.aeval T) T.charpoly
      = ∑ i ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff i • T ^ i := Polynomial.aeval_eq_sum_range _
  rw [hExpand] at hCH
  have hMat : ∑ i ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff i • T ^ (n + i) = 0 := by
    calc ∑ i ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff i • T ^ (n + i)
        = ∑ i ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff i • (T ^ n * T ^ i) := by
          refine Finset.sum_congr rfl ?_
          intro i _; rw [pow_add]
      _ = T ^ n * ∑ i ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff i • T ^ i := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl ?_
          intro i _; exact (Matrix.mul_smul _ _ _).symm
      _ = T ^ n * 0 := by rw [hCH]
      _ = 0 := Matrix.mul_zero _
  have hDot : u ⬝ᵥ (∑ i ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff i • T ^ (n + i)) *ᵥ v = 0 := by
    rw [hMat, Matrix.zero_mulVec, dotProduct_zero]
  rw [Matrix.sum_mulVec, dotProduct_sum] at hDot
  convert hDot using 1
  refine Finset.sum_congr rfl ?_
  intro i _
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- **Sandwich recurrence at a factored charpoly.**  When `T.charpoly` is
known to factor as `X^a · (X-1)^b`, the sandwich recurrence rewrites
coefficient-by-coefficient into the factored form. -/
lemma sandwich_recurrence_of_charpoly_factored
    (T : Matrix ι ι ℚ) (u v : ι → ℚ) (a b : ℕ)
    (h_factor : T.charpoly = X^a * (X - 1)^b) (n : ℕ) :
    ∑ i ∈ Finset.range ((X^a * (X - 1)^b : Polynomial ℚ).natDegree + 1),
      (X^a * (X - 1)^b : Polynomial ℚ).coeff i * (u ⬝ᵥ (T^(n + i)) *ᵥ v) = 0 := by
  have h := sandwich_charpoly_recurrence T u v n
  rw [h_factor] at h
  exact h

/-- **Eventually polynomial from charpoly factorization** (Task E.γ.b abstract
entry point).  Given `T : Matrix ι ι ℚ` with `T.charpoly = X^a · (X-1)^b` and
`b ≥ 1`, the sandwich `n ↦ u ⬝ᵥ T^n *ᵥ v` agrees on `{m ≥ a}` with a
polynomial of natDegree ≤ b - 1.

Direct composition of `sandwich_recurrence_of_charpoly_factored` (feeding the
factored form into the `X^a · (X-1)^{k+1}`-recurrence hypothesis at
`k := b - 1`) with `tail_polyFit_of_X_pow_mul_X_sub_one_pow`.

**Interpretation.**  `a` is the multiplicity of eigenvalue 0 (transient
block), `b` is the multiplicity of eigenvalue 1 (frozen block).  The paper's
`lem:quotient` for the sequel's transfer matrix has `a = |transient states|`
and `b = |frozen orbits| = 2^m` for the trivial-character block. -/
theorem eventually_polynomial_of_charpoly_factored
    (T : Matrix ι ι ℚ) (u v : ι → ℚ)
    (a b : ℕ) (hb : 1 ≤ b) (h_factor : T.charpoly = X^a * (X - 1)^b) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ b - 1 ∧
      ∀ m : ℕ, a ≤ m → u ⬝ᵥ (T^m) *ᵥ v = p.eval (m : ℚ) := by
  set k := b - 1 with hk_def
  have hkb : k + 1 = b := by omega
  have h_factor' : T.charpoly = X^a * (X - 1)^(k + 1) := by
    rw [hkb]; exact h_factor
  have h_rec : ∀ n, ∑ i ∈ Finset.range
      ((X^a * (X - 1)^(k + 1) : Polynomial ℚ).natDegree + 1),
      (X^a * (X - 1)^(k + 1) : Polynomial ℚ).coeff i *
        (u ⬝ᵥ (T^(n + i)) *ᵥ v) = 0 := by
    intro n
    exact sandwich_recurrence_of_charpoly_factored T u v a (k + 1) h_factor' n
  exact tail_polyFit_of_X_pow_mul_X_sub_one_pow (fun n => u ⬝ᵥ (T^n) *ᵥ v) a k h_rec

/-! ## Binary-diagonal charpoly (Task E.γ.b concrete bridge)

The paper's `lem:quotient` reduces the spectrum of `T_0^{trivial}` to
`{0, 1}`: on the trivial-character block, every 3-cycle collapses to a
self-loop, so `T_0^{trivial}` is the identity on the `2^m` frozen orbits and
zero on the transient ones — a diagonal matrix in the orbit basis with
entries in `{0, 1}`.  This section supplies the abstract fact that any
diagonal matrix over `ℚ` with entries in `{0, 1}` has charpoly
`X^{|zeros|} · (X - 1)^{|ones|}`, and composes with
`eventually_polynomial_of_charpoly_factored` to give the polynomial witness.

Task E.γ.b remaining work: instantiate `T := T_0^{trivial}` (a matrix on
orbit space) and establish that it is diagonal with `d i ∈ {0, 1}`, using
`SequelTransferInst.T0_quotient_action`. -/

/-- **Charpoly of a diagonal `ℚ`-matrix with entries in `{0, 1}`** factors as
`X^{|zeros|} · (X - 1)^{|ones|}`. -/
theorem charpoly_diagonal_binary (d : ι → ℚ)
    (hd : ∀ i, d i = 0 ∨ d i = 1) :
    (Matrix.diagonal d).charpoly
      = X ^ (Finset.univ.filter (fun i : ι => d i = 0)).card
      * (X - 1) ^ (Finset.univ.filter (fun i : ι => d i = 1)).card := by
  rw [Matrix.charpoly_diagonal]
  set S₀ : Finset ι := Finset.univ.filter (fun i => d i = 0)
  set S₁ : Finset ι := Finset.univ.filter (fun i => d i = 1)
  have h_disj : Disjoint S₀ S₁ := by
    rw [Finset.disjoint_left]
    intro i hi₀ hi₁
    rw [Finset.mem_filter] at hi₀ hi₁
    have : (0 : ℚ) = 1 := hi₀.2.symm.trans hi₁.2
    norm_num at this
  have h_union : S₀ ∪ S₁ = Finset.univ := by
    ext i
    rw [Finset.mem_union]
    simp only [Finset.mem_univ, iff_true]
    simp only [S₀, S₁, Finset.mem_filter, Finset.mem_univ, true_and]
    exact hd i
  rw [← h_union, Finset.prod_union h_disj]
  have h_prod_zero : ∏ i ∈ S₀, (Polynomial.X - Polynomial.C (d i))
      = Polynomial.X ^ S₀.card := by
    rw [← Finset.prod_const]
    apply Finset.prod_congr rfl
    intro i hi
    rw [Finset.mem_filter] at hi
    rw [hi.2]; simp
  have h_prod_one : ∏ i ∈ S₁, (Polynomial.X - Polynomial.C (d i))
      = (Polynomial.X - 1) ^ S₁.card := by
    rw [← Finset.prod_const]
    apply Finset.prod_congr rfl
    intro i hi
    rw [Finset.mem_filter] at hi
    rw [hi.2]; simp
  rw [h_prod_zero, h_prod_one]

/-- **Eventually polynomial from binary-diagonal matrix.**  A ℚ-matrix that
is diagonal with entries in `{0, 1}` and has at least one entry equal to `1`
admits, for any boundary vectors `u v`, a polynomial `p` with
`natDegree p ≤ (# ones) - 1` such that `u ⬝ᵥ T^m *ᵥ v = p.eval (m : ℚ)` for
every `m` at least the number of zero entries.

Direct composition of `charpoly_diagonal_binary` with
`eventually_polynomial_of_charpoly_factored`. -/
theorem eventually_polynomial_of_diagonal_binary
    (d : ι → ℚ) (hd : ∀ i, d i = 0 ∨ d i = 1)
    (u v : ι → ℚ)
    (h_ones : 1 ≤ (Finset.univ.filter (fun i : ι => d i = 1)).card) :
    ∃ p : Polynomial ℚ,
      p.natDegree ≤ (Finset.univ.filter (fun i : ι => d i = 1)).card - 1 ∧
      ∀ m : ℕ, (Finset.univ.filter (fun i : ι => d i = 0)).card ≤ m →
        u ⬝ᵥ ((Matrix.diagonal d)^m) *ᵥ v = p.eval (m : ℚ) := by
  apply eventually_polynomial_of_charpoly_factored (Matrix.diagonal d) u v
    (Finset.univ.filter (fun i : ι => d i = 0)).card
    (Finset.univ.filter (fun i : ι => d i = 1)).card
    h_ones
  exact charpoly_diagonal_binary d hd

end CayleyHamiltonSandwich

end OrigamiCone.Sequel

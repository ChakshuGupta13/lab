import Mathlib

/-!
# Sequel: matrix-level foundation for iterated `lem:poles`

Standalone formalisation of the **matrix-level infrastructure** for the
arbitrary-`d` generalisation of `Lemma lem:poles` of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

This module supplies the matrix-valued primitives needed to lift
`SequelPolesConv`'s d=1 scalar `p²`-recurrence on `Rseq(n) := ∑_{j=0}^{n} u^⊤
T^j A T^{n-j} v` to an arbitrary-`d` `p^{d+1}`-recurrence on the `d`-fold
convolutional sum. The key idea, due to the paper's Neumann-series proof of
`lem:poles`, is that each `[x^d]` coefficient of the resolvent `(I - z T -
z x A - O(x²))^{-1}` is a `d`-fold convolution `∑_{j₁+⋯+j_{d+1}=n} T^{j₁} A
T^{j₂} A ⋯ T^{j_{d+1}}` of `T`-powers separated by `A`-insertions, and
applying the charpoly action `p(E_n)` to such a `d`-fold convolution reduces
the convolution order by one — at the cost of a "boundary" term whose
`n`-shift dependence has bounded width. Iterating `d+1` times leaves only
boundary-of-boundary terms, all killed by Cayley-Hamilton.

The matrix-level form of this argument lets us prove the inductive step
**once** at the level of arbitrary matrix-valued sequences `S : ℕ → Matrix
ι ι R`, then specialise to scalar generating sequences via `u ⬝ᵥ (· *ᵥ v)`.
This module supplies that matrix-level core.

## Theorems

* `charActMat T S n := ∑_k p_k • S(n+k)` where `p := T.charpoly` is the
  matrix-valued analogue of the scalar `charAct` from `SequelPolesConv`.
* `convolveMat T A S n := ∑_{j=0..n} T^j * A * S(n-j)` is the matrix-valued
  once-convolution of an `A`-insertion against an arbitrary matrix-valued
  sequence `S`.
* `BoundaryMat T A S n := ∑_k p_k • ∑_{t<k} T^(n+1+t) * A * S(k-1-t)` is the
  boundary that arises when `charActMat` is applied to `convolveMat T A S`.
* `convolveMat_split` : the `n+k` split of `convolveMat` (index reindex from
  the second half of the range).
* `charActMat_convolveMat_decomp` (**key decomposition**): for any matrix-
  valued sequence `S`,
  `charActMat T (convolveMat T A S) = convolveMat T A (charActMat T S) + BoundaryMat T A S`.
  This is the inductive step in disguise: applying `charActMat` to a once-
  convolution against `S` produces a once-convolution against `charActMat S`
  (pushing one `charAct` through the inner `S`) **plus** a boundary that is
  itself killed by one more application of `charActMat` (`charActMat_BoundaryMat_eq_zero`).
* `ch_shift_mat` : matrix Cayley-Hamilton at shift `m`, `∑_k p_k • T^(m+k) = 0`.
  Re-derived in place (matches `SequelPolesConv.ch_shift_mat` definitionally).
* `charActMat_BoundaryMat_eq_zero` : applying `charActMat` to `BoundaryMat`
  yields zero. Mirrors the d=1 boundary-killer
  (`SequelPolesConv.boundary_charAct_eq_zero`), generalised from scalar to
  matrix-valued sequences.
* `charActMat_add`, `charActMat_zero` : linearity of `charActMat` in the
  sequence argument.
* `charActIter T S d n` : the `d`-fold iterate of `charActMat T`, defined
  recursively on `d`. The arbitrary-`d` `lem:poles` reduces to proving that
  `charActIter T (d-fold convolveMat against T^·) (d+1) ≡ 0`.

## How the next-session induction will use this

The arbitrary-`d` `p^{d+1}`-recurrence on the `d`-fold convolutional sum
`RseqMat T A d n := convolveMat T A (RseqMat T A (d-1)) n` (with
`RseqMat T A 0 n := T^n`) follows from this module's primitives by induction
on `d`:

* **Base** (`d = 0`) : `RseqMat T A 0 n = T^n`, so `charActMat T (RseqMat T A 0) n
  = ∑_k p_k • T^(n+k) = 0` by `ch_shift_mat`.
* **Step** (`d → d+1`) : assume `charActIter T (RseqMat T A d) (d+1) ≡ 0`.
  Then `charActIter T (RseqMat T A (d+1)) (d+2)
  = charActMat T (charActIter T (convolveMat T A (RseqMat T A d)) (d+1))`.
  Telescoping cleanly via `charActMat_convolveMat_decomp` + `charActIter_add`
  (the d-fold linearity, immediate from `charActMat_add` by induction): at
  each step one application of `charActMat` is consumed splitting the live
  `convolveMat T A S` into `convolveMat T A (charActMat T S) + BoundaryMat T A S`,
  so at most ONE `convolveMat` and ONE `BoundaryMat` term are alive at any
  step. After `d+1` steps the `convolveMat` term has its inner `S` fully
  charpoly-acted to zero (by the inductive hypothesis at one less depth), and
  the `d+2`-th step kills the surviving boundary via
  `charActMat_BoundaryMat_eq_zero`.

The formalisation of the induction itself (defining `RseqMat`, threading the
linear-telescoping invariant) is a focused follow-up session of order
30-40 lines of Lean, not a multi-day effort: at any step there is at
most one convolveMat-term and one Boundary-term alive, so the induction
carries a two-component invariant rather than a growing collection.

## Scope

* All matrix-level primitives (`charActMat`, `convolveMat`, `BoundaryMat`,
  `charActIter`) and the key decomposition + boundary-killer + linearity
  lemmas (single-step `charActMat_add` and iterated `charActIter_add`) are
  proved end-to-end (no `sorry`).
* The **`d`-fold convolutional sum** `RseqMat T A d` and the **arbitrary-`d`
  recurrence theorem** are NOT formalised here; they require the induction
  outlined above. With the primitives in this module the induction
  carries a two-component invariant (one live `convolveMat` term and one
  live `BoundaryMat` term per step) and is expected to fit in 30-40 lines
  of Lean. This is the natural follow-up session.
* The connection to the paper's actual `[x^d]` Neumann-series coefficient —
  i.e. that `[x^d](u^⊤ T_m(x)^N v)` is a polynomial in `RseqMat T A d (·)` —
  is the PowerSeries Leibniz identification, still downstream (and still
  shift-corrected per the off-by-one documented in `SequelPolesConv`).
* Per the discipline, this module imports `Mathlib` only.

No `sorry`; check with
`#print axioms OrigamiCone.Sequel.charActMat_convolveMat_decomp`.
-/

namespace OrigamiCone.Sequel

open Matrix Polynomial

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Matrix-valued charpoly action**: `charActMat T S n := ∑_k p_k • S(n+k)`
where `p := T.charpoly`. The matrix-valued analogue of the scalar `charAct`
implicit in `SequelPolesConv.Rseq_charAct_eq_boundary`. -/
noncomputable def charActMat (T : Matrix ι ι R) (S : ℕ → Matrix ι ι R) (n : ℕ) :
    Matrix ι ι R :=
  ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
    T.charpoly.coeff k • S (n + k)

/-- **Matrix-valued once-convolution**: `convolveMat T A S n := ∑_{j=0..n} T^j *
A * S(n-j)`. For the canonical choice `S(n) = T^n` we recover the scalar
`SequelPolesConv.Rseq` after sandwiching with `u`/`v`; for the recursive
choice `S(n) = convolveMat T A T^· (n)` we obtain the d=2 convolution, and so on. -/
noncomputable def convolveMat (T A : Matrix ι ι R) (S : ℕ → Matrix ι ι R) (n : ℕ) :
    Matrix ι ι R :=
  ∑ j ∈ Finset.range (n + 1), T ^ j * A * S (n - j)

/-- **Boundary** produced by one application of `charActMat` to `convolveMat T A
S`. The `n`-dependence enters only via `T^(n+1+t)` in the `T`-power; the
`S`-argument `k - 1 - t` and the `t < k` index range are bounded by
`p.natDegree`, independent of `n`. This boundedness is what makes
`charActMat_BoundaryMat_eq_zero` provable. -/
noncomputable def BoundaryMat (T A : Matrix ι ι R) (S : ℕ → Matrix ι ι R) (n : ℕ) :
    Matrix ι ι R :=
  ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
    T.charpoly.coeff k •
      ∑ t ∈ Finset.range k, T ^ (n + 1 + t) * A * S (k - 1 - t)

/-- **Index split** of the matrix-valued once-convolution: `convolveMat T A S (n
+ k) = (j ≤ n contribution) + (j > n boundary contribution)`. The matrix-
valued analogue of `SequelPolesConv.Rseq_split`. -/
lemma convolveMat_split (T A : Matrix ι ι R) (S : ℕ → Matrix ι ι R) (n k : ℕ) :
    convolveMat T A S (n + k)
      = (∑ j ∈ Finset.range (n + 1), T ^ j * A * S (n + k - j))
        + (∑ t ∈ Finset.range k, T ^ (n + 1 + t) * A * S (k - 1 - t)) := by
  unfold convolveMat
  rw [show n + k + 1 = (n + 1) + k from by ring, Finset.sum_range_add]
  congr 1
  refine Finset.sum_congr rfl ?_
  intro t _; congr 2; omega

-- Typed helper for `Finset.smul_sum` in the `Matrix`-over-`R` setting; the
-- bare lemma leaves `DistribSMul` underdetermined inside `simp_rw`.
set_option linter.unusedSectionVars false in
private lemma smul_sum_explicit {α : Type*} (s : Finset α) (b : R)
    (f : α → Matrix ι ι R) :
    b • ∑ x ∈ s, f x = ∑ x ∈ s, b • f x := Finset.smul_sum

/-- **Key decomposition**: applying `charActMat T` to `convolveMat T A S`
splits as the convolution against the once-charpoly-acted `S` plus the
boundary contribution.

    `charActMat T (convolveMat T A S) = convolveMat T A (charActMat T S) + BoundaryMat T A S`

This is the *inductive step* of the iterated `lem:poles` argument: one
application of `charActMat` pushes a `charAct` through the inner `S` (the
first term — which is killed by one fewer remaining `charActMat` applications
when `S` is itself a deeper convolution) plus a boundary (which is killed by
one more `charActMat` application via `charActMat_BoundaryMat_eq_zero`).
-/
lemma charActMat_convolveMat_decomp (T A : Matrix ι ι R) (S : ℕ → Matrix ι ι R)
    (n : ℕ) :
    charActMat T (convolveMat T A S) n
      = convolveMat T A (charActMat T S) n + BoundaryMat T A S n := by
  show ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • convolveMat T A S (n + k)
    = convolveMat T A (charActMat T S) n + BoundaryMat T A S n
  simp_rw [convolveMat_split T A S n]
  simp_rw [smul_add, Finset.sum_add_distrib]
  congr 1
  -- First piece: ∑_k p_k • ∑_j (j ≤ n contribution) = convolveMat T A (charActMat T S) n.
  unfold convolveMat charActMat
  rw [show (∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff k • ∑ j ∈ Finset.range (n + 1),
              T ^ j * A * S (n + k - j))
        = ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            ∑ j ∈ Finset.range (n + 1),
              T.charpoly.coeff k • (T ^ j * A * S (n + k - j)) from
      Finset.sum_congr rfl (fun k _ => smul_sum_explicit _ _ _)]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl ?_
  intro j hj
  rw [Finset.mem_range] at hj
  have hreidx : ∀ k, n + k - j = (n - j) + k := by intro k; omega
  simp_rw [hreidx]
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [Matrix.mul_smul]

/-- **Matrix Cayley-Hamilton at shift `m`**: `∑_k p_k • T^(m+k) = 0`. Re-derived
in-place to keep the module self-contained (matches
`SequelPolesConv.ch_shift_mat` and `SequelPoles`'s own derivation
definitionally; the in-place form avoids the documented kernel `whnf`
timeout when aliasing across modules). -/
lemma ch_shift_mat (T : Matrix ι ι R) (m : ℕ) :
    ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • T ^ (m + k) = 0 := by
  have hCH : (Polynomial.aeval T) T.charpoly = 0 := Matrix.aeval_self_charpoly _
  have hExpand : (Polynomial.aeval T) T.charpoly
      = ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff l • T ^ l := Polynomial.aeval_eq_sum_range _
  calc ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff k • T ^ (m + k)
      = ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff k • (T ^ m * T ^ k) := by
        refine Finset.sum_congr rfl ?_
        intro k _; rw [pow_add]
    _ = T ^ m * ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff k • T ^ k := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl ?_
        intro k _; rw [Matrix.mul_smul]
    _ = T ^ m * 0 := by rw [← hExpand, hCH]
    _ = 0 := Matrix.mul_zero _

/-- **Boundary killer** (matrix-valued): applying one further `charActMat T` to
`BoundaryMat T A S` yields zero. Each boundary summand has the form
`T^(n+l+1+t) * (A * S(k-1-t))` with `(A * S(k-1-t))` independent of both `n`
and `l`; reindexing `n+l+1+t = (n+1+t)+l` and factoring the `(A * S(k-1-t))`
on the right reduces each `(k, t)` slice to a `ch_shift_mat` application at
shift `n+1+t`.

Mirrors `SequelPolesConv.boundary_charAct_eq_zero` (scalar version), now
generalised so the `S`-factor on the right can be any matrix-valued sequence
(not just `T^·`). This generality is what lets the iterated `lem:poles`
induction reuse this lemma at every recursion depth. -/
lemma charActMat_BoundaryMat_eq_zero (T A : Matrix ι ι R) (S : ℕ → Matrix ι ι R)
    (n : ℕ) :
    charActMat T (BoundaryMat T A S) n = 0 := by
  show ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff l • BoundaryMat T A S (n + l) = 0
  -- Push p_l into the outer (k-) sum of BoundaryMat.
  rw [show (∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff l • BoundaryMat T A S (n + l))
        = ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
            ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff l • (T.charpoly.coeff k •
                ∑ t ∈ Finset.range k,
                  T ^ (n + l + 1 + t) * A * S (k - 1 - t)) from
      Finset.sum_congr rfl (fun l _ => smul_sum_explicit _ _ _)]
  -- Swap l ↔ k.
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero ?_
  intro k _
  -- For fixed k: commute p_l and p_k, pull p_k out.
  rw [show (∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff l • (T.charpoly.coeff k •
              ∑ t ∈ Finset.range k, T ^ (n + l + 1 + t) * A * S (k - 1 - t)))
        = T.charpoly.coeff k •
            ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff l •
                ∑ t ∈ Finset.range k,
                  T ^ (n + l + 1 + t) * A * S (k - 1 - t) from ?_]
  swap
  · rw [smul_sum_explicit]
    refine Finset.sum_congr rfl ?_
    intro l _
    rw [smul_comm]
  -- Push p_l into the t-sum, then swap l ↔ t.
  rw [show (∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff l •
              ∑ t ∈ Finset.range k, T ^ (n + l + 1 + t) * A * S (k - 1 - t))
        = ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
            ∑ t ∈ Finset.range k,
              T.charpoly.coeff l • (T ^ (n + l + 1 + t) * A * S (k - 1 - t)) from
      Finset.sum_congr rfl (fun l _ => smul_sum_explicit _ _ _)]
  rw [Finset.sum_comm]
  -- For each fixed t, factor out (A * S(k-1-t)) on the right and apply ch_shift_mat.
  have hinner : ∑ t ∈ Finset.range k,
      ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
        T.charpoly.coeff l • (T ^ (n + l + 1 + t) * A * S (k - 1 - t)) = 0 := by
    refine Finset.sum_eq_zero ?_
    intro t _
    set Y := A * S (k - 1 - t)
    have hreidx : ∀ l, n + l + 1 + t = (n + 1 + t) + l := by intro l; ring
    have hassoc : ∀ l,
        T ^ (n + l + 1 + t) * A * S (k - 1 - t)
        = T ^ ((n + 1 + t) + l) * Y := by
      intro l
      rw [hreidx]
      show T ^ ((n + 1 + t) + l) * A * S (k - 1 - t) = T ^ ((n + 1 + t) + l) * Y
      rw [Matrix.mul_assoc]
    simp_rw [hassoc]
    have h := ch_shift_mat T (n + 1 + t)
    calc ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff l • (T ^ ((n + 1 + t) + l) * Y)
        = (∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff l • T ^ ((n + 1 + t) + l)) * Y := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl ?_
          intro l _; exact (Matrix.smul_mul _ _ _).symm
      _ = 0 * Y := by rw [h]
      _ = 0 := Matrix.zero_mul _
  rw [hinner, smul_zero]

/-- **Linearity** of `charActMat` in the sequence argument. -/
lemma charActMat_add (T : Matrix ι ι R) (S₁ S₂ : ℕ → Matrix ι ι R) (n : ℕ) :
    charActMat T (fun m => S₁ m + S₂ m) n
      = charActMat T S₁ n + charActMat T S₂ n := by
  show ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • (S₁ (n + k) + S₂ (n + k))
    = (∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
        T.charpoly.coeff k • S₁ (n + k))
      + ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
        T.charpoly.coeff k • S₂ (n + k)
  simp_rw [smul_add, Finset.sum_add_distrib]

/-- `charActMat` of the zero sequence is zero. -/
lemma charActMat_zero (T : Matrix ι ι R) (n : ℕ) :
    charActMat T (fun _ => (0 : Matrix ι ι R)) n = 0 := by
  show ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • (0 : Matrix ι ι R) = 0
  simp

/-- **Iterated charpoly action**: `charActIter T S d n` applies `charActMat T`
`d`-fold. The `d+1`-fold action killing a `d`-fold convolutional sum is the
arbitrary-`d` form of `lem:poles`. -/
noncomputable def charActIter (T : Matrix ι ι R) (S : ℕ → Matrix ι ι R) :
    ℕ → ℕ → Matrix ι ι R
  | 0, n => S n
  | d + 1, n => charActMat T (charActIter T S d) n

/-- **Iterated linearity** of `charActIter` in the sequence argument. By
induction on `d` from `charActMat_add`; the inductive step substitutes the
IH and re-applies `charActMat_add` to the resulting sum. Needed by the
arbitrary-`d` induction sketched in the module docstring (each step
applies `charActMat T` to a sum `convolveMat T A S + BoundaryMat T A S`,
and the d-fold version of this is exactly `charActIter_add`). -/
lemma charActIter_add (T : Matrix ι ι R) (S₁ S₂ : ℕ → Matrix ι ι R) :
    ∀ d n, charActIter T (fun m => S₁ m + S₂ m) d n
      = charActIter T S₁ d n + charActIter T S₂ d n := by
  intro d
  induction d with
  | zero => intro n; rfl
  | succ d IH =>
    intro n
    show charActMat T (charActIter T (fun m => S₁ m + S₂ m) d) n
      = charActMat T (charActIter T S₁ d) n + charActMat T (charActIter T S₂ d) n
    have hpoint : (fun m => charActIter T (fun m => S₁ m + S₂ m) d m)
        = fun m => charActIter T S₁ d m + charActIter T S₂ d m := by
      funext m; exact IH m
    rw [show (charActIter T (fun m => S₁ m + S₂ m) d)
          = fun m => charActIter T S₁ d m + charActIter T S₂ d m from hpoint]
    exact charActMat_add T (charActIter T S₁ d) (charActIter T S₂ d) n

end OrigamiCone.Sequel

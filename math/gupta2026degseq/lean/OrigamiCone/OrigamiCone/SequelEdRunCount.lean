import Mathlib
import OrigamiCone.SequelEdPolyFit

/-!
# Sequel: run-length count is eventually polynomial (Task E.δ.b)

The combinatorial arm of paper `lem:uniform` (§8) reduces, after
active-column contraction, to counting the ways a "type" with `r` frozen runs
extends to a width-`n` grid: each run takes any length `≥ 1`, with the runs
summing to `n - c` (where `c` is the number of columns in the contracted
type).  By standard stars-and-bars, that count is the number of positive
compositions of `n - c` into `r` parts, namely `(n - c - 1).choose (r - 1)`.
The paper's claim is that this count, as a function of `n`, agrees with a
polynomial of degree `r - 1 ≤ d - 2`.

This module formalises that **polynomial-agreement** step, the crux of the
degree bound: for any offset `c` and degree parameter `k`, the function
`n ↦ (n - c).choose k` agrees on `{n ≥ c}` with a polynomial of natDegree
`≤ k`.  The explicit witness is `(choosePolyℚ k).comp (X - C c)`.

The identification "positive-composition count `= (n - c - 1).choose (r - 1)`"
is the standard stars-and-bars bijection (a `Finset.card` computation, cited
but not re-proved here — Mathlib lacks a direct `finAntidiagonal`-card lemma,
and the bijection is orthogonal to the polynomiality that `lem:uniform`
actually invokes).

## Theorems

* `shiftedChoosePoly (c k : ℕ) : Polynomial ℚ` — `(choosePolyℚ k).comp (X - C c)`.
* `shiftedChoosePoly_natDegree_le (c k : ℕ) : (shiftedChoosePoly c k).natDegree ≤ k`.
* `shiftedChoosePoly_eval (c k n) (hn : c ≤ n) : eval n = (n - c).choose k`.
* **`runCount_eventually_polynomial`** (∃-packaged, `lem:uniform` shape):
  for offset `c` and run-degree `k`, there is a polynomial `p` of natDegree
  `≤ k` with `((n - c).choose k : ℚ) = p.eval (n : ℚ)` for every `n ≥ c`.
* **`degreeBound_assembly`** — sum-of-types assembly: if `F` decomposes on
  `{n ≥ N}` as `∑ t, mult t · (n - c t).choose (k t)` with each `k t ≤ D`,
  then `F` agrees on `{n ≥ N}` with a polynomial of natDegree ≤ D.

## Substrate

Uses `SequelEdPolyFit.choosePolyℚ` (polynomial lift of `n ↦ n.choose k`,
natDegree ≤ k, evaluating to `n.choose k`) + Mathlib `Polynomial.comp`
degree/eval lemmas.  Standalone; imports only `OrigamiCone.SequelEdPolyFit`.

## Role in Task E.δ

This closes the "polynomial in `n` of degree `r - 1`" claim (paper
`lem:uniform`, per-axis-degree paragraph, final sentence).  Combined with
`SequelEdActiveCol.numActiveColumns_le_numExtrema` (the `r ≤ d - 1` bound via
`#active ≤ #extrema`), the degree bound `≤ d - 2` follows once the
frozen-run contraction is formalised (deferred: needs the `SequelFrozen`
column-substrate bridge).

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
Check with `#print axioms OrigamiCone.Sequel.runCount_eventually_polynomial`.
-/

namespace OrigamiCone.Sequel

open Polynomial Finset

/-- Shifted binomial polynomial: `(choosePolyℚ k).comp (X - C c)`.
Evaluates to `(n - c).choose k` for `n ≥ c`, and has natDegree ≤ k. -/
noncomputable def shiftedChoosePoly (c k : ℕ) : Polynomial ℚ :=
  (choosePolyℚ k).comp (Polynomial.X - Polynomial.C (c : ℚ))

/-- `shiftedChoosePoly c k` has natDegree ≤ k. -/
lemma shiftedChoosePoly_natDegree_le (c k : ℕ) :
    (shiftedChoosePoly c k).natDegree ≤ k := by
  unfold shiftedChoosePoly
  calc ((choosePolyℚ k).comp (Polynomial.X - Polynomial.C (c : ℚ))).natDegree
      ≤ (choosePolyℚ k).natDegree * (Polynomial.X - Polynomial.C (c : ℚ)).natDegree :=
        Polynomial.natDegree_comp_le
    _ ≤ k * 1 := by
        gcongr
        · exact choosePolyℚ_natDegree_le k
        · rw [Polynomial.natDegree_X_sub_C]
    _ = k := Nat.mul_one k

/-- The binomial count `(n - c).choose k` (as ℚ) equals `shiftedChoosePoly c k`
at `n`, for every `n ≥ c`. -/
lemma shiftedChoosePoly_eval (c k : ℕ) (n : ℕ) (hn : c ≤ n) :
    (shiftedChoosePoly c k).eval (n : ℚ) = ((n - c).choose k : ℚ) := by
  unfold shiftedChoosePoly
  rw [Polynomial.eval_comp, Polynomial.eval_sub, Polynomial.eval_X,
      Polynomial.eval_C]
  have hcast : (n : ℚ) - (c : ℚ) = ((n - c : ℕ) : ℚ) := by
    rw [Nat.cast_sub hn]
  rw [hcast, choosePolyℚ_eval_nat]

/-- **Run-length count is eventually polynomial** (paper `lem:uniform`,
per-axis-degree step).  For any offset `c` and run-degree parameter `k`, the
binomial count `n ↦ (n - c).choose k` agrees on `{n ≥ c}` with a single
polynomial of natDegree `≤ k`.

Applied in `lem:uniform` with `c` the width of a contracted type and
`k = r - 1` the number of frozen runs minus one: since `r ≤ #active columns
≤ d - 1` (via `SequelEdActiveCol.numActiveColumns_le_numExtrema`), the count
has degree `≤ d - 2`. -/
theorem runCount_eventually_polynomial (c k : ℕ) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ k ∧
      ∀ n : ℕ, c ≤ n → ((n - c).choose k : ℚ) = p.eval (n : ℚ) := by
  refine ⟨shiftedChoosePoly c k, shiftedChoosePoly_natDegree_le c k, ?_⟩
  intro n hn
  exact (shiftedChoosePoly_eval c k n hn).symm

/-! ## Degree-bound assembly

Packages the per-type run-count contributions into the paper's degree bound.
The paper's `lem:uniform` writes `E_d(m, ·)` (for `n` large) as a finite sum
`∑_C P_C`, each `P_C` the run-length count of a type `C` with `r_C ≤ d - 1`
runs, hence a polynomial of degree `r_C - 1 ≤ d - 2`.  A sum of such
polynomials has degree `≤ d - 2`.  This section supplies that assembly
abstractly, parameterised over the type set. -/

/-- **Degree-bound assembly** (paper `lem:uniform`, "the count `E_d(m, ·)`
agrees with the single polynomial `∑_C P_C`, of degree at most `d - 2`").
If `F : ℕ → ℚ` decomposes on `{n ≥ N}` as a finite sum of run-count
contributions `∑ t, mult t · (n - c t).choose (k t)`, each with `k t ≤ D`
and `c t ≤ N`, then `F` agrees on `{n ≥ N}` with a polynomial of
natDegree ≤ D.

Applied in `lem:uniform` with `D = d - 2`, `k t = r_t - 1 ≤ d - 2` (via the
runs bound `numFrozenRuns_lt_numExtrema`), `mult t` the multiplicity of type
`t`, and `c t` the width of the contracted type.  The type-decomposition
`hdecomp` itself (the contraction map + finite type enumeration) is the
remaining combinatorial substrate, deferred. -/
theorem degreeBound_assembly
    {ι : Type*} (types : Finset ι) (mult : ι → ℚ) (c k : ι → ℕ) (D N : ℕ)
    (F : ℕ → ℚ)
    (hk : ∀ t ∈ types, k t ≤ D)
    (hc : ∀ t ∈ types, c t ≤ N)
    (hdecomp : ∀ n, N ≤ n →
      F n = ∑ t ∈ types, mult t * ((n - c t).choose (k t) : ℚ)) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ n : ℕ, N ≤ n → F n = p.eval (n : ℚ) := by
  refine ⟨∑ t ∈ types, Polynomial.C (mult t) * shiftedChoosePoly (c t) (k t), ?_, ?_⟩
  · apply Polynomial.natDegree_sum_le_of_forall_le
    intro t ht
    by_cases hm : mult t = 0
    · simp [hm]
    · rw [Polynomial.natDegree_C_mul hm]
      exact (shiftedChoosePoly_natDegree_le (c t) (k t)).trans (hk t ht)
  · intro n hn
    rw [hdecomp n hn, Polynomial.eval_finset_sum]
    apply Finset.sum_congr rfl
    intro t ht
    rw [Polynomial.eval_mul, Polynomial.eval_C]
    rw [shiftedChoosePoly_eval (c t) (k t) n (le_trans (hc t ht) hn)]

end OrigamiCone.Sequel

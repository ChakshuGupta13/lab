import OrigamiCone.SequelCompositionCount
import Mathlib.Algebra.Polynomial.BigOperators

/-!
# Substrate 4/5: Contraction-type data structure + counting polynomial API

Abstract data type for the "type C" of the paper's Lemma 8.5 (`lem:uniform`),
with the polynomial-counting interface substrate 5 (`SequelUniformOnsetProof`)
will assemble.

A `ContractionType` bundles two natural numbers:
* `width` — the number of columns in the contracted type `C` (`w_C`);
* `runCount` — the number of frozen runs in `C`, each of which contracts to
  a single column and can be re-extended to any length `≥ 1`.

Runs are required nonzero (`1 ≤ runCount`).  This restricts to "extendable"
types; rigid types (`runCount = 0`, unique width) are outside the polynomial
regime and belong to substrate 5's assembly, not here.

## Counting

For each type `C` and target width `n ≥ C.width`, the extensions of `C` to
width `n` correspond to run-length vectors `(l_1, …, l_r_C)` with each
`l_i ≥ 1` summing to `n - w_C + r_C`, of which there are
`Nat.choose (n − w_C + r_C − 1) (r_C − 1)`.  Substrate 1
(`composition_count_as_poly`) supplies the polynomial witness.

## Interface for substrate 5

Given a finite set `S` of contraction types and `n ≥ max_{C ∈ S} C.width`,
* `sum_eval_of_le`  — `∑ C ∈ S, count C n = (∑ C ∈ S, poly C).eval n` over ℚ;
* `sum_natDegree_le`  — if every `C.runCount ≤ D + 1`, the sum polynomial has
  natural degree at most `D`.

Substrate 5 will assemble a finite set `S(d, m)` of types from `Ed(m, ·)`,
prove the decomposition identity, and apply the sum interface with `D := d`
to obtain the uniform-onset polynomial witness.

All axiom-clean: `[propext, Classical.choice, Quot.sound]`, no `sorry`.
`Classical.choice` enters through the noncomputable `poly` witness of
`composition_count_as_poly`.
-/

namespace OrigamiCone.Sequel

open Polynomial

/-- **Contraction type**: a bookkeeping record for a contracted height-function
type `C` in the paper's Lemma 8.5.  Stores the type's total width `w_C` and
its number of frozen runs `r_C ≥ 1`.

Rigid types (`r_C = 0`, unique width, contributing a single value at a single
`n`) are handled outside this abstraction by substrate 5. -/
structure ContractionType where
  width : ℕ
  runCount : ℕ
  runCount_pos : 1 ≤ runCount
  deriving DecidableEq

namespace ContractionType

/-- **Composition count**: for `n ≥ C.width`, the number of ways to extend
type `C` to width `n` by choosing a positive integer length for each of its
`C.runCount` frozen runs.  Equals `Nat.choose (n − w + r − 1) (r − 1)` where
`w = C.width` and `r = C.runCount`. -/
def count (C : ContractionType) (n : ℕ) : ℕ :=
  Nat.choose (n - C.width + C.runCount - 1) (C.runCount - 1)

/-- **Counting polynomial** for a contraction type.  The polynomial witness
from `composition_count_as_poly`, packaged as a definition. -/
noncomputable def poly (C : ContractionType) : Polynomial ℚ :=
  Classical.choose
    (composition_count_as_poly C.width C.runCount C.runCount_pos)

/-- Natural-degree bound: `poly C` has degree at most `runCount − 1`. -/
theorem poly_natDegree_le (C : ContractionType) :
    C.poly.natDegree ≤ C.runCount - 1 :=
  (Classical.choose_spec
    (composition_count_as_poly C.width C.runCount C.runCount_pos)).1

/-- Evaluation identity: for `n ≥ C.width`, `poly C` evaluated at `n` equals
the composition count `count C n`, cast to `ℚ`. -/
theorem poly_eval_of_le (C : ContractionType) {n : ℕ} (hn : C.width ≤ n) :
    (C.count n : ℚ) = C.poly.eval (n : ℚ) :=
  (Classical.choose_spec
    (composition_count_as_poly C.width C.runCount C.runCount_pos)).2 n hn

/-- **Finset-sum evaluation identity**: for a finset `S` of contraction types
and `n` at least the width of every `C ∈ S`, the total composition count
equals the evaluation of the sum polynomial at `n` (over `ℚ`). -/
theorem sum_eval_of_le (S : Finset ContractionType) {n : ℕ}
    (hn : ∀ C ∈ S, C.width ≤ n) :
    ((∑ C ∈ S, C.count n : ℕ) : ℚ) = (∑ C ∈ S, C.poly).eval (n : ℚ) := by
  rw [Polynomial.eval_finset_sum]
  push_cast
  exact Finset.sum_congr rfl (fun C hCS => C.poly_eval_of_le (hn C hCS))

/-- **Finset-sum degree bound**: if every `C ∈ S` has `runCount ≤ D + 1`,
the sum polynomial `∑ C ∈ S, poly C` has natural degree at most `D`. -/
theorem sum_natDegree_le (S : Finset ContractionType) {D : ℕ}
    (hD : ∀ C ∈ S, C.runCount ≤ D + 1) :
    (∑ C ∈ S, C.poly).natDegree ≤ D := by
  refine Polynomial.natDegree_sum_le_of_forall_le _ _ (fun C hCS => ?_)
  refine (C.poly_natDegree_le).trans ?_
  have := hD C hCS
  omega

end ContractionType

end OrigamiCone.Sequel

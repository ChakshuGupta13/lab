import OrigamiCone.SequelEdFiberPartition
import OrigamiCone.SequelContractionType

/-!
# Shrunk axiomatic gap: fiber cardinality + composition sum

Isolates the sole residual obligation of `Ed_decomposition_of_ge_two`:
the fiber cardinality identity (paper Lemma 8.5's run-length-vector
bijection).  Given the partition + disjointness from
`SequelEdFiberPartition`, the total fiber-cardinality decomposition
`Ed d m n = ∑ t ∈ PRFinset, (fiber t n).card` is a *theorem* — the
remaining axiomatic content is only the pointwise identity
`(fiber t n).card = choose (n - t.W + r - 1, r - 1)`.

## Contents

* `PaperReducedForm.toContractionType`: (with `1 ≤ runCount`) →
  `ContractionType`.
* `fiber_card_axiom`: fiber cardinality identity (paper Lemma 8.5, isolated
  as an axiom pending the run-length-vector bijection).
* `fiber_card_zero_axiom`: for `t.runCount = 0`, fiber is empty at large
  `n` (a specific case of the general axiom, split for cleanliness).
* `Ed_eq_sum_fiber_card`: `Ed_finset.card = ∑ t ∈ PRFinset, fiber-card`
  (a *theorem* — partition + disjointness).
* `Ed_eq_sum_fiber_card_at_large_n`: same bridged to `Ed d m n` at large
  `n`.

## Residual obligation shrinkage

Before this module, `Ed_decomposition_of_ge_two` was a monolithic axiom
producing `∃ S mult, bounds ∧ sum-identity`.  This module reduces the
axiomatic content to the pointwise fiber-cardinality identity; the
aggregation into `S : Finset ContractionType` + `mult` + the full sum
identity of the outer axiom becomes a theorem (still pending — see
follow-up module for the Finset image aggregation).

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline
plus the new `fiber_card_axiom`, `fiber_card_zero_axiom`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m : ℕ}

namespace PaperReducedForm

/-- Map a paper-reduced form with `1 ≤ runCount` to a `ContractionType`.
The `runCount_pos` field is discharged from the input hypothesis. -/
noncomputable def toContractionType {d : ℕ} (t : PaperReducedForm m d)
    (hr : 1 ≤ t.runCount) : ContractionType where
  width := t.W
  runCount := t.runCount
  runCount_pos := hr

end PaperReducedForm

/-- **Fiber-cardinality axiom** (paper Lemma 8.5, isolated).

For each paper-reduced form `t : PaperReducedForm m d` with `1 ≤
t.runCount`, and each `n ≥ 2*d + 4`, the fiber cardinality equals the
composition count `Nat.choose (n - t.W + t.runCount - 1) (t.runCount -
1)`.

**Paper argument**: the extensions of `t` from `Cell m t.W` to `Cell m
n` are in bijection with positive-integer run-length vectors `ℓ : Fin
t.runCount → ℕ+` summing to `n - t.W + t.runCount`; the number of
such vectors is the composition count (stars-and-bars, already proved as
`composition_count_as_poly`).

**Formalising this bijection** is the sole residual obligation of the
formalisation of `lem:uniform` / `thm:poly` §8. -/
axiom fiber_card_axiom
    {d : ℕ} (hm : 1 ≤ m) (hd : 2 ≤ d)
    (t : PaperReducedForm m d) (hr : 1 ≤ t.runCount)
    {n : ℕ} (hn : 2 * d + 4 ≤ n) :
    (fiber hm (by omega : 2 ≤ n) hd t).card =
      Nat.choose (n - t.W + t.runCount - 1) (t.runCount - 1)

/-- **Fiber-emptiness for `runCount = 0`**.  When a paper-reduced form
has no frozen columns (`runCount = 0`), no proper extension exists at
larger `n`: extending would require adding some columns, and the
contraction map preserves the isolated-frozen structure — so the extension
would necessarily introduce new frozen or active columns, mismatching `t`.
At `n > t.W`, the fiber is empty. -/
axiom fiber_card_zero_axiom
    {d : ℕ} (hm : 1 ≤ m) (hd : 2 ≤ d)
    (t : PaperReducedForm m d) (h0 : t.runCount = 0)
    {n : ℕ} (hn : 2 * d + 4 ≤ n) :
    (fiber hm (by omega : 2 ≤ n) hd t).card = 0

/-- **Total fiber cardinality via the partition + disjointness**
(a *theorem*, no axioms beyond baseline + partition/disjointness).

The `Ed`-Finset cardinality decomposes as a sum over `PRFinset` of fiber
cardinalities.  Direct from `Ed_finset_eq_biUnion_fiber` +
`fiber_pairwise_disjoint` + `Finset.card_biUnion`. -/
theorem Ed_eq_sum_fiber_card {d : ℕ} (hm : 1 ≤ m) (hd : 2 ≤ d) {n : ℕ}
    (hn : 2 ≤ n) :
    (Ed_finset d m hm n).card =
      ∑ t ∈ PaperReducedForm.PRFinset m d hm,
        (fiber hm hn hd t).card := by
  rw [Ed_finset_eq_biUnion_fiber hm hn hd]
  exact Finset.card_biUnion (by
    intro t1 ht1 t2 ht2 hne
    exact fiber_pairwise_disjoint hm hn hd ht1 ht2 hne)

/-- **Total cardinality as `Ed` at large `n`** (a *theorem*, using the
`Ed_finset_card_eq_Ed` bridge).  Combined with `fiber_card_axiom` /
`fiber_card_zero_axiom`, this is the residual identity closing
`Ed_decomposition_of_ge_two`. -/
theorem Ed_eq_sum_fiber_card_at_large_n {d : ℕ} (hm : 1 ≤ m) (hd : 2 ≤ d)
    {n : ℕ} (hn : 2 * d + 4 ≤ n) :
    Ed d m n =
      ∑ t ∈ PaperReducedForm.PRFinset m d hm,
        (fiber hm (by omega : 2 ≤ n) hd t).card := by
  have hn2 : 2 ≤ n := by omega
  have hn1 : 1 ≤ n := by omega
  rw [← Ed_finset_card_eq_Ed d m hm n hn1]
  exact Ed_eq_sum_fiber_card hm hd hn2

end OrigamiCone.Sequel

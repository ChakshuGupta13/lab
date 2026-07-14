import OrigamiCone.SequelEdCanonicalReducedForm
import OrigamiCone.SequelEdPRFinset
import OrigamiCone.SequelEdUniformFamily

/-!
# Fiber Finset + partition + disjointness

Extends `SequelEdCanonicalReducedForm` with the fiber structure needed to
apply `Ed_polynomial_of_partition` (from `SequelEdUniformFamily`) — the
`Ed`-Finset partitions over the `PRFinset` of paper-reduced forms via the
`canonicalReducedForm` fiber map, with the standard function-preimage
disjointness.

## Contents

* `mem_Ed_finset_iff`: `h ∈ Ed_finset d m hm n ↔ IsCanonicalHeight h ∧
  numExtrema h = d` for `1 ≤ n`.
* `fiber`: `PaperReducedForm m d → Finset (Cell m n → ℤ)`, the fiber of
  `canonicalReducedForm` over `t` inside `Ed_finset d m hm n`.
* `fiber_subset_Ed_finset`: `fiber t n ⊆ Ed_finset d m hm n`.
* `mem_fiber_iff`: membership characterisation.
* `Ed_finset_eq_biUnion_fiber`: `Ed_finset = biUnion (PRFinset) fiber`.
* `fiber_pairwise_disjoint`: distinct paper-reduced forms have disjoint
  fibers (function-preimage + proof irrelevance).

Together these are the `hpart` + `hdisj` hypotheses of
`Ed_polynomial_of_partition`.  The remaining `hfiber` hypothesis
(`(fiber t n).card = (n - (W - r + 1)).choose (r - 1)`) is the paper's
Lemma 8.5 combinatorial content — a run-length-vector bijection, not
proved here.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline
(no new).
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m : ℕ}

/-- **Membership in `Ed_finset`** for the nonempty range `1 ≤ n`.
Elements of `Ed_finset d m hm n` are exactly the canonical `d`-extremum
height functions on `Cell m n`. -/
theorem mem_Ed_finset_iff (d m : ℕ) (hm : 1 ≤ m) {n : ℕ} (hn : 1 ≤ n)
    (h : Cell m n → ℤ) :
    h ∈ Ed_finset d m hm n ↔ IsCanonicalHeight h ∧ numExtrema h = d := by
  unfold Ed_finset
  rw [dif_pos hn]
  exact Set.Finite.mem_toFinset _

/-- **The fiber of `canonicalReducedForm` over `t`** inside `Ed_finset`.
Uses classical decidability (the equality `canonicalReducedForm ... = t`
mentions a `noncomputable` function). -/
noncomputable def fiber (hm : 1 ≤ m) {n d : ℕ} (hn : 2 ≤ n) (hd : 2 ≤ d)
    (t : PaperReducedForm m d) : Finset (Cell m n → ℤ) := by
  classical
  exact (Ed_finset d m hm n).filter (fun h =>
    ∃ (hC : IsCanonicalHeight h) (hE : numExtrema h = d),
      canonicalReducedForm hm hn hd h hC hE = t)

/-- The fiber is a subset of the ambient `Ed_finset`. -/
theorem fiber_subset_Ed_finset (hm : 1 ≤ m) {n d : ℕ} (hn : 2 ≤ n)
    (hd : 2 ≤ d) (t : PaperReducedForm m d) :
    fiber hm hn hd t ⊆ Ed_finset d m hm n := by
  unfold fiber
  classical
  intro h hmem
  exact (Finset.mem_filter.mp hmem).1

/-- **Membership in the fiber** — extract the equality along with the
canonicity/extremum witnesses. -/
theorem mem_fiber_iff (hm : 1 ≤ m) {n d : ℕ} (hn : 2 ≤ n) (hd : 2 ≤ d)
    (t : PaperReducedForm m d) (h : Cell m n → ℤ) :
    h ∈ fiber hm hn hd t ↔
      h ∈ Ed_finset d m hm n ∧
      ∃ (hC : IsCanonicalHeight h) (hE : numExtrema h = d),
        canonicalReducedForm hm hn hd h hC hE = t := by
  unfold fiber
  classical
  exact Finset.mem_filter

/-- **Partition** — `Ed_finset` is the disjoint union of the fibers over
`PRFinset`.  Every `d`-extremum height function has *some* paper-reduced
form via `canonicalReducedForm`, so it lands in the corresponding
fiber. -/
theorem Ed_finset_eq_biUnion_fiber (hm : 1 ≤ m) {n d : ℕ} (hn : 2 ≤ n)
    (hd : 2 ≤ d) :
    Ed_finset d m hm n =
      (PaperReducedForm.PRFinset m d hm).biUnion (fun t => fiber hm hn hd t) := by
  classical
  ext h
  constructor
  · intro hmem
    have hn1 : 1 ≤ n := by omega
    have hmem' := (mem_Ed_finset_iff d m hm hn1 h).mp hmem
    obtain ⟨hC, hE⟩ := hmem'
    let t := canonicalReducedForm hm hn hd h hC hE
    rw [Finset.mem_biUnion]
    refine ⟨t, PaperReducedForm.mem_PRFinset hm t, ?_⟩
    rw [mem_fiber_iff]
    exact ⟨hmem, hC, hE, rfl⟩
  · intro hmem
    rw [Finset.mem_biUnion] at hmem
    obtain ⟨t, _htmem, hfib⟩ := hmem
    exact fiber_subset_Ed_finset hm hn hd t hfib

/-- **Disjointness** — distinct paper-reduced forms have disjoint fibers.
Follows from proof irrelevance on the canonicity + extremum witnesses:
if `h` lands in both `fiber t₁` and `fiber t₂`, the two calls of
`canonicalReducedForm` on the same `h` (with the same, proof-irrelevant
prop witnesses) return the same value, forcing `t₁ = t₂`. -/
theorem fiber_pairwise_disjoint (hm : 1 ≤ m) {n d : ℕ} (hn : 2 ≤ n)
    (hd : 2 ≤ d) :
    (↑(PaperReducedForm.PRFinset m d hm) : Set (PaperReducedForm m d)).PairwiseDisjoint
      (fun t => fiber hm hn hd t) := by
  classical
  intro t1 _ht1 t2 _ht2 hne
  refine Finset.disjoint_filter.mpr ?_
  intro h _hmem hex1 hex2
  obtain ⟨hC1, hE1, heq1⟩ := hex1
  obtain ⟨hC2, hE2, heq2⟩ := hex2
  -- Proof irrelevance: `hC1 = hC2` and `hE1 = hE2` (both are Props on the
  -- same `h`).  Hence `canonicalReducedForm ... hC1 hE1
  -- = canonicalReducedForm ... hC2 hE2`, so `t1 = t2`, contradicting `hne`.
  apply hne
  rw [← heq1, ← heq2]

end OrigamiCone.Sequel

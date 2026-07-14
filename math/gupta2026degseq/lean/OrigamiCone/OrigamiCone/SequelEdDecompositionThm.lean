import OrigamiCone.SequelEdFiberCardAxiom

/-!
# `Ed_decomposition_of_ge_two` as a theorem

Closes the residual monolithic axiom `Ed_decomposition_of_ge_two` (from
`SequelUniformOnsetProof`) into a theorem, modulo the two fine-grained
fiber-cardinality axioms `fiber_card_axiom` and `fiber_card_zero_axiom`
(from `SequelEdFiberCardAxiom`).

## Proof strategy

1. `Ed d m n = ∑ t ∈ PRFinset, (fiber t n).card` (theorem —
   `Ed_eq_sum_fiber_card_at_large_n`).
2. Split PRFinset into `PRPos` (runCount ≥ 1) + `PRZero` (runCount = 0);
   the zero part vanishes by `fiber_card_zero_axiom`.
3. Aggregate the positive part by `toCT : PaperReducedForm →
   ContractionType`; `SPos := image` and `multPos := preimage count`.
4. Partition-by-image: `∑ t ∈ PRPos, count(toCT t) = ∑ C ∈ SPos, mult(C)
   · count C n` via `Finset.sum_biUnion`.
5. Combined with `fiber_card_axiom`, yields the axiom shape.

## Contents

* `PaperReducedForm.toCT`, `toCT_width`, `toCT_runCount_of_pos`:
  `PaperReducedForm → ContractionType` (total, uses `max 1` for the dummy).
* `Ed_decomposition_of_ge_two_thm`: the axiom shape, now a theorem.

No `sorry`.  Axioms of `Ed_decomposition_of_ge_two_thm`:
`[propext, Classical.choice, Quot.sound, fiber_card_axiom,
fiber_card_zero_axiom]`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m : ℕ}

namespace PaperReducedForm

/-- Total map `PaperReducedForm m d → ContractionType`, using `max 1
t.runCount` for the `runCount_pos` field.  For `t.runCount ≥ 1` this
agrees with the natural conversion; for `t.runCount = 0`, produces a
`⟨t.W, 1, _⟩` sentinel (whose fibers contribute zero at large `n` by
`fiber_card_zero_axiom`). -/
noncomputable def toCT {d : ℕ} (t : PaperReducedForm m d) : ContractionType where
  width := t.W
  runCount := max 1 t.runCount
  runCount_pos := le_max_left _ _

@[simp] theorem toCT_width {d : ℕ} (t : PaperReducedForm m d) :
    t.toCT.width = t.W := rfl

theorem toCT_runCount_of_pos {d : ℕ} (t : PaperReducedForm m d)
    (hr : 1 ≤ t.runCount) : t.toCT.runCount = t.runCount := by
  simp [toCT, hr]

end PaperReducedForm

open Classical

variable {d : ℕ}

/-- Positive-runCount subset of `PRFinset`. -/
private noncomputable def PRPos (d m : ℕ) (hm : 1 ≤ m) :
    Finset (PaperReducedForm m d) :=
  (PaperReducedForm.PRFinset m d hm).filter (fun t => 1 ≤ t.runCount)

/-- Zero-runCount subset of `PRFinset`. -/
private noncomputable def PRZero (d m : ℕ) (hm : 1 ≤ m) :
    Finset (PaperReducedForm m d) :=
  (PaperReducedForm.PRFinset m d hm).filter (fun t => t.runCount = 0)

/-- Contraction types = image of `PRPos` under `toCT`. -/
private noncomputable def SPos (d m : ℕ) (hm : 1 ≤ m) : Finset ContractionType :=
  (PRPos d m hm).image PaperReducedForm.toCT

/-- Multiplicity of `C`: `|{t ∈ PRPos : toCT t = C}|`. -/
private noncomputable def multPos (d m : ℕ) (hm : 1 ≤ m)
    (C : ContractionType) : ℕ :=
  ((PRPos d m hm).filter (fun t => t.toCT = C)).card

private theorem mem_PRPos_iff (d m : ℕ) (hm : 1 ≤ m) (t : PaperReducedForm m d) :
    t ∈ PRPos d m hm ↔ 1 ≤ t.runCount := by
  simp [PRPos, PaperReducedForm.mem_PRFinset]

private theorem mem_PRZero_iff (d m : ℕ) (hm : 1 ≤ m) (t : PaperReducedForm m d) :
    t ∈ PRZero d m hm ↔ t.runCount = 0 := by
  simp [PRZero, PaperReducedForm.mem_PRFinset]

/-- Every `C ∈ SPos` has `runCount ≤ d + 1`. -/
private theorem SPos_runCount_le (d m : ℕ) (hm : 1 ≤ m) (hm2 : 2 ≤ m)
    (C : ContractionType) (hC : C ∈ SPos d m hm) :
    C.runCount ≤ d + 1 := by
  unfold SPos at hC
  rw [Finset.mem_image] at hC
  obtain ⟨t, htPos, hteq⟩ := hC
  rw [mem_PRPos_iff] at htPos
  rw [← hteq, PaperReducedForm.toCT_runCount_of_pos t htPos]
  exact t.runCount_le_d_add_one (by omega)

/-- Every `C ∈ SPos` has `width ≤ 2*d + 3`. -/
private theorem SPos_width_le (d m : ℕ) (hm : 1 ≤ m)
    (C : ContractionType) (hC : C ∈ SPos d m hm) :
    C.width ≤ 2 * d + 3 := by
  unfold SPos at hC
  rw [Finset.mem_image] at hC
  obtain ⟨t, _, hteq⟩ := hC
  rw [← hteq, PaperReducedForm.toCT_width]
  exact t.hW_upper

/-- Fiber cardinality via `toCT.count`. -/
private theorem fiber_card_eq_count_toCT
    (hm : 1 ≤ m) (hd : 2 ≤ d) (t : PaperReducedForm m d) (hr : 1 ≤ t.runCount)
    {n : ℕ} (hn : 2 * d + 4 ≤ n) :
    (fiber hm (by omega : 2 ≤ n) hd t).card = t.toCT.count n := by
  rw [fiber_card_axiom hm hd t hr hn]
  simp [ContractionType.count, PaperReducedForm.toCT_width,
        PaperReducedForm.toCT_runCount_of_pos t hr]

/-- Split PRFinset into positive-runCount + zero-runCount. -/
private theorem PRFinset_split (d m : ℕ) (hm : 1 ≤ m) :
    PaperReducedForm.PRFinset m d hm = PRPos d m hm ∪ PRZero d m hm := by
  ext t
  simp only [Finset.mem_union, mem_PRPos_iff, mem_PRZero_iff,
             PaperReducedForm.mem_PRFinset, true_iff]
  omega

private theorem PRPos_disjoint_PRZero (d m : ℕ) (hm : 1 ≤ m) :
    Disjoint (PRPos d m hm) (PRZero d m hm) := by
  rw [Finset.disjoint_left]
  intro t ht1 ht2
  rw [mem_PRPos_iff] at ht1
  rw [mem_PRZero_iff] at ht2
  omega

/-- Partition PRPos by toCT image (a disjoint biUnion of preimages). -/
private theorem PRPos_eq_biUnion_preimage (d m : ℕ) (hm : 1 ≤ m) :
    PRPos d m hm = (SPos d m hm).biUnion
      (fun C => (PRPos d m hm).filter (fun t => t.toCT = C)) := by
  ext t
  simp only [Finset.mem_biUnion, Finset.mem_filter]
  constructor
  · intro htPos
    refine ⟨t.toCT, ?_, htPos, rfl⟩
    unfold SPos
    exact Finset.mem_image.mpr ⟨t, htPos, rfl⟩
  · rintro ⟨_C, _hC, htPos, _⟩
    exact htPos

private theorem preimage_disjoint (d m : ℕ) (hm : 1 ≤ m) :
    (↑(SPos d m hm) : Set ContractionType).PairwiseDisjoint
      (fun C => (PRPos d m hm).filter (fun t => t.toCT = C)) := by
  intros C1 _hC1 C2 _hC2 hne
  rw [Function.onFun, Finset.disjoint_left]
  intros t h1 h2
  rw [Finset.mem_filter] at h1 h2
  apply hne
  rw [← h1.2, h2.2]

/-- Sum over PRPos of fiber-cards = sum over SPos of `mult · count`. -/
private theorem sum_PRPos_fiber_card_eq (hm : 1 ≤ m) (hd : 2 ≤ d)
    {n : ℕ} (hn : 2 * d + 4 ≤ n) :
    ((∑ t ∈ PRPos d m hm,
        (fiber hm (by omega : 2 ≤ n) hd t).card : ℕ) : ℚ) =
      ∑ C ∈ SPos d m hm, (multPos d m hm C : ℚ) * (C.count n : ℚ) := by
  rw [PRPos_eq_biUnion_preimage d m hm,
      Finset.sum_biUnion (preimage_disjoint d m hm)]
  push_cast
  refine Finset.sum_congr rfl (fun C _hC => ?_)
  have h_inner : ∀ t ∈ (PRPos d m hm).filter (fun t => t.toCT = C),
      ((fiber hm (by omega : 2 ≤ n) hd t).card : ℚ) = (C.count n : ℚ) := by
    intro t htMem
    rw [Finset.mem_filter] at htMem
    obtain ⟨htPos, htEq⟩ := htMem
    rw [mem_PRPos_iff] at htPos
    rw [fiber_card_eq_count_toCT hm hd t htPos hn, htEq]
  rw [Finset.sum_congr rfl h_inner]
  rw [Finset.sum_const]
  unfold multPos
  push_cast
  ring

/-- **`Ed_decomposition_of_ge_two` as a theorem** (paper `lem:uniform`
`§8.5` decomposition, closed modulo `fiber_card_axiom` +
`fiber_card_zero_axiom`).

Same shape as the existing axiom in `SequelUniformOnsetProof`, but proved
from the fine-grained fiber-cardinality axioms + partition + disjointness. -/
theorem Ed_decomposition_of_ge_two_thm (d : ℕ) (hd : 2 ≤ d) :
    ∀ m : ℕ, 2 ≤ m →
      ∃ (S : Finset ContractionType) (mult : ContractionType → ℕ),
        (∀ C ∈ S, C.runCount ≤ d + 1) ∧
        (∀ C ∈ S, C.width ≤ 2 * d + 3) ∧
        (∀ n : ℕ, 2 * d + 4 ≤ n →
          (Ed d m n : ℚ) = ∑ C ∈ S, (mult C : ℚ) * (C.count n : ℚ)) := by
  intro m hm
  have hm1 : 1 ≤ m := by omega
  refine ⟨SPos d m hm1, multPos d m hm1, ?_, ?_, ?_⟩
  · exact fun C hC => SPos_runCount_le d m hm1 hm C hC
  · exact fun C hC => SPos_width_le d m hm1 C hC
  · intro n hn
    rw [Ed_eq_sum_fiber_card_at_large_n hm1 hd hn]
    push_cast
    rw [PRFinset_split d m hm1,
        Finset.sum_union (PRPos_disjoint_PRZero d m hm1)]
    push_cast
    have hzero : ∑ t ∈ PRZero d m hm1,
        ((fiber hm1 (by omega : 2 ≤ n) hd t).card : ℚ) = 0 := by
      refine Finset.sum_eq_zero (fun t ht => ?_)
      rw [mem_PRZero_iff] at ht
      have := fiber_card_zero_axiom hm1 hd t ht hn
      exact_mod_cast this
    rw [hzero, add_zero]
    have := sum_PRPos_fiber_card_eq hm1 hd hn (d := d) (m := m)
    push_cast at this
    exact this

end OrigamiCone.Sequel

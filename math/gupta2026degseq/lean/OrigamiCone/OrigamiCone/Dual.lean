import OrigamiCone.DegreeExtrema

/-!
# The colour-inversion automorphism `h ↦ -h`

The paper's degree-sequence proofs use the involution `h ↦ -h` (colour inversion,
"swapping colours 1 and 2 and fixing 0") as an automorphism of the origami flip
graph that exchanges maxima and minima.  This module formalises that involution
in the height-function model:

* `isHeight_neg`        — `-h` is a height function when `h` is;
* `strictMax_neg_iff` / `strictMin_neg_iff` — `-h` turns maxima into minima and
  vice versa;
* `neighbors_neg`       — `h ↦ -h` is a degree-preserving bijection on the flip
  graph, so `deg(-h) = deg(h)`.

These are reused by the degree-3 and degree-4 characterizations (the (2,1)↔(1,2)
and unique-max/unique-min dualities).  No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- `-h` is a height function when `h` is. -/
lemma isHeight_neg {h : Cell m n → ℤ} (hh : IsHeight h) :
    IsHeight (fun v => -h v) := by
  intro p q hpq
  have hpq' := hh p q hpq
  rw [show (-h p) - (-h q) = -(h p - h q) by ring, abs_neg]
  exact hpq'

/-- A strict local maximum of `-h` is exactly a strict local minimum of `h`. -/
lemma strictMax_neg_iff {h : Cell m n → ℤ} {v : Cell m n} :
    IsStrictLocalMax (fun w => -h w) v ↔ IsStrictLocalMin h v := by
  constructor
  · intro hmax u hu
    have := hmax u hu
    simp only at this
    linarith
  · intro hmin u hu
    have := hmin u hu
    simp only
    linarith

/-- A strict local minimum of `-h` is exactly a strict local maximum of `h`. -/
lemma strictMin_neg_iff {h : Cell m n → ℤ} {v : Cell m n} :
    IsStrictLocalMin (fun w => -h w) v ↔ IsStrictLocalMax h v := by
  constructor
  · intro hmin u hu
    have := hmin u hu
    simp only at this
    linarith
  · intro hmax u hu
    have := hmax u hu
    simp only
    linarith

/-- The flip-graph adjacency is preserved by colour inversion: `h'` is a neighbour
of `h` iff `-h'` is a neighbour of `-h`. -/
lemma ofgAdj_neg_iff {h h' : Cell m n → ℤ} :
    OFGAdj (fun v => -h v) h' ↔ OFGAdj h (fun v => -h' v) := by
  unfold OFGAdj
  constructor
  · rintro ⟨hheight, v, hagree, hne⟩
    refine ⟨isHeight_neg hheight, v, ?_, ?_⟩
    · intro w hw; have := hagree w hw; simp only at this ⊢; omega
    · simp only at hne ⊢; omega
  · rintro ⟨hheight, v, hagree, hne⟩
    refine ⟨?_, v, ?_, ?_⟩
    · have := isHeight_neg hheight; simpa using this
    · intro w hw; have := hagree w hw; simp only at this ⊢; omega
    · simp only at hne ⊢; omega

/-- **Colour inversion preserves degree.** `h ↦ -h` is a bijection of the flip
graph, so `deg(-h) = deg(h)`. -/
lemma neighbors_neg (h : Cell m n → ℤ) :
    (neighbors (fun v => -h v)).ncard = (neighbors h).ncard := by
  have himg : neighbors (fun v => -h v) = (fun f => (fun v => -f v)) '' neighbors h := by
    ext h'
    simp only [neighbors, Set.mem_setOf_eq, Set.mem_image]
    constructor
    · intro hAdj
      refine ⟨fun v => -h' v, ?_, ?_⟩
      · exact (ofgAdj_neg_iff).mp hAdj
      · funext v; simp
    · rintro ⟨f, hf, rfl⟩
      exact (ofgAdj_neg_iff (h' := fun v => -f v)).mpr (by simpa using hf)
  rw [himg, Set.ncard_image_of_injective]
  intro a b hab
  funext v
  have := congrFun hab v
  simpa using this

end OrigamiCone

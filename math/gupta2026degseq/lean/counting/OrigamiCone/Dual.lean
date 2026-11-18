import OrigamiCone.DegreeExtrema

namespace OrigamiCone

variable {m n : ℕ}

lemma isHeight_neg {h : Cell m n → ℤ} (hh : IsHeight h) :
    IsHeight (fun v => -h v) := by
  intro p q hpq
  have hpq' := hh p q hpq
  rw [show (-h p) - (-h q) = -(h p - h q) by ring, abs_neg]
  exact hpq'

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

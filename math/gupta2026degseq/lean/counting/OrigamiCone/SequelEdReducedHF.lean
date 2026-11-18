import OrigamiCone.SequelEdColumnPartition

namespace OrigamiCone.Sequel

open OrigamiCone

structure ReducedHF (m : ℕ) where

  W : ℕ

  hW : 2 ≤ W

  h : Cell m W → ℤ

  isHeight : IsHeight h

  isCanonical : ∀ p : Cell m W, p.1.val = 0 → p.2.val = 0 → h p = 0

  reduced : ∀ j : Fin W, ∀ (hj1 : j.val + 1 < W),
    ¬ (frozenColumn h j ∧ frozenColumn h ⟨j.val + 1, hj1⟩)

namespace ReducedHF

variable {m : ℕ}

theorem isCanonicalHeight (r : ReducedHF m) : IsCanonicalHeight r.h :=
  ⟨r.isHeight, r.isCanonical⟩

theorem frozenColumn_iff_frozenRunStart (r : ReducedHF m) (hm : 0 < m)
    (j : Fin r.W) :
    frozenColumn r.h j ↔ frozenRunStart r.h j := by
  constructor
  ·
    intro hfrz
    have hfrz' := hfrz
    obtain ⟨hj0, hj1, _⟩ := hfrz'
    refine ⟨?_, ?_⟩
    · exact frozenColumn_not_active r.h j hfrz
    · right
      refine ⟨hj0, ?_⟩
      by_cases hb : j.val - 1 = 0
      ·
        have heq : (⟨j.val - 1, by omega⟩ : Fin r.W) = ⟨0, by omega⟩ :=
          Fin.ext (by simp [hb])
        rw [heq]
        exact firstColumn_active r.h r.isHeight hm r.hW
      ·
        by_contra hna
        have hj1' : (j.val - 1) + 1 < r.W := by omega
        have hfrz_prev : frozenColumn r.h ⟨j.val - 1, by omega⟩ := by
          rcases column_active_or_frozen r.h r.isHeight hm r.hW
              ⟨j.val - 1, by omega⟩ with hact | hfrz
          · exact absurd hact hna
          · exact hfrz
        apply r.reduced ⟨j.val - 1, by omega⟩ hj1'
        refine ⟨hfrz_prev, ?_⟩
        have heq : (⟨(j.val - 1) + 1, hj1'⟩ : Fin r.W) = j :=
          Fin.ext (by simp; omega)
        rw [heq]; exact hfrz
  ·
    intro ⟨hna, hstart⟩
    rcases hstart with h0 | ⟨hj0, _⟩
    · exfalso
      apply hna
      have heq : j = (⟨0, by omega⟩ : Fin r.W) := Fin.ext (by simp [h0])
      rw [heq]
      exact firstColumn_active r.h r.isHeight hm r.hW
    ·
      by_cases hlast : j.val + 1 = r.W
      · exfalso
        apply hna
        have heq : j = (⟨r.W - 1, by omega⟩ : Fin r.W) :=
          Fin.ext (by simp; omega)
        rw [heq]
        exact lastColumn_active r.h r.isHeight hm r.hW
      · have hj1 : j.val + 1 < r.W := by have := j.isLt; omega
        exact (frozenColumn_iff_inactive r.h r.isHeight j hj0 hj1 hm).mpr hna

theorem numFrozenColumns_eq_numFrozenRuns (r : ReducedHF m) (hm : 0 < m) :
    numFrozenColumns r.h = numFrozenRuns r.h := by
  unfold numFrozenColumns numFrozenRuns
  congr 1
  apply Finset.filter_congr
  intro j _
  exact r.frozenColumn_iff_frozenRunStart hm j

theorem width_le_two_d_sub_one (r : ReducedHF m) (hm : 0 < m)
    (d : ℕ) (hd : numExtrema r.h = d) :
    r.W ≤ 2 * d - 1 := by
  have h_partition := numActive_add_numFrozen_eq_n r.h r.isHeight hm r.hW
  have h_frozen_runs := r.numFrozenColumns_eq_numFrozenRuns hm
  have h_runs_bound := numFrozenRuns_le_of_isHeight r.h r.isHeight hm r.hW
  have h_active_bound := numActiveColumns_le_of_numExtrema_eq r.h d hd
  omega

end ReducedHF

end OrigamiCone.Sequel

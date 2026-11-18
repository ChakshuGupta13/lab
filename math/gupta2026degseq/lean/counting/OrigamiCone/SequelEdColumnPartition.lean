import OrigamiCone.SequelEdFrozenBridge
import OrigamiCone.SequelEdBoundary

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m n : ℕ}

instance decFrozenColumn {h : Cell m n → ℤ} : DecidablePred (frozenColumn h) := by
  intro j; unfold frozenColumn; infer_instance

def numFrozenColumns (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (frozenColumn h)).card

theorem boundary_not_frozenColumn (h : Cell m n → ℤ)
    (j : Fin n) (hb : j.val = 0 ∨ j.val + 1 = n) :
    ¬ frozenColumn h j := by
  rintro ⟨hj0, hj1, _⟩
  rcases hb with h1 | h1 <;> omega

theorem active_disjoint_frozen (h : Cell m n → ℤ) (j : Fin n)
    (ha : activeColumn h j) (hf : frozenColumn h j) : False :=
  frozenColumn_not_active h j hf ha

theorem column_active_or_frozen (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) (j : Fin n) :
    activeColumn h j ∨ frozenColumn h j := by
  by_cases hb : j.val = 0
  · left
    have heq : j = (⟨0, by omega⟩ : Fin n) := Fin.ext (by simp [hb])
    rw [heq]; exact firstColumn_active h hh hm hn
  · by_cases hb' : j.val + 1 = n
    · left
      have heq : j = (⟨n - 1, by omega⟩ : Fin n) :=
        Fin.ext (by simp [Fin.val_mk]; omega)
      rw [heq]; exact lastColumn_active h hh hm hn
    ·
      by_cases ha : activeColumn h j
      · exact Or.inl ha
      · exact Or.inr ((frozenColumn_iff_inactive h hh j
          (by omega) (by omega) hm).mpr ha)

theorem numActive_add_numFrozen_eq_n (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    numActiveColumns h + numFrozenColumns h = n := by
  unfold numActiveColumns numFrozenColumns
  rw [← Finset.card_union_of_disjoint]
  · have huniv : (Finset.univ.filter (activeColumn h))
        ∪ (Finset.univ.filter (frozenColumn h)) = Finset.univ := by
      ext j
      simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
      exact ⟨fun _ => trivial, fun _ => column_active_or_frozen h hh hm hn j⟩
    rw [huniv, Finset.card_univ, Fintype.card_fin]
  · rw [Finset.disjoint_filter]
    intro j _ hactive hfrozen
    exact active_disjoint_frozen h j hactive hfrozen

theorem numFrozenColumns_eq_sub_numActive (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    numFrozenColumns h = n - numActiveColumns h := by
  have := numActive_add_numFrozen_eq_n h hh hm hn
  omega

theorem numActiveColumns_le_of_numExtrema_eq (h : Cell m n → ℤ) (d : ℕ)
    (hd : numExtrema h = d) : numActiveColumns h ≤ d := by
  rw [← hd]; exact numActiveColumns_le_numExtrema h

theorem numFrozenColumns_ge_sub_numExtrema (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) (d : ℕ) (hd : numExtrema h = d) :
    n - d ≤ numFrozenColumns h := by
  rw [numFrozenColumns_eq_sub_numActive h hh hm hn]
  have := numActiveColumns_le_of_numExtrema_eq h d hd
  omega

theorem numFrozenRuns_le_of_isHeight (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    numFrozenRuns h + 1 ≤ numActiveColumns h := by
  apply numFrozenRuns_le h (by omega)
  · exact firstColumn_active h hh hm hn
  · exact lastColumn_active h hh hm hn

theorem numFrozenRuns_lt_of_numExtrema_eq (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) (d : ℕ) (hd : numExtrema h = d) :
    numFrozenRuns h + 1 ≤ d := by
  have h1 := numFrozenRuns_le_of_isHeight h hh hm hn
  have h2 := numActiveColumns_le_of_numExtrema_eq h d hd
  omega

end OrigamiCone.Sequel

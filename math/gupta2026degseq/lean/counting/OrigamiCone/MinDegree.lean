import OrigamiCone.DegreeExtrema

namespace OrigamiCone

variable {m n : ℕ}

lemma exists_strictLocalMax [Nonempty (Cell m n)] {h : Cell m n → ℤ}
    (hh : IsHeight h) : ∃ w, IsStrictLocalMax h w := by
  obtain ⟨w, hw⟩ := Finite.exists_max h
  refine ⟨w, fun u hu => ?_⟩
  have hle := hw u
  have h1 := hh w u hu
  rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega

lemma exists_strictLocalMin [Nonempty (Cell m n)] {h : Cell m n → ℤ}
    (hh : IsHeight h) : ∃ w, IsStrictLocalMin h w := by
  obtain ⟨w, hw⟩ := Finite.exists_min h
  refine ⟨w, fun u hu => ?_⟩
  have hge := hw u
  have h1 := hh w u hu
  rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega

lemma two_le_extrema {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 2 ≤ m * n) :
    2 ≤ (Finset.univ.filter (IsStrictLocalExtremum h)).card := by
  have hm : 0 < m := Nat.pos_of_ne_zero (by rintro rfl; simp at hmn)
  have hn : 0 < n := Nat.pos_of_ne_zero (by rintro rfl; simp at hmn)
  haveI : Nonempty (Cell m n) := ⟨(⟨0, hm⟩, ⟨0, hn⟩)⟩
  obtain ⟨wmax, hmax⟩ := exists_strictLocalMax hh
  obtain ⟨wmin, hmin⟩ := exists_strictLocalMin hh
  have hne : wmax ≠ wmin := by
    rintro rfl
    exact max_min_excl hmn hmax hmin
  have hmax_mem : wmax ∈ Finset.univ.filter (IsStrictLocalExtremum h) :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, Or.inl hmax⟩
  have hmin_mem : wmin ∈ Finset.univ.filter (IsStrictLocalExtremum h) :=
    Finset.mem_filter.mpr ⟨Finset.mem_univ _, Or.inr hmin⟩
  have h1lt : 1 < (Finset.univ.filter (IsStrictLocalExtremum h)).card :=
    Finset.one_lt_card.mpr ⟨wmax, hmax_mem, wmin, hmin_mem, hne⟩
  omega

theorem min_degree_ge_two {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 2 ≤ m * n) :
    2 ≤ (neighbors h).ncard := by
  rw [degree_eq_extrema hh hmn]
  exact two_le_extrema hh hmn

end OrigamiCone

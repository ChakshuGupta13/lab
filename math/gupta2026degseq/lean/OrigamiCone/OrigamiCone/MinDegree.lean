import OrigamiCone.DegreeExtrema

/-!
# Minimum degree (Lemma 3.1, lower bound)

Formalisation of the lower-bound half of **Lemma 3.1** of the paper:

> For `mn ≥ 3`, every vertex of `OFG(M_{m,n})` has degree at least 2 (in
> particular none of degree 0 or 1); and degree 2 occurs.

The reusable structural content is the **lower bound**: a height function is
non-constant on a grid with `≥ 2` cells, so its global maximum is a strict local
maximum and its global minimum is a strict local minimum, and these are distinct
— giving at least two strict local extrema, hence degree at least 2 by the
degree–extrema correspondence (`degree_eq_extrema`).

The *attainment* half ("degree 2 occurs") is, in the paper, a consequence of the
degree-2 count (Theorem 3.2), itself a corollary of the cone classification: a
cone at a corner has exactly two extrema.  We formalise attainment alongside the
cone classification rather than here, mirroring the paper's own deferral.

**Quotient caveat** (see `DegreeExtrema`): `min_degree_ge_two` is proved at
`mn ≥ 2` for the *unquotiented* height-flip graph, where it holds at every grid
including `M_{1,2}` (degree exactly 2).  This matches the paper's OFG only for
`mn ≥ 3`; at `M_{1,2}` the paper's quotient OFG vertex has degree 1, below this
bound, because the global rotation collapses the two single-edge flips.  The
paper accordingly states Lemma 3.1 for `mn ≥ 3`.  Every consumer of this lemma
works at `m, n ≥ 2` (so `mn ≥ 4 ≥ 3`), inside the agreement regime.

Results:
* `exists_strictLocalMax`, `exists_strictLocalMin` — the global extrema of a
  height function are strict local extrema;
* `two_le_extrema` — at least two strict local extrema, for `mn ≥ 2`;
* `min_degree_ge_two` — degree at least 2.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- The global maximum of a height function is a strict local maximum. -/
lemma exists_strictLocalMax [Nonempty (Cell m n)] {h : Cell m n → ℤ}
    (hh : IsHeight h) : ∃ w, IsStrictLocalMax h w := by
  obtain ⟨w, hw⟩ := Finite.exists_max h
  refine ⟨w, fun u hu => ?_⟩
  have hle := hw u
  have h1 := hh w u hu
  rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega

/-- The global minimum of a height function is a strict local minimum. -/
lemma exists_strictLocalMin [Nonempty (Cell m n)] {h : Cell m n → ℤ}
    (hh : IsHeight h) : ∃ w, IsStrictLocalMin h w := by
  obtain ⟨w, hw⟩ := Finite.exists_min h
  refine ⟨w, fun u hu => ?_⟩
  have hge := hw u
  have h1 := hh w u hu
  rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1 with h2 | h2 <;> omega

/-- **For `mn ≥ 2`, every height function has at least two strict local
extrema.** A strict local maximum and a strict local minimum exist, and they are
distinct (a vertex with a neighbour cannot be both, `max_min_excl`). -/
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

/-- **Lemma 3.1 (Minimum degree), lower bound.** For `mn ≥ 2`, every vertex of
the origami flip graph has degree at least 2; in particular there are no vertices
of degree 0 or 1. -/
theorem min_degree_ge_two {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 2 ≤ m * n) :
    2 ≤ (neighbors h).ncard := by
  rw [degree_eq_extrema hh hmn]
  exact two_le_extrema hh hmn

end OrigamiCone

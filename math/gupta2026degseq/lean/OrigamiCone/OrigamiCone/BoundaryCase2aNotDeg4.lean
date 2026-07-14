import OrigamiCone.Degree4
import OrigamiCone.BoundaryCase2aUnified

/-!
# The three-maxima bridge: `lem:boundary` case 2a ⟹ not degree 4

Combines the case 2a corner sub-case (`case2a_anyCorner_exists_three_maxima`)
with `Degree4.degree_four_iff` to give the paper's `lem:boundary`
conclusion (for the corner sub-case): the flip-graph vertex `cpe p_B p_I δ`
has degree ≠ 4 whenever `p_B` is a corner and `p_I` is interior (under
parity + right-side activity).

## Results

* `three_maxima_not_deg4`: general bridge — if `h` has a three-element
  Finset of strict local maxima, then `h`'s flip-graph degree
  (`neighbors h).ncard`) is not 4.
* `case2a_anyCorner_not_deg4`: the paper's case 2a conclusion —
  `(neighbors (cpe p_B p_I δ)).ncard ≠ 4` for `p_B` any corner and
  `p_I` interior under parity + activity.

## What this file gets us

For the flip-graph enumeration in `lem:boundary`, the "N(p_1, p_2) = 0"
statement is:  when p_B is a corner and p_I is interior, no configuration
(p_B, p_I, δ) contributes to the degree-4 count. `case2a_anyCorner_not_
deg4` is precisely that statement in the corner sub-case.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Three-maxima bridge.** A height function with at least three
strict local maxima cannot have flip-graph degree 4 (which would require
exactly two). -/
theorem three_maxima_not_deg4 (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h)
    (h_three : ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s, IsStrictLocalMax h c) :
    (neighbors h).ncard ≠ 4 := by
  intro hdeg4
  obtain ⟨hmax2, _⟩ := (degree_four_iff hm hn hh).mp hdeg4
  obtain ⟨s, hcard, hmem⟩ := h_three
  -- s ⊆ (filter IsStrictLocalMax) : Finset (Cell m n)
  have h_le : s ⊆ Finset.univ.filter (IsStrictLocalMax h) := by
    intro c hc
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ c, hmem c hc⟩
  have h_ge3 : 3 ≤ (Finset.univ.filter (IsStrictLocalMax h)).card := by
    rw [← hcard]
    exact Finset.card_le_card h_le
  omega

/-- **`lem:boundary` case 2a (corner sub-case): the flip-graph degree at
`cpe p_B p_I δ` is not 4.** For any grid corner `p_B` and any interior
`p_I`, under parity + right-side activity, the envelope `cpe p_B p_I δ`
has ≥3 strict local maxima and thus cannot correspond to a degree-4
vertex of the flip graph.

This is the paper's `N(p_1, p_2) = 0` conclusion (for the corner sub-case
of the "one interior, one boundary" scenario).  Case-2b (`p_B` on an
edge) is a separate build; the both-interior case is closed by
`bothInterior_exists_four_maxima` (Boundary, c62790f). -/
theorem case2a_anyCorner_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {p_B p_I : Cell m n}
    (hpB_corner : IsCorner p_B) (h_I : IsInterior p_I) (δ : ℤ)
    (hparity : (δ - gdist p_B p_I) % 2 = 0)
    (hact : δ < gdist p_B p_I) :
    (neighbors (cpe p_B p_I δ)).ncard ≠ 4 := by
  have hh : IsHeight (cpe p_B p_I δ) := cpe_isHeight p_B p_I δ hparity
  have h_three := case2a_anyCorner_exists_three_maxima hm hn hpB_corner h_I δ
                    hparity hact
  exact three_maxima_not_deg4 (by omega) (by omega) hh h_three

end OrigamiCone

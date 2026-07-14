import OrigamiCone.BoundaryCase2aNotDeg4
import OrigamiCone.BoundaryCase2bNotDeg4
import OrigamiCone.Boundary
import OrigamiCone.Degree4

/-!
# `lem:boundary` — final capstone

Records the paper's Boundary Lemma end-to-end at the N=0 (flip-graph
degree ≠ 4) level, decomposed by which apex is boundary:

* **Case 1: both apexes interior**: `bothInterior_not_deg4` (this file).
* **Case 2a: `p_B` corner, `p_I` interior**: `case2a_anyCorner_not_deg4`
  (6105e500).
* **Case 2b: `p_B` non-corner boundary, `p_I` interior**:
  `case2b_anyEdge_not_deg4` (40346f11).

Together, these three theorems cover every configuration in which at
least one apex is interior — exactly the paper's `lem:boundary` scope.

## Results

* `k_maxima_not_deg4`: general bridge — any k-element Finset of strict
  local maxima with k ≥ 3 rules out degree 4.
* `bothInterior_not_deg4`: N=0 for the both-interior case.
* `lem_boundary_one_interior`: unified case-2 (one interior apex + one
  boundary apex): N=0 for both `p_B` corner and `p_B` non-corner
  boundary via `IsCorner ∨ IsBoundaryNonCorner` case-split.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **k-maxima bridge (k ≥ 3).** A height function with a k-element
Finset of strict local maxima cannot have flip-graph degree 4. -/
theorem k_maxima_not_deg4 (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) {k : ℕ} (hk : 3 ≤ k)
    (h_k : ∃ s : Finset (Cell m n), s.card = k ∧
      ∀ c ∈ s, IsStrictLocalMax h c) :
    (neighbors h).ncard ≠ 4 := by
  intro hdeg4
  obtain ⟨hmax2, _⟩ := (degree_four_iff hm hn hh).mp hdeg4
  obtain ⟨s, hcard, hmem⟩ := h_k
  have h_le : s ⊆ Finset.univ.filter (IsStrictLocalMax h) := by
    intro c hc
    exact Finset.mem_filter.mpr ⟨Finset.mem_univ c, hmem c hc⟩
  have h_ge_k : k ≤ (Finset.univ.filter (IsStrictLocalMax h)).card := by
    rw [← hcard]
    exact Finset.card_le_card h_le
  omega

/-- **`lem:boundary` case 1 (both apexes interior): N=0.** -/
theorem bothInterior_not_deg4 (hm : 2 ≤ m) (hn : 2 ≤ n)
    {p_1 p_2 : Cell m n} {δ : ℤ}
    (h1 : IsInterior p_1) (h2 : IsInterior p_2)
    (hparity : (δ - gdist p_1 p_2) % 2 = 0) :
    (neighbors (cpe p_1 p_2 δ)).ncard ≠ 4 := by
  have hh : IsHeight (cpe p_1 p_2 δ) := cpe_isHeight p_1 p_2 δ hparity
  have h_four := bothInterior_exists_four_maxima hm hn h1 h2 (δ := δ)
  exact k_maxima_not_deg4 hm hn hh (by omega : 3 ≤ 4) h_four

/-- **`lem:boundary` case 2 (`p_B` boundary + `p_I` interior): N=0.**
Unified over `p_B` any boundary cell (corner OR non-corner). -/
theorem lem_boundary_one_interior
    (hm : 3 ≤ m) (hn : 3 ≤ n)
    {p_B p_I : Cell m n}
    (h_pB_bnd : IsCorner p_B ∨ IsBoundaryNonCorner p_B)
    (h_I : IsInterior p_I) (δ : ℤ)
    (hparity : (δ - gdist p_B p_I) % 2 = 0)
    (hact : δ < gdist p_B p_I) :
    (neighbors (cpe p_B p_I δ)).ncard ≠ 4 := by
  rcases h_pB_bnd with h_corner | h_edge
  · exact case2a_anyCorner_not_deg4 hm hn h_corner h_I δ hparity hact
  · exact case2b_anyEdge_not_deg4 hm hn h_edge h_I δ hparity hact

end OrigamiCone

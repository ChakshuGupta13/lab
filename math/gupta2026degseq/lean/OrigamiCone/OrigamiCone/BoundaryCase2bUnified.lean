import OrigamiCone.BoundaryOneInteriorEdge
import OrigamiCone.BoundaryOneInteriorBotEdge
import OrigamiCone.BoundaryOneInteriorSideEdges
import OrigamiCone.Degree3

/-!
# `lem:boundary` case 2b: unified any-edge two-far-corner existence

Packages the four edge-specific two-far-corner theorems (top / bottom /
left / right) into a single statement parametrised by
`IsBoundaryNonCorner p_B`. Case-splits on which edge `p_B` lies on and
dispatches to the corresponding edge-specific theorem.

## Result

* `case2b_anyEdge_two_farCorners`: for `p_B` any non-corner boundary
  cell (`IsBoundaryNonCorner p_B`) and `p_I` interior, `cpe p_B p_I δ`
  admits at least two distinct strict local maxima at "far" corners of
  the grid (the two corners not incident to `p_B`'s side).

## Complement

Combined with `case2a_anyCorner_exists_three_maxima` (corner case, gives
≥ 3 maxima) and `bothInterior_exists_four_maxima` (gives ≥ 4 maxima),
this file rounds out `lem:boundary`'s existence-of-maxima structure at
the two-maxima level for the edge sub-case. The paper's third-max
via dual-envelope contradiction remains as an independent substrate
build.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`lem:boundary` case 2b, unified two-far-corner existence.** For any
non-corner boundary cell `p_B` and any interior `p_I`, `cpe p_B p_I δ`
has at least two distinct strict local maxima (the "far corners" on the
side opposite `p_B`). Dispatches to the four edge-specific theorems
`oneInterior_TopEdge_/_BotEdge_/_LeftEdge_/_RightEdge_two_*_max`. -/
theorem case2b_anyEdge_two_farCorners
    {p_B : Cell m n} (h_pB : IsBoundaryNonCorner p_B)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    ∃ v₁ v₂ : Cell m n, v₁ ≠ v₂ ∧
      IsStrictLocalMax (cpe p_B p_I δ) v₁ ∧
      IsStrictLocalMax (cpe p_B p_I δ) v₂ := by
  -- Interior p_I forces m ≥ 3 and n ≥ 3.
  have hm : 3 ≤ m := by
    have := h_I.1; have := h_I.2.1; omega
  have hn : 3 ≤ n := by
    have := h_I.2.2.1; have := h_I.2.2.2; omega
  -- Case-split IsBoundaryNonCorner into 4 physical edges.
  rcases h_pB with ⟨hRow_ep, hCol_notEp⟩ | ⟨hRow_notEp, hCol_ep⟩
  · -- p_B.1 endpoint, p_B.2 non-endpoint: top or bottom edge.
    have hcol_pos : 1 ≤ p_B.2.val := by
      unfold IsEndpoint at hCol_notEp; push_neg at hCol_notEp
      have := hCol_notEp.1; omega
    have hcol_lt : p_B.2.val + 1 < n := by
      unfold IsEndpoint at hCol_notEp; push_neg at hCol_notEp
      have h1 := hCol_notEp.2
      have h2 : p_B.2.val < n := p_B.2.isLt
      omega
    rcases hRow_ep with hrow0 | hrowm
    · -- Top edge: p_B = (0, cB).
      have hpB_eq : p_B =
          ((⟨0, by omega⟩ : Fin m), (⟨p_B.2.val, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext hrow0
        · exact Fin.ext rfl
      rw [hpB_eq]
      obtain ⟨h_TL, h_TR, hne⟩ :=
        oneInterior_TopEdge_two_bottom_max (m := m) (cB := p_B.2.val)
          (by omega) hcol_pos hcol_lt h_I δ
      exact ⟨((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)),
             ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
             hne, h_TL, h_TR⟩
    · -- Bottom edge: p_B = (m-1, cB).
      have hpB_eq : p_B =
          ((⟨m - 1, by omega⟩ : Fin m), (⟨p_B.2.val, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · apply Fin.ext
          have := p_B.1.isLt; dsimp only; omega
        · exact Fin.ext rfl
      rw [hpB_eq]
      obtain ⟨h_TL, h_TR, hne⟩ :=
        oneInterior_BotEdge_two_top_max (m := m) (n := n) (cB := p_B.2.val)
          (by omega) hcol_pos hcol_lt h_I δ
      exact ⟨((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)),
             ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
             hne, h_TL, h_TR⟩
  · -- p_B.1 non-endpoint, p_B.2 endpoint: left or right edge.
    have hrow_pos : 1 ≤ p_B.1.val := by
      unfold IsEndpoint at hRow_notEp; push_neg at hRow_notEp
      have := hRow_notEp.1; omega
    have hrow_lt : p_B.1.val + 1 < m := by
      unfold IsEndpoint at hRow_notEp; push_neg at hRow_notEp
      have h1 := hRow_notEp.2
      have h2 : p_B.1.val < m := p_B.1.isLt
      omega
    rcases hCol_ep with hcol0 | hcoln
    · -- Left edge: p_B = (rB, 0).
      have hpB_eq : p_B =
          ((⟨p_B.1.val, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext rfl
        · exact Fin.ext hcol0
      rw [hpB_eq]
      obtain ⟨h_TR, h_BR, hne⟩ :=
        oneInterior_LeftEdge_two_right_max (m := m) (n := n) (rB := p_B.1.val)
          (by omega) hrow_pos hrow_lt h_I δ
      exact ⟨((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
             ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
             hne, h_TR, h_BR⟩
    · -- Right edge: p_B = (rB, n-1).
      have hpB_eq : p_B =
          ((⟨p_B.1.val, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext rfl
        · apply Fin.ext
          have := p_B.2.isLt; dsimp only; omega
      rw [hpB_eq]
      obtain ⟨h_TL, h_BL, hne⟩ :=
        oneInterior_RightEdge_two_left_max (m := m) (n := n) (rB := p_B.1.val)
          (by omega) hrow_pos hrow_lt h_I δ
      exact ⟨((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)),
             ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)),
             hne, h_TL, h_BL⟩

/-- **`lem:boundary` case 2b, Finset-form.** Uniform-shape counterpart to
`case2a_anyCorner_exists_three_maxima` (Boundary.lean's
`bothInterior_exists_four_maxima` shape). For `p_B` a non-corner
boundary cell and `p_I` interior, `cpe p_B p_I δ` has a two-element
Finset of strict local maxima. Ready-to-use in enumeration arguments
alongside the corner (3-maxima) and both-interior (4-maxima) forms. -/
theorem case2b_anyEdge_exists_two_maxima
    {p_B : Cell m n} (h_pB : IsBoundaryNonCorner p_B)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    ∃ s : Finset (Cell m n), s.card = 2 ∧
      ∀ c ∈ s, IsStrictLocalMax (cpe p_B p_I δ) c := by
  obtain ⟨v₁, v₂, hne, hv₁, hv₂⟩ :=
    case2b_anyEdge_two_farCorners h_pB h_I δ
  refine ⟨{v₁, v₂}, ?_, ?_⟩
  · rw [Finset.card_insert_of_notMem (by simp [hne])]
    simp
  · intro c hc
    simp only [Finset.mem_insert, Finset.mem_singleton] at hc
    rcases hc with rfl | rfl
    · exact hv₁
    · exact hv₂

end OrigamiCone

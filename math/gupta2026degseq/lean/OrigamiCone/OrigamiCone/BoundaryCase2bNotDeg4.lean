import OrigamiCone.BoundaryCase2bTopEdgeUnified
import OrigamiCone.BoundaryCase2bBotEdgeNotDeg4
import OrigamiCone.BoundaryCase2bLeftEdgeUnified
import OrigamiCone.BoundaryCase2bRightEdgeNotDeg4
import OrigamiCone.Degree3

/-!
# `lem:boundary` case 2b unified: N=0 for any boundary-non-corner `p_B`

Packages the four edge-specific N=0 theorems (top / bottom / left / right)
into a single any-edge N=0 statement parametrised by `IsBoundaryNonCorner
p_B`. Analog of `case2a_anyCorner_not_deg4` for the edge sub-case.

## Result

* `case2b_anyEdge_not_deg4`: for any `p_B : IsBoundaryNonCorner` and
  `p_I` interior, parity + right-active, `(neighbors cpe).ncard ≠ 4`.

Combined with `case2a_anyCorner_not_deg4` (6105e500) and
`bothInterior_exists_four_maxima` (c62790f), `lem:boundary` is now
formalised for ALL configurations at the N=0 level.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`lem:boundary` case 2b any-edge N=0.** For any non-corner boundary
cell `p_B` (`IsBoundaryNonCorner p_B`), any interior `p_I`, parity +
right-active, `cpe p_B p_I δ` has flip-graph degree ≠ 4. -/
theorem case2b_anyEdge_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {p_B p_I : Cell m n}
    (h_pB_edge : IsBoundaryNonCorner p_B) (h_I : IsInterior p_I) (δ : ℤ)
    (hparity : (δ - gdist p_B p_I) % 2 = 0)
    (hact : δ < gdist p_B p_I) :
    (neighbors (cpe p_B p_I δ)).ncard ≠ 4 := by
  rcases h_pB_edge with ⟨hRow_ep, hCol_notEp⟩ | ⟨hRow_notEp, hCol_ep⟩
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
    · -- Top edge.
      have hpB_eq : p_B =
          ((⟨0, by omega⟩ : Fin m), (⟨p_B.2.val, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext hrow0
        · exact Fin.ext rfl
      rw [hpB_eq] at hparity hact ⊢
      exact case2b_TopEdge_not_deg4 hm hn hcol_pos hcol_lt h_I δ hparity hact
    · -- Bottom edge.
      have hpB_eq : p_B =
          ((⟨m - 1, by omega⟩ : Fin m), (⟨p_B.2.val, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · apply Fin.ext
          have := p_B.1.isLt; dsimp only; omega
        · exact Fin.ext rfl
      rw [hpB_eq] at hparity hact ⊢
      exact case2b_BotEdge_not_deg4 hm hn hcol_pos hcol_lt h_I δ hparity hact
  · -- p_B.2 endpoint, p_B.1 non-endpoint: left or right edge.
    have hrow_pos : 1 ≤ p_B.1.val := by
      unfold IsEndpoint at hRow_notEp; push_neg at hRow_notEp
      have := hRow_notEp.1; omega
    have hrow_lt : p_B.1.val + 1 < m := by
      unfold IsEndpoint at hRow_notEp; push_neg at hRow_notEp
      have h1 := hRow_notEp.2
      have h2 : p_B.1.val < m := p_B.1.isLt
      omega
    rcases hCol_ep with hcol0 | hcoln
    · -- Left edge.
      have hpB_eq : p_B =
          ((⟨p_B.1.val, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext rfl
        · exact Fin.ext hcol0
      rw [hpB_eq] at hparity hact ⊢
      exact case2b_LeftEdge_not_deg4 hm hn hrow_pos hrow_lt h_I δ hparity hact
    · -- Right edge.
      have hpB_eq : p_B =
          ((⟨p_B.1.val, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext rfl
        · apply Fin.ext
          have := p_B.2.isLt; dsimp only; omega
      rw [hpB_eq] at hparity hact ⊢
      exact case2b_RightEdge_not_deg4 hm hn hrow_pos hrow_lt h_I δ hparity hact

end OrigamiCone

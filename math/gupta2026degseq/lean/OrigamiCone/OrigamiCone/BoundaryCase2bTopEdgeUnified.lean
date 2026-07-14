import OrigamiCone.BoundaryCase2bTopEdgeRNotDeg4
import OrigamiCone.BoundaryCase2bTopEdgeLNotDeg4

/-!
# Case 2b top-edge: unified N=0 (any `s vs cB`)

Combines the two sub-cases R (`s ≥ cB`) and L (`s ≤ cB`) into a single
top-edge case-2b N=0 theorem. Since ℕ satisfies `s ≥ cB ∨ s ≤ cB` for
every pair (by trichotomy), the two sub-cases cover the full top-edge
case 2b.

## Result

* `case2b_TopEdge_not_deg4`: for `p_B = (0, cB)` on the top edge, any
  `p_I` interior, parity + right-active, `cpe p_B p_I δ` has
  flip-graph degree ≠ 4 (paper's N=0 conclusion for top-edge).

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b top-edge: N=0 (any `s vs cB`).** For `p_B = (0, cB)` on
the top edge (with `1 ≤ cB ≤ n - 2`) and `p_I` interior, `cpe p_B p_I δ`
has flip-graph degree ≠ 4 under parity + right-active. -/
theorem case2b_TopEdge_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  -- Case-split ℕ trichotomy on p_I.2.val vs cB.
  rcases (by omega : cB ≤ p_I.2.val ∨ p_I.2.val < cB) with h_s_ge_cB | h_s_lt_cB
  · exact case2b_TopEdge_R_not_deg4 hm hn hcB_pos hcB_lt h_I δ h_s_ge_cB
      hparity hact
  · exact case2b_TopEdge_L_not_deg4 hm hn hcB_pos hcB_lt h_I δ
      (Nat.le_of_lt h_s_lt_cB) hparity hact

end OrigamiCone

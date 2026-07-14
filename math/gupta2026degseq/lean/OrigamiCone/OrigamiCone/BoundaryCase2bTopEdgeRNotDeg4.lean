import OrigamiCone.BoundaryOneInteriorEdge
import OrigamiCone.BoundaryCase2bTopEdgeRAssembly
import OrigamiCone.BoundaryCase2aNotDeg4

/-!
# Case 2b sub-case `s ≥ cB`: three-maxima Finset + N=0 bridge

Composes the two-bottom-corner maxima (from `BoundaryOneInteriorEdge`)
with the third max at `(i₀, n - 1)` (from `BoundaryCase2bTopEdgeRAssembly`)
into a Finset of size 3 for the top-edge sub-case `s ≥ cB`. Then applies
`three_maxima_not_deg4` to reach the paper's `N(p_1, p_2) = 0`
conclusion for this sub-case.

## Results

* `case2b_TopEdge_R_exists_three_maxima`: Finset-form three-maxima
  existence for top-edge `p_B` with `s ≥ cB`.
* `case2b_TopEdge_R_not_deg4`: flip-graph degree ≠ 4 (paper's N=0
  conclusion) for the sub-case.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b sub-case `s ≥ cB`: three-element Finset of strict local
maxima.** Under top-edge `p_B = (0, cB)`, p_I interior, s ≥ cB, parity,
right-active, `cpe p_B p_I δ` has a three-element Finset of strict
local maxima. -/
theorem case2b_TopEdge_R_exists_three_maxima
    (hm : 2 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_ge_cB : cB ≤ p_I.2.val)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s,
        IsStrictLocalMax
          (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
               p_I δ) c := by
  obtain ⟨h_BL, h_BR, hne_bl_br⟩ :=
    oneInterior_TopEdge_two_bottom_max (m := m) (cB := cB) hm hcB_pos hcB_lt h_I δ
  obtain ⟨i₀, hi₀_lt, h_col_max⟩ :=
    case2b_TopEdge_R_third_max_grid hm hn hcB_pos hcB_lt h_I δ h_s_ge_cB
      hparity hact
  -- Three cells: (m-1, 0), (m-1, n-1), (i₀, n-1). Pairwise distinct:
  --   BL vs BR: col differs (0 vs n-1).
  --   BL vs (i₀, n-1): col differs.
  --   BR vs (i₀, n-1): row differs (m-1 vs i₀ < p_I.1.val < m-1).
  have hi₀_lt_mm1 : i₀.val < m - 1 := by
    have := h_I.2.1  -- p_I.1.val + 1 < m
    omega
  refine ⟨{((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)),
           ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
           (i₀, (⟨n - 1, by omega⟩ : Fin n))}, ?_, ?_⟩
  · have hne1 : ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ≠
                (((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) :
                  Cell m n) := hne_bl_br
    have hne2 : ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ≠
                ((i₀, (⟨n - 1, by omega⟩ : Fin n)) : Cell m n) := by
      intro heq
      have := congrArg (fun c : Cell m n => c.2.val) heq
      dsimp at this
      omega
    have hne3 : ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ≠
                ((i₀, (⟨n - 1, by omega⟩ : Fin n)) : Cell m n) := by
      intro heq
      have := congrArg (fun c : Cell m n => c.1.val) heq
      dsimp at this
      omega
    rw [Finset.card_insert_of_notMem (by simp [hne1, hne2]),
        Finset.card_insert_of_notMem (by simp [hne3])]
    simp
  · intro c hc
    simp only [Finset.mem_insert, Finset.mem_singleton] at hc
    rcases hc with rfl | rfl | rfl
    · exact h_BL
    · exact h_BR
    · exact h_col_max

/-- **Case 2b sub-case `s ≥ cB`: not degree 4 (paper's N=0).** For
top-edge `p_B` with `s ≥ cB`, `cpe p_B p_I δ` has flip-graph degree
≠ 4. -/
theorem case2b_TopEdge_R_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_ge_cB : cB ≤ p_I.2.val)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  have hh : IsHeight (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                          p_I δ) :=
    cpe_isHeight _ p_I δ hparity
  have h_three := case2b_TopEdge_R_exists_three_maxima (by omega) hn hcB_pos
                    hcB_lt h_I δ h_s_ge_cB hparity hact
  exact three_maxima_not_deg4 (by omega) (by omega) hh h_three

end OrigamiCone

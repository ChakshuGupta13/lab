import OrigamiCone.BoundaryOneInteriorTopEdgeCol
import OrigamiCone.BoundaryOneInteriorColumn
import OrigamiCone.Parity

/-!
# Case 2b's third max — grid-lift for sub-case `s ≥ cB`

Lifts `oneInterior_TopEdge_col_third_max_R` (col-strict-max in column
`n - 1`) to a full-grid `IsStrictLocalMax` via the standard case-split
on adjacency directions + step-inward-lowers primitive.

## Result

* `case2b_TopEdge_R_third_max_grid`: for `p_B = (0, cB)` top edge with
  `s ≥ cB` (p_I weakly right of p_B), parity + right-active, exists
  `i₀ : Fin m` with `i₀.val < p_I.1.val` and grid-`IsStrictLocalMax`
  at `(i₀, n - 1)`.

The proof pattern matches `case2a_TL_col_strictLocalMax` exactly.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b's third max lifted to grid `IsStrictLocalMax`, sub-case
`s ≥ cB`.** Analog of `case2a_TL_col_strictLocalMax`. -/
theorem case2b_TopEdge_R_third_max_grid
    (hm : 2 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_ge_cB : cB ≤ p_I.2.val)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    ∃ i₀ : Fin m, i₀.val < p_I.1.val ∧
      IsStrictLocalMax
        (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
        (i₀, (⟨n - 1, by omega⟩ : Fin n)) := by
  obtain ⟨i₀, hi₀_lt, h_right_col, h_left_col⟩ :=
    oneInterior_TopEdge_col_third_max_R hm hn hcB_pos hcB_lt h_I δ h_s_ge_cB
      hparity hact
  refine ⟨i₀, hi₀_lt, ?_⟩
  intro u hadj
  set v : Cell m n := (i₀, (⟨n - 1, by omega⟩ : Fin n)) with hv
  have hisht := cpe_isHeight ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                             p_I δ hparity
  have habs := hisht v u hadj
  rw [abs_eq (by norm_num : (0:ℤ) ≤ 1)] at habs
  suffices h_lt :
      cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ u <
      cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ v by
    rcases habs with h | h
    · linarith
    · linarith
  have hadj' : (((v.1.val : ℤ) - u.1.val).natAbs +
                ((v.2.val : ℤ) - u.2.val).natAbs : ℤ) = 1 := by
    have := hadj; unfold adj gdist at this; exact_mod_cast this
  have hu1 := u.1.isLt
  have hu2 := u.2.isLt
  by_cases h1 : u.1.val = i₀.val
  · by_cases h2 : u.2.val = n - 2
    · -- u = (i₀, n - 2): use step-inward-lowers for top-edge p_B.
      have hu_eq : u = (i₀, (⟨n - 2, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext h1
        · exact Fin.ext h2
      rw [hu_eq]
      simpa [hv] using oneInterior_TopEdge_lastCol_stepIn_lower hm hn hcB_lt
                        h_I δ i₀
    · exfalso
      have hv2 : v.2.val = n - 1 := by simp [hv]
      have hd0_or_1 : u.2.val = n - 1 ∨ u.2.val = n - 2 ∨ u.2.val + 2 ≤ n - 1 := by
        omega
      rcases hd0_or_1 with h | h | h
      · have : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by simp [hv, h]
        have hrow0 : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by simp [hv, h1]
        omega
      · exact h2 h
      · have : ((v.2.val : ℤ) - u.2.val).natAbs ≥ 2 := by
          simp only [hv2]; omega
        have hrow0 : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by simp [hv, h1]
        omega
  · have hu2_eq_v2 : u.2.val = n - 1 := by
      by_contra hne
      have hv2 : v.2.val = n - 1 := by simp [hv]
      have hcol_ne : ((v.2.val : ℤ) - u.2.val).natAbs ≥ 1 := by
        simp only [hv2]; omega
      have hrow_ne : ((v.1.val : ℤ) - u.1.val).natAbs ≥ 1 := by
        have : v.1.val = i₀.val := by simp [hv]
        omega
      omega
    have hrow_diff : ((v.1.val : ℤ) - u.1.val).natAbs = 1 := by
      have hcol0 : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by
        have hv2 : v.2.val = n - 1 := by simp [hv]
        simp [hv2, hu2_eq_v2]
      omega
    have hi₀val : v.1.val = i₀.val := by simp [hv]
    have hu1_case : u.1.val = i₀.val + 1 ∨ u.1.val + 1 = i₀.val := by
      have : ((i₀.val : ℤ) - u.1.val).natAbs = 1 := by rw [← hi₀val]; exact hrow_diff
      omega
    rcases hu1_case with h | h
    · have h_i0succ : i₀.val + 1 < m := by
        have := u.1.isLt; omega
      have := h_right_col h_i0succ
      have hu_eq : u = (⟨i₀.val + 1, h_i0succ⟩, (⟨n - 1, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext h
        · exact Fin.ext hu2_eq_v2
      rw [hu_eq]
      simpa [hv] using this
    · have hi₀pos : 0 < i₀.val := by omega
      have hi₀_eq : i₀.val - 1 = u.1.val := by omega
      have hi₀m1_lt_m : i₀.val - 1 < m := by omega
      rcases h_left_col with h_z | h_up
      · omega
      · have hu_eq : u = (⟨i₀.val - 1, hi₀m1_lt_m⟩, (⟨n - 1, by omega⟩ : Fin n)) := by
          apply Prod.ext
          · exact Fin.ext hi₀_eq.symm
          · exact Fin.ext hu2_eq_v2
        rw [hu_eq]
        simpa [hv] using h_up

end OrigamiCone

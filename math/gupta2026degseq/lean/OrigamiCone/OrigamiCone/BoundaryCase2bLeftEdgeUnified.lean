import OrigamiCone.BoundaryOneInteriorLeftEdgeRow
import OrigamiCone.BoundaryOneInteriorSideEdges
import OrigamiCone.BoundaryCase2aNotDeg4
import OrigamiCone.RowReflectTransport

/-!
# Case 2b left-edge N=0 — full assembly for both sub-cases

Grid-lift + Finset + N=0 for left edge, both sub-cases (r ≥ rB, r ≤ rB).

## Results

* `case2b_LeftEdge_R_third_max_grid`: grid-lift of the row-third-max
  from `BoundaryOneInteriorLeftEdgeRow` for sub-case R.
* `case2b_LeftEdge_R_exists_three_maxima`: 3-elem Finset for sub-case R.
* `case2b_LeftEdge_R_not_deg4`: N=0 for sub-case R.
* `case2b_LeftEdge_L_exists_three_maxima` and `_L_not_deg4`: sub-case
  L (r ≤ rB) via rowRefl transport.
* `case2b_LeftEdge_not_deg4`: unified left-edge N=0.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b left-edge sub-case R (r ≥ rB): grid IsStrictLocalMax at
`(m - 1, j₀)`.** Analog of `case2b_TopEdge_R_third_max_grid`. -/
theorem case2b_LeftEdge_R_third_max_grid
    (hm : 3 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_r_ge_rB : rB ≤ p_I.1.val)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ j₀ : Fin n, j₀.val < p_I.2.val ∧
      IsStrictLocalMax
        (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
        ((⟨m - 1, by omega⟩ : Fin m), j₀) := by
  obtain ⟨j₀, hj₀_lt, h_right_row, h_left_row⟩ :=
    oneInterior_LeftEdge_row_third_max_R hm hn hrB_pos hrB_lt h_I δ h_r_ge_rB
      hparity hact
  refine ⟨j₀, hj₀_lt, ?_⟩
  intro u hadj
  set v : Cell m n := ((⟨m - 1, by omega⟩ : Fin m), j₀) with hv
  have hisht := cpe_isHeight ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                             p_I δ hparity
  have habs := hisht v u hadj
  rw [abs_eq (by norm_num : (0:ℤ) ≤ 1)] at habs
  suffices h_lt :
      cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ u <
      cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ v by
    rcases habs with h | h
    · linarith
    · linarith
  have hadj' : (((v.1.val : ℤ) - u.1.val).natAbs +
                ((v.2.val : ℤ) - u.2.val).natAbs : ℤ) = 1 := by
    have := hadj; unfold adj gdist at this; exact_mod_cast this
  have hu1 := u.1.isLt
  have hu2 := u.2.isLt
  -- Case: u.2 = j₀ (same column), u.1 = m-2 (inward)
  by_cases h2 : u.2.val = j₀.val
  · by_cases h1 : u.1.val = m - 2
    · have hu_eq : u = ((⟨m - 2, by omega⟩ : Fin m), j₀) := by
        apply Prod.ext
        · exact Fin.ext h1
        · exact Fin.ext h2
      rw [hu_eq]
      simpa [hv] using oneInterior_LeftEdge_lastRow_stepIn_lower hm hn hrB_lt
                        h_I δ j₀
    · exfalso
      have hv1 : v.1.val = m - 1 := by simp [hv]
      have hd0_or_1 : u.1.val = m - 1 ∨ u.1.val = m - 2 ∨ u.1.val + 2 ≤ m - 1 := by
        omega
      rcases hd0_or_1 with h | h | h
      · have : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by simp [hv, h]
        have hcol0 : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by simp [hv, h2]
        omega
      · exact h1 h
      · have : ((v.1.val : ℤ) - u.1.val).natAbs ≥ 2 := by
          simp only [hv1]; omega
        have hcol0 : ((v.2.val : ℤ) - u.2.val).natAbs = 0 := by simp [hv, h2]
        omega
  · -- u.2 ≠ j₀: row stays at m - 1
    have hu1_eq_v1 : u.1.val = m - 1 := by
      by_contra hne
      have hv1 : v.1.val = m - 1 := by simp [hv]
      have hrow_ne : ((v.1.val : ℤ) - u.1.val).natAbs ≥ 1 := by
        simp only [hv1]; omega
      have hcol_ne : ((v.2.val : ℤ) - u.2.val).natAbs ≥ 1 := by
        have : v.2.val = j₀.val := by simp [hv]
        omega
      omega
    have hcol_diff : ((v.2.val : ℤ) - u.2.val).natAbs = 1 := by
      have hrow0 : ((v.1.val : ℤ) - u.1.val).natAbs = 0 := by
        have hv1 : v.1.val = m - 1 := by simp [hv]
        simp [hv1, hu1_eq_v1]
      omega
    have hj₀val : v.2.val = j₀.val := by simp [hv]
    have hu2_case : u.2.val = j₀.val + 1 ∨ u.2.val + 1 = j₀.val := by
      have : ((j₀.val : ℤ) - u.2.val).natAbs = 1 := by rw [← hj₀val]; exact hcol_diff
      omega
    rcases hu2_case with h | h
    · have h_j0succ : j₀.val + 1 < n := by
        have := u.2.isLt; omega
      have := h_right_row h_j0succ
      have hu_eq : u = ((⟨m - 1, by omega⟩ : Fin m), ⟨j₀.val + 1, h_j0succ⟩) := by
        apply Prod.ext
        · exact Fin.ext hu1_eq_v1
        · exact Fin.ext h
      rw [hu_eq]
      simpa [hv] using this
    · have hj₀pos : 0 < j₀.val := by omega
      have hj₀_eq : j₀.val - 1 = u.2.val := by omega
      have hj₀m1_lt_n : j₀.val - 1 < n := by omega
      rcases h_left_row with h_z | h_left
      · omega
      · have hu_eq : u = ((⟨m - 1, by omega⟩ : Fin m), ⟨j₀.val - 1, hj₀m1_lt_n⟩) := by
          apply Prod.ext
          · exact Fin.ext hu1_eq_v1
          · exact Fin.ext hj₀_eq.symm
        rw [hu_eq]
        simpa [hv] using h_left

/-- **Sub-case R (r ≥ rB): 3-element Finset of strict local maxima.** -/
theorem case2b_LeftEdge_R_exists_three_maxima
    (hm : 3 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_r_ge_rB : rB ≤ p_I.1.val)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s,
        IsStrictLocalMax
          (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
               p_I δ) c := by
  obtain ⟨h_TR, h_BR, hne_tr_br⟩ :=
    oneInterior_LeftEdge_two_right_max (m := m) (rB := rB) hn hrB_pos hrB_lt h_I δ
  obtain ⟨j₀, hj₀_lt, h_row_max⟩ :=
    case2b_LeftEdge_R_third_max_grid hm hn hrB_pos hrB_lt h_I δ h_r_ge_rB
      hparity hact
  have hj₀_lt_nm1 : j₀.val < n - 1 := by
    have := h_I.2.2.2  -- p_I.2.val + 1 < n
    omega
  refine ⟨{((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
           ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)),
           ((⟨m - 1, by omega⟩ : Fin m), j₀)}, ?_, ?_⟩
  · have hne1 : ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ≠
                (((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) :
                  Cell m n) := hne_tr_br
    have hne2 : ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ≠
                (((⟨m - 1, by omega⟩ : Fin m), j₀) : Cell m n) := by
      intro heq
      have := congrArg (fun c : Cell m n => c.1.val) heq
      dsimp at this
      omega
    have hne3 : ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ≠
                (((⟨m - 1, by omega⟩ : Fin m), j₀) : Cell m n) := by
      intro heq
      have := congrArg (fun c : Cell m n => c.2.val) heq
      dsimp at this
      omega
    rw [Finset.card_insert_of_notMem (by simp [hne1, hne2]),
        Finset.card_insert_of_notMem (by simp [hne3])]
    simp
  · intro c hc
    simp only [Finset.mem_insert, Finset.mem_singleton] at hc
    rcases hc with rfl | rfl | rfl
    · exact h_TR
    · exact h_BR
    · exact h_row_max

/-- **Sub-case R (r ≥ rB): N=0.** -/
theorem case2b_LeftEdge_R_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_r_ge_rB : rB ≤ p_I.1.val)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  have hh : IsHeight (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                          p_I δ) :=
    cpe_isHeight _ p_I δ hparity
  have h_three := case2b_LeftEdge_R_exists_three_maxima hm (by omega) hrB_pos
                    hrB_lt h_I δ h_r_ge_rB hparity hact
  exact three_maxima_not_deg4 (by omega) (by omega) hh h_three

/-- **Sub-case L (r ≤ rB): 3-element Finset via rowRefl transport.** -/
theorem case2b_LeftEdge_L_exists_three_maxima
    (hm : 3 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_r_le_rB : p_I.1.val ≤ rB)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s,
        IsStrictLocalMax
          (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
               p_I δ) c := by
  set p_I' : Cell m n := rowRefl p_I with hp_I'_def
  have h_I' : IsInterior p_I' := h_I.rowRefl
  set rB' : ℕ := m - 1 - rB with hrB'_def
  have hrB'_pos : 1 ≤ rB' := by show 1 ≤ m - 1 - rB; omega
  have hrB'_lt : rB' + 1 < m := by show m - 1 - rB + 1 < m; omega
  have h_r'_ge_rB' : rB' ≤ p_I'.1.val := by
    have hp_I1_lt : p_I.1.val < m := p_I.1.isLt
    have h_r'_val : p_I'.1.val = m - 1 - p_I.1.val := by
      show (Fin.rev p_I.1).val = _
      rw [Fin.val_rev]; omega
    show m - 1 - rB ≤ p_I'.1.val
    rw [h_r'_val]; omega
  set p_B_orig : Cell m n :=
    ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) with hp_B_orig_def
  set p_B_R : Cell m n :=
    ((⟨rB', by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) with hp_B_R_def
  have h_rowRefl_pB : rowRefl p_B_R = p_B_orig := by
    apply Prod.ext
    · show (Fin.rev (⟨rB', by omega⟩ : Fin m)) = ⟨rB, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
      show m - (rB' + 1) = rB
      omega
    · rfl
  have h_rowRefl_pI : rowRefl p_I' = p_I := rowRefl_involutive p_I
  have h_gdist_eq : gdist p_B_R p_I' = gdist p_B_orig p_I := by
    conv_rhs => rw [← h_rowRefl_pB, ← h_rowRefl_pI]
    exact (rowRefl_gdist p_B_R p_I').symm
  have hparity_R : (δ - gdist p_B_R p_I') % 2 = 0 := by rw [h_gdist_eq]; exact hparity
  have hact_R : δ < gdist p_B_R p_I' := by rw [h_gdist_eq]; exact hact
  obtain ⟨s_R, hcard, hmem⟩ :=
    case2b_LeftEdge_R_exists_three_maxima hm hn hrB'_pos hrB'_lt h_I' δ
      h_r'_ge_rB' hparity_R hact_R
  refine ⟨s_R.image rowRefl, ?_, ?_⟩
  · rw [Finset.card_image_of_injective _ rowRefl.injective]; exact hcard
  · intro c hc
    obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
    have h_R_max := hmem v hv_in
    have h_transp := cpe_strictMax_rowRefl h_R_max
    rw [h_rowRefl_pB, h_rowRefl_pI] at h_transp
    rw [← hv_eq]
    exact h_transp

/-- **Sub-case L (r ≤ rB): N=0.** -/
theorem case2b_LeftEdge_L_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_r_le_rB : p_I.1.val ≤ rB)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  have hh : IsHeight (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                          p_I δ) :=
    cpe_isHeight _ p_I δ hparity
  have h_three := case2b_LeftEdge_L_exists_three_maxima hm (by omega) hrB_pos
                    hrB_lt h_I δ h_r_le_rB hparity hact
  exact three_maxima_not_deg4 (by omega) (by omega) hh h_three

/-- **Left edge: unified N=0** via trichotomy over `r vs rB`. -/
theorem case2b_LeftEdge_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  rcases (by omega : rB ≤ p_I.1.val ∨ p_I.1.val < rB) with h_r_ge_rB | h_r_lt_rB
  · exact case2b_LeftEdge_R_not_deg4 hm hn hrB_pos hrB_lt h_I δ h_r_ge_rB
      hparity hact
  · exact case2b_LeftEdge_L_not_deg4 hm hn hrB_pos hrB_lt h_I δ
      (Nat.le_of_lt h_r_lt_rB) hparity hact

end OrigamiCone

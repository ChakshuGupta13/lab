import OrigamiCone.BoundaryCase2bLeftEdgeUnified
import OrigamiCone.ColReflectTransport

/-!
# Case 2b right edge: N=0 via `colRefl` transport

Left-symmetric twin. Right-edge `p_B = (rB, n - 1)` transports to left-edge
`(rB, 0)` via colRefl (which holds row, reverses column).

## Result

* `case2b_RightEdge_not_deg4`: for `p_B = (rB, n - 1)` right edge, p_I
  interior, parity + right-active, `(neighbors cpe).ncard ≠ 4`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b right-edge N=0 via colRefl transport.** -/
theorem case2b_RightEdge_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  set p_I' : Cell m n := colRefl p_I with hp_I'_def
  have h_I' : IsInterior p_I' := h_I.colRefl
  set p_B_orig : Cell m n :=
    ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) with hp_B_orig_def
  set p_B_left : Cell m n :=
    ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) with hp_B_left_def
  have h_colRefl_pB : colRefl p_B_left = p_B_orig := by
    apply Prod.ext
    · rfl
    · show (Fin.rev (⟨0, by omega⟩ : Fin n)) = ⟨n - 1, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
  have h_colRefl_pI : colRefl p_I' = p_I := colRefl_involutive p_I
  have h_gdist_eq : gdist p_B_left p_I' = gdist p_B_orig p_I := by
    conv_rhs => rw [← h_colRefl_pB, ← h_colRefl_pI]
    exact (colRefl_gdist p_B_left p_I').symm
  have hparity_left : (δ - gdist p_B_left p_I') % 2 = 0 := by
    rw [h_gdist_eq]; exact hparity
  have hact_left : δ < gdist p_B_left p_I' := by
    rw [h_gdist_eq]; exact hact
  intro hdeg4
  -- Case-split on r vs rB (for p_I'), dispatch to R or L exists-three-maxima.
  rcases (by omega : rB ≤ p_I'.1.val ∨ p_I'.1.val < rB) with h_r_ge_rB | h_r_lt_rB
  · obtain ⟨s_left, hcard, hmem⟩ :=
      case2b_LeftEdge_R_exists_three_maxima hm (by omega) hrB_pos hrB_lt h_I' δ
        h_r_ge_rB hparity_left hact_left
    have hh : IsHeight (cpe p_B_orig p_I δ) := by
      rw [hp_B_orig_def]
      exact cpe_isHeight _ p_I δ hparity
    apply three_maxima_not_deg4 (by omega) (by omega) hh
      ⟨s_left.image colRefl, ?_, ?_⟩ hdeg4
    · rw [Finset.card_image_of_injective _ colRefl.injective]; exact hcard
    · intro c hc
      obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
      have h_left_max := hmem v hv_in
      have h_transp := cpe_strictMax_colRefl h_left_max
      rw [h_colRefl_pB, h_colRefl_pI] at h_transp
      rw [← hv_eq]; exact h_transp
  · obtain ⟨s_left, hcard, hmem⟩ :=
      case2b_LeftEdge_L_exists_three_maxima hm (by omega) hrB_pos hrB_lt h_I' δ
        (Nat.le_of_lt h_r_lt_rB) hparity_left hact_left
    have hh : IsHeight (cpe p_B_orig p_I δ) := by
      rw [hp_B_orig_def]
      exact cpe_isHeight _ p_I δ hparity
    apply three_maxima_not_deg4 (by omega) (by omega) hh
      ⟨s_left.image colRefl, ?_, ?_⟩ hdeg4
    · rw [Finset.card_image_of_injective _ colRefl.injective]; exact hcard
    · intro c hc
      obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
      have h_left_max := hmem v hv_in
      have h_transp := cpe_strictMax_colRefl h_left_max
      rw [h_colRefl_pB, h_colRefl_pI] at h_transp
      rw [← hv_eq]; exact h_transp

end OrigamiCone

import OrigamiCone.BoundaryCase2bTopEdgeRNotDeg4
import OrigamiCone.ColReflectTransport

/-!
# Case 2b sub-case `s ≤ cB` (leftward `p_I`) via `colRefl` transport

Row-symmetric analog of `BoundaryCase2bTopEdgeRNotDeg4` for the sub-case
where `p_I`'s column `s` is at most `cB` (p_I weakly LEFT of p_B).
Follows from the sub-case R result by transporting via `colRefl`:
`colRefl(0, cB) = (0, n-1-cB)` and `colRefl(r, s) = (r, n-1-s)`, so
`s ≥ cB` transports to `s' ≤ cB'` with `s' = n-1-s`, `cB' = n-1-cB`.

## Results

* `case2b_TopEdge_L_exists_three_maxima`: three-maxima Finset for
  top-edge `p_B` with `s ≤ cB`.
* `case2b_TopEdge_L_not_deg4`: (neighbors cpe).ncard ≠ 4.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b sub-case `s ≤ cB`: three-element Finset of strict local maxima**
via colRefl transport from sub-case R. -/
theorem case2b_TopEdge_L_exists_three_maxima
    (hm : 2 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_le_cB : p_I.2.val ≤ cB)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s,
        IsStrictLocalMax
          (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
               p_I δ) c := by
  -- Setup for colRefl transport: p_I' := colRefl p_I, p_B' := (0, n-1-cB).
  set p_I' : Cell m n := colRefl p_I with hp_I'_def
  have h_I' : IsInterior p_I' := h_I.colRefl
  set cB' : ℕ := n - 1 - cB with hcB'_def
  have hcB'_pos : 1 ≤ cB' := by
    show 1 ≤ n - 1 - cB
    omega
  have hcB'_lt : cB' + 1 < n := by
    show n - 1 - cB + 1 < n
    omega
  have h_s'_ge_cB' : cB' ≤ p_I'.2.val := by
    have hp_I2_lt : p_I.2.val < n := p_I.2.isLt
    have h_s'_val : p_I'.2.val = n - 1 - p_I.2.val := by
      show (Fin.rev p_I.2).val = _
      rw [Fin.val_rev]; omega
    show n - 1 - cB ≤ p_I'.2.val
    rw [h_s'_val]
    omega
  -- The transported p_B is (0, cB') in the top-edge form.
  set p_B_orig : Cell m n := ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
    with hp_B_orig_def
  set p_B_R : Cell m n := ((⟨0, by omega⟩ : Fin m), (⟨cB', by omega⟩ : Fin n))
    with hp_B_R_def
  -- gdist p_B_R p_I' = gdist p_B_orig p_I (via colRefl_gdist since colRefl
  -- p_B_R = p_B_orig, colRefl p_I' = p_I).
  have h_colRefl_pB : colRefl p_B_R = p_B_orig := by
    apply Prod.ext
    · rfl  -- row unchanged under colRefl
    · show (Fin.rev (⟨cB', by omega⟩ : Fin n)) = ⟨cB, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
      show n - (cB' + 1) = cB
      omega
  have h_colRefl_pI : colRefl p_I' = p_I := colRefl_involutive p_I
  have h_gdist_eq : gdist p_B_R p_I' = gdist p_B_orig p_I := by
    conv_rhs => rw [← h_colRefl_pB, ← h_colRefl_pI]
    exact (colRefl_gdist p_B_R p_I').symm
  have hparity_R : (δ - gdist p_B_R p_I') % 2 = 0 := by rw [h_gdist_eq]; exact hparity
  have hact_R : δ < gdist p_B_R p_I' := by rw [h_gdist_eq]; exact hact
  -- Apply sub-case R at p_B_R, p_I'.
  obtain ⟨s_R, hcard, hmem⟩ :=
    case2b_TopEdge_R_exists_three_maxima hm hn hcB'_pos hcB'_lt h_I' δ
      h_s'_ge_cB' hparity_R hact_R
  -- Transport via colRefl: image the Finset, use colRefl_cpe.
  refine ⟨s_R.image colRefl, ?_, ?_⟩
  · rw [Finset.card_image_of_injective _ colRefl.injective]; exact hcard
  · intro c hc
    obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
    have h_R_max := hmem v hv_in
    have h_transp := cpe_strictMax_colRefl h_R_max
    -- h_transp : IsStrictLocalMax (cpe (colRefl p_B_R) (colRefl p_I') δ) (colRefl v)
    -- Rewrite colRefl p_B_R = p_B_orig, colRefl p_I' = p_I.
    rw [h_colRefl_pB, h_colRefl_pI] at h_transp
    rw [← hv_eq]
    exact h_transp

/-- **Case 2b sub-case `s ≤ cB`: not degree 4.** Paper's N=0 conclusion
for the leftward-p_I sub-case, via three_maxima_not_deg4. -/
theorem case2b_TopEdge_L_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (h_s_le_cB : p_I.2.val ≤ cB)
    (hparity :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  have hh : IsHeight (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                          p_I δ) :=
    cpe_isHeight _ p_I δ hparity
  have h_three := case2b_TopEdge_L_exists_three_maxima (by omega) hn hcB_pos
                    hcB_lt h_I δ h_s_le_cB hparity hact
  exact three_maxima_not_deg4 (by omega) (by omega) hh h_three

end OrigamiCone

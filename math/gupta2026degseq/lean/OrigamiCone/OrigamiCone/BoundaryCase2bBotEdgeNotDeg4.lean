import OrigamiCone.BoundaryCase2bTopEdgeUnified
import OrigamiCone.RowReflectTransport

/-!
# Case 2b bottom edge: N=0 via `rowRefl` transport

Row-symmetric twin of `BoundaryCase2bTopEdgeUnified`. Bottom-edge
`p_B = (m - 1, cB)` transports to top-edge under `rowRefl`, so the
top-edge N=0 result immediately transports to the bottom edge.

## Result

* `case2b_BotEdge_not_deg4`: for `p_B = (m - 1, cB)` on the bottom
  edge and `p_I` interior + parity + right-active, `(neighbors cpe).
  ncard ≠ 4`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b bottom-edge N=0 via rowRefl transport.** -/
theorem case2b_BotEdge_not_deg4
    (hm : 3 ≤ m) (hn : 3 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) % 2 = 0)
    (hact : δ < gdist ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I) :
    (neighbors (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n))
                    p_I δ)).ncard ≠ 4 := by
  -- Setup rowRefl transport: p_I' := rowRefl p_I; p_B' := (0, cB) (top edge).
  set p_I' : Cell m n := rowRefl p_I with hp_I'_def
  have h_I' : IsInterior p_I' := h_I.rowRefl
  set p_B_orig : Cell m n :=
    ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) with hp_B_orig_def
  set p_B_top : Cell m n :=
    ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) with hp_B_top_def
  -- rowRefl p_B_top = p_B_orig.
  have h_rowRefl_pB : rowRefl p_B_top = p_B_orig := by
    apply Prod.ext
    · show (Fin.rev (⟨0, by omega⟩ : Fin m)) = ⟨m - 1, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
    · rfl
  have h_rowRefl_pI : rowRefl p_I' = p_I := rowRefl_involutive p_I
  -- gdist p_B_top p_I' = gdist p_B_orig p_I via rowRefl isometry.
  have h_gdist_eq : gdist p_B_top p_I' = gdist p_B_orig p_I := by
    conv_rhs => rw [← h_rowRefl_pB, ← h_rowRefl_pI]
    exact (rowRefl_gdist p_B_top p_I').symm
  have hparity_top : (δ - gdist p_B_top p_I') % 2 = 0 := by
    rw [h_gdist_eq]; exact hparity
  have hact_top : δ < gdist p_B_top p_I' := by
    rw [h_gdist_eq]; exact hact
  -- Apply top-edge unified N=0 at (cB, p_I').
  have h_top_not_deg4 :=
    case2b_TopEdge_not_deg4 hm hn hcB_pos hcB_lt h_I' δ hparity_top hact_top
  -- Transport the "not deg 4" via the flip-graph iso induced by rowRefl.
  -- Reformulate: cpe p_B_orig p_I δ = cpe (rowRefl p_B_top) (rowRefl p_I') δ
  --   = cpe p_B_top p_I' δ ∘ rowRefl (pointwise via rowRefl_cpe).
  -- Since rowRefl is an isometry-bijection of the grid, neighbors.ncard is
  -- preserved. However, formalising the neighbors-transport is involved;
  -- an alternative: directly derive the three-maxima existence for p_B_orig
  -- from p_B_top by transporting the Finset via rowRefl.image.
  -- Simpler route: contradict with three-maxima via three_maxima_not_deg4.
  intro hdeg4
  -- three_maxima_not_deg4 applied at p_B_orig would need a 3-elem Finset of
  -- strict local max of cpe p_B_orig p_I δ. Transport top-edge 3-elem Finset
  -- via rowRefl:
  -- (Extract p_B_top's Finset first, but case2b_TopEdge_not_deg4 gives only
  -- the ≠ 4 conclusion. Need the Finset form.)
  -- Rebuild by dispatching to R or L three-maxima Finset + transport:
  rcases (by omega : cB ≤ p_I'.2.val ∨ p_I'.2.val < cB) with h_s_ge_cB | h_s_lt_cB
  · obtain ⟨s_top, hcard, hmem⟩ :=
      case2b_TopEdge_R_exists_three_maxima (by omega) hn hcB_pos hcB_lt h_I' δ
        h_s_ge_cB hparity_top hact_top
    -- Transport to p_B_orig via rowRefl.
    have hh : IsHeight (cpe p_B_orig p_I δ) := by
      rw [hp_B_orig_def]
      exact cpe_isHeight _ p_I δ hparity
    apply three_maxima_not_deg4 (by omega) (by omega) hh
      ⟨s_top.image rowRefl, ?_, ?_⟩ hdeg4
    · rw [Finset.card_image_of_injective _ rowRefl.injective]; exact hcard
    · intro c hc
      obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
      have h_top_max := hmem v hv_in
      have h_transp := cpe_strictMax_rowRefl h_top_max
      rw [h_rowRefl_pB, h_rowRefl_pI] at h_transp
      rw [← hv_eq]; exact h_transp
  · obtain ⟨s_top, hcard, hmem⟩ :=
      case2b_TopEdge_L_exists_three_maxima (by omega) hn hcB_pos hcB_lt h_I' δ
        (Nat.le_of_lt h_s_lt_cB) hparity_top hact_top
    have hh : IsHeight (cpe p_B_orig p_I δ) := by
      rw [hp_B_orig_def]
      exact cpe_isHeight _ p_I δ hparity
    apply three_maxima_not_deg4 (by omega) (by omega) hh
      ⟨s_top.image rowRefl, ?_, ?_⟩ hdeg4
    · rw [Finset.card_image_of_injective _ rowRefl.injective]; exact hcard
    · intro c hc
      obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
      have h_top_max := hmem v hv_in
      have h_transp := cpe_strictMax_rowRefl h_top_max
      rw [h_rowRefl_pB, h_rowRefl_pI] at h_transp
      rw [← hv_eq]; exact h_transp

end OrigamiCone

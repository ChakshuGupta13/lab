import OrigamiCone.BoundaryOneInterior
import OrigamiCone.BoundaryCase2aAssembly
import OrigamiCone.RowReflectTransport

/-!
# Case 2a for the BL corner sub-case — via `rowRefl` transport

With `p_B = (m - 1, 0)` (bottom-left corner) and `p_I` interior, case 2a
of `lem:boundary` follows from the TL result by transporting via
`rowRefl`: `rowRefl(TL) = BL`, `rowRefl(BR) = TR`, so `rowRefl` sends
the TL frame's opposite corner `(m - 1, n - 1)` to the BL frame's
opposite corner `(0, n - 1)`.

## Results

* `oneInterior_BLcorner_opposite_max`: (0, n-1) is a strict local max
  under `cpe BL p_I δ`.
* `case2a_BL_exists_three_maxima`: three-element Finset of strict local
  maxima under parity + right-side activity.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **BL-corner opposite-max via `rowRefl` transport.** -/
theorem oneInterior_BLcorner_opposite_max (hm : 2 ≤ m) (hn : 2 ≤ n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  set p_I' : Cell m n := rowRefl p_I with hp_I'
  have h_I' : IsInterior p_I' := h_I.rowRefl
  have h_TL := oneInterior_TLcorner_opposite_max hm hn h_I' δ
  have h_transp := cpe_strictMax_rowRefl h_TL
  have e_TL_to_BL : rowRefl ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) =
                    ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
    apply Prod.ext
    · show (Fin.rev (⟨0, by omega⟩ : Fin m)) = ⟨m - 1, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
    · rfl
  have e_BR_to_TR : rowRefl ((⟨m - 1, by omega⟩ : Fin m),
                             (⟨n - 1, by omega⟩ : Fin n)) =
                    ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
    apply Prod.ext
    · show (Fin.rev (⟨m - 1, by omega⟩ : Fin m)) = ⟨0, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]; dsimp only; omega
    · rfl
  have e_p_I : rowRefl p_I' = p_I := rowRefl_involutive p_I
  rw [e_TL_to_BL, e_BR_to_TR, e_p_I] at h_transp
  exact h_transp

/-- **Case 2a for the BL corner: Finset-form three-maxima existence.** -/
theorem case2a_BL_exists_three_maxima
    (hm : 3 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I)
        % 2 = 0)
    (hact : δ < gdist ((⟨m - 1, by omega⟩ : Fin m),
                       (⟨0, by omega⟩ : Fin n)) p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s,
        IsStrictLocalMax
          (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n))
               p_I δ) c := by
  set p_I' : Cell m n := rowRefl p_I with hp_I'_def
  have h_I' : IsInterior p_I' := h_I.rowRefl
  have h_gdist_eq :
      gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I' =
      gdist ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I := by
    have e_TL_eq : ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) =
                   rowRefl ((⟨m - 1, by omega⟩ : Fin m),
                            (⟨0, by omega⟩ : Fin n)) := by
      apply Prod.ext
      · show (⟨0, by omega⟩ : Fin m) = Fin.rev (⟨m - 1, by omega⟩ : Fin m)
        apply Fin.ext
        show (0 : ℕ) = (Fin.rev _).val
        rw [Fin.val_rev]; dsimp only; omega
      · rfl
    rw [e_TL_eq]
    exact rowRefl_gdist _ _
  have hparity_TL :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I') % 2 = 0 := by
    rw [h_gdist_eq]; exact hparity
  have hact_TL :
      δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I' := by
    rw [h_gdist_eq]; exact hact
  obtain ⟨s_TL, hcard, hmem⟩ :=
    case2a_TL_exists_three_maxima hm hn h_I' δ hparity_TL hact_TL
  refine ⟨s_TL.image rowRefl, ?_, ?_⟩
  · rw [Finset.card_image_of_injective _ rowRefl.injective]
    exact hcard
  · intro c hc
    obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
    have h_TL_max := hmem v hv_in
    have h_transp := cpe_strictMax_rowRefl h_TL_max
    have e_TL_to_BL : rowRefl ((⟨0, by omega⟩ : Fin m),
                                (⟨0, by omega⟩ : Fin n)) =
                      ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
      apply Prod.ext
      · show (Fin.rev (⟨0, by omega⟩ : Fin m)) = ⟨m - 1, by omega⟩
        apply Fin.ext
        show (Fin.rev _).val = _
        rw [Fin.val_rev]
      · rfl
    have e_p_I' : rowRefl p_I' = p_I := rowRefl_involutive p_I
    rw [e_TL_to_BL, e_p_I'] at h_transp
    rw [← hv_eq]
    exact h_transp

end OrigamiCone

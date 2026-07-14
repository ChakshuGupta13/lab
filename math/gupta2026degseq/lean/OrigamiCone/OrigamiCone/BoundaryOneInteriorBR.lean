import OrigamiCone.BoundaryOneInterior
import OrigamiCone.BoundaryCase2aAssembly
import OrigamiCone.CellReflectTransport

/-!
# Case 2a for the BR corner sub-case — via `cellRefl` transport

Demonstrates using `CellReflectTransport.cpe_strictMax_cellRefl` to lift
the TL-corner case-2a opposite-max theorem to the BR-corner sub-case
without repeating the proof. The transport pattern:
  cellRefl(TL) = BR, cellRefl(BR) = TL — so a strict local max at
  BR under `cpe_TL p_I δ` transports (via cellRefl on both apexes and
  the max point) to a strict local max at TL under
  `cpe_BR (cellRefl p_I) δ`.

## Result

* `oneInterior_BRcorner_opposite_max`: with `p_B = (m-1, n-1)` (BR) and
  `p_I` interior, the OPPOSITE corner `(0, 0)` is a strict local max of
  `cpe p_B p_I δ`.

## What this file is (and is not)

This module is a WORKED EXAMPLE of the `cellRefl` transport pattern.
For the full BR corner case 2a (all three maxima + capstone), the same
pattern applies mechanically — starting from `case2a_TL_three_
strictLocalMax` and transporting through `cellRefl`. That full derivation
is left for a follow-up commit; here we prove the opposite-max piece as
the minimal existence-proof-of-concept.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **BR-corner opposite-max via `cellRefl` transport.** With `p_B` at the
bottom-right corner `(m - 1, n - 1)` and `p_I` interior, the OPPOSITE
corner `(0, 0)` is a strict local max of `cpe p_B p_I δ`. Derived by
`cellRefl`-transporting `oneInterior_TLcorner_opposite_max`. -/
theorem oneInterior_BRcorner_opposite_max (hm : 2 ≤ m) (hn : 2 ≤ n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
  -- Under cellRefl, `p_I` maps to a fresh interior point `p_I'`.
  set p_I' : Cell m n := cellRefl p_I with hp_I'
  have h_I' : IsInterior p_I' := h_I.cellRefl
  -- Apply the TL case at p_I' via cellRefl to move to the BR frame.
  have h_TL := oneInterior_TLcorner_opposite_max hm hn h_I' δ
  -- Transport through cellRefl: cpe TL p_I' δ strict-max at (m-1, n-1)
  --   → cpe (cellRefl TL) (cellRefl p_I') δ strict-max at cellRefl(m-1, n-1).
  have h_transp := cpe_strictMax_cellRefl h_TL
  -- Reduce cellRefl of each specific cell.
  have e_TL : cellRefl ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) =
              ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
    apply Prod.ext
    · show (Fin.rev (⟨0, by omega⟩ : Fin m)) = ⟨m - 1, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
    · show (Fin.rev (⟨0, by omega⟩ : Fin n)) = ⟨n - 1, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
  have e_BR : cellRefl ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
              = ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
    apply Prod.ext
    · show (Fin.rev (⟨m - 1, by omega⟩ : Fin m)) = ⟨0, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
      dsimp only
      omega
    · show (Fin.rev (⟨n - 1, by omega⟩ : Fin n)) = ⟨0, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
      dsimp only
      omega
  have e_p_I : cellRefl p_I' = p_I := cellRefl_involutive p_I
  rw [e_TL, e_BR, e_p_I] at h_transp
  exact h_transp

/-- **Case 2a for the BR corner: Finset-form three-maxima existence.**
Transports `case2a_TL_exists_three_maxima` via `cellRefl`. With
`p_B = (m - 1, n - 1)` (BR corner) and `p_I` interior, under parity +
right-side activity, `cpe p_B p_I δ` has a three-element Finset of
strict local maxima. -/
theorem case2a_BR_exists_three_maxima
    (hm : 3 ≤ m) (hn : 3 ≤ n) {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ)
    (hparity :
      (δ - gdist ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I)
        % 2 = 0)
    (hact : δ < gdist ((⟨m - 1, by omega⟩ : Fin m),
                       (⟨n - 1, by omega⟩ : Fin n)) p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s,
        IsStrictLocalMax
          (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n))
               p_I δ) c := by
  -- Set up p_I' := cellRefl p_I (interior since IsInterior.cellRefl).
  set p_I' : Cell m n := cellRefl p_I with hp_I'_def
  have h_I' : IsInterior p_I' := h_I.cellRefl
  -- The TL frame's p_B = (0, 0) = cellRefl(BR); its p_I is p_I'. Compute
  -- gdist p_B_TL p_I' = gdist (cellRefl BR) (cellRefl p_I) = gdist BR p_I,
  -- so parity and activity transport pointwise.
  have h_gdist_eq :
      gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I' =
      gdist ((⟨m - 1, by omega⟩ : Fin m),
             (⟨n - 1, by omega⟩ : Fin n)) p_I := by
    have e_TL : ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) =
                cellRefl ((⟨m - 1, by omega⟩ : Fin m),
                          (⟨n - 1, by omega⟩ : Fin n)) := by
      apply Prod.ext
      · show (⟨0, by omega⟩ : Fin m) = Fin.rev (⟨m - 1, by omega⟩ : Fin m)
        apply Fin.ext
        show (0 : ℕ) = (Fin.rev _).val
        rw [Fin.val_rev]; dsimp only; omega
      · show (⟨0, by omega⟩ : Fin n) = Fin.rev (⟨n - 1, by omega⟩ : Fin n)
        apply Fin.ext
        show (0 : ℕ) = (Fin.rev _).val
        rw [Fin.val_rev]; dsimp only; omega
    rw [e_TL]
    exact cellRefl_gdist _ _
  have hparity_TL :
      (δ - gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I') % 2 = 0 := by
    rw [h_gdist_eq]; exact hparity
  have hact_TL :
      δ < gdist ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I' := by
    rw [h_gdist_eq]; exact hact
  -- Apply the TL Finset theorem at p_I'.
  obtain ⟨s_TL, hcard, hmem⟩ :=
    case2a_TL_exists_three_maxima hm hn h_I' δ hparity_TL hact_TL
  -- Transport the Finset via cellRefl.image; use Finset.card_image_of_injective
  -- since cellRefl is a bijection.
  refine ⟨s_TL.image cellRefl, ?_, ?_⟩
  · rw [Finset.card_image_of_injective _ cellRefl.injective]
    exact hcard
  · intro c hc
    obtain ⟨v, hv_in, hv_eq⟩ := Finset.mem_image.mp hc
    have h_TL_max := hmem v hv_in
    -- Transport: cpe TL p_I' δ strict max at v → cpe (cellRefl TL) (cellRefl p_I') δ strict max at cellRefl v.
    have h_transp := cpe_strictMax_cellRefl h_TL_max
    -- cellRefl TL = BR, cellRefl p_I' = p_I.
    have e_TL_to_BR : cellRefl ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) =
                      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
      apply Prod.ext
      · show (Fin.rev (⟨0, by omega⟩ : Fin m)) = ⟨m - 1, by omega⟩
        apply Fin.ext
        show (Fin.rev _).val = _
        rw [Fin.val_rev]
      · show (Fin.rev (⟨0, by omega⟩ : Fin n)) = ⟨n - 1, by omega⟩
        apply Fin.ext
        show (Fin.rev _).val = _
        rw [Fin.val_rev]
    have e_p_I' : cellRefl p_I' = p_I := cellRefl_involutive p_I
    rw [e_TL_to_BR, e_p_I'] at h_transp
    rw [← hv_eq]
    exact h_transp

end OrigamiCone

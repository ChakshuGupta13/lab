import OrigamiCone.BoundaryOneInteriorEdge
import OrigamiCone.RowReflectTransport

/-!
# Case 2b for the bottom edge — via `rowRefl` transport

Row-symmetric twin of `BoundaryOneInteriorEdge.lean`. With `p_B = (m - 1, cB)`
on the bottom edge (1 ≤ cB ≤ n - 2) and `p_I` interior, the two TOP corners
(0, 0) and (0, n - 1) are strict local maxima of `cpe p_B p_I δ`. Follows
from the top-edge result by `rowRefl` transport (rowRefl holds the column,
flips the row).

## Results

* `oneInterior_BotEdge_topLeft_max`, `oneInterior_BotEdge_topRight_max`.
* `oneInterior_BotEdge_two_top_max`: packaging with pairwise distinctness.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b (bottom edge) FIRST maximum.** -/
theorem oneInterior_BotEdge_topLeft_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
  -- Transport the top-edge BL max via rowRefl: rowRefl(0, cB) = (m-1, cB),
  -- rowRefl(m-1, 0) = (0, 0), and rowRefl p_I' = p_I for p_I' := rowRefl p_I.
  set p_I' : Cell m n := rowRefl p_I with hp_I'
  have h_I' : IsInterior p_I' := h_I.rowRefl
  have h_top := oneInterior_TopEdge_bottomLeft_max hm hn hcB_pos hcB_lt h_I' δ
  have h_transp := cpe_strictMax_rowRefl h_top
  have e_top_pB : rowRefl ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) =
                  ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) := by
    apply Prod.ext
    · show (Fin.rev (⟨0, by omega⟩ : Fin m)) = ⟨m - 1, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]
    · rfl
  have e_BL_to_TL : rowRefl ((⟨m - 1, by omega⟩ : Fin m),
                             (⟨0, by omega⟩ : Fin n)) =
                    ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
    apply Prod.ext
    · show (Fin.rev (⟨m - 1, by omega⟩ : Fin m)) = ⟨0, by omega⟩
      apply Fin.ext
      show (Fin.rev _).val = _
      rw [Fin.val_rev]; dsimp only; omega
    · rfl
  have e_p_I : rowRefl p_I' = p_I := rowRefl_involutive p_I
  rw [e_top_pB, e_BL_to_TL, e_p_I] at h_transp
  exact h_transp

/-- **Case 2b (bottom edge) SECOND maximum.** -/
theorem oneInterior_BotEdge_topRight_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  set p_I' : Cell m n := rowRefl p_I with hp_I'
  have h_I' : IsInterior p_I' := h_I.rowRefl
  have h_top := oneInterior_TopEdge_bottomRight_max hm hn hcB_pos hcB_lt h_I' δ
  have h_transp := cpe_strictMax_rowRefl h_top
  have e_top_pB : rowRefl ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) =
                  ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) := by
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
  rw [e_top_pB, e_BR_to_TR, e_p_I] at h_transp
  exact h_transp

/-- **Case 2b (bottom edge): both top corners are pairwise-distinct strict
local maxima.** Packaging theorem: with `p_B = (m - 1, cB)` on the bottom
edge (`1 ≤ cB ≤ n - 2`) and `p_I` interior, `(0, 0)` and `(0, n - 1)` are
both `IsStrictLocalMax` of `cpe p_B p_I δ`, pairwise distinct. -/
theorem oneInterior_BotEdge_two_top_max
    (hm : 2 ≤ m) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ∧
    IsStrictLocalMax
      (cpe ((⟨m - 1, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ∧
    ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ≠
      (((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) :
        Cell m n) := by
  have hn : 2 ≤ n := by omega
  refine ⟨oneInterior_BotEdge_topLeft_max hm hn hcB_pos hcB_lt h_I δ,
          oneInterior_BotEdge_topRight_max hm hn hcB_pos hcB_lt h_I δ,
          ?_⟩
  intro heq
  have := congrArg (fun c : Cell m n => c.2.val) heq
  dsimp at this
  omega

end OrigamiCone

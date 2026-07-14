import OrigamiCone.BoundaryOneInterior
import OrigamiCone.BoundaryCase2aAssembly
import OrigamiCone.BoundaryOneInteriorBR
import OrigamiCone.BoundaryOneInteriorTR
import OrigamiCone.BoundaryOneInteriorBL

/-!
# `lem:boundary` case 2a: unified any-corner three-maxima existence

Packages the four corner sub-cases (TL / BR / TR / BL) into a single
theorem parametrised by the corner. Case-splits `IsCorner p_B` and
dispatches to the appropriate corner-specific Finset theorem.

## Result

* `case2a_anyCorner_exists_three_maxima`: for `p_B` any grid corner
  (`IsCorner p_B`) and `p_I` interior, under parity + right-side
  activity, `cpe p_B p_I δ` has a three-element Finset of strict local
  maxima.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`lem:boundary` case 2a, unified corner sub-case.** For any grid
corner `p_B` (`IsCorner p_B`) and any interior `p_I`, under parity +
right-side activity, `cpe p_B p_I δ` admits a three-element Finset of
strict local maxima. Dispatches to `case2a_TL_/BR_/TR_/BL_exists_three
_maxima` by case-splitting `IsCorner`. -/
theorem case2a_anyCorner_exists_three_maxima
    (hm : 3 ≤ m) (hn : 3 ≤ n) {p_B p_I : Cell m n}
    (hpB_corner : IsCorner p_B) (h_I : IsInterior p_I) (δ : ℤ)
    (hparity : (δ - gdist p_B p_I) % 2 = 0)
    (hact : δ < gdist p_B p_I) :
    ∃ s : Finset (Cell m n), s.card = 3 ∧
      ∀ c ∈ s, IsStrictLocalMax (cpe p_B p_I δ) c := by
  obtain ⟨hrow, hcol⟩ := hpB_corner
  have hp_B1 : p_B.1.val < m := p_B.1.isLt
  have hp_B2 : p_B.2.val < n := p_B.2.isLt
  rcases hrow with hrow0 | hrowm
  · rcases hcol with hcol0 | hcoln
    · -- p_B = TL = (0, 0)
      have hpB_eq : p_B = ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext hrow0
        · exact Fin.ext hcol0
      rw [hpB_eq] at hparity hact ⊢
      exact case2a_TL_exists_three_maxima hm hn h_I δ hparity hact
    · -- p_B = TR = (0, n-1)
      have hpB_eq : p_B = ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext hrow0
        · exact Fin.ext hcoln
      rw [hpB_eq] at hparity hact ⊢
      exact case2a_TR_exists_three_maxima hm hn h_I δ hparity hact
  · rcases hcol with hcol0 | hcoln
    · -- p_B = BL = (m-1, 0)
      have hpB_eq : p_B = ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext hrowm
        · exact Fin.ext hcol0
      rw [hpB_eq] at hparity hact ⊢
      exact case2a_BL_exists_three_maxima hm hn h_I δ hparity hact
    · -- p_B = BR = (m-1, n-1)
      have hpB_eq : p_B = ((⟨m - 1, by omega⟩ : Fin m),
                           (⟨n - 1, by omega⟩ : Fin n)) := by
        apply Prod.ext
        · exact Fin.ext hrowm
        · exact Fin.ext hcoln
      rw [hpB_eq] at hparity hact ⊢
      exact case2a_BR_exists_three_maxima hm hn h_I δ hparity hact

end OrigamiCone

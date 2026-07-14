import OrigamiCone.Basic
import OrigamiCone.ConePair
import OrigamiCone.Boundary

/-!
# `rowRefl` transport: partial (row-axis) reflection

Row-axis-only analog of `AcellReflect.cellRefl` / `ColReflectTransport.colRefl`.
Reflects the row coordinate via `Fin.rev`; the column stays. On corners,
`rowRefl` sends `TL ↔ BL` and `TR ↔ BR` (holding the column).

## Definition + isometry facts

* `rowRefl : Cell m n ≃ Cell m n`, `(i, j) ↦ (Fin.rev i, j)`; involution.
* `rowRefl_gdist / rowRefl_adj / rowRefl_cpe / IsStrictLocalMax.rowRefl_
  transport / cpe_strictMax_rowRefl`: same shape as the `cellRefl` and
  `colRefl` transport suites.
* `IsInterior.rowRefl`, `IsCorner.rowRefl`: predicate transports.

## Application

Combined with `case2a_TL_exists_three_maxima`, immediately gives the BL
corner three-maxima result. See `BoundaryOneInteriorBL.lean` for the
worked instantiation.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Row-axis reflection of the grid.** Reverses the row via `Fin.rev`;
holds the column. Involution. -/
def rowRefl : Cell m n ≃ Cell m n where
  toFun v := (Fin.rev v.1, v.2)
  invFun v := (Fin.rev v.1, v.2)
  left_inv v := by ext <;> simp [Fin.rev_rev]
  right_inv v := by ext <;> simp [Fin.rev_rev]

lemma rowRefl_involutive (v : Cell m n) : rowRefl (rowRefl v) = v :=
  rowRefl.left_inv v

/-- **`rowRefl` is a `gdist`-isometry.** -/
lemma rowRefl_gdist (p q : Cell m n) :
    gdist (rowRefl p) (rowRefl q) = gdist p q := by
  have hp1 : p.1.val < m := p.1.isLt
  have hq1 : q.1.val < m := q.1.isLt
  have e_p1 : (OrigamiCone.rowRefl p).1.val = m - (p.1.val + 1) :=
    Fin.val_rev p.1
  have e_q1 : (OrigamiCone.rowRefl q).1.val = m - (q.1.val + 1) :=
    Fin.val_rev q.1
  have e_p2 : (OrigamiCone.rowRefl p).2 = p.2 := rfl
  have e_q2 : (OrigamiCone.rowRefl q).2 = q.2 := rfl
  have hrow_eq :
      (((m - (p.1.val + 1) : ℕ) : ℤ) - ((m - (q.1.val + 1) : ℕ) : ℤ)).natAbs
        = ((p.1.val : ℤ) - q.1.val).natAbs := by omega
  unfold gdist
  rw [e_p1, e_q1, e_p2, e_q2, hrow_eq]

/-- **`rowRefl` preserves adjacency.** -/
lemma rowRefl_adj (p q : Cell m n) : adj (rowRefl p) (rowRefl q) ↔ adj p q := by
  unfold adj
  rw [rowRefl_gdist]

/-- **`rowRefl` transports `cpe` via the apexes.** -/
lemma rowRefl_cpe (p_B p_I : Cell m n) (δ : ℤ) (v : Cell m n) :
    cpe p_B p_I δ v =
      cpe (rowRefl p_B) (rowRefl p_I) δ (rowRefl v) := by
  unfold cpe
  rw [show gdist p_B v = gdist (rowRefl p_B) (rowRefl v) from
        (rowRefl_gdist p_B v).symm,
      show gdist p_I v = gdist (rowRefl p_I) (rowRefl v) from
        (rowRefl_gdist p_I v).symm]

/-- **Strict-local-max transports through `rowRefl`.** -/
lemma IsStrictLocalMax.rowRefl_transport {f : Cell m n → ℤ} {v : Cell m n}
    (h : IsStrictLocalMax f v) :
    IsStrictLocalMax (fun w => f (rowRefl w)) (rowRefl v) := by
  intro u hadj
  have hadj' : adj v (rowRefl u) := by
    apply (rowRefl_adj v (rowRefl u)).mp
    rw [rowRefl_involutive u]
    exact hadj
  have := h _ hadj'
  simp only
  rw [rowRefl_involutive v]
  exact this

/-- **`cpe` strict-local-max transports through simultaneous apex-and-cell
row reflection.** -/
theorem cpe_strictMax_rowRefl {p_B p_I : Cell m n} {δ : ℤ} {v : Cell m n}
    (h : IsStrictLocalMax (cpe p_B p_I δ) v) :
    IsStrictLocalMax (cpe (rowRefl p_B) (rowRefl p_I) δ) (rowRefl v) := by
  have h' : IsStrictLocalMax (fun w => cpe p_B p_I δ (rowRefl w)) (rowRefl v) :=
    h.rowRefl_transport
  intro u hadj
  have hprev := h' u hadj
  simp only at hprev
  rw [rowRefl_involutive v] at hprev
  have e_pB : rowRefl (rowRefl p_B) = p_B := rowRefl_involutive p_B
  have e_pI : rowRefl (rowRefl p_I) = p_I := rowRefl_involutive p_I
  have lhs_rw :
      cpe (rowRefl p_B) (rowRefl p_I) δ u = cpe p_B p_I δ (rowRefl u) := by
    rw [rowRefl_cpe (rowRefl p_B) (rowRefl p_I) δ u, e_pB, e_pI]
  have rhs_rw :
      cpe (rowRefl p_B) (rowRefl p_I) δ (rowRefl v) = cpe p_B p_I δ v := by
    rw [rowRefl_cpe (rowRefl p_B) (rowRefl p_I) δ (rowRefl v),
        e_pB, e_pI, rowRefl_involutive]
  rw [lhs_rw, rhs_rw]
  exact hprev

/-- **`rowRefl` preserves `IsInterior`.** -/
lemma IsInterior.rowRefl {p : Cell m n} (hp : IsInterior p) :
    IsInterior (rowRefl p) := by
  obtain ⟨h1, h2, h3, h4⟩ := hp
  have h1' : p.1.val < m := p.1.isLt
  have e_row : (OrigamiCone.rowRefl p).1.val = m - (p.1.val + 1) :=
    Fin.val_rev p.1
  have e_col : (OrigamiCone.rowRefl p).2 = p.2 := rfl
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [e_row]; omega
  · rw [e_row]; omega
  · rw [e_col]; exact h3
  · rw [e_col]; exact h4

/-- **`rowRefl` preserves `IsCorner`.** -/
lemma IsCorner.rowRefl {v : Cell m n} (hv : IsCorner v) :
    IsCorner (rowRefl v) := by
  obtain ⟨hrow, hcol⟩ := hv
  have h1 : v.1.val < m := v.1.isLt
  have e_row : (OrigamiCone.rowRefl v).1.val = m - (v.1.val + 1) :=
    Fin.val_rev v.1
  have e_col : (OrigamiCone.rowRefl v).2 = v.2 := rfl
  refine ⟨?_, ?_⟩
  · rcases hrow with h | h
    · right; rw [e_row]; omega
    · left; rw [e_row]; omega
  · rw [e_col]; exact hcol

end OrigamiCone

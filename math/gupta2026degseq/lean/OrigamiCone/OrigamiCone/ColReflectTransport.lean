import OrigamiCone.Basic
import OrigamiCone.ConePair
import OrigamiCone.Boundary

/-!
# `colRefl` transport: partial (column-axis) reflection

Column-axis-only analog of `AcellReflect.cellRefl`. Reflects the column
coordinate via `Fin.rev`; the row coordinate stays. On corners, `colRefl`
sends `TL ↔ TR` and `BL ↔ BR` (holding the row).

## Definition + isometry facts

* `colRefl : Cell m n ≃ Cell m n`, `(i, j) ↦ (i, Fin.rev j)`; involution.
* `colRefl_gdist / colRefl_adj / colRefl_cpe / IsStrictLocalMax.colRefl_
  transport / cpe_strictMax_colRefl`: same shape as the `cellRefl`
  transport suite (`CellReflectTransport`), but on the column axis only.
* `IsInterior.colRefl`, `IsCorner.colRefl`: predicate transports.

## Application

Combined with `case2a_TL_exists_three_maxima`, immediately gives the TR
corner three-maxima result. See `BoundaryOneInteriorTR.lean` for the
worked instantiation.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Column-axis reflection of the grid.** Holds the row, reverses the
column via `Fin.rev`. Involution. -/
def colRefl : Cell m n ≃ Cell m n where
  toFun v := (v.1, Fin.rev v.2)
  invFun v := (v.1, Fin.rev v.2)
  left_inv v := by ext <;> simp [Fin.rev_rev]
  right_inv v := by ext <;> simp [Fin.rev_rev]

lemma colRefl_involutive (v : Cell m n) : colRefl (colRefl v) = v :=
  colRefl.left_inv v

/-- **`colRefl` is a `gdist`-isometry.** -/
lemma colRefl_gdist (p q : Cell m n) :
    gdist (colRefl p) (colRefl q) = gdist p q := by
  have hp2 : p.2.val < n := p.2.isLt
  have hq2 : q.2.val < n := q.2.isLt
  have e_p1 : (OrigamiCone.colRefl p).1 = p.1 := rfl
  have e_q1 : (OrigamiCone.colRefl q).1 = q.1 := rfl
  have e_p2 : (OrigamiCone.colRefl p).2.val = n - (p.2.val + 1) :=
    Fin.val_rev p.2
  have e_q2 : (OrigamiCone.colRefl q).2.val = n - (q.2.val + 1) :=
    Fin.val_rev q.2
  have hcol_eq :
      (((n - (p.2.val + 1) : ℕ) : ℤ) - ((n - (q.2.val + 1) : ℕ) : ℤ)).natAbs
        = ((p.2.val : ℤ) - q.2.val).natAbs := by omega
  unfold gdist
  rw [e_p1, e_q1, e_p2, e_q2, hcol_eq]

/-- **`colRefl` preserves adjacency.** -/
lemma colRefl_adj (p q : Cell m n) : adj (colRefl p) (colRefl q) ↔ adj p q := by
  unfold adj
  rw [colRefl_gdist]

/-- **`colRefl` transports `cpe` via the apexes.** -/
lemma colRefl_cpe (p_B p_I : Cell m n) (δ : ℤ) (v : Cell m n) :
    cpe p_B p_I δ v =
      cpe (colRefl p_B) (colRefl p_I) δ (colRefl v) := by
  unfold cpe
  rw [show gdist p_B v = gdist (colRefl p_B) (colRefl v) from
        (colRefl_gdist p_B v).symm,
      show gdist p_I v = gdist (colRefl p_I) (colRefl v) from
        (colRefl_gdist p_I v).symm]

/-- **Strict-local-max transports through `colRefl`.** -/
lemma IsStrictLocalMax.colRefl_transport {f : Cell m n → ℤ} {v : Cell m n}
    (h : IsStrictLocalMax f v) :
    IsStrictLocalMax (fun w => f (colRefl w)) (colRefl v) := by
  intro u hadj
  have hadj' : adj v (colRefl u) := by
    apply (colRefl_adj v (colRefl u)).mp
    rw [colRefl_involutive u]
    exact hadj
  have := h _ hadj'
  simp only
  rw [colRefl_involutive v]
  exact this

/-- **`cpe` strict-local-max transports through simultaneous apex-and-cell
column reflection.** -/
theorem cpe_strictMax_colRefl {p_B p_I : Cell m n} {δ : ℤ} {v : Cell m n}
    (h : IsStrictLocalMax (cpe p_B p_I δ) v) :
    IsStrictLocalMax (cpe (colRefl p_B) (colRefl p_I) δ) (colRefl v) := by
  have h' : IsStrictLocalMax (fun w => cpe p_B p_I δ (colRefl w)) (colRefl v) :=
    h.colRefl_transport
  intro u hadj
  have hprev := h' u hadj
  simp only at hprev
  rw [colRefl_involutive v] at hprev
  have e_pB : colRefl (colRefl p_B) = p_B := colRefl_involutive p_B
  have e_pI : colRefl (colRefl p_I) = p_I := colRefl_involutive p_I
  have lhs_rw :
      cpe (colRefl p_B) (colRefl p_I) δ u = cpe p_B p_I δ (colRefl u) := by
    rw [colRefl_cpe (colRefl p_B) (colRefl p_I) δ u, e_pB, e_pI]
  have rhs_rw :
      cpe (colRefl p_B) (colRefl p_I) δ (colRefl v) = cpe p_B p_I δ v := by
    rw [colRefl_cpe (colRefl p_B) (colRefl p_I) δ (colRefl v),
        e_pB, e_pI, colRefl_involutive]
  rw [lhs_rw, rhs_rw]
  exact hprev

/-- **`colRefl` preserves `IsInterior`.** -/
lemma IsInterior.colRefl {p : Cell m n} (hp : IsInterior p) :
    IsInterior (colRefl p) := by
  obtain ⟨h1, h2, h3, h4⟩ := hp
  have h2' : p.2.val < n := p.2.isLt
  have e_row : (OrigamiCone.colRefl p).1 = p.1 := rfl
  have e_col : (OrigamiCone.colRefl p).2.val = n - (p.2.val + 1) :=
    Fin.val_rev p.2
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [e_row]; exact h1
  · rw [e_row]; exact h2
  · rw [e_col]; omega
  · rw [e_col]; omega

/-- **`colRefl` preserves `IsCorner`.** -/
lemma IsCorner.colRefl {v : Cell m n} (hv : IsCorner v) :
    IsCorner (colRefl v) := by
  obtain ⟨hrow, hcol⟩ := hv
  have h2 : v.2.val < n := v.2.isLt
  have e_row : (OrigamiCone.colRefl v).1 = v.1 := rfl
  have e_col : (OrigamiCone.colRefl v).2.val = n - (v.2.val + 1) :=
    Fin.val_rev v.2
  refine ⟨?_, ?_⟩
  · rw [e_row]; exact hrow
  · rcases hcol with h | h
    · right; rw [e_col]; omega
    · left; rw [e_col]; omega

end OrigamiCone

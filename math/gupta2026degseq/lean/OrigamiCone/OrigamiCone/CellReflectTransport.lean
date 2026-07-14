import OrigamiCone.AcellReflect
import OrigamiCone.ConePair
import OrigamiCone.Boundary

/-!
# `cellRefl` transport: gdist / adj / cpe / IsStrictLocalMax

`AcellReflect` introduced the involution `cellRefl : Cell m n ≃ Cell m n`,
`(i, j) ↦ (Fin.rev i, Fin.rev j)`. This module records the ISOMETRY facts
that let us transport strict-local-max results across the involution.

## Results

* `cellRefl_gdist`: `gdist (cellRefl p) (cellRefl q) = gdist p q`. The
  involution is an isometry of the grid graph.
* `cellRefl_adj`: adjacency is preserved (`adj p q ↔ adj (cellRefl p) (cellRefl q)`).
* `cellRefl_cpe`: `cpe p_B p_I δ ∘ cellRefl = cpe (cellRefl p_B) (cellRefl p_I) δ`.
* `IsStrictLocalMax.cellRefl`: `IsStrictLocalMax f v → IsStrictLocalMax
  (f ∘ cellRefl) (cellRefl v)`.
* `cpe_strictMax_cellRefl`: applying `cellRefl` to both apexes AND the max
  point transports the strict-max property.
* `IsInterior.cellRefl`: interior cells map to interior cells under `cellRefl`.
* `IsCorner.cellRefl`: corner cells map to corner cells under `cellRefl`.

## Application (out of scope here; example)

Combined with `oneInterior_TLcorner_opposite_max`, this immediately gives
the BR-corner opposite-max theorem: `cellRefl (TL) = BR`, `cellRefl (BR) =
TL`, so a TL-apex strict-max at BR transports to a BR-apex strict-max at
TL. See the follow-up `BoundaryOneInteriorBR.lean` (if landed) for the
worked example.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`cellRefl` is a `gdist`-isometry.** -/
lemma cellRefl_gdist (p q : Cell m n) :
    gdist (cellRefl p) (cellRefl q) = gdist p q := by
  have hp1 : p.1.val < m := p.1.isLt
  have hp2 : p.2.val < n := p.2.isLt
  have hq1 : q.1.val < m := q.1.isLt
  have hq2 : q.2.val < n := q.2.isLt
  have e1 : (OrigamiCone.cellRefl p).1.val = m - (p.1.val + 1) :=
    Fin.val_rev p.1
  have e2 : (OrigamiCone.cellRefl p).2.val = n - (p.2.val + 1) :=
    Fin.val_rev p.2
  have e3 : (OrigamiCone.cellRefl q).1.val = m - (q.1.val + 1) :=
    Fin.val_rev q.1
  have e4 : (OrigamiCone.cellRefl q).2.val = n - (q.2.val + 1) :=
    Fin.val_rev q.2
  -- Component-wise natAbs equalities (omega handles Nat.sub + Int.natAbs).
  have hrow_eq :
      (((m - (p.1.val + 1) : ℕ) : ℤ) - ((m - (q.1.val + 1) : ℕ) : ℤ)).natAbs
        = ((p.1.val : ℤ) - q.1.val).natAbs := by omega
  have hcol_eq :
      (((n - (p.2.val + 1) : ℕ) : ℤ) - ((n - (q.2.val + 1) : ℕ) : ℤ)).natAbs
        = ((p.2.val : ℤ) - q.2.val).natAbs := by omega
  unfold gdist
  rw [e1, e2, e3, e4]
  rw [hrow_eq, hcol_eq]

/-- **`cellRefl` preserves adjacency.** -/
lemma cellRefl_adj (p q : Cell m n) : adj (cellRefl p) (cellRefl q) ↔ adj p q := by
  unfold adj
  rw [cellRefl_gdist]

/-- **`cellRefl` transports `cpe` via the apexes.** -/
lemma cellRefl_cpe (p_B p_I : Cell m n) (δ : ℤ) (v : Cell m n) :
    cpe p_B p_I δ v =
      cpe (cellRefl p_B) (cellRefl p_I) δ (cellRefl v) := by
  unfold cpe
  rw [show gdist p_B v = gdist (cellRefl p_B) (cellRefl v) from
        (cellRefl_gdist p_B v).symm,
      show gdist p_I v = gdist (cellRefl p_I) (cellRefl v) from
        (cellRefl_gdist p_I v).symm]

/-- **Strict-local-max transports through `cellRefl`.** If `f` has a strict
local max at `v`, then `f ∘ cellRefl` has a strict local max at `cellRefl v`. -/
lemma IsStrictLocalMax.cellRefl_transport {f : Cell m n → ℤ} {v : Cell m n}
    (h : IsStrictLocalMax f v) :
    IsStrictLocalMax (fun w => f (cellRefl w)) (cellRefl v) := by
  intro u hadj
  -- adj (cellRefl v) u → adj v (cellRefl u) (using cellRefl_adj + involutivity of u)
  have hadj' : adj v (cellRefl u) := by
    apply (cellRefl_adj v (cellRefl u)).mp
    rw [cellRefl_involutive u]
    exact hadj
  have := h _ hadj'
  simp only
  -- goal: f (cellRefl u) = f (cellRefl (cellRefl v)) - 1
  rw [cellRefl_involutive v]
  exact this

/-- **`cpe` strict-local-max transports through simultaneous apex-and-cell
reflection.** If `cpe p_B p_I δ` has a strict local max at `v`, then
`cpe (cellRefl p_B) (cellRefl p_I) δ` has a strict local max at `cellRefl v`. -/
theorem cpe_strictMax_cellRefl {p_B p_I : Cell m n} {δ : ℤ} {v : Cell m n}
    (h : IsStrictLocalMax (cpe p_B p_I δ) v) :
    IsStrictLocalMax (cpe (cellRefl p_B) (cellRefl p_I) δ) (cellRefl v) := by
  have h' : IsStrictLocalMax (fun w => cpe p_B p_I δ (cellRefl w)) (cellRefl v) :=
    h.cellRefl_transport
  intro u hadj
  have hprev := h' u hadj
  simp only at hprev
  -- hprev : cpe p_B p_I δ (cellRefl u) = cpe p_B p_I δ (cellRefl (cellRefl v)) - 1
  rw [cellRefl_involutive v] at hprev
  -- hprev : cpe p_B p_I δ (cellRefl u) = cpe p_B p_I δ v - 1
  have e_pB : cellRefl (cellRefl p_B) = p_B := cellRefl_involutive p_B
  have e_pI : cellRefl (cellRefl p_I) = p_I := cellRefl_involutive p_I
  have lhs_rw :
      cpe (cellRefl p_B) (cellRefl p_I) δ u = cpe p_B p_I δ (cellRefl u) := by
    rw [cellRefl_cpe (cellRefl p_B) (cellRefl p_I) δ u, e_pB, e_pI]
  have rhs_rw :
      cpe (cellRefl p_B) (cellRefl p_I) δ (cellRefl v) = cpe p_B p_I δ v := by
    rw [cellRefl_cpe (cellRefl p_B) (cellRefl p_I) δ (cellRefl v),
        e_pB, e_pI, cellRefl_involutive]
  rw [lhs_rw, rhs_rw]
  exact hprev

/-- **`cellRefl` preserves `IsInterior`.** -/
lemma IsInterior.cellRefl {p : Cell m n} (hp : IsInterior p) :
    IsInterior (cellRefl p) := by
  obtain ⟨h1, h2, h3, h4⟩ := hp
  have h1' : p.1.val < m := p.1.isLt
  have h2' : p.2.val < n := p.2.isLt
  have e1 : (OrigamiCone.cellRefl p).1.val = m - (p.1.val + 1) := by
    show (Fin.rev p.1).val = _
    exact Fin.val_rev p.1
  have e2 : (OrigamiCone.cellRefl p).2.val = n - (p.2.val + 1) := by
    show (Fin.rev p.2).val = _
    exact Fin.val_rev p.2
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [e1]; omega
  · rw [e1]; omega
  · rw [e2]; omega
  · rw [e2]; omega

/-- **`cellRefl` preserves `IsCorner`.** -/
lemma IsCorner.cellRefl {v : Cell m n} (hv : IsCorner v) :
    IsCorner (cellRefl v) := by
  obtain ⟨hrow, hcol⟩ := hv
  have h1 : v.1.val < m := v.1.isLt
  have h2 : v.2.val < n := v.2.isLt
  have e1 : (OrigamiCone.cellRefl v).1.val = m - (v.1.val + 1) := by
    show (Fin.rev v.1).val = _
    exact Fin.val_rev v.1
  have e2 : (OrigamiCone.cellRefl v).2.val = n - (v.2.val + 1) := by
    show (Fin.rev v.2).val = _
    exact Fin.val_rev v.2
  refine ⟨?_, ?_⟩
  · rcases hrow with h | h
    · right; rw [e1]; omega
    · left; rw [e1]; omega
  · rcases hcol with h | h
    · right; rw [e2]; omega
    · left; rw [e2]; omega

end OrigamiCone

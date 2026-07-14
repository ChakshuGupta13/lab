import OrigamiCone.AcellDual

/-!
# Strict-extremum witnesses for `acell` at the opposite corners

The modules `AcellGradient.lean` and `AcellMin.lean` proved the
**uniqueness** of the strict local extrema of the antidiagonal `acell`:
any strict local max must be at `(m − 1, n − 1)`, and any strict local
min must be at `(0, 0)`.  Missing from those theorems were the dual
**existence** statements — that the two opposite corners are *actually*
strict local extrema.  This module closes that gap.

Combined with the uniqueness theorems (for `m, n ≥ 2`), the strict-extremum
picture for `acell` is now fully formal:

  `(m − 1, n − 1)` is the unique strict local max,
  `(0, 0)`         is the unique strict local min.

The existence halves below carry the weaker hypothesis `m, n ≥ 1` (the
weakest under which the corner cells `Fin.mk (m − 1)` and `Fin.mk (n − 1)`
type-check); when `m = n = 1` the grid is a singleton and the cell `(0, 0)`
is vacuously both a strict local max and a strict local min.  Downstream
uses combining existence with uniqueness must supply `m, n ≥ 2`.

Both existence proofs reduce to `coneC_max_at` via the cone
identifications `acell_eq_coneC_top` (top-right) and
`negAcell_eq_coneC_bottom` (bottom-left, after a sign flip turning the
cone's max into `acell`'s min).

Results:
* `acell_strictMax_topRight` — `IsStrictLocalMax acell (⟨m − 1, _⟩, ⟨n − 1, _⟩)`.
* `acell_strictMin_origin`   — `IsStrictLocalMin acell (⟨0, _⟩, ⟨0, _⟩)`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **The top-right corner is a strict local maximum of `acell`.**

For `m, n ≥ 1`, the cell `(m − 1, n − 1)` is a strict local max of the
antidiagonal `acell v = v.1.val + v.2.val`: every grid-neighbour has value
exactly one less.  Combined with `acell_unique_max` (same hypothesis), the
top-right corner is the **unique** strict local maximum. -/
theorem acell_strictMax_topRight (hm : 1 ≤ m) (hn : 1 ≤ n) :
    IsStrictLocalMax (acell (m := m) (n := n))
      (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩) := by
  rw [acell_eq_coneC_top hm hn]
  exact coneC_max_at _ _

/-- **The origin is a strict local minimum of `acell`.**

For `m, n ≥ 1`, the cell `(0, 0)` is a strict local min of `acell`: every
grid-neighbour has value exactly one more.  Combined with `acell_unique_min`
(which strengthens the hypothesis to `m, n ≥ 2`), the origin is the
**unique** strict local minimum for `m, n ≥ 2`.

The proof uses the cone identification `−acell = coneC ⟨0, 0⟩ 0` to convert
the existence claim about `acell`'s min into a `coneC_max_at` instance for
the negated function. -/
theorem acell_strictMin_origin (hm : 1 ≤ m) (hn : 1 ≤ n) :
    IsStrictLocalMin (acell (m := m) (n := n))
      (⟨0, by omega⟩, ⟨0, by omega⟩) := by
  -- `-acell = coneC ⟨0,0⟩ 0`, and `coneC_max_at` says this has a strict
  -- local max at its apex `⟨0,0⟩`.  A strict local max of `-acell` at a
  -- point is exactly a strict local min of `acell` at that point.
  have h_max :
      IsStrictLocalMax (fun v : Cell m n => -acell v)
        (⟨0, by omega⟩, ⟨0, by omega⟩) := by
    rw [negAcell_eq_coneC_bottom hm hn]
    exact coneC_max_at _ _
  -- Convert: `(−acell) u = (−acell) o − 1` ↔ `acell u = acell o + 1`.
  intro u hu
  have h_eq := h_max u hu
  -- `h_eq : -acell u = -acell ⟨0,0⟩ - 1`
  -- Goal   : acell u = acell ⟨0,0⟩ + 1
  linarith

end OrigamiCone

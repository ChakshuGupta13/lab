import Mathlib
import OrigamiCone.SequelEdActiveCol

/-!
# Sequel: frozen columns on the height substrate (Task E.δ.e)

The combinatorial arm of paper `lem:uniform` classifies middle columns as
active (carrying an extremum) or frozen (extremum-free).  Paper `lem:frozen`
gives the classification over the `ℤ/3` colouring picture; this module supplies
the `⟸` direction (**frozen ⟹ extremum-free ⟹ inactive**) directly on the
height-function substrate `Cell m n → ℤ`, matching the `activeColumn` predicate
of `SequelEdActiveCol` (and hence the run-count machinery) without passing
through the colouring substrate of `SequelFrozen`.

## The height-substrate frozen condition

An interior column `j` (both horizontal neighbours present, `0 < j` and
`j + 1 < n`) is **frozen** if at every row `i` the horizontal neighbours are
symmetric about the cell value:
`h(i, j-1) + h(i, j+1) = 2 h(i, j)`.

Equivalently the two horizontal height differences `h(i,j-1) - h(i,j)` and
`h(i,j+1) - h(i,j)` sum to zero, so they have opposite signs: one neighbour is
higher, the other lower.  A strict local maximum needs every neighbour lower
and a strict local minimum needs every neighbour higher; either is contradicted
by a single opposite-sign horizontal pair.  Hence a frozen column carries no
extremum and is not active.

## Theorems

* `frozenColumn (h : Cell m n → ℤ) (j : Fin n) : Prop` — the height-substrate
  frozen condition (interior column with symmetric horizontal neighbours).
* **`frozenColumn_not_active`** — frozen ⟹ not active (paper `lem:frozen`
  `⟸`).  Notably needs no `IsHeight` hypothesis: the symmetric-neighbour
  condition alone forces the opposite-sign pair that kills both extremum
  types.

## Role in Task E.δ

Bridges the run-count machinery (`SequelEdActiveCol`, `SequelEdRunCount`,
which reason over `activeColumn` on `Cell m n → ℤ`) to the frozen
classification, on the SAME substrate — closing the substrate-mismatch gap
noted in earlier Task E.δ commits (the `SequelFrozen` colouring picture lives
over `ℕ → ZMod 3`).  The remaining combinatorial work is the frozen-run
CONTRACTION map (each maximal frozen run collapses to one column preserving
height differences) and the finite type enumeration feeding
`degreeBound_assembly`'s `hdecomp`.

## Substrate

Imports only `OrigamiCone.SequelEdActiveCol` (for `activeColumn`, transitively
`Cell`, `adj`, `gdist`, `IsStrictLocalMax/Min`).  Standalone.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
Check with `#print axioms OrigamiCone.Sequel.frozenColumn_not_active`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

/-- An interior column `j` (both horizontal neighbours present) is **frozen**
in the height function `h` if at every row the left and right horizontal
neighbours are symmetric about the cell value:
`h(i, j-1) + h(i, j+1) = 2 h(i, j)`.  On such a column the two horizontal
neighbours go in opposite directions, so no cell of the column is a strict
local extremum. -/
def frozenColumn (h : Cell m n → ℤ) (j : Fin n) : Prop :=
  ∃ (_ : 0 < j.val) (hj1 : j.val + 1 < n),
    ∀ i : Fin m,
      h (i, ⟨j.val - 1, by omega⟩) + h (i, ⟨j.val + 1, hj1⟩) = 2 * h (i, j)

/-- **Frozen ⟹ inactive** (paper `lem:frozen`, `⟸` direction, height
substrate).  A frozen interior column carries no strict local extremum, hence
is not active.

The proof needs no `IsHeight` hypothesis: at each cell the frozen condition
makes the two horizontal height differences sum to zero, so they have opposite
signs.  A strict maximum forces both neighbours strictly lower (both
differences negative) and a strict minimum forces both strictly higher (both
positive); either contradicts the zero sum with opposite signs. -/
theorem frozenColumn_not_active (h : Cell m n → ℤ)
    (j : Fin n) (hfrz : frozenColumn h j) :
    ¬ activeColumn h j := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  rintro ⟨i, hext⟩
  set jL : Fin n := ⟨j.val - 1, by omega⟩ with hjL
  set jR : Fin n := ⟨j.val + 1, hj1⟩ with hjR
  have hadjL : adj ((i, j) : Cell m n) (i, jL) := by
    unfold adj gdist; simp only [hjL]; omega
  have hadjR : adj ((i, j) : Cell m n) (i, jR) := by
    unfold adj gdist; simp only [hjR]; omega
  have hsum : (h (i, jL) - h (i, j)) + (h (i, jR) - h (i, j)) = 0 := by
    have hs := hsym i; omega
  rcases hext with hmax | hmin
  · -- strict max: both horizontal neighbours are `h(i,j) - 1`, sum of diffs = -2.
    have hL := hmax (i, jL) hadjL
    have hR := hmax (i, jR) hadjR
    omega
  · -- strict min: both horizontal neighbours are `h(i,j) + 1`, sum of diffs = +2.
    have hL := hmin (i, jL) hadjL
    have hR := hmin (i, jR) hadjR
    omega

/-- Contrapositive convenience: an active interior column is not frozen. -/
theorem active_not_frozenColumn (h : Cell m n → ℤ) (j : Fin n)
    (hact : activeColumn h j) : ¬ frozenColumn h j :=
  fun hfrz => frozenColumn_not_active h j hfrz hact

end OrigamiCone.Sequel

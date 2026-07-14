import OrigamiCone.SequelEdColumnPartition

/-!
# Sequel: reduced height function type (Task E.δ.h substrate foundation)

The paper's `lem:uniform` classifies height functions on `Gmn` by their
"type": the contraction obtained by collapsing each maximal frozen run of
columns to a single column.  This module defines the abstract type of such
*reduced* height functions (fully-contracted: no two adjacent frozen columns)
as a Lean structure, banking the foundation for the substrate-heavy
contraction map.

## The `ReducedHF` structure

`ReducedHF m` is a canonical height function on `m × W` such that no two
adjacent interior columns are both frozen.  The paper's contraction produces
a `ReducedHF m` from any `Cell m n → ℤ` height function, and the recovery
map extends each frozen column to a run of length ≥ 1 (with the run-length
tuple summing to the deficit `n - W + r`).

Width `W`, height function `h`, and the reducedness condition are structure
fields.  The width bound `W ≤ 2 · numExtrema r.h - 1` (paper `lem:uniform`)
is not proved here; it follows from `numActive_add_numFrozen_eq_n` +
`numFrozenRuns_le_of_isHeight` + the reduced-case identification of frozen
columns with frozen-run starts, deferred to a follow-up module.

## Theorems

* `ReducedHF.isCanonicalHeight` — the full `IsCanonicalHeight` predicate,
  packaged from the structure fields.
* `ReducedHF.frozenColumn_iff_frozenRunStart` — in a reduced h.f., a column
  is frozen iff it is a frozen-run start (each frozen column is isolated,
  hence its own maximal run).
* `ReducedHF.numFrozenColumns_eq_numFrozenRuns` — the count identity from
  the previous iff.
* **`ReducedHF.width_le_two_d_sub_one`** — the paper's key `lem:uniform`
  width bound: `W ≤ 2 · numExtrema r.h - 1`.

## Substrate

Imports `SequelEdColumnPartition` (for `frozenColumn`).  Standalone.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open OrigamiCone

/-- A **reduced height function** on `m` rows: a canonical height function on
`m × W` (with `W ≥ 2`, so at least two boundary columns) such that no two
adjacent interior columns are both frozen.  This is the paper's fully-
contracted "type" in `lem:uniform`, indexing the type-fibers of the
contraction map.

The width `W` is a data field; the paper's bound `W ≤ 2 · numExtrema r.h - 1`
is a derived property, not baked into the definition. -/
structure ReducedHF (m : ℕ) where
  /-- Column count. -/
  W : ℕ
  /-- Width is at least 2 (boundary columns must exist as distinct columns
  for the classification to be meaningful). -/
  hW : 2 ≤ W
  /-- The height function. -/
  h : Cell m W → ℤ
  /-- Height function property: adjacent differences are ±1. -/
  isHeight : IsHeight h
  /-- Canonical: the origin cell has height 0. -/
  isCanonical : ∀ p : Cell m W, p.1.val = 0 → p.2.val = 0 → h p = 0
  /-- Reduced: no two adjacent columns are both frozen. -/
  reduced : ∀ j : Fin W, ∀ (hj1 : j.val + 1 < W),
    ¬ (frozenColumn h j ∧ frozenColumn h ⟨j.val + 1, hj1⟩)

namespace ReducedHF

variable {m : ℕ}

/-- Package the two canonicity fields into the ambient `IsCanonicalHeight`
predicate.  Convenience for downstream lemmas that consume `IsCanonicalHeight`
uniformly. -/
theorem isCanonicalHeight (r : ReducedHF m) : IsCanonicalHeight r.h :=
  ⟨r.isHeight, r.isCanonical⟩

/-- **In a reduced h.f., a column is frozen iff it is a frozen-run start.**
Both directions use the column partition `column_active_or_frozen` (interior
inactive ⟺ frozen) plus the reducedness condition (no adjacent frozens
forces the predecessor of a frozen interior column to be active). -/
theorem frozenColumn_iff_frozenRunStart (r : ReducedHF m) (hm : 0 < m)
    (j : Fin r.W) :
    frozenColumn r.h j ↔ frozenRunStart r.h j := by
  constructor
  · -- frozen ⟹ run start
    intro hfrz
    have hfrz' := hfrz
    obtain ⟨hj0, hj1, _⟩ := hfrz'
    refine ⟨?_, ?_⟩
    · exact frozenColumn_not_active r.h j hfrz
    · right
      refine ⟨hj0, ?_⟩
      by_cases hb : j.val - 1 = 0
      · -- boundary predecessor: active by `firstColumn_active`
        have heq : (⟨j.val - 1, by omega⟩ : Fin r.W) = ⟨0, by omega⟩ :=
          Fin.ext (by simp [hb])
        rw [heq]
        exact firstColumn_active r.h r.isHeight hm r.hW
      · -- interior predecessor: use reducedness to exclude frozen adjacency
        by_contra hna
        have hj1' : (j.val - 1) + 1 < r.W := by omega
        have hfrz_prev : frozenColumn r.h ⟨j.val - 1, by omega⟩ := by
          rcases column_active_or_frozen r.h r.isHeight hm r.hW
              ⟨j.val - 1, by omega⟩ with hact | hfrz
          · exact absurd hact hna
          · exact hfrz
        apply r.reduced ⟨j.val - 1, by omega⟩ hj1'
        refine ⟨hfrz_prev, ?_⟩
        have heq : (⟨(j.val - 1) + 1, hj1'⟩ : Fin r.W) = j :=
          Fin.ext (by simp; omega)
        rw [heq]; exact hfrz
  · -- run start ⟹ frozen
    intro ⟨hna, hstart⟩
    rcases hstart with h0 | ⟨hj0, _⟩
    · exfalso
      apply hna
      have heq : j = (⟨0, by omega⟩ : Fin r.W) := Fin.ext (by simp [h0])
      rw [heq]
      exact firstColumn_active r.h r.isHeight hm r.hW
    · -- interior j: use column partition
      by_cases hlast : j.val + 1 = r.W
      · exfalso
        apply hna
        have heq : j = (⟨r.W - 1, by omega⟩ : Fin r.W) :=
          Fin.ext (by simp; omega)
        rw [heq]
        exact lastColumn_active r.h r.isHeight hm r.hW
      · have hj1 : j.val + 1 < r.W := by have := j.isLt; omega
        exact (frozenColumn_iff_inactive r.h r.isHeight j hj0 hj1 hm).mpr hna

/-- **Number of frozen columns equals number of frozen-run starts** in a
reduced h.f.  Direct consequence of `frozenColumn_iff_frozenRunStart`. -/
theorem numFrozenColumns_eq_numFrozenRuns (r : ReducedHF m) (hm : 0 < m) :
    numFrozenColumns r.h = numFrozenRuns r.h := by
  unfold numFrozenColumns numFrozenRuns
  congr 1
  apply Finset.filter_congr
  intro j _
  exact r.frozenColumn_iff_frozenRunStart hm j

/-- **Width bound** (paper `lem:uniform`).  A reduced h.f. with `d` extrema
has `W ≤ 2 * d - 1` columns.  Proof: `W = #active + #frozen` (partition
identity) `≤ #active + #runs` (frozen ↔ run-start in reduced case)
`≤ d + (d - 1)` (`#active ≤ d` and `#runs + 1 ≤ #active ≤ d`). -/
theorem width_le_two_d_sub_one (r : ReducedHF m) (hm : 0 < m)
    (d : ℕ) (hd : numExtrema r.h = d) :
    r.W ≤ 2 * d - 1 := by
  have h_partition := numActive_add_numFrozen_eq_n r.h r.isHeight hm r.hW
  have h_frozen_runs := r.numFrozenColumns_eq_numFrozenRuns hm
  have h_runs_bound := numFrozenRuns_le_of_isHeight r.h r.isHeight hm r.hW
  have h_active_bound := numActiveColumns_le_of_numExtrema_eq r.h d hd
  omega

end ReducedHF

end OrigamiCone.Sequel

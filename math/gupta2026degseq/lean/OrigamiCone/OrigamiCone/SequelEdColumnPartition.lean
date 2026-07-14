import OrigamiCone.SequelEdFrozenBridge
import OrigamiCone.SequelEdBoundary

/-!
# Sequel: column partition identity (paper §8 classification loop)

The paper's `lem:uniform` proof opens with the classification: each column of a
height function is either **active** (carries an extremum) or **frozen** (an
interior column whose horizontal neighbours are symmetric about the cell
value).  This module closes the classification loop by uniting the previously
formalized pieces:

* boundary columns are active (`lem:boundary`, `SequelEdBoundary`);
* interior columns are frozen iff not active (`lem:frozen` full iff,
  `SequelEdFrozenBridge`);
* frozen columns are not active (`lem:frozen` `⟸`, `SequelEdFrozenCol`).

Together these give the column-count identity `#active + #frozen = n`.

## Theorems

* `decFrozenColumn` (instance) — `frozenColumn` is decidable, powering the
  `numFrozenColumns` counting definition below and any future contraction
  algorithm.
* `numFrozenColumns` — count of frozen columns.
* `boundary_not_frozenColumn` — a boundary column (either endpoint) is never
  frozen, since `frozenColumn` requires both `0 < j.val` and `j.val + 1 < n`.
* `active_disjoint_frozen` — a column is never both active and frozen (any
  substrate, from `frozenColumn_not_active` in `SequelEdFrozenCol`).
* `column_active_or_frozen` — for a height function on `Gmn` with `m ≥ 1` and
  `n ≥ 2`, every column is either active or frozen (boundary→active by
  `lem:boundary`, interior→one-of-two by `lem:frozen` iff).
* **`numActive_add_numFrozen_eq_n`** — the counting identity: the number of
  active columns plus the number of frozen columns equals `n`.  This is the
  substrate-side accounting `lem:uniform` uses when partitioning height
  functions by their active-column pattern.
* `numFrozenColumns_eq_sub_numActive`, `numActiveColumns_le_of_numExtrema_eq`,
  `numFrozenColumns_ge_sub_numExtrema`, `numFrozenRuns_le_of_isHeight`,
  `numFrozenRuns_lt_of_numExtrema_eq` — corollaries.  Once the extremum count
  is pinned (`numExtrema h = d`), the identity forces `n − d ≤ #frozen` and
  `#runs + 1 ≤ d`, packaging the paper's `lem:uniform` numerical intuition.

## Substrate

Imports `SequelEdFrozenBridge` (for the full iff `frozenColumn_iff_inactive`)
and `SequelEdBoundary` (for `firstColumn_active`, `lastColumn_active`).
Standalone; no other Sequel imports.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m n : ℕ}

/-- `frozenColumn` is decidable: its dependent existentials over `Nat`
inequalities and the universal `∀ i : Fin m` reduce to decidable predicates.
Enables `numFrozenColumns` and any future column-scan algorithm. -/
instance decFrozenColumn {h : Cell m n → ℤ} : DecidablePred (frozenColumn h) := by
  intro j; unfold frozenColumn; infer_instance

/-- Number of frozen columns of `h`. -/
def numFrozenColumns (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (frozenColumn h)).card

/-- A **boundary** column (`j.val = 0` or `j.val + 1 = n`) is never frozen: the
`frozenColumn` predicate requires both `0 < j.val` and `j.val + 1 < n`, so
either endpoint is excluded. -/
theorem boundary_not_frozenColumn (h : Cell m n → ℤ)
    (j : Fin n) (hb : j.val = 0 ∨ j.val + 1 = n) :
    ¬ frozenColumn h j := by
  rintro ⟨hj0, hj1, _⟩
  rcases hb with h1 | h1 <;> omega

/-- **Active and frozen are disjoint.**  No column carries an extremum and is
frozen at the same time (immediate from `frozenColumn_not_active`, which needs
no `IsHeight` hypothesis). -/
theorem active_disjoint_frozen (h : Cell m n → ℤ) (j : Fin n)
    (ha : activeColumn h j) (hf : frozenColumn h j) : False :=
  frozenColumn_not_active h j hf ha

/-- **Every column is active or frozen** (paper §8 classification, height
substrate).  For a height function on `Gmn` with `m ≥ 1` and `n ≥ 2`, each
column falls into exactly one of the two categories: boundary columns are
active by `lem:boundary`, and interior columns are active or frozen by
`lem:frozen`'s full biconditional. -/
theorem column_active_or_frozen (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) (j : Fin n) :
    activeColumn h j ∨ frozenColumn h j := by
  by_cases hb : j.val = 0
  · left
    have heq : j = (⟨0, by omega⟩ : Fin n) := Fin.ext (by simp [hb])
    rw [heq]; exact firstColumn_active h hh hm hn
  · by_cases hb' : j.val + 1 = n
    · left
      have heq : j = (⟨n - 1, by omega⟩ : Fin n) :=
        Fin.ext (by simp [Fin.val_mk]; omega)
      rw [heq]; exact lastColumn_active h hh hm hn
    · -- interior: 0 < j.val ∧ j.val + 1 < n
      by_cases ha : activeColumn h j
      · exact Or.inl ha
      · exact Or.inr ((frozenColumn_iff_inactive h hh j
          (by omega) (by omega) hm).mpr ha)

/-- **Column-count identity** (paper §8 classification loop).  For a height
function on `Gmn` with `m ≥ 1` and `n ≥ 2`, the number of active columns plus
the number of frozen columns equals the total column count `n`.  The two sets
partition `Fin n` disjointly (via `active_disjoint_frozen`) and cover it
completely (via `column_active_or_frozen`), so their filter-cards sum to `n`. -/
theorem numActive_add_numFrozen_eq_n (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    numActiveColumns h + numFrozenColumns h = n := by
  unfold numActiveColumns numFrozenColumns
  rw [← Finset.card_union_of_disjoint]
  · have huniv : (Finset.univ.filter (activeColumn h))
        ∪ (Finset.univ.filter (frozenColumn h)) = Finset.univ := by
      ext j
      simp only [Finset.mem_union, Finset.mem_filter, Finset.mem_univ, true_and]
      exact ⟨fun _ => trivial, fun _ => column_active_or_frozen h hh hm hn j⟩
    rw [huniv, Finset.card_univ, Fintype.card_fin]
  · rw [Finset.disjoint_filter]
    intro j _ hactive hfrozen
    exact active_disjoint_frozen h j hactive hfrozen

/-! ## Corollaries — extremum-count bounds

Once the extremum count is pinned (`numExtrema h = d`), the partition identity
combined with `numActiveColumns_le_numExtrema` (paper `lem:uniform` active-column
pigeonhole, `SequelEdActiveCol`) gives concrete lower bounds on the number of
frozen columns and frozen runs.  These are the numerical statements the
substrate-heavy contraction map will consume when partitioning height functions
into types indexed by their active-column pattern.
-/

/-- Subtraction form of the identity: `#frozen = n − #active`. -/
theorem numFrozenColumns_eq_sub_numActive (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    numFrozenColumns h = n - numActiveColumns h := by
  have := numActive_add_numFrozen_eq_n h hh hm hn
  omega

/-- `#active ≤ d` when `#extrema = d` (paper `lem:uniform`, active-column
pigeonhole, hypothesis form). -/
theorem numActiveColumns_le_of_numExtrema_eq (h : Cell m n → ℤ) (d : ℕ)
    (hd : numExtrema h = d) : numActiveColumns h ≤ d := by
  rw [← hd]; exact numActiveColumns_le_numExtrema h

/-- **Most columns are frozen when extrema are few.**  Given `#extrema = d`,
the partition identity forces `n − d ≤ #frozen`.  This is the paper's key
intuition for `lem:uniform`: as `n` grows with `d` fixed, the height function
becomes overwhelmingly frozen. -/
theorem numFrozenColumns_ge_sub_numExtrema (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) (d : ℕ) (hd : numExtrema h = d) :
    n - d ≤ numFrozenColumns h := by
  rw [numFrozenColumns_eq_sub_numActive h hh hm hn]
  have := numActiveColumns_le_of_numExtrema_eq h d hd
  omega

/-- **Frozen-runs bound with automatic boundary discharge** (paper `lem:uniform`
runs pigeonhole, IsHeight form).  Packages `SequelEdActiveCol.numFrozenRuns_le`
by discharging the boundary hypotheses from `lem:boundary`. -/
theorem numFrozenRuns_le_of_isHeight (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) :
    numFrozenRuns h + 1 ≤ numActiveColumns h := by
  apply numFrozenRuns_le h (by omega)
  · exact firstColumn_active h hh hm hn
  · exact lastColumn_active h hh hm hn

/-- **`#runs + 1 ≤ d`** when `#extrema = d` (paper `lem:uniform`, runs bound
under the type-count hypothesis).  Combines `numFrozenRuns_le_of_isHeight` with
the active-column pigeonhole. -/
theorem numFrozenRuns_lt_of_numExtrema_eq (h : Cell m n → ℤ) (hh : IsHeight h)
    (hm : 0 < m) (hn : 1 < n) (d : ℕ) (hd : numExtrema h = d) :
    numFrozenRuns h + 1 ≤ d := by
  have h1 := numFrozenRuns_le_of_isHeight h hh hm hn
  have h2 := numActiveColumns_le_of_numExtrema_eq h d hd
  omega

end OrigamiCone.Sequel

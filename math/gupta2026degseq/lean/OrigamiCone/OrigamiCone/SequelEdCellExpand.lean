import OrigamiCone.SequelEdContractAt
import OrigamiCone.SequelEd

/-!
# Sequel: cell-expansion map skipping the frozen column (E.δ.h contract map)

The atomic `contractAt` step (`SequelEdContractAt`) skips a frozen column `j`
in a height function on `Cell m n`, producing one on `Cell m (n - 1)`.  This
module supplies the paired **cell-expansion** map `cellExpand : Cell m (n-1) →
Cell m n` which injects `contractAt`'s domain back into `h`'s domain, skipping
column `j`.  The map is used to establish extremum count preservation
(`contractAt_numExtrema_eq`, deferred).

## Theorems

* `frozenColumn_no_extremum` — pointwise form: no cell of a frozen column is a
  strict local extremum.
* `cellExpand` — the injection.  Maps `(i, j')` to `(i, j'.val)` when
  `j'.val < j.val` and to `(i, j'.val + 1)` when `j'.val ≥ j.val`.
* `cellExpand_left`, `cellExpand_right` — value equations.
* **`cellExpand_ne_j`** — the image misses column `j`.
* **`cellExpand_injective`** — the map is injective.

## Substrate

Imports `SequelEdContractAt`.  Standalone.

No `sorry`.  Axioms: `[propext, Quot.sound]` (no `Classical.choice`).
-/

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

/-- **Frozen columns have no extrema** (pointwise form of
`frozenColumn_not_active`).  Any cell in a frozen column fails to be a strict
local extremum. -/
theorem frozenColumn_no_extremum (h : Cell m n → ℤ) (j : Fin n) (hfrz : frozenColumn h j) :
    ∀ i : Fin m, ¬ IsStrictLocalExtremum h (i, j) := by
  intro i hext
  exact frozenColumn_not_active h j hfrz ⟨i, hext⟩

/-- **Cell expansion** paired with `contractAt`: the injection
`Cell m (n - 1) → Cell m n` skipping column `j`.  The image misses exactly the
column `j` — the removed frozen column — matching `contractAt`'s domain
reduction. -/
def cellExpand (j : Fin n) (p : Cell m (n - 1)) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n) :
    Cell m n :=
  if p.2.val < j.val then
    (p.1, ⟨p.2.val, by have := p.2.isLt; omega⟩)
  else
    (p.1, ⟨p.2.val + 1, by have := p.2.isLt; omega⟩)

/-- Value of `cellExpand` on the "left" (column before `j`). -/
theorem cellExpand_left (j : Fin n) (p : Cell m (n - 1)) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n)
    (hp : p.2.val < j.val) :
    cellExpand j p hj0 hj1 = (p.1, ⟨p.2.val, by have := p.2.isLt; omega⟩) := by
  unfold cellExpand; simp [hp]

/-- Value of `cellExpand` on the "right" (column at or after `j`). -/
theorem cellExpand_right (j : Fin n) (p : Cell m (n - 1)) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n)
    (hp : ¬ p.2.val < j.val) :
    cellExpand j p hj0 hj1 = (p.1, ⟨p.2.val + 1, by have := p.2.isLt; omega⟩) := by
  unfold cellExpand; simp [hp]

/-- `cellExpand`'s image never lands at column `j`.  Both branches produce
either `< j.val` or `> j.val`, never `= j.val`. -/
theorem cellExpand_ne_j (j : Fin n) (p : Cell m (n - 1)) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n) :
    (cellExpand j p hj0 hj1).2 ≠ j := by
  intro hc
  have hval := congrArg Fin.val hc
  unfold cellExpand at hval
  by_cases hlt : p.2.val < j.val
  · simp [hlt] at hval; omega
  · simp [hlt] at hval; omega

/-- `cellExpand` is injective.  Distinct `(i, j')` map to distinct grid cells
because the map preserves the row and the column-value formula is monotone. -/
theorem cellExpand_injective (j : Fin n) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n) :
    Function.Injective (cellExpand (m := m) j · hj0 hj1) := by
  intro p q hpq
  have hval1 := congrArg Prod.fst hpq
  have hval2 := congrArg Prod.snd hpq
  have hval2v : (cellExpand j p hj0 hj1).2.val = (cellExpand j q hj0 hj1).2.val :=
    congrArg Fin.val hval2
  unfold cellExpand at hval1 hval2v
  by_cases hlp : p.2.val < j.val
  · by_cases hlq : q.2.val < j.val
    · simp [hlp, hlq] at hval1 hval2v
      exact Prod.ext hval1 (Fin.ext hval2v)
    · simp [hlp, hlq] at hval1 hval2v
      omega
  · by_cases hlq : q.2.val < j.val
    · simp [hlp, hlq] at hval1 hval2v
      omega
    · simp [hlp, hlq] at hval1 hval2v
      exact Prod.ext hval1 (Fin.ext (by omega))

/-- `contractAt` preserves canonicity: if `h` is canonical then so is
`contractAt h hm j hfrz`.  The origin of the new grid `(⟨0, hm⟩, ⟨0, _⟩)` has
`p.2.val = 0 < j.val`, so `contractAt_left` gives the value as
`h(⟨0, hm⟩, ⟨0, _⟩) = 0` by `h`'s canonicity. -/
theorem contractAt_isCanonicalHeight (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (hcanon : IsCanonicalHeight h) :
    IsCanonicalHeight (contractAt h hm j hfrz) := by
  refine ⟨contractAt_isHeight h hh hm j hfrz, ?_⟩
  intro p hp1 hp2
  have hj0 : 0 < j.val := hfrz.1
  have hval : contractAt h hm j hfrz p
      = h (p.1, ⟨p.2.val, by have := p.2.isLt; have hj1 := hfrz.2.1; omega⟩) := by
    apply contractAt_left h hm j hfrz p
    rw [hp2]; exact hj0
  rw [hval]
  exact hcanon.2 _ hp1 (by simp [hp2])

/-- **`cellExpand` surjects onto `q.2 ≠ j`.**  Every cell in `Cell m n` not at
column `j` is the `cellExpand`-image of some `p : Cell m (n-1)`.  Combined with
`cellExpand_ne_j`, this makes `cellExpand` a bijection between `Cell m (n-1)`
and `{q : Cell m n | q.2 ≠ j}`. -/
theorem cellExpand_surjOn (j : Fin n) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n)
    (q : Cell m n) (hqj : q.2 ≠ j) :
    ∃ p : Cell m (n - 1), cellExpand j p hj0 hj1 = q := by
  by_cases hlt : q.2.val < j.val
  · refine ⟨(q.1, ⟨q.2.val, by have := q.2.isLt; omega⟩), ?_⟩
    unfold cellExpand
    simp [hlt]
  · have hqjval : q.2.val ≠ j.val := fun h => hqj (Fin.ext h)
    have hgt : q.2.val > j.val := by omega
    refine ⟨(q.1, ⟨q.2.val - 1, by have := q.2.isLt; omega⟩), ?_⟩
    have hpge : ¬ q.2.val - 1 < j.val := by omega
    unfold cellExpand
    simp only [hpge, if_false]
    exact Prod.ext rfl (Fin.ext (by simp only [Fin.val_mk]; omega))

/-- **Diff preservation for column-preserving cell pairs.**  If `p.2.val =
q.2.val` (both in the same branch relative to `j`), the `contractAt`-difference
equals the `h`-difference at the `cellExpand`-images.  The `frozenSlope` shift
cancels because both cells are in the same branch.

Foundation for the eventual extremum-preservation bijection: this closes the
row-neighbour direction of the diff match (adjacent rows same column). -/
theorem contractAt_diff_row (h : Cell m n → ℤ) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j)
    (p q : Cell m (n - 1)) (hp2v : p.2.val = q.2.val) :
    contractAt h hm j hfrz q - contractAt h hm j hfrz p
      = h (cellExpand j q hfrz.1 hfrz.2.1) - h (cellExpand j p hfrz.1 hfrz.2.1) := by
  obtain ⟨hj0, hj1, _⟩ := hfrz
  by_cases hlt : p.2.val < j.val
  · have hlt' : q.2.val < j.val := hp2v ▸ hlt
    rw [contractAt_left _ _ _ _ p hlt, contractAt_left _ _ _ _ q hlt']
    rw [cellExpand_left _ _ hj0 hj1 hlt, cellExpand_left _ _ hj0 hj1 hlt']
  · have hlt' : ¬ q.2.val < j.val := hp2v ▸ hlt
    rw [contractAt_right _ _ _ _ p hlt, contractAt_right _ _ _ _ q hlt']
    rw [cellExpand_right _ _ hj0 hj1 hlt, cellExpand_right _ _ hj0 hj1 hlt']
    ring

/-- **Right-side slope of a frozen column.**  The `+1`-side slope also equals
`frozenSlope`; follows from the frozen sum identity `2 h(i, j) = h(i, j-1) +
h(i, j+1)` plus the left-side slope `frozenSlope_eq`. -/
theorem frozenSlope_eq_right (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (i : Fin m) :
    h (i, ⟨j.val + 1, hfrz.2.1⟩) - h (i, j) = frozenSlope h hm j hfrz := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  have hL := frozenSlope_eq h hh hm j ⟨hj0, hj1, hsym⟩ i
  have hs := hsym i
  linarith

/-- **Diff preservation for same-branch cell pairs.**  For any `p, q` where
both are on the same branch of the frozen column (either both `.2.val < j.val`
or both `≥ j.val`), the `contractAt`-diff equals the `h`-diff at the
`cellExpand`-images.  Generalizes `contractAt_diff_row` (which required equal
columns, a special case where the same-branch condition is automatic). -/
theorem contractAt_diff_same_branch (h : Cell m n → ℤ) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p q : Cell m (n - 1))
    (hbr : (p.2.val < j.val ∧ q.2.val < j.val) ∨ (¬ p.2.val < j.val ∧ ¬ q.2.val < j.val)) :
    contractAt h hm j hfrz q - contractAt h hm j hfrz p
      = h (cellExpand j q hfrz.1 hfrz.2.1) - h (cellExpand j p hfrz.1 hfrz.2.1) := by
  obtain ⟨hj0, hj1, _⟩ := hfrz
  rcases hbr with ⟨hpL, hqL⟩ | ⟨hpR, hqR⟩
  · rw [contractAt_left _ _ _ _ p hpL, contractAt_left _ _ _ _ q hqL]
    rw [cellExpand_left _ _ hj0 hj1 hpL, cellExpand_left _ _ hj0 hj1 hqL]
  · rw [contractAt_right _ _ _ _ p hpR, contractAt_right _ _ _ _ q hqR]
    rw [cellExpand_right _ _ hj0 hj1 hpR, cellExpand_right _ _ hj0 hj1 hqR]
    ring

/-- **Seam diff, left-to-right.**  For same-row cells `p` on the left of `j`
(with `p.2.val + 1 = j.val`) and `q` on the right of `j` (with `q.2.val =
j.val`) — adjacent across the seam in `Cell m (n-1)` — the `contractAt`-diff
matches the `h`-diff from `cellExpand p` to the frozen cell `(p.1, j)`.  The
h-image on the right is NOT `cellExpand q` (which would give a diff of `2k`,
because it lies 2 columns away from `cellExpand p`), but the frozen cell
itself, giving the correct diff of `k`. -/
theorem contractAt_diff_seam_LtoR (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j)
    (p q : Cell m (n - 1)) (hp1 : p.1 = q.1)
    (hpv : p.2.val + 1 = j.val) (hqv : q.2.val = j.val) :
    contractAt h hm j hfrz q - contractAt h hm j hfrz p
      = h (p.1, j) - h (cellExpand j p hfrz.1 hfrz.2.1) := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  have hpL : p.2.val < j.val := by omega
  have hqR : ¬ q.2.val < j.val := by omega
  rw [contractAt_left _ _ _ _ p hpL, contractAt_right _ _ _ _ q hqR]
  rw [cellExpand_left _ _ hj0 hj1 hpL]
  have hqFin : (⟨q.2.val + 1, by have := q.2.isLt; omega⟩ : Fin n) = ⟨j.val + 1, hj1⟩ := by
    apply Fin.ext; simp only [Fin.val_mk]; omega
  have hpFin : (⟨p.2.val, by have := p.2.isLt; omega⟩ : Fin n)
      = ⟨j.val - 1, by omega⟩ := by
    apply Fin.ext; simp only [Fin.val_mk]; omega
  rw [hqFin, hpFin, ← hp1]
  have hR := frozenSlope_eq_right h hh hm j ⟨hj0, hj1, hsym⟩ p.1
  linarith

/-- **Seam diff, right-to-left.**  Mirror of `contractAt_diff_seam_LtoR`: `p`
on the right of `j` (with `p.2.val = j.val`), `q` on the left (with `q.2.val +
1 = j.val`).  Same conclusion: the h-image on the left of `cellExpand p` is
the frozen cell `(p.1, j)`, not `cellExpand q`. -/
theorem contractAt_diff_seam_RtoL (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j)
    (p q : Cell m (n - 1)) (hp1 : p.1 = q.1)
    (hpv : p.2.val = j.val) (hqv : q.2.val + 1 = j.val) :
    contractAt h hm j hfrz q - contractAt h hm j hfrz p
      = h (p.1, j) - h (cellExpand j p hfrz.1 hfrz.2.1) := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  have hpR : ¬ p.2.val < j.val := by omega
  have hqL : q.2.val < j.val := by omega
  rw [contractAt_right _ _ _ _ p hpR, contractAt_left _ _ _ _ q hqL]
  rw [cellExpand_right _ _ hj0 hj1 hpR]
  have hqFin : (⟨q.2.val, by have := q.2.isLt; omega⟩ : Fin n)
      = ⟨j.val - 1, by omega⟩ := by
    apply Fin.ext; simp only [Fin.val_mk]; omega
  have hpFin : (⟨p.2.val + 1, by have := p.2.isLt; omega⟩ : Fin n)
      = ⟨j.val + 1, hj1⟩ := by
    apply Fin.ext; simp only [Fin.val_mk]; omega
  rw [hqFin, hpFin, ← hp1]
  have hL := frozenSlope_eq h hh hm j ⟨hj0, hj1, hsym⟩ p.1
  linarith

end OrigamiCone.Sequel

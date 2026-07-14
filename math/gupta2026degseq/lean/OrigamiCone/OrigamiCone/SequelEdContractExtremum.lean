import OrigamiCone.SequelEdCellExpand

/-!
# Sequel: extremum bijection at a frozen column (Task E.δ.h contract map)

The atomic `contractAt` step (`SequelEdContractAt`) produces a valid height
function on the smaller grid `Cell m (n - 1)`.  For the paper's `lem:uniform`
proof, the crucial property is that this atomic step preserves the number of
strict local extrema.  This module builds the pointwise extremum bijection
that will feed the count-preservation `Finset.card_bij` argument.

## Theorems

* `adj_row_or_col` — every adjacency decomposes into row-adj (same column) or
  col-adj (same row).
* `adj_of_row_col_diff`, `adj_of_col_row_diff` — building `adj` from
  same-row / same-col plus a natAbs-1 difference on the other axis.
* `cellExpand_adj_same_branch` — `cellExpand` preserves adjacency for
  same-branch pairs.
* `cellExpand_adj_frozenCell` — `cellExpand p` is adjacent to the frozen cell
  `(p.1, j)` when `p` is at the seam boundary.
* **`contractAt_isSLM_of_h_max`** — if `cellExpand p` is a strict local max of
  `h`, then `p` is a strict local max of `contractAt`.
* **`contractAt_isSLm_of_h_min`** — symmetric for strict local min.
* **`h_max_of_contractAt_isSLM`** — the forward direction: if `p` is a strict
  local max of `contractAt`, then `cellExpand p` is a strict local max of `h`.
* **`h_min_of_contractAt_isSLm`** — symmetric for strict local min.
* **`contractAt_isSLE_iff`** — the full pointwise extremum bijection.
* **`contractAt_numExtrema_eq`** — extremum count preservation.

## Deferred to a follow-up

* Iterated `contract` to full paper `lem:uniform`.

## Substrate

Imports `SequelEdCellExpand` (for `cellExpand`, `contractAt`, and the diff
lemmas).  Standalone.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

/-- Auxiliary: decompose `adj p q` in `Cell m N` into col-adj + row-adj cases. -/
lemma adj_row_or_col {N : ℕ} {p q : Cell m N} (h : adj p q) :
    (p.1 = q.1 ∧ ((p.2.val : ℤ) - q.2.val).natAbs = 1) ∨
    (p.2 = q.2 ∧ ((p.1.val : ℤ) - q.1.val).natAbs = 1) := by
  unfold adj gdist at h
  set a := ((p.1.val : ℤ) - q.1.val).natAbs with ha
  set b := ((p.2.val : ℤ) - q.2.val).natAbs with hb
  have hab : (a : ℤ) + b = 1 := by exact_mod_cast h
  by_cases h1 : a = 0
  · left
    have hbv : b = 1 := by omega
    refine ⟨?_, hbv⟩
    apply Fin.ext
    have hz : ((p.1.val : ℤ) - q.1.val) = 0 := by
      have := Int.natAbs_eq_zero.mp h1; omega
    omega
  · right
    have hav : a = 1 := by omega
    have hbv : b = 0 := by omega
    refine ⟨?_, hav⟩
    apply Fin.ext
    have hz : ((p.2.val : ℤ) - q.2.val) = 0 := by
      have := Int.natAbs_eq_zero.mp hbv; omega
    omega

/-- Building `adj` from equal-row + col-diff-natAbs-1. -/
lemma adj_of_row_col_diff {N : ℕ} {p q : Cell m N}
    (hrow : p.1 = q.1) (hcol : ((p.2.val : ℤ) - q.2.val).natAbs = 1) : adj p q := by
  have hrowV : p.1.val = q.1.val := by rw [hrow]
  unfold adj gdist
  omega

/-- Building `adj` from equal-col + row-diff-natAbs-1. -/
lemma adj_of_col_row_diff {N : ℕ} {p q : Cell m N}
    (hcol : p.2 = q.2) (hrow : ((p.1.val : ℤ) - q.1.val).natAbs = 1) : adj p q := by
  have hcolV : p.2.val = q.2.val := by rw [hcol]
  unfold adj gdist
  omega

/-- `cellExpand` preserves adjacency for same-branch pairs. -/
lemma cellExpand_adj_same_branch (j : Fin n) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n)
    (p q : Cell m (n - 1)) (hadj : adj p q)
    (hbr : (p.2.val < j.val ∧ q.2.val < j.val) ∨ (¬ p.2.val < j.val ∧ ¬ q.2.val < j.val)) :
    adj (cellExpand j p hj0 hj1) (cellExpand j q hj0 hj1) := by
  rcases hbr with ⟨hpL, hqL⟩ | ⟨hpR, hqR⟩
  · rw [cellExpand_left _ _ hj0 hj1 hpL, cellExpand_left _ _ hj0 hj1 hqL]
    unfold adj gdist at hadj ⊢
    push_cast at hadj ⊢
    exact hadj
  · rw [cellExpand_right _ _ hj0 hj1 hpR, cellExpand_right _ _ hj0 hj1 hqR]
    unfold adj gdist at hadj ⊢
    push_cast at hadj ⊢
    convert hadj using 3
    omega

/-- `cellExpand p` is adjacent to the frozen cell `(p.1, j)` iff `p` is at the
seam boundary (either `p.2.val + 1 = j.val` on the left or `p.2.val = j.val` on
the right). -/
lemma cellExpand_adj_frozenCell (j : Fin n) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n)
    (p : Cell m (n - 1)) (hpb : p.2.val + 1 = j.val ∨ p.2.val = j.val) :
    adj (cellExpand j p hj0 hj1) (p.1, j) := by
  rcases hpb with hL | hR
  · have hpL : p.2.val < j.val := by omega
    rw [cellExpand_left _ _ hj0 hj1 hpL]
    refine adj_of_row_col_diff rfl ?_
    simp only [Fin.val_mk]
    omega
  · have hpR : ¬ p.2.val < j.val := by omega
    rw [cellExpand_right _ _ hj0 hj1 hpR]
    refine adj_of_row_col_diff rfl ?_
    simp only [Fin.val_mk]
    omega

/-- **Reverse direction of extremum bijection (strict local max).**  If
`cellExpand p` is a strict local max of `h`, then `p` is a strict local max of
`contractAt`.

Case analysis on `adj p q`: row-adjacent (same column) or column-adjacent
(same row).  In the row-adj case, both `p` and `q` have the same column so
they're trivially on the same branch of `j`; use `contractAt_diff_same_branch`
+ `cellExpand_adj_same_branch`.  In the col-adj case, either both cells are
on the same branch (again same-branch path) or they cross the seam (use
`contractAt_diff_seam_LtoR` / `RtoL` + `cellExpand_adj_frozenCell` applied to
`(p.1, j)`). -/
theorem contractAt_isSLM_of_h_max (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p : Cell m (n - 1))
    (hmax : IsStrictLocalMax h (cellExpand j p hfrz.1 hfrz.2.1)) :
    IsStrictLocalMax (contractAt h hm j hfrz) p := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  intro q hqadj
  rcases adj_row_or_col hqadj with ⟨hp1eq, hb⟩ | ⟨hp2eq, ha⟩
  · -- Col-adj
    by_cases hbr : (p.2.val < j.val ∧ q.2.val < j.val) ∨ (¬ p.2.val < j.val ∧ ¬ q.2.val < j.val)
    · -- Same branch
      have hdiff := contractAt_diff_same_branch h hm j ⟨hj0, hj1, hsym⟩ p q hbr
      have hexp_adj := cellExpand_adj_same_branch j hj0 hj1 p q hqadj hbr
      have hval := hmax _ hexp_adj
      linarith
    · -- Seam crossing
      push_neg at hbr
      obtain ⟨hnAA, hnBB⟩ := hbr
      by_cases hpL : p.2.val < j.val
      · -- Seam LtoR
        have hqR : ¬ q.2.val < j.val := by
          have := hnAA hpL; omega
        have hqeq : q.2.val = p.2.val + 1 := by
          have hnat : ((p.2.val : ℤ) - q.2.val).natAbs = 1 := hb; omega
        have hpv : p.2.val + 1 = j.val := by omega
        have hqv : q.2.val = j.val := by omega
        have hdiff := contractAt_diff_seam_LtoR h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1eq hpv hqv
        have hexp_adj := cellExpand_adj_frozenCell j hj0 hj1 p (Or.inl hpv)
        have hval := hmax (p.1, j) hexp_adj
        linarith
      · -- Seam RtoL
        push_neg at hpL
        have hqL : q.2.val < j.val := hnBB hpL
        have hpvv : p.2.val = q.2.val + 1 := by
          have hnat : ((p.2.val : ℤ) - q.2.val).natAbs = 1 := hb; omega
        have hpv : p.2.val = j.val := by omega
        have hqv : q.2.val + 1 = j.val := by omega
        have hdiff := contractAt_diff_seam_RtoL h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1eq hpv hqv
        have hexp_adj := cellExpand_adj_frozenCell j hj0 hj1 p (Or.inr hpv)
        have hval := hmax (p.1, j) hexp_adj
        linarith
  · -- Row-adj: same column, trivially same branch
    have hp2v : p.2.val = q.2.val := by rw [hp2eq]
    have hbr : (p.2.val < j.val ∧ q.2.val < j.val) ∨ (¬ p.2.val < j.val ∧ ¬ q.2.val < j.val) := by
      by_cases hpL : p.2.val < j.val
      · exact Or.inl ⟨hpL, hp2v ▸ hpL⟩
      · exact Or.inr ⟨hpL, hp2v ▸ hpL⟩
    have hdiff := contractAt_diff_same_branch h hm j ⟨hj0, hj1, hsym⟩ p q hbr
    have hexp_adj := cellExpand_adj_same_branch j hj0 hj1 p q hqadj hbr
    have hval := hmax _ hexp_adj
    linarith

/-- **Reverse direction of extremum bijection (strict local min).**  Symmetric
version of `contractAt_isSLM_of_h_max`. -/
theorem contractAt_isSLm_of_h_min (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p : Cell m (n - 1))
    (hmin : IsStrictLocalMin h (cellExpand j p hfrz.1 hfrz.2.1)) :
    IsStrictLocalMin (contractAt h hm j hfrz) p := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  intro q hqadj
  rcases adj_row_or_col hqadj with ⟨hp1eq, hb⟩ | ⟨hp2eq, ha⟩
  · by_cases hbr : (p.2.val < j.val ∧ q.2.val < j.val) ∨ (¬ p.2.val < j.val ∧ ¬ q.2.val < j.val)
    · have hdiff := contractAt_diff_same_branch h hm j ⟨hj0, hj1, hsym⟩ p q hbr
      have hexp_adj := cellExpand_adj_same_branch j hj0 hj1 p q hqadj hbr
      have hval := hmin _ hexp_adj
      linarith
    · push_neg at hbr
      obtain ⟨hnAA, hnBB⟩ := hbr
      by_cases hpL : p.2.val < j.val
      · have hqR : ¬ q.2.val < j.val := by
          have := hnAA hpL; omega
        have hqeq : q.2.val = p.2.val + 1 := by
          have hnat : ((p.2.val : ℤ) - q.2.val).natAbs = 1 := hb; omega
        have hpv : p.2.val + 1 = j.val := by omega
        have hqv : q.2.val = j.val := by omega
        have hdiff := contractAt_diff_seam_LtoR h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1eq hpv hqv
        have hexp_adj := cellExpand_adj_frozenCell j hj0 hj1 p (Or.inl hpv)
        have hval := hmin (p.1, j) hexp_adj
        linarith
      · push_neg at hpL
        have hqL : q.2.val < j.val := hnBB hpL
        have hpvv : p.2.val = q.2.val + 1 := by
          have hnat : ((p.2.val : ℤ) - q.2.val).natAbs = 1 := hb; omega
        have hpv : p.2.val = j.val := by omega
        have hqv : q.2.val + 1 = j.val := by omega
        have hdiff := contractAt_diff_seam_RtoL h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1eq hpv hqv
        have hexp_adj := cellExpand_adj_frozenCell j hj0 hj1 p (Or.inr hpv)
        have hval := hmin (p.1, j) hexp_adj
        linarith
  · have hp2v : p.2.val = q.2.val := by rw [hp2eq]
    have hbr : (p.2.val < j.val ∧ q.2.val < j.val) ∨ (¬ p.2.val < j.val ∧ ¬ q.2.val < j.val) := by
      by_cases hpL : p.2.val < j.val
      · exact Or.inl ⟨hpL, hp2v ▸ hpL⟩
      · exact Or.inr ⟨hpL, hp2v ▸ hpL⟩
    have hdiff := contractAt_diff_same_branch h hm j ⟨hj0, hj1, hsym⟩ p q hbr
    have hexp_adj := cellExpand_adj_same_branch j hj0 hj1 p q hqadj hbr
    have hval := hmin _ hexp_adj
    linarith

/-- **Reverse direction of extremum bijection.**  If `cellExpand p` is a strict
local extremum of `h`, then `p` is a strict local extremum of `contractAt`. -/
theorem contractAt_isSLE_of_h_extremum (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p : Cell m (n - 1))
    (hext : IsStrictLocalExtremum h (cellExpand j p hfrz.1 hfrz.2.1)) :
    IsStrictLocalExtremum (contractAt h hm j hfrz) p := by
  rcases hext with hmax | hmin
  · exact Or.inl (contractAt_isSLM_of_h_max h hh hm j hfrz p hmax)
  · exact Or.inr (contractAt_isSLm_of_h_min h hh hm j hfrz p hmin)

/-- Given `adj (cellExpand p) u` where `u` is not in the frozen column,
extract a cell `q ∈ Cell m (n-1)` such that `cellExpand q = u`, `adj p q`, and
`p, q` are on the same branch.  Different branches are impossible under
`adj (cellExpand p) u` because the frozen column separates them by at least 2. -/
lemma adj_cellExpand_of_not_frozen (j : Fin n) (hj0 : 0 < j.val) (hj1 : j.val + 1 < n)
    (p : Cell m (n - 1)) (u : Cell m n)
    (hadj : adj (cellExpand j p hj0 hj1) u) (hufrz : u.2 ≠ j) :
    ∃ q : Cell m (n - 1), cellExpand j q hj0 hj1 = u ∧ adj p q ∧
    ((p.2.val < j.val ∧ q.2.val < j.val) ∨ (¬ p.2.val < j.val ∧ ¬ q.2.val < j.val)) := by
  obtain ⟨q, hq⟩ := cellExpand_surjOn j hj0 hj1 u hufrz
  refine ⟨q, hq, ?_, ?_⟩
  · rw [← hq] at hadj
    by_cases hpL : p.2.val < j.val
    · by_cases hqL : q.2.val < j.val
      · rw [cellExpand_left _ _ hj0 hj1 hpL, cellExpand_left _ _ hj0 hj1 hqL] at hadj
        unfold adj gdist at hadj ⊢
        push_cast at hadj ⊢
        exact hadj
      · exfalso
        push_neg at hqL
        have hqR : ¬ q.2.val < j.val := by omega
        rw [cellExpand_left _ _ hj0 hj1 hpL, cellExpand_right _ _ hj0 hj1 hqR] at hadj
        unfold adj gdist at hadj
        simp only [Prod.fst, Prod.snd, Fin.val_mk] at hadj
        have h_col : ((p.2.val : ℤ) - (q.2.val + 1)).natAbs ≥ 2 := by
          have hle : (p.2.val : ℤ) - (q.2.val + 1) ≤ -2 := by push_cast; omega
          omega
        omega
    · push_neg at hpL
      have hpR : ¬ p.2.val < j.val := by omega
      by_cases hqL : q.2.val < j.val
      · exfalso
        rw [cellExpand_right _ _ hj0 hj1 hpR, cellExpand_left _ _ hj0 hj1 hqL] at hadj
        unfold adj gdist at hadj
        simp only [Prod.fst, Prod.snd, Fin.val_mk] at hadj
        have h_col : ((p.2.val + 1 : ℤ) - q.2.val).natAbs ≥ 2 := by
          have hge : (p.2.val + 1 : ℤ) - q.2.val ≥ 2 := by push_cast; omega
          omega
        omega
      · push_neg at hqL
        have hqR : ¬ q.2.val < j.val := by omega
        rw [cellExpand_right _ _ hj0 hj1 hpR, cellExpand_right _ _ hj0 hj1 hqR] at hadj
        unfold adj gdist at hadj ⊢
        push_cast at hadj ⊢
        convert hadj using 3
        omega
  · rw [← hq] at hadj
    by_cases hpL : p.2.val < j.val
    · by_cases hqL : q.2.val < j.val
      · exact Or.inl ⟨hpL, hqL⟩
      · exfalso
        push_neg at hqL
        have hqR : ¬ q.2.val < j.val := by omega
        rw [cellExpand_left _ _ hj0 hj1 hpL, cellExpand_right _ _ hj0 hj1 hqR] at hadj
        unfold adj gdist at hadj
        simp only [Prod.fst, Prod.snd, Fin.val_mk] at hadj
        have h_col : ((p.2.val : ℤ) - (q.2.val + 1)).natAbs ≥ 2 := by
          have hle : (p.2.val : ℤ) - (q.2.val + 1) ≤ -2 := by push_cast; omega
          omega
        omega
    · push_neg at hpL
      have hpR : ¬ p.2.val < j.val := by omega
      by_cases hqL : q.2.val < j.val
      · exfalso
        rw [cellExpand_right _ _ hj0 hj1 hpR, cellExpand_left _ _ hj0 hj1 hqL] at hadj
        unfold adj gdist at hadj
        simp only [Prod.fst, Prod.snd, Fin.val_mk] at hadj
        have h_col : ((p.2.val + 1 : ℤ) - q.2.val).natAbs ≥ 2 := by
          have hge : (p.2.val + 1 : ℤ) - q.2.val ≥ 2 := by push_cast; omega
          omega
        omega
      · push_neg at hqL
        have hqR : ¬ q.2.val < j.val := by omega
        exact Or.inr ⟨hpR, hqR⟩

/-- **Forward direction of extremum bijection (strict local max).**  If `p` is
a strict local max of `contractAt`, then `cellExpand p` is a strict local max
of `h`.

Case analysis on `adj (cellExpand p) u`: if `u.2 = j` (u is the frozen cell
`(p.1, j)`), then `p` is at the seam boundary and we use
`contractAt_diff_seam_LtoR`/`RtoL` with `q = (p.1, ⟨j.val, _⟩)` or
`q = (p.1, ⟨j.val - 1, _⟩)` in `Cell m (n-1)`.  If `u.2 ≠ j`, we use
`adj_cellExpand_of_not_frozen` to find `q` with `cellExpand q = u` and same
branch as `p`, then apply `contractAt_diff_same_branch`. -/
theorem h_max_of_contractAt_isSLM (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p : Cell m (n - 1))
    (hmax : IsStrictLocalMax (contractAt h hm j hfrz) p) :
    IsStrictLocalMax h (cellExpand j p hfrz.1 hfrz.2.1) := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  intro u hadj
  by_cases hufrz : u.2 = j
  · have hcellne : (cellExpand j p hj0 hj1).2 ≠ j := cellExpand_ne_j j p hj0 hj1
    rcases adj_row_or_col hadj with ⟨hrow_eq, hcol⟩ | ⟨hcol_eq, _⟩
    · have hcellp1 : (cellExpand j p hj0 hj1).1 = p.1 := by
        by_cases hpL : p.2.val < j.val
        · rw [cellExpand_left _ _ hj0 hj1 hpL]
        · rw [cellExpand_right _ _ hj0 hj1 (by omega : ¬ p.2.val < j.val)]
      have hu1 : u.1 = p.1 := hcellp1 ▸ hrow_eq.symm
      have hueq : u = (p.1, j) := Prod.ext hu1 hufrz
      by_cases hpL : p.2.val < j.val
      · have hcellp2v : (cellExpand j p hj0 hj1).2.val = p.2.val := by
          rw [cellExpand_left _ _ hj0 hj1 hpL]
        have hpv : p.2.val + 1 = j.val := by
          have hcolShow := hcol
          rw [hufrz, hcellp2v] at hcolShow
          have h_p_lt : (p.2.val : ℤ) < j.val := by push_cast; omega
          have : (p.2.val : ℤ) - j.val < 0 := by omega
          omega
        let q : Cell m (n - 1) := (p.1, ⟨j.val, by omega⟩)
        have hp1q : p.1 = q.1 := rfl
        have hqv : q.2.val = j.val := rfl
        have hqadj : adj p q := by
          refine adj_of_row_col_diff rfl ?_
          show ((p.2.val : ℤ) - j.val).natAbs = 1
          push_cast; omega
        have hdiff := contractAt_diff_seam_LtoR h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1q hpv hqv
        have hval := hmax q hqadj
        rw [hueq]
        linarith
      · push_neg at hpL
        have hpR : ¬ p.2.val < j.val := by omega
        have hcellp2v : (cellExpand j p hj0 hj1).2.val = p.2.val + 1 := by
          rw [cellExpand_right _ _ hj0 hj1 hpR]
        have hpv : p.2.val = j.val := by
          have hcolShow := hcol
          rw [hufrz, hcellp2v] at hcolShow
          have h_p_ge : (p.2.val : ℤ) ≥ j.val := by push_cast; omega
          have : (p.2.val + 1 : ℤ) - j.val > 0 := by omega
          omega
        let q : Cell m (n - 1) := (p.1, ⟨j.val - 1, by omega⟩)
        have hp1q : p.1 = q.1 := rfl
        have hqv : q.2.val + 1 = j.val := by
          show j.val - 1 + 1 = j.val
          omega
        have hqadj : adj p q := by
          refine adj_of_row_col_diff rfl ?_
          show ((p.2.val : ℤ) - (j.val - 1 : ℕ)).natAbs = 1
          push_cast; omega
        have hdiff := contractAt_diff_seam_RtoL h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1q hpv hqv
        have hval := hmax q hqadj
        rw [hueq]
        linarith
    · exfalso
      apply hcellne
      rw [hcol_eq, hufrz]
  · obtain ⟨q, hqEq, hqadj, hbr⟩ := adj_cellExpand_of_not_frozen j hj0 hj1 p u hadj hufrz
    have hdiff := contractAt_diff_same_branch h hm j ⟨hj0, hj1, hsym⟩ p q hbr
    have hval := hmax q hqadj
    rw [← hqEq]
    linarith

/-- **Forward direction of extremum bijection (strict local min).**  Symmetric
version of `h_max_of_contractAt_isSLM`. -/
theorem h_min_of_contractAt_isSLm (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p : Cell m (n - 1))
    (hmin : IsStrictLocalMin (contractAt h hm j hfrz) p) :
    IsStrictLocalMin h (cellExpand j p hfrz.1 hfrz.2.1) := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  intro u hadj
  by_cases hufrz : u.2 = j
  · have hcellne : (cellExpand j p hj0 hj1).2 ≠ j := cellExpand_ne_j j p hj0 hj1
    rcases adj_row_or_col hadj with ⟨hrow_eq, hcol⟩ | ⟨hcol_eq, _⟩
    · have hcellp1 : (cellExpand j p hj0 hj1).1 = p.1 := by
        by_cases hpL : p.2.val < j.val
        · rw [cellExpand_left _ _ hj0 hj1 hpL]
        · rw [cellExpand_right _ _ hj0 hj1 (by omega : ¬ p.2.val < j.val)]
      have hu1 : u.1 = p.1 := hcellp1 ▸ hrow_eq.symm
      have hueq : u = (p.1, j) := Prod.ext hu1 hufrz
      by_cases hpL : p.2.val < j.val
      · have hcellp2v : (cellExpand j p hj0 hj1).2.val = p.2.val := by
          rw [cellExpand_left _ _ hj0 hj1 hpL]
        have hpv : p.2.val + 1 = j.val := by
          have hcolShow := hcol
          rw [hufrz, hcellp2v] at hcolShow
          have h_p_lt : (p.2.val : ℤ) < j.val := by push_cast; omega
          have : (p.2.val : ℤ) - j.val < 0 := by omega
          omega
        let q : Cell m (n - 1) := (p.1, ⟨j.val, by omega⟩)
        have hp1q : p.1 = q.1 := rfl
        have hqv : q.2.val = j.val := rfl
        have hqadj : adj p q := by
          refine adj_of_row_col_diff rfl ?_
          show ((p.2.val : ℤ) - j.val).natAbs = 1
          push_cast; omega
        have hdiff := contractAt_diff_seam_LtoR h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1q hpv hqv
        have hval := hmin q hqadj
        rw [hueq]
        linarith
      · push_neg at hpL
        have hpR : ¬ p.2.val < j.val := by omega
        have hcellp2v : (cellExpand j p hj0 hj1).2.val = p.2.val + 1 := by
          rw [cellExpand_right _ _ hj0 hj1 hpR]
        have hpv : p.2.val = j.val := by
          have hcolShow := hcol
          rw [hufrz, hcellp2v] at hcolShow
          have h_p_ge : (p.2.val : ℤ) ≥ j.val := by push_cast; omega
          have : (p.2.val + 1 : ℤ) - j.val > 0 := by omega
          omega
        let q : Cell m (n - 1) := (p.1, ⟨j.val - 1, by omega⟩)
        have hp1q : p.1 = q.1 := rfl
        have hqv : q.2.val + 1 = j.val := by
          show j.val - 1 + 1 = j.val
          omega
        have hqadj : adj p q := by
          refine adj_of_row_col_diff rfl ?_
          show ((p.2.val : ℤ) - (j.val - 1 : ℕ)).natAbs = 1
          push_cast; omega
        have hdiff := contractAt_diff_seam_RtoL h hh hm j ⟨hj0, hj1, hsym⟩ p q hp1q hpv hqv
        have hval := hmin q hqadj
        rw [hueq]
        linarith
    · exfalso
      apply hcellne
      rw [hcol_eq, hufrz]
  · obtain ⟨q, hqEq, hqadj, hbr⟩ := adj_cellExpand_of_not_frozen j hj0 hj1 p u hadj hufrz
    have hdiff := contractAt_diff_same_branch h hm j ⟨hj0, hj1, hsym⟩ p q hbr
    have hval := hmin q hqadj
    rw [← hqEq]
    linarith

/-- **Forward direction of extremum bijection.**  If `p` is a strict local
extremum of `contractAt`, then `cellExpand p` is a strict local extremum of
`h`. -/
theorem h_extremum_of_contractAt_isSLE (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p : Cell m (n - 1))
    (hext : IsStrictLocalExtremum (contractAt h hm j hfrz) p) :
    IsStrictLocalExtremum h (cellExpand j p hfrz.1 hfrz.2.1) := by
  rcases hext with hmax | hmin
  · exact Or.inl (h_max_of_contractAt_isSLM h hh hm j hfrz p hmax)
  · exact Or.inr (h_min_of_contractAt_isSLm h hh hm j hfrz p hmin)

/-- **Pointwise extremum bijection.**  `p` is a strict local extremum of
`contractAt` iff `cellExpand p` is a strict local extremum of `h`. -/
theorem contractAt_isSLE_iff (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) (p : Cell m (n - 1)) :
    IsStrictLocalExtremum (contractAt h hm j hfrz) p ↔
    IsStrictLocalExtremum h (cellExpand j p hfrz.1 hfrz.2.1) :=
  ⟨h_extremum_of_contractAt_isSLE h hh hm j hfrz p,
   contractAt_isSLE_of_h_extremum h hh hm j hfrz p⟩

/-- **Extremum count preservation.**  The atomic `contractAt` step at a
frozen column preserves the total number of strict local extrema.  Proved by
`Finset.card_bij` with `cellExpand` as the bijection: forward via
`h_extremum_of_contractAt_isSLE`; surjectivity via `cellExpand_surjOn` +
`frozenColumn_no_extremum` (h-extrema never sit in the frozen column) +
`contractAt_isSLE_of_h_extremum`. -/
theorem contractAt_numExtrema_eq (h : Cell m n → ℤ) (hh : IsHeight h) (hm : 0 < m)
    (j : Fin n) (hfrz : frozenColumn h j) :
    (Finset.univ.filter (IsStrictLocalExtremum (contractAt h hm j hfrz))).card =
    (Finset.univ.filter (IsStrictLocalExtremum h)).card := by
  obtain ⟨hj0, hj1, hsym⟩ := hfrz
  apply Finset.card_bij (fun p _ => cellExpand j p hj0 hj1)
  · intro p hp
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
    exact h_extremum_of_contractAt_isSLE h hh hm j ⟨hj0, hj1, hsym⟩ p hp
  · intro p1 _ p2 _ heq
    exact cellExpand_injective j hj0 hj1 heq
  · intro u hu
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hu
    have hufrz : u.2 ≠ j := by
      intro h_eq
      have hueq : u = (u.1, j) := Prod.ext rfl h_eq
      exact frozenColumn_no_extremum h j ⟨hj0, hj1, hsym⟩ u.1 (hueq ▸ hu)
    obtain ⟨p, hp⟩ := cellExpand_surjOn j hj0 hj1 u hufrz
    refine ⟨p, ?_, hp⟩
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    exact contractAt_isSLE_of_h_extremum h hh hm j ⟨hj0, hj1, hsym⟩ p (hp ▸ hu)

end OrigamiCone.Sequel

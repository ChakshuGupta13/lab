import Mathlib.Tactic.Linarith
import OrigamiCone.Diameter
import OrigamiCone.RearrLipschitz1
import OrigamiCone.RearrPairwise
import OrigamiCone.Reduction

/-!
# Column-sort: definition, `1`-Lipschitz preservation, and multiset preservation
(Sub-4c of `prop:monotone`)

Third piece of the assembly for `prop:monotone` (Section 4, main.tex
L1024-1056).  Mirror of `RowSort.lean` (cc9b5e2) + `RowSortMultiset.lean`
(f9a3601) for the **column-sort**.

Definitions and theorems are direct transpositions of the row-sort ones:
each column is independently replaced by its increasing rearrangement.
The paper's L1054-1056 claim about the column step is:

> "After the column step, columns are sorted and `1`-Lipschitz by~(i)
> applied to columns; adjacent columns of the row-sorted matrix are
> entrywise ordered, so by~(iii) rows stay nondecreasing, and by~(ii)
> rows stay `1`-Lipschitz."

The `1`-Lipschitz and multiset preservation halves are this module.  The
nondecreasing-in-both-coordinates conclusion (paper L1057) belongs to the
final assembly Sub-4d.

Definitions:
* `colOf` — column `j` of `φ` as a function `Fin m → ℤ`.
* `colSort` — the grid function with each column sorted.

Results:
* `colSort_isLipschitz1` — `IsLipschitz1 φ → IsLipschitz1 (colSort φ)`.
* `colSort_sum_eq` — `∑ v, colSort φ v = ∑ v, φ v`.
* `colSort_disp_eq` — `disp (colSort φ) K = disp φ K`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-! ## Definitions -/

/-- Column `j` of `φ` viewed as a function `Fin m → ℤ`. -/
def colOf (φ : Cell m n → ℤ) (j : Fin n) : Fin m → ℤ :=
  fun i => φ (i, j)

/-- The column-sort of `φ`: each column replaced by its increasing
rearrangement.  Mirror of `rowSort` (cc9b5e2) with row/column roles
swapped. -/
def colSort (φ : Cell m n → ℤ) : Cell m n → ℤ :=
  fun v => (colOf φ v.2 ∘ Tuple.sort (colOf φ v.2)) v.1

/-! ## Adjacency dichotomy reuse -/

/-- **Adjacency dichotomy on the grid.**  Duplicate of `adj_cases` from
`RowSort.lean` (private there).  Two adjacent cells either share a row
(and differ by one in column) or share a column (and differ by one in
row). -/
private lemma adj_cases' {p q : Cell m n} (h : adj p q) :
    (p.1 = q.1 ∧ ((p.2.val : ℤ) - q.2.val).natAbs = 1) ∨
      (((p.1.val : ℤ) - q.1.val).natAbs = 1 ∧ p.2 = q.2) := by
  unfold adj gdist at h
  have h_sum : ((p.1.val : ℤ) - q.1.val).natAbs
                + ((p.2.val : ℤ) - q.2.val).natAbs = 1 := by exact_mod_cast h
  rcases (Nat.eq_zero_or_pos ((p.1.val : ℤ) - q.1.val).natAbs) with h1 | h1
  · have hp1q1 : p.1.val = q.1.val := by
      have : ((p.1.val : ℤ) - q.1.val) = 0 := by omega
      omega
    refine Or.inl ⟨Fin.ext hp1q1, ?_⟩
    omega
  · have hp1q1 : ((p.1.val : ℤ) - q.1.val).natAbs = 1 := by omega
    have hp2q2 : ((p.2.val : ℤ) - q.2.val).natAbs = 0 := by omega
    have hp2eq : p.2.val = q.2.val := by
      have : ((p.2.val : ℤ) - q.2.val) = 0 := by omega
      omega
    exact Or.inr ⟨hp1q1, Fin.ext hp2eq⟩

/-- Columns of a `1`-Lipschitz grid function are path-`1`-Lipschitz.
Mirror of `rowOf_pathLipschitz1`. -/
private lemma colOf_pathLipschitz1 {φ : Cell m n → ℤ}
    (hφ : IsLipschitz1 φ) (j : Fin n) : PathLipschitz1 (colOf φ j) := by
  intro i i' hii'
  have h_adj : adj ((i, j) : Cell m n) ((i', j) : Cell m n) := by
    unfold adj gdist
    have hi : ((i.val : ℤ) - i'.val).natAbs = 1 := by
      have h_succ : i.val + 1 = i'.val := hii'
      omega
    have hj : ((j.val : ℤ) - j.val).natAbs = 0 := by omega
    show (((((i.val : ℤ) - i'.val).natAbs + ((j.val : ℤ) - j.val).natAbs) : ℕ) : ℤ) = 1
    rw [hi, hj]; rfl
  exact hφ _ _ h_adj

/-- Adjacent columns of a `1`-Lipschitz grid function are pairwise within `1`.
Mirror of `rowOf_pairwise_le`. -/
private lemma colOf_pairwise_le {φ : Cell m n → ℤ} (hφ : IsLipschitz1 φ)
    {j j' : Fin n} (hjj' : ((j.val : ℤ) - j'.val).natAbs = 1) (k : Fin m) :
    |colOf φ j k - colOf φ j' k| ≤ 1 := by
  have h_adj : adj ((k, j) : Cell m n) ((k, j') : Cell m n) := by
    unfold adj gdist
    have hk : ((k.val : ℤ) - k.val).natAbs = 0 := by omega
    show (((((k.val : ℤ) - k.val).natAbs + ((j.val : ℤ) - j'.val).natAbs) : ℕ) : ℤ) = 1
    rw [hk, hjj']; rfl
  exact hφ _ _ h_adj

/-! ## Main results -/

/-- **The column-sort preserves the `1`-Lipschitz property.**  Mirror of
`rowSort_isLipschitz1` (cc9b5e2) with row/column roles swapped.

Adjacency case-split via `adj_cases'`:
* Horizontal edge (same row, adjacent columns): apply fact (ii)
  (`sort_pairwise_abs_le_one`) to the two sorted columns.
* Vertical edge (adjacent rows, same column): apply fact (i)
  (`sort_pathLipschitz1`) to the sorted column. -/
theorem colSort_isLipschitz1 {φ : Cell m n → ℤ} (hφ : IsLipschitz1 φ) :
    IsLipschitz1 (colSort φ) := by
  intro p q hpq
  rcases adj_cases' hpq with ⟨hrow, hcol⟩ | ⟨hrow, hcol⟩
  · -- Horizontal edge: p.1 = q.1, |p.2 - q.2|.natAbs = 1.
    -- Apply fact (ii) to columns p.2 and q.2.
    have h_pair : ∀ k, |colOf φ p.2 k - colOf φ q.2 k| ≤ 1 :=
      colOf_pairwise_le hφ hcol
    have h_sort := sort_pairwise_abs_le_one h_pair p.1
    show |(colOf φ p.2 ∘ Tuple.sort (colOf φ p.2)) p.1
            - (colOf φ q.2 ∘ Tuple.sort (colOf φ q.2)) q.1| ≤ 1
    rw [show q.1 = p.1 from hrow.symm]
    exact h_sort
  · -- Vertical edge: |p.1 - q.1|.natAbs = 1, p.2 = q.2.
    -- Both sides use the SAME sorted column p.2; apply fact (i).
    have h_path : PathLipschitz1 (colOf φ p.2 ∘ Tuple.sort (colOf φ p.2)) :=
      sort_pathLipschitz1 (colOf_pathLipschitz1 hφ p.2)
    show |(colOf φ p.2 ∘ Tuple.sort (colOf φ p.2)) p.1
            - (colOf φ q.2 ∘ Tuple.sort (colOf φ q.2)) q.1| ≤ 1
    rw [show q.2 = p.2 from hcol.symm]
    have h_adj_idx : (p.1.val + 1 = q.1.val) ∨ (q.1.val + 1 = p.1.val) := by omega
    rcases h_adj_idx with h_left | h_right
    · exact h_path p.1 q.1 h_left
    · rw [abs_sub_comm]
      exact h_path q.1 p.1 h_right

/-- **Grid-sum invariance of `colSort`.**  Mirror of `rowSort_sum_eq`
(f9a3601).  Each column is permuted, so the total grid sum is unchanged.

Proof: split `∑ v : Cell m n, _` as `∑ j, ∑ i, _` via
`Fintype.sum_prod_type` followed by `Finset.sum_comm`; for each fixed
column `j`, the inner sum is `Equiv.sum_comp (Tuple.sort (colOf φ j))`
applied to `colOf φ j`. -/
theorem colSort_sum_eq (φ : Cell m n → ℤ) :
    ∑ v : Cell m n, colSort φ v = ∑ v : Cell m n, φ v := by
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type]
  -- Goal: ∑ i, ∑ j, colSort φ (i, j) = ∑ i, ∑ j, φ (i, j).
  -- Swap both iterated sums so the OUTER index is the column j, then
  -- use Equiv.sum_comp on the inner column-sum.
  rw [Finset.sum_comm]
  conv_rhs => rw [Finset.sum_comm]
  -- Goal: ∑ j, ∑ i, colSort φ (i, j) = ∑ j, ∑ i, φ (i, j).
  refine Finset.sum_congr rfl fun j _ => ?_
  show ∑ i : Fin m, colOf φ j (Tuple.sort (colOf φ j) i) = ∑ i : Fin m, φ (i, j)
  rw [Equiv.sum_comp (Tuple.sort (colOf φ j)) (colOf φ j)]
  rfl

/-- **Dispersion invariance of `colSort`.**  Mirror of `rowSort_disp_eq`
(f9a3601).  Same technique as `colSort_sum_eq`, applied with the per-column
function `i ↦ |colOf φ j i - K|`. -/
theorem colSort_disp_eq (φ : Cell m n → ℤ) (K : ℤ) :
    disp (colSort φ) K = disp φ K := by
  unfold disp
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type]
  rw [Finset.sum_comm]
  conv_rhs => rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun j _ => ?_
  show ∑ i : Fin m, |colOf φ j (Tuple.sort (colOf φ j) i) - K|
      = ∑ i : Fin m, |φ (i, j) - K|
  rw [Equiv.sum_comp (Tuple.sort (colOf φ j)) (fun i => |colOf φ j i - K|)]
  rfl

end OrigamiCone

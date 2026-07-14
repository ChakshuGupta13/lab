import OrigamiCone.RearrMonotone
import OrigamiCone.RowSort
import OrigamiCone.RowSortMultiset
import OrigamiCone.ColumnSort

/-!
# Monotone reduction: row-sort then column-sort (Sub-4d of `prop:monotone`)

Final piece of the assembly for `prop:monotone` (Section 4, main.tex
L1024-1056).  Combines:

* Sub-1 (`RearrLipschitz1`, 8d50116) — fact (i).
* Sub-2 (`RearrPairwise`, ee4baf6)   — fact (ii).
* Sub-3 (`RearrMonotone`, 616546d)   — fact (iii).
* Sub-4a (`RowSort`, cc9b5e2)        — rowSort `1`-Lipschitz preservation.
* Sub-4b (`RowSortMultiset`, f9a3601) — rowSort sum + dispersion preservation.
* Sub-4c (`ColumnSort`, 393de3b)     — colSort def + Lipschitz/sum/disp preservation.

This module defines the **monotone reduction** `monoReduce φ := colSort (rowSort φ)`
and proves the four headline properties asserted by `prop:monotone`:

* `monoReduce_isLipschitz1` — preserves the integer `1`-Lipschitz property.
* `monoReduce_sum_eq`       — preserves the grid sum.
* `monoReduce_disp_eq`      — preserves the dispersion.
* `monoReduce_monotone_row` — nondecreasing in the row coordinate.
* `monoReduce_monotone_col` — nondecreasing in the column coordinate.

The first three are one-line compositions of Sub-4a / Sub-4b / Sub-4c.
`monoReduce_monotone_row` is direct from `Tuple.monotone_sort` on each
column.  `monoReduce_monotone_col` is the only non-trivial step:
after the row-sort, the (row-sorted) value at row `k`, column `j`,
is `≤` the value at row `k`, column `j+1` (each row is sorted), so the
two columns are pointwise ordered.  By fact (iii) (`sort_pointwise_le`)
the column-sort preserves that pointwise order at each row.  This is
the paper's L1054-1056 argument.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- The **monotone reduction** of `φ`: row-sort each row, then column-sort
each column.  Defined as `colSort ∘ rowSort` to match the paper's L1037
construction order. -/
def monoReduce (φ : Cell m n → ℤ) : Cell m n → ℤ := colSort (rowSort φ)

/-! ## Conservation properties (one-line compositions) -/

/-- **The monotone reduction preserves the `1`-Lipschitz property.**
Composition of `rowSort_isLipschitz1` (cc9b5e2) and
`colSort_isLipschitz1` (393de3b). -/
theorem monoReduce_isLipschitz1 {φ : Cell m n → ℤ} (hφ : IsLipschitz1 φ) :
    IsLipschitz1 (monoReduce φ) :=
  colSort_isLipschitz1 (rowSort_isLipschitz1 hφ)

/-- **The monotone reduction preserves the grid sum.**  Composition of
`rowSort_sum_eq` and `colSort_sum_eq` (f9a3601, 393de3b). -/
theorem monoReduce_sum_eq (φ : Cell m n → ℤ) :
    ∑ v : Cell m n, monoReduce φ v = ∑ v : Cell m n, φ v := by
  unfold monoReduce
  rw [colSort_sum_eq, rowSort_sum_eq]

/-- **The monotone reduction preserves dispersion.**  Composition of
`rowSort_disp_eq` and `colSort_disp_eq` (f9a3601, 393de3b).  This is the
paper's "the grid value-multiset, and hence `\disp`, is unchanged"
(L1037-1041), the form `prop:monotone` ultimately cites. -/
theorem monoReduce_disp_eq (φ : Cell m n → ℤ) (K : ℤ) :
    disp (monoReduce φ) K = disp φ K := by
  unfold monoReduce
  rw [colSort_disp_eq, rowSort_disp_eq]

/-! ## Coordinate-monotonicity -/

/-- **The monotone reduction is nondecreasing in the row coordinate.**

Direct from `Tuple.monotone_sort (colOf (rowSort φ) j)`: each column of
`monoReduce φ = colSort (rowSort φ)` is sorted (the `colSort` step). -/
theorem monoReduce_monotone_row (φ : Cell m n → ℤ) (j : Fin n)
    {i i' : Fin m} (hii' : i ≤ i') :
    monoReduce φ (i, j) ≤ monoReduce φ (i', j) := by
  -- `monoReduce φ (i, j) = colSort (rowSort φ) (i, j)
  --                     = (colOf (rowSort φ) j ∘ Tuple.sort (colOf (rowSort φ) j)) i`.
  -- Apply `Tuple.monotone_sort` to the column-of (rowSort φ) at column j.
  show (colOf (rowSort φ) j ∘ Tuple.sort (colOf (rowSort φ) j)) i
      ≤ (colOf (rowSort φ) j ∘ Tuple.sort (colOf (rowSort φ) j)) i'
  exact Tuple.monotone_sort (colOf (rowSort φ) j) hii'

/-- **The monotone reduction is nondecreasing in the column coordinate.**

After the row-sort, each row `rowSort φ (k, ·)` is sorted (this is what
`rowSort` does); hence for `j ≤ j'`,
`rowSort φ (k, j) ≤ rowSort φ (k, j')` for every row `k`.  Equivalently,
`colOf (rowSort φ) j k ≤ colOf (rowSort φ) j' k` for every `k`, i.e.,
the two columns are pointwise ordered.  By fact (iii) `sort_pointwise_le`,
the column-sort preserves that pointwise ordering at each row index `i`.

This is the paper's L1054-1056 step. -/
theorem monoReduce_monotone_col (φ : Cell m n → ℤ) (i : Fin m)
    {j j' : Fin n} (hjj' : j ≤ j') :
    monoReduce φ (i, j) ≤ monoReduce φ (i, j') := by
  -- Step 1: the two columns of `rowSort φ` are pointwise ordered.
  have h_col_le : ∀ k : Fin m, colOf (rowSort φ) j k ≤ colOf (rowSort φ) j' k := by
    intro k
    -- `colOf (rowSort φ) j k = rowSort φ (k, j) = (rowOf φ k ∘ sort (rowOf φ k)) j`
    --                                          ≤ (rowOf φ k ∘ sort (rowOf φ k)) j'
    --                                          = colOf (rowSort φ) j' k.
    show (rowOf φ k ∘ Tuple.sort (rowOf φ k)) j
        ≤ (rowOf φ k ∘ Tuple.sort (rowOf φ k)) j'
    exact Tuple.monotone_sort (rowOf φ k) hjj'
  -- Step 2: by fact (iii) `sort_pointwise_le`, the column-sort preserves
  -- this pointwise order at every row.
  show (colOf (rowSort φ) j ∘ Tuple.sort (colOf (rowSort φ) j)) i
      ≤ (colOf (rowSort φ) j' ∘ Tuple.sort (colOf (rowSort φ) j')) i
  exact sort_pointwise_le h_col_le i

end OrigamiCone

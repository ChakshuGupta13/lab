import Mathlib.Tactic.Linarith
import OrigamiCone.Diameter
import OrigamiCone.RowSort

/-!
# Row-sort preserves the grid sum and dispersion (Sub-4b of `prop:monotone`)

Second piece of the assembly for `prop:monotone` (Section 4, main.tex
L1024-1056).  Building on Sub-4a (`RowSort.lean`, `cc9b5e2`) which
defined `rowSort` and proved the `IsLipschitz1` preservation, this module
proves the **value-multiset preservation** in the only form the paper
needs:

* `rowSort_sum_eq` — grid sum invariance: `∑ v, rowSort φ v = ∑ v, φ v`.
* `rowSort_disp_eq` — dispersion invariance: `disp (rowSort φ) K = disp φ K`.

The proof is one application of `Equiv.sum_comp` per row, after splitting
the grid sum as iterated sums over rows.  Since each row's sort is an
`Equiv.Perm`, the per-row sum is preserved by `Equiv.sum_comp`.

The paper's prop:monotone uses precisely `rowSort_disp_eq` (and the same
for `colSort`) to conclude that the monotone rearrangement has the same
dispersion as the original — the value-multiset preservation step.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Grid-sum invariance of `rowSort`.**  Since `rowSort` permutes the
entries of each row, the total sum over the grid is unchanged.

Proof: split `∑ v : Cell m n, _` as `∑ i, ∑ j, _` via `Finset.sum_product`;
for each fixed row `i`, the inner sum is `Equiv.sum_comp (Tuple.sort (rowOf φ i))`
applied to `rowOf φ i`. -/
theorem rowSort_sum_eq (φ : Cell m n → ℤ) :
    ∑ v : Cell m n, rowSort φ v = ∑ v : Cell m n, φ v := by
  -- Split the sum over `Cell m n = Fin m × Fin n` as iterated sums.
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type]
  -- Now goal: ∑ i, ∑ j, rowSort φ (i, j) = ∑ i, ∑ j, φ (i, j).
  refine Finset.sum_congr rfl fun i _ => ?_
  -- For fixed row `i`, apply Equiv.sum_comp to the sort permutation.
  show ∑ j : Fin n, rowOf φ i (Tuple.sort (rowOf φ i) j) = ∑ j : Fin n, φ (i, j)
  rw [Equiv.sum_comp (Tuple.sort (rowOf φ i)) (rowOf φ i)]
  rfl

/-- **Dispersion invariance of `rowSort`.**  The dispersion sum
`∑ v, |φ v - K|` is preserved by `rowSort`, since each row's sort is a
permutation that just reorders the per-row `|· - K|` values.

Proof: the same `Fintype.sum_prod_type` + per-row `Equiv.sum_comp`
technique as `rowSort_sum_eq`, applied with the per-row function
`j ↦ |rowOf φ i j - K|`. -/
theorem rowSort_disp_eq (φ : Cell m n → ℤ) (K : ℤ) :
    disp (rowSort φ) K = disp φ K := by
  -- The grid sum of `|rowSort φ v - K|` equals the grid sum of
  -- `|φ v - K|`.  We apply `Equiv.sum_comp` per row to the row-of
  -- `|φ - K|`, similarly to `rowSort_sum_eq`.
  unfold disp
  rw [Fintype.sum_prod_type, Fintype.sum_prod_type]
  refine Finset.sum_congr rfl fun i _ => ?_
  show ∑ j : Fin n, |rowOf φ i (Tuple.sort (rowOf φ i) j) - K|
      = ∑ j : Fin n, |φ (i, j) - K|
  rw [Equiv.sum_comp (Tuple.sort (rowOf φ i)) (fun j => |rowOf φ i j - K|)]
  rfl

end OrigamiCone

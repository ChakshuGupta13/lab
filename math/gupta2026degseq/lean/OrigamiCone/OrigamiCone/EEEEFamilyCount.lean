import Mathlib.Order.Interval.Finset.Nat
import Mathlib.Tactic.Ring
import OrigamiCone.Deg4FamilySum

/-!
# EE|EE family count arithmetic assembly (Sub-6e of `thm:deg4count`)

The paper's **`thm:deg4count`** family `EE|EE` (paper L658-696) analyses the
three geometric sub-cases:

* **Same-side minima** (both edge apexes on the same side): ruled out —
  every maximum lands at a corner, killing the `EE|EE` classification.
* **Adjacent-side minima** (one apex on top, one on left, or symmetric):
  ruled out — the opposite corner is always a strict local maximum, so
  again no `EE|EE` pair.
* **Opposite-side minima** (e.g., top and bottom, distinct columns): the
  only surviving case. Contribution matrix is `(n-2) × (n-2)` tridiagonal:
  * Diagonal (`c₁ = c₂`): value `m - 2`, contributing `(m-2)(n-2)`.
  * Off-diagonals (`|c₁ - c₂| = 1`): value `m - 3`, `2(n-3)` such cells,
    contributing `2(m-3)(n-3)`.
  * Bandwidth-2 or greater: zero contribution.

Sum for one orientation (top-bottom): `(m-2)(n-2) + 2(m-3)(n-3)`.
Doubling for the symmetric left-right orientation:
`family EE|EE = 2(m-2)(n-2) + 4(m-3)(n-3)`.

This module handles the **integer arithmetic** of the assembly.  The
underlying *geometric* facts (the same/adjacent-side exclusion arguments
and the opposite-side tridiagonal structure) are the content of the Ridge
Lemma (`lem:ridge`, formalised in `RidgeMax.lean`) and the per-pair ridge
enumeration (paper L658-696).

Results:
* `ee_diagonal_count` — the (n-2)×(n-2) tridiagonal has `n − 2` diagonal
  cells `{(c, c) : c ∈ Icc 2 (n-1)}`, cardinality via `Nat.card_Icc`.
* `ee_offdiagonal_pair_count` — the ordered pairs `(c, c+1)` with both
  endpoints in `Icc 2 (n-1)` number `n − 3`; doubled for symmetry, the
  full off-diagonal `{|c₁ − c₂| = 1}` has `2(n − 3)` cells.
* `ee_geometric_sum_matches_closed_form` — wires the two cardinality
  lemmas into the closed one-orientation sum
  `diag_count · (m-2) + 2 · offdiag_pair_count · (m-3) =
  (n-2)(m-2) + 2(n-3)(m-3)`.  Doubling gives `familyEEEE m n`.
* `family_EEEE_decomposition` — the polynomial identity
  `2 · ((m-2)(n-2) + 2(m-3)(n-3)) = familyEEEE m n` in `ℤ`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Diagonal cell count of the EE|EE tridiagonal contribution matrix.**
The diagonal `{(c, c) : c ∈ Icc 2 (n-1)}` has `n − 2` cells, matching the
paper's "the diagonal has `n − 2` entries" (L681). -/
theorem ee_diagonal_count (hn : 3 ≤ n) :
    (Finset.Icc 2 (n - 1) : Finset ℕ).card = n - 2 := by
  rw [Nat.card_Icc]; omega

/-- **Ordered-pair off-diagonal count.**  Ordered pairs `(c₁, c₂)` with
`c₂ = c₁ + 1` and both in `Icc 2 (n-1)` are parametrized by `c₁ ∈ Icc 2
(n-2)` (equivalently `c₂ ∈ Icc 3 (n-1)`), of which there are `n − 3`.
Doubled for the symmetric `(c₂ = c₁ − 1)` sub-diagonal (each contributing
the same cell value `m − 3`), the full off-diagonal `{|c₁ − c₂| = 1}` has
`2(n − 3)` cells (paper L681). -/
theorem ee_offdiagonal_pair_count (hn : 3 ≤ n) :
    (Finset.Icc 2 (n - 2) : Finset ℕ).card = n - 3 := by
  rw [Nat.card_Icc]; omega

/-- **Geometric-sum form of the EE|EE one-orientation contribution.**  Wires
the diagonal and off-diagonal cardinality lemmas into the closed form.

For opposite-side minima, one orientation (top-bottom) contributes
`|diag_cells| · (m-2) + 2 · |offdiag_pairs| · (m-3)`.  By the cardinality
lemmas this equals `(n-2)(m-2) + 2(n-3)(m-3)`, the paper's one-orientation
tridiagonal sum (L695). -/
theorem ee_geometric_sum_matches_closed_form (_hm : 3 ≤ m) (hn : 3 ≤ n) :
    (Finset.Icc 2 (n - 1) : Finset ℕ).card * (m - 2)
      + 2 * (Finset.Icc 2 (n - 2) : Finset ℕ).card * (m - 3)
      = (n - 2) * (m - 2) + 2 * (n - 3) * (m - 3) := by
  rw [ee_diagonal_count hn, ee_offdiagonal_pair_count hn]

/-- **EE|EE family count assembly.** The tridiagonal contribution matrix for
opposite-side minima sums to `(m-2)(n-2) + 2(m-3)(n-3)` for one orientation.
Doubling for the second orientation gives the closed-form
`familyEEEE m n = 2(m-2)(n-2) + 4(m-3)(n-3)`. Stated as a polynomial identity
in `ℤ`. -/
theorem family_EEEE_decomposition (_hm : 3 ≤ m) (_hn : 3 ≤ n) :
    (2 : ℤ) * (((m - 2) * (n - 2) : ℤ) + 2 * ((m - 3) * (n - 3) : ℤ))
      = familyEEEE m n := by
  unfold familyEEEE
  ring

end OrigamiCone

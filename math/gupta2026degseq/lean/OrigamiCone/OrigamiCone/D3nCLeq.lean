import OrigamiCone.DSquareMiddle

/-!
# `D(3, n)`: unified middle-range per-term `cLeq` formula (Section 4)

Building toward the next paper closed form `D(3, n) = ⌊3n²/4⌋ + 2`
(Section 4 of the paper).  This module supplies the **unified middle-range
per-term formula** for the cumulative sublevel count on the `3 × n`
antidiagonal grid:

  `cLeq acell ℓ = 3ℓ`   for every `ℓ ∈ [1, n − 1]` (with `n ≥ 2`).

The formula unifies two regimes that are normally separate:
- `ℓ = 1`: small triangle (`cLeq_acell_triangle`) gives `(1 + 1)(1 + 2)/2 = 3 = 3·1`.
- `ℓ ∈ [2, n − 1]`: middle band (`cLeq_acell_middle`) gives
  `3(ℓ + 1) − 3·2/2 = 3ℓ`.

Both formulas agree at `ℓ = 2`, where they overlap.

The boundary terms `ℓ = 0` (triangle endpoint, `c = 1`) and `ℓ = n`
(suffix endpoint, `c = 3n − 1`) require separate treatment and are
handled in the subsequent module assembling the full `D(3, n)`.

Results:
* `cLeq_acell_three_mid` — `cLeq (acell (m := 3) (n := n)) ℓ = 3ℓ`
  for `ℓ ∈ [1, n − 1]`.

No `sorry`.
-/

namespace OrigamiCone

variable {n : ℕ}

/-- **Unified middle-range `cLeq` formula for the `3 × n` antidiagonal.**

For `1 ≤ ℓ` and `ℓ + 1 ≤ n` (equivalently `ℓ ∈ [1, n − 1]`, which forces
`n ≥ 2`), the cumulative sublevel count of the antidiagonal
`acell : Cell 3 n → ℤ` is `3ℓ`.  Combines the small-triangle formula
(at `ℓ = 1`) and the middle-band trapezoidal formula (for `ℓ ∈ [2, n − 1]`)
into a single closed form, which both regimes agree on. -/
theorem cLeq_acell_three_mid (ℓ : ℕ) (h1 : 1 ≤ ℓ) (h2 : ℓ + 1 ≤ n) :
    cLeq (acell (m := 3) (n := n)) (ℓ : ℤ) = 3 * (ℓ : ℤ) := by
  rcases (by omega : ℓ = 1 ∨ 2 ≤ ℓ) with hℓ1 | hℓ2
  · -- ℓ = 1: triangle case.
    subst hℓ1
    have h_three : (1 : ℕ) < 3 := by decide
    have h_n : (1 : ℕ) < n := by omega
    rw [cLeq_acell_triangle (m := 3) (n := n) 1 h_three h_n]
    norm_num
  · -- ℓ ≥ 2: middle band case.
    have hℓ_lo : 3 ≤ ℓ + 1 := by omega
    rw [cLeq_acell_middle (m := 3) (n := n) ℓ hℓ_lo h2]
    push_cast
    ring

end OrigamiCone

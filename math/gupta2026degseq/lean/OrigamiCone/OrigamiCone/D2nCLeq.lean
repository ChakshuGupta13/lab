import OrigamiCone.DSquareMiddle

/-!
# `D(2, n)`: unified per-term `cLeq` formula (Section 4)

Building toward the next paper closed form `D(2, n) = ⌈n²/2⌉` (Section 4 of
the paper).  This module supplies the **unified per-term formula** for the
cumulative sublevel count on the `2 × n` antidiagonal grid:

  `cLeq acell ℓ = 2ℓ + 1`  for every `ℓ ∈ [0, n - 1]` (with `n ≥ 1`).

The formula unifies two regimes that are normally separate:
- `ℓ = 0`: triangle (`cLeq_acell_triangle`) gives `(0+1)(0+2)/2 = 1 = 2·0 + 1`.
- `ℓ ∈ [1, n - 1]`: middle band (`cLeq_acell_middle`) gives
  `2(ℓ+1) - 2·1/2 = 2ℓ + 1`.

Both formulas agree, so the unified expression `c_ℓ = 2ℓ + 1` holds throughout.
This is the per-term primitive on which the `D(2, n) = ⌈n²/2⌉` sum builds
(the next module — `D2n.lean` — will combine this with the median
characterisation and a parity-casework sum identity).

Results:
* `cLeq_acell_two` — `cLeq (acell (m := 2) (n := n)) ℓ = 2ℓ + 1` for
  `ℓ ∈ [0, n)`.

No `sorry`.
-/

namespace OrigamiCone

variable {n : ℕ}

/-- **Unified per-term `cLeq` formula for the `2 × n` antidiagonal.**

For `ℓ < n` (which forces `n ≥ 1`), the cumulative sublevel count of the
antidiagonal `acell : Cell 2 n → ℤ` is `2ℓ + 1`.  Combines the small-triangle
formula (at `ℓ = 0`) and the middle-band trapezoidal formula (for
`ℓ ∈ [1, n - 1]`) into a single closed form, which both regimes agree on. -/
theorem cLeq_acell_two (ℓ : ℕ) (hℓ : ℓ < n) :
    cLeq (acell (m := 2) (n := n)) (ℓ : ℤ) = 2 * (ℓ : ℤ) + 1 := by
  rcases Nat.eq_zero_or_pos ℓ with hℓ0 | hℓ_pos
  · -- ℓ = 0: triangle case.
    subst hℓ0
    have h_two : (0 : ℕ) < 2 := by decide
    rw [cLeq_acell_triangle (m := 2) (n := n) 0 h_two hℓ]
    norm_num
  · -- ℓ ≥ 1: middle band case.
    have hℓ_lo : 2 ≤ ℓ + 1 := by omega
    have hℓ_hi : ℓ + 1 ≤ n := by omega
    rw [cLeq_acell_middle (m := 2) (n := n) ℓ hℓ_lo hℓ_hi]
    push_cast
    ring

end OrigamiCone

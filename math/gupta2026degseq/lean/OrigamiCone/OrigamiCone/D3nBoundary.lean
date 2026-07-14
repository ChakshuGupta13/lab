import OrigamiCone.D3nCLeq

/-!
# `D(3, n)`: boundary `cLeq` lemmas (Section 4)

Two short boundary lemmas for the `3 × n` antidiagonal, completing the
per-term `cLeq` picture started in `D3nCLeq.lean`:

  `cLeq acell 0 = 1`         (triangle base, requires `n ≥ 1`).
  `cLeq acell n = 3n − 1`    (suffix endpoint, requires `n ≥ 1`).

Combined with `cLeq_acell_three_mid` (`D3nCLeq.lean`, `ecc935a`), these
give the full per-term `cLeq` formula across the level range `[0, n]` for
the `3 × n` grid:

  `ℓ = 0`: c = 1
  `ℓ ∈ [1, n − 1]`: c = 3ℓ
  `ℓ = n`: c = 3n − 1

This is the per-term primitive on which the `D(3, n) = ⌊3n²/4⌋ + 2` sum
identity (next module) builds.

Results:
* `cLeq_acell_three_zero` — `cLeq (acell (m := 3) (n := n)) 0 = 1`.
* `cLeq_acell_three_top` — `cLeq (acell (m := 3) (n := n)) n = 3n − 1`.

No `sorry`.
-/

namespace OrigamiCone

variable {n : ℕ}

/-- **Triangle base for the `3 × n` antidiagonal.**  For `n ≥ 1`,
`cLeq acell 0 = 1` (only the corner cell `(0, 0)` has antidiagonal value 0).
Direct specialisation of `cLeq_acell_triangle` at `ℓ = 0`. -/
theorem cLeq_acell_three_zero (hn : 1 ≤ n) :
    cLeq (acell (m := 3) (n := n)) (0 : ℤ) = 1 := by
  have h_three : (0 : ℕ) < 3 := by decide
  have h_n : (0 : ℕ) < n := hn
  have h := cLeq_acell_triangle (m := 3) (n := n) 0 h_three h_n
  rw [show ((0 : ℕ) : ℤ) = (0 : ℤ) from rfl] at h
  rw [h]
  norm_num

/-- **Suffix endpoint for the `3 × n` antidiagonal.**  For `n ≥ 1`,
`cLeq acell n = 3n − 1`.  Direct specialisation of `cLeq_acell_suffix`
at `ℓ = n`. -/
theorem cLeq_acell_three_top (hn : 1 ≤ n) :
    cLeq (acell (m := 3) (n := n)) (n : ℤ) = 3 * (n : ℤ) - 1 := by
  -- cLeq_acell_suffix: cLeq acell ℓ = mn - (m+n-2-ℓ)(m+n-1-ℓ)/2.
  -- For m=3, ℓ=n: requires hℓlo : m-2 ≤ ℓ = 1 ≤ n (= hn), hℓlo' : n-2 ≤ n
  -- (trivial), hℓhi : ℓ + 3 ≤ m + n = n + 3 ≤ 3 + n (trivial).
  have hℓ_lo : (3 : ℕ) - 2 ≤ n := by omega
  have hℓ_lo' : n - 2 ≤ n := by omega
  have hℓ_hi : n + 3 ≤ 3 + n := by omega
  have h := cLeq_acell_suffix (m := 3) (n := n) n hℓ_lo hℓ_lo' hℓ_hi
  rw [h]
  -- Goal: ((3 * n : ℕ) : ℤ) - ((3 + n - 2 - n) * (3 + n - 1 - n) / 2 : ℤ)
  --     = 3 * (n : ℤ) - 1.
  -- (3 + n - 2 - n) = 1, (3 + n - 1 - n) = 2, so the product is 1*2 = 2,
  -- and 2 / 2 = 1 exactly.  push_cast normalises the (3 * n : ℕ) cast;
  -- omega handles the integer division 2 / 2 = 1 (which `ring` does not).
  push_cast
  omega

end OrigamiCone

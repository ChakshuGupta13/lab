import OrigamiCone.AcellCount
import OrigamiCone.AcellReflect
import OrigamiCone.DSquareMiddle

/-!
# `D(4, n)`: per-term `cLeq` formulas for the `4 × n` antidiagonal

For the `4 × n` grid (`Cell 4 n`), the antidiagonal `acell v = v.1.val + v.2.val`
takes values in `[0, n + 2]` (max attained at cell `(3, n − 1)`).  The per-term
counts `c_ℓ := |{v : Cell 4 n | acell v ≤ ℓ}|` follow the **piecewise** pattern:

| `ℓ`                  | `c_ℓ`              | source primitive          |
|----------------------|--------------------|---------------------------|
| `0`                  | `1`                | `cLeq_acell_triangle`     |
| `1`                  | `3`                | `cLeq_acell_triangle`     |
| `2`                  | `6`                | `cLeq_acell_triangle`     |
| `[3, n − 1]`         | `4ℓ − 2`           | `cLeq_acell_middle`       |
| `n`                  | `4n − 3`           | `cLeq_acell_suffix`       |
| `n + 1`              | `4n − 1`           | `cLeq_acell_suffix`       |

This module specialises the three generic per-level primitives
(`cLeq_acell_triangle`, `cLeq_acell_middle`, `cLeq_acell_suffix`) to the
`m = 4` case, yielding clean `ℓ`-named lemmas the downstream sum-identity and
assembly modules can invoke without re-deriving each cast.

In contrast to `D(3, n)` (where the prefix triangle entry at `ℓ = 1` *coincides*
with the middle formula at `ℓ = 1`, yielding the unified `c_ℓ = 3ℓ` formula in
`D3nCLeq.lean`), `m = 4` has *three* distinct prefix entries (`1, 3, 6`) that
do not all match the middle formula `4ℓ − 2`: only at `ℓ = 2` does the prefix
`T(3) = 6` coincide with `4 · 2 − 2 = 6`.  So no unified formula exists; each
prefix entry needs its own lemma.

Results (all `m = 4` per-term `cLeq`):
* `cLeq_acell_four_zero` (`n ≥ 1`) — `c_0 = 1`.
* `cLeq_acell_four_one`  (`n ≥ 2`) — `c_1 = 3`.
* `cLeq_acell_four_two`  (`n ≥ 3`) — `c_2 = 6`.
* `cLeq_acell_four_mid`  (`3 ≤ ℓ`, `ℓ + 1 ≤ n`) — `c_ℓ = 4ℓ − 2`.
* `cLeq_acell_four_top_minus_one` (`n ≥ 2`) — `c_n = 4n − 3`.
* `cLeq_acell_four_top` (`n ≥ 1`) — `c_{n+1} = 4n − 1`.

The sum-identity for the middle band and the final `D(4, n) = n² + 4 + [n odd]`
assembly are deferred to sibling modules.

No `sorry`.
-/

namespace OrigamiCone

variable {n : ℕ}

/-! ## Prefix: triangle entries at `ℓ = 0, 1, 2` -/

/-- **Prefix `ℓ = 0`** for `m = 4`: `c_0 = 1` (only the cell `(0, 0)`). -/
theorem cLeq_acell_four_zero (hn : 1 ≤ n) :
    cLeq (acell (m := 4) (n := n)) 0 = 1 := by
  have htri := cLeq_acell_triangle (m := 4) (n := n) 0 (by decide) hn
  -- htri : cLeq acell 0 = (0 + 1) * (0 + 2) / 2 = 1.
  simpa using htri

/-- **Prefix `ℓ = 1`** for `m = 4`: `c_1 = 3` (cells `(0,0), (0,1), (1,0)`). -/
theorem cLeq_acell_four_one (hn : 2 ≤ n) :
    cLeq (acell (m := 4) (n := n)) 1 = 3 := by
  have htri := cLeq_acell_triangle (m := 4) (n := n) 1 (by decide) (by omega)
  -- htri : cLeq acell 1 = (1 + 1) * (1 + 2) / 2 = 3.
  simpa using htri

/-- **Prefix `ℓ = 2`** for `m = 4`: `c_2 = 6`. -/
theorem cLeq_acell_four_two (hn : 3 ≤ n) :
    cLeq (acell (m := 4) (n := n)) 2 = 6 := by
  have htri := cLeq_acell_triangle (m := 4) (n := n) 2 (by decide) (by omega)
  -- htri : cLeq acell 2 = (2 + 1) * (2 + 2) / 2 = 6.
  simpa using htri

/-! ## Middle band: `ℓ ∈ [3, n − 1]` -/

/-- **Middle band** for `m = 4`: `c_ℓ = 4ℓ − 2` for every `ℓ ∈ [3, n − 1]`.

This requires `n ≥ 4` (implied by `3 ≤ ℓ` and `ℓ + 1 ≤ n`).  Derived from
`cLeq_acell_middle` at `m = 4`. -/
theorem cLeq_acell_four_mid (ℓ : ℕ) (hℓ_lo : 3 ≤ ℓ) (hℓ_hi : ℓ + 1 ≤ n) :
    cLeq (acell (m := 4) (n := n)) (ℓ : ℤ) = 4 * (ℓ : ℤ) - 2 := by
  have hmid := cLeq_acell_middle (m := 4) (n := n) ℓ (by omega) hℓ_hi
  -- hmid : cLeq acell ℓ = 4 * (ℓ + 1) - 4 * (4 - 1) / 2.
  -- 4 * 3 / 2 = 6, so RHS = 4(ℓ+1) - 6 = 4ℓ + 4 - 6 = 4ℓ - 2.
  rw [hmid]; push_cast; ring

/-! ## Suffix: top two entries at `ℓ = n` and `ℓ = n + 1` -/

/-- **Suffix `ℓ = n`** for `m = 4`: `c_n = 4n − 3`.  Valid for `n ≥ 2`. -/
theorem cLeq_acell_four_top_minus_one (hn : 2 ≤ n) :
    cLeq (acell (m := 4) (n := n)) (n : ℤ) = 4 * (n : ℤ) - 3 := by
  -- Apply suffix at ℓ = n; need m - 2 ≤ n (i.e., 2 ≤ n, holds), n - 2 ≤ n
  -- (trivial), and ℓ + 3 ≤ m + n (i.e., n + 3 ≤ 4 + n, holds).
  have hsuf := cLeq_acell_suffix (m := 4) (n := n) n (by omega) (by omega)
    (by omega)
  -- (4 + n - 2 - n) = 2, (4 + n - 1 - n) = 3, 2 * 3 / 2 = 3 in ℤ.
  -- push_cast normalises the casts; omega handles ℤ ediv 6 / 2 = 3.
  rw [hsuf]
  push_cast
  omega

/-- **Suffix `ℓ = n + 1`** for `m = 4`: `c_{n+1} = 4n − 1`.  Valid for
`n ≥ 1`.  The level `ℓ = n + 1` is the second-largest antidiagonal level
(one below the max `n + 2`); only the cell `(3, n − 1)` (with the maximum
`acell` value) is excluded. -/
theorem cLeq_acell_four_top (hn : 1 ≤ n) :
    cLeq (acell (m := 4) (n := n)) ((n + 1 : ℕ) : ℤ) = 4 * (n : ℤ) - 1 := by
  -- Apply suffix at ℓ = n + 1; need m - 2 ≤ n + 1 (holds for n ≥ 1),
  -- n - 2 ≤ n + 1 (trivial), and ℓ + 3 ≤ m + n (i.e., n + 4 ≤ 4 + n, holds).
  have hsuf := cLeq_acell_suffix (m := 4) (n := n) (n + 1) (by omega)
    (by omega) (by omega)
  -- (4 + n - 2 - (n+1)) = 1, (4 + n - 1 - (n+1)) = 2, 1 * 2 / 2 = 1 in ℤ.
  -- push_cast normalises the casts; omega handles ℤ ediv 2 / 2 = 1.
  rw [hsuf]
  push_cast
  omega

end OrigamiCone

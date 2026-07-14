import Mathlib.Order.Interval.Finset.Nat
import OrigamiCone.Deg4FamilySum

/-!
# CE|EE family count assembly (Sub-6c of `thm:deg4count`)

The paper's **`thm:deg4count`** linear family `CE|EE` (paper L568-582) splits
the per-(corner, non-incident-side) admissibility count into two regimes:

* **`c = 2` regime** (antidiagonal arm exits on the left/top side at length 2
  for every admissible δ; horizontal arm sweeps rows): contributes
  `(Icc 2 (m-2)).card = m - 3` configurations.
* **`c ≥ 3` regime** (antidiagonal exits on the bottom/right side at length 2
  only at the largest admissible δ; one configuration per such `c`):
  contributes `(Icc 3 (n-1)).card = n - 3` configurations.

Their sum is `m + n - 6` per (corner, non-incident-side) bucket.  Four corners
× two non-incident sides per corner × `(m + n − 6)` configurations gives the
closed-form count `familyCEEE m n = 8 (m + n − 6)`.

This module performs the **integer arithmetic** of the assembly only.  The
underlying *geometric* facts that the L-shaped ridge has the claimed structure
and that `ℓ = 2` configurations are precisely the degree-4-yielding ones are
the content of the Ridge Lemma (`lem:ridge`, formalised in `RidgeMax.lean`)
and the per-(corner, side) ridge enumeration (paper L568-582).

Results:
* `ceee_per_bucket_count` — `|Icc 2 (m-2)| + |Icc 3 (n-1)| = m + n - 6`
  for `m, n ≥ 3` (the two-regime sum).
* `family_CEEE_decomposition` — `8 * (per-bucket count : ℤ) = familyCEEE m n`
  (factor `8` from four corners × two non-incident sides).

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Per-(corner, non-incident-side) configuration count for `CE|EE`.**
The two regimes (`c = 2` antidiagonal-fixed sweep over rows; `c ≥ 3` single
configuration per `c`) sum to `m + n − 6` for any `m, n ≥ 3`. -/
theorem ceee_per_bucket_count (hm : 3 ≤ m) (hn : 3 ≤ n) :
    (Finset.Icc 2 (m - 2)).card + (Finset.Icc 3 (n - 1)).card = m + n - 6 := by
  rw [Nat.card_Icc, Nat.card_Icc]
  -- (m - 2 + 1 - 2) + (n - 1 + 1 - 3) = m + n - 6 in ℕ.
  omega

/-- **CE|EE family count assembly.**  Four corners × two non-incident sides
per corner = `8` (corner, non-incident-side) buckets, each contributing
`m + n − 6` configurations.  This gives the closed-form family count
`familyCEEE m n = 8 (m + n − 6)`. -/
theorem family_CEEE_decomposition (hm : 3 ≤ m) (hn : 3 ≤ n) :
    8 * (((Finset.Icc 2 (m - 2)).card + (Finset.Icc 3 (n - 1)).card : ℕ) : ℤ)
      = familyCEEE m n := by
  rw [ceee_per_bucket_count hm hn]
  -- Goal: 8 * ((m + n - 6 : ℕ) : ℤ) = 8 * ((m : ℤ) + n - 6).
  unfold familyCEEE
  have hcast : ((m + n - 6 : ℕ) : ℤ) = (m : ℤ) + n - 6 := by omega
  rw [hcast]

end OrigamiCone

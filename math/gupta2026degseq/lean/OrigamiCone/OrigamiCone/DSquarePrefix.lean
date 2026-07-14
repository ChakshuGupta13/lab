import OrigamiCone.MedianSumHelpers
import OrigamiCone.AcellReflect
import OrigamiCone.Median

/-!
# `D(m, m)`: prefix half of the median sum (Section 4)

Building toward the headline closed form `D(m, m) = (m³ − m)/3` of Section 4
of the paper.  This module evaluates the **prefix half** of the median sum:

  `∑ ℓ ∈ [0, m − 1), min(c_ℓ, m² − c_ℓ) = ∑ j ∈ [0, m − 1), (j + 1)(j + 2)/2`,

where `c_ℓ = cLeq acell ℓ`.  On the prefix range `ℓ ∈ [0, m − 1)`:
* The small-triangle closed form gives `c_ℓ = (ℓ + 1)(ℓ + 2)/2`
  (`cLeq_acell_triangle`).
* On the same range, `2 · c_ℓ ≤ m²` (the prefix-of-min property), so
  `min(c_ℓ, m² − c_ℓ) = c_ℓ`.

The companion file `DSquareSuffix.lean` evaluates the symmetric suffix half,
and `DSquare.lean` combines them with the pyramid identity to conclude `D_mm`.

Results:
* `acell_prefix_term_le` — for `j < m − 1`, `(j + 1)(j + 2) ≤ m · m`;
* `acell_prefix_min_eq` — per-term reduction
  `min((j+1)(j+2)/2) (m·m − (j+1)(j+2)/2) = (j+1)(j+2)/2`;
* `acell_prefix_sum` — the prefix half of the median sum.

No `sorry`.
-/

namespace OrigamiCone

variable {m : ℕ}

/-- **Prefix product bound** (private helper).  For `j < m - 1`,
`(j + 1)(j + 2) ≤ m * m` in `ℕ`.  Note: `j < m - 1` in `ℕ` forces `m ≥ 2`. -/
private lemma acell_prefix_term_le_nat (j : ℕ) (hj : j < m - 1) :
    (j + 1) * (j + 2) ≤ m * m := by
  have hm : 2 ≤ m := by omega
  have h1 : j + 1 ≤ m - 1 := by omega
  have h2 : j + 2 ≤ m := by omega
  calc (j + 1) * (j + 2)
      ≤ (m - 1) * m := Nat.mul_le_mul h1 h2
    _ ≤ m * m := Nat.mul_le_mul_right m (by omega)

/-- **Prefix product bound** (ℤ cast).  For `j < m - 1`,
`(j + 1)(j + 2) ≤ m * m` in `ℤ`. -/
lemma acell_prefix_term_le (j : ℕ) (hj : j < m - 1) :
    ((j : ℤ) + 1) * ((j : ℤ) + 2) ≤ ((m * m : ℕ) : ℤ) := by
  have := acell_prefix_term_le_nat j hj
  exact_mod_cast this

/-- **Prefix per-term min reduction.**  For `j < m - 1`, the per-level `min`
in the median sum equals the smaller branch `(j + 1)(j + 2)/2` (which is the
`cLeq acell j` value by the small-triangle closed form). -/
lemma acell_prefix_min_eq (j : ℕ) (hj : j < m - 1) :
    min (((j : ℤ) + 1) * ((j : ℤ) + 2) / 2)
        (((m * m : ℕ) : ℤ) - ((j : ℤ) + 1) * ((j : ℤ) + 2) / 2)
      = ((j : ℤ) + 1) * ((j : ℤ) + 2) / 2 := by
  -- 2 * cLeq = (j+1)(j+2) ≤ m·m, so cLeq ≤ m·m − cLeq.
  have h2 := int_two_mul_div_consec (j : ℤ)
  have hle := acell_prefix_term_le j hj
  apply min_eq_left
  linarith

/-- **Prefix half of the median sum.**  On the prefix range `[0, m − 1)`
(in `ℤ`), the median-sum integrand `min(c_ℓ, m² − c_ℓ)` reduces to the
small-triangle closed form, yielding a `ℕ`-indexed sum of triangle numbers
`∑ j ∈ range (m − 1), (j + 1)(j + 2)/2`. -/
theorem acell_prefix_sum (hm : 2 ≤ m) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) ((m : ℤ) - 1),
        min (cLeq (acell (m := m) (n := m)) ℓ)
            (((m * m : ℕ) : ℤ) - cLeq (acell (m := m) (n := m)) ℓ)
      = ∑ j ∈ Finset.range (m - 1),
          (((j : ℤ) + 1) * ((j : ℤ) + 2) / 2) := by
  -- Reindex Ico (0 : ℤ) (m - 1 : ℤ) as range ((m - 1 : ℤ).toNat = m - 1).
  have h0_le : (0 : ℤ) ≤ ((m : ℤ) - 1) := by
    have : (1 : ℤ) ≤ m := by exact_mod_cast (by omega : 1 ≤ m)
    linarith
  rw [sum_Ico_int_eq_sum_range ((m : ℤ) - 1) h0_le]
  -- Reduce ((m : ℤ) - 1).toNat to (m - 1 : ℕ).
  have hcast_m1 : ((m - 1 : ℕ) : ℤ) = (m : ℤ) - 1 := by
    have : (1 : ℕ) ≤ m := by omega
    push_cast [Nat.cast_sub this]; ring
  have htoNat : ((m : ℤ) - 1).toNat = m - 1 := by
    rw [← hcast_m1, Int.toNat_natCast]
  rw [htoNat]
  -- Per-term reduction.
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.mem_range] at hj
  have hj_lt_m : j < m := by omega
  rw [cLeq_acell_triangle j hj_lt_m hj_lt_m]
  exact acell_prefix_min_eq j hj

end OrigamiCone

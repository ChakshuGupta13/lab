import OrigamiCone.DSquarePrefix

/-!
# `D(m, m)`: suffix half of the median sum (Section 4)

Building toward the headline closed form `D(m, m) = (m³ − m)/3`.  This module
evaluates the **suffix half** of the median sum and reflects it onto the
prefix shape:

  `∑ ℓ ∈ [m − 1, 2(m − 1)), min(c_ℓ, m² − c_ℓ) = ∑ j ∈ [0, m − 1), (j+1)(j+2)/2`.

On the suffix range, the upper-suffix closed form (`cLeq_acell_suffix`) gives
`c_ℓ = m² − X(ℓ)` with `X(ℓ) := (2m − 2 − ℓ)(2m − 1 − ℓ)/2`, and
`X(ℓ) ≤ m² − X(ℓ)`, so `min = X(ℓ)`.  Reindexing `ℓ = j + (m − 1)` gives
`X(j + m − 1) = ((m − 2 − j) + 1)((m − 2 − j) + 2)/2`, whose reflection
`j ↦ m − 2 − j` (via `Finset.sum_range_reflect`) yields the prefix shape.

Results:
* `acell_suffix_term_le` — for `j < m − 1`,
  `((m − 2 − j : ℤ) + 1) * ((m − 2 − j : ℤ) + 2) ≤ m · m`;
* `acell_suffix_sum` — the suffix half, reduced and reflected to the prefix.

No `sorry`.
-/

namespace OrigamiCone

variable {m : ℕ}

/-- **Suffix product bound** (ℤ).  For `j < m - 1` (which forces `m ≥ 2`),
`((m - 2 - j : ℤ) + 1) * ((m - 2 - j : ℤ) + 2) ≤ m · m`. -/
lemma acell_suffix_term_le (j : ℕ) (hj : j < m - 1) :
    ((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2) ≤ ((m * m : ℕ) : ℤ) := by
  have hm : 2 ≤ m := by omega
  have hj_z : (j : ℤ) ≥ 0 := Int.natCast_nonneg _
  have hj2_le : (j : ℤ) + 2 ≤ m := by exact_mod_cast (by omega : j + 2 ≤ m)
  have h1 : ((m : ℤ) - 2 - j + 1) ≤ (m - 1 : ℤ) := by linarith
  have h2 : ((m : ℤ) - 2 - j + 2) ≤ (m : ℤ) := by linarith
  have hb : (0 : ℤ) ≤ ((m : ℤ) - 2 - j + 2) := by linarith
  have h_inter : ((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2)
               ≤ ((m - 1 : ℤ)) * ((m : ℤ)) :=
    mul_le_mul h1 h2 hb (by linarith)
  have h_mm : ((m - 1 : ℤ)) * ((m : ℤ)) ≤ ((m * m : ℕ) : ℤ) := by
    push_cast; nlinarith
  linarith

/-- **Suffix half of the median sum** (post-reflection).  On the suffix range
`[m − 1, 2(m − 1))` (in `ℤ`), the median-sum integrand reduces — via
`cLeq_acell_suffix` — to the upper-triangle contribution
`((m − 2 − j) + 1)((m − 2 − j) + 2)/2`, which reflects to the prefix shape
`(j + 1)(j + 2)/2`. -/
theorem acell_suffix_sum (hm : 2 ≤ m) :
    ∑ ℓ ∈ Finset.Ico ((m : ℤ) - 1) (2 * ((m : ℤ) - 1)),
        min (cLeq (acell (m := m) (n := m)) ℓ)
            (((m * m : ℕ) : ℤ) - cLeq (acell (m := m) (n := m)) ℓ)
      = ∑ j ∈ Finset.range (m - 1),
          (((j : ℤ) + 1) * ((j : ℤ) + 2) / 2) := by
  -- Reindex Ico (m-1) (2(m-1)) as range ((m-1) - (m-1)).toNat = (m-1).toNat = m-1.
  have hab : ((m : ℤ) - 1) ≤ 2 * ((m : ℤ) - 1) := by
    have : (1 : ℤ) ≤ m := by exact_mod_cast (by omega : 1 ≤ m)
    linarith
  rw [sum_Ico_int_shift ((m : ℤ) - 1) (2 * ((m : ℤ) - 1)) hab]
  -- (2(m-1) - (m-1)).toNat = (m-1).toNat = m-1.
  have hcast_m1 : ((m - 1 : ℕ) : ℤ) = (m : ℤ) - 1 := by
    have : (1 : ℕ) ≤ m := by omega
    push_cast [Nat.cast_sub this]; ring
  have hshift_eq : 2 * ((m : ℤ) - 1) - ((m : ℤ) - 1) = ((m - 1 : ℕ) : ℤ) := by
    rw [hcast_m1]; ring
  have htoNat : (2 * ((m : ℤ) - 1) - ((m : ℤ) - 1)).toNat = m - 1 := by
    rw [hshift_eq, Int.toNat_natCast]
  rw [htoNat]
  -- Reduce each suffix term to ((m - 2 - j : ℤ) + 1) * ((m - 2 - j : ℤ) + 2) / 2,
  -- then apply Finset.sum_range_reflect to match the prefix shape.
  rw [show
    ∑ j ∈ Finset.range (m - 1),
        min (cLeq (acell (m := m) (n := m)) ((j : ℤ) + ((m : ℤ) - 1)))
            (((m * m : ℕ) : ℤ) - cLeq (acell (m := m) (n := m))
              ((j : ℤ) + ((m : ℤ) - 1)))
      = ∑ j ∈ Finset.range (m - 1),
          (((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2) / 2) from ?_]
  · -- Reflection j ↦ (m-1)-1-j = m-2-j brings ((m-2-j)+1)((m-2-j)+2) to (j+1)(j+2).
    have hreflect := Finset.sum_range_reflect
      (fun k => (((k : ℤ) + 1) * ((k : ℤ) + 2) / 2)) (m - 1)
    -- hreflect: ∑ j ∈ range (m-1), f ((m-1)-1-j) = ∑ j ∈ range (m-1), f j
    -- Match: ((m-1)-1-j : ℤ) + 1 = (m : ℤ) - 2 - j + 1 (when j ≤ m-2).
    have hrw : ∀ j ∈ Finset.range (m - 1),
        ((((((m - 1) - 1 - j : ℕ) : ℤ) + 1) * ((((m - 1) - 1 - j : ℕ) : ℤ) + 2) / 2)
            : ℤ)
          = ((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2) / 2 := by
      intro j hj
      rw [Finset.mem_range] at hj
      have hcast : (((m - 1) - 1 - j : ℕ) : ℤ) = (m : ℤ) - 2 - j := by
        have h1 : (1 : ℕ) ≤ m - 1 := by omega
        have h2 : 1 + j ≤ m - 1 := by omega
        have hstep : ((m - 1) - 1 - j : ℕ) = (m - 1) - 1 - j := rfl
        rw [hstep]
        have : ((((m - 1) - 1 - j : ℕ) : ℤ)) = ((m - 1 - 1 - j : ℕ) : ℤ) := rfl
        rw [this]
        have h3 : ((m - 1 - 1 - j : ℕ) : ℤ) = ((m : ℤ) - 1 - 1 - j) := by
          have hsub1 : (1 : ℕ) ≤ m := by omega
          have hsub2 : (1 : ℕ) ≤ m - 1 := by omega
          have hsub3 : (j : ℕ) ≤ m - 1 - 1 := by omega
          rw [Nat.cast_sub hsub3, Nat.cast_sub hsub2, Nat.cast_sub hsub1]
          push_cast; ring
        rw [h3]; ring
      rw [hcast]
    rw [Finset.sum_congr rfl hrw] at hreflect
    -- hreflect now: ∑ j ∈ range (m-1), ((m-2-j)+1)((m-2-j)+2)/2 = ∑ j (j+1)(j+2)/2.
    exact hreflect
  · -- Per-term reduction: min cLeq (m² - cLeq) = X where X = ((m-2-j)+1)((m-2-j)+2)/2.
    apply Finset.sum_congr rfl
    intro j hj
    rw [Finset.mem_range] at hj
    -- Translate (j : ℤ) + ((m : ℤ) - 1) into a ℕ index ℓ = j + (m - 1).
    set ℓ : ℕ := j + (m - 1) with hℓdef
    have hcast_ℓ : ((j : ℤ) + ((m : ℤ) - 1)) = (ℓ : ℤ) := by
      rw [hℓdef]
      have : (1 : ℕ) ≤ m := by omega
      push_cast [Nat.cast_sub this]; ring
    rw [hcast_ℓ]
    have hℓ_lo : m - 2 ≤ ℓ := by rw [hℓdef]; omega
    have hℓ_hi : ℓ + 3 ≤ m + m := by rw [hℓdef]; omega
    rw [cLeq_acell_suffix ℓ hℓ_lo hℓ_lo hℓ_hi]
    -- cLeq = (m*m) - (m+m-2-ℓ)*(m+m-1-ℓ)/2.  Rewrite the argument shape:
    -- (m + m - 2 - ℓ : ℤ) = (m : ℤ) - 2 - j + 1, (m + m - 1 - ℓ) = (m : ℤ) - 2 - j + 2.
    have hj2_le : j + 1 + 1 ≤ m := by omega
    have h_mn_2 : ((m : ℤ) + m - 2 - ℓ) = ((m : ℤ) - 2 - j + 1) := by
      rw [hℓdef]
      have h1 : (1 : ℕ) ≤ m := by omega
      push_cast [Nat.cast_sub h1, Nat.cast_sub hj2_le]; ring
    have h_mn_1 : ((m : ℤ) + m - 1 - ℓ) = ((m : ℤ) - 2 - j + 2) := by
      rw [hℓdef]
      have h1 : (1 : ℕ) ≤ m := by omega
      push_cast [Nat.cast_sub h1, Nat.cast_sub hj2_le]; ring
    rw [h_mn_2, h_mn_1]
    -- Goal: min ((m*m) - X) (m² - ((m*m) - X)) = X, where X = ((m-2-j)+1)((m-2-j)+2)/2.
    -- Simplifies to: min (m² - X) X = X via min_eq_right, given X ≤ m² - X.
    set X : ℤ := ((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2) / 2 with hXdef
    have h_2X_le : 2 * X ≤ ((m * m : ℕ) : ℤ) := by
      rw [hXdef]
      have h2X_eq : 2 * (((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2) / 2)
                  = ((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2) := by
        have hev : (2 : ℤ) ∣ ((m : ℤ) - 2 - j + 1) * ((m : ℤ) - 2 - j + 2) := by
          have h := int_two_dvd_consec ((m : ℤ) - 2 - j)
          -- int_two_dvd_consec gives 2 ∣ (k+1)*(k+2), match k = m - 2 - j.
          exact h
        exact Int.mul_ediv_cancel' hev
      have h_prod_le := acell_suffix_term_le j hj
      linarith
    rw [show ((m * m : ℕ) : ℤ)
            - (((m * m : ℕ) : ℤ) - X) = X from by ring]
    exact min_eq_right (by linarith)

end OrigamiCone

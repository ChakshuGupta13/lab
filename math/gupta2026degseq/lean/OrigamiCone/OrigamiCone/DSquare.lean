import OrigamiCone.DSquareSuffix

/-!
# `D(m, m) = (m³ − m)/3` (Section 4)

The headline closed form of Section 4 of the paper:

  `D(m, m) := \min_K \sum_{v \in \mathrm{Cell}\,m\,m} |\mathrm{acell}(v) - K|
            = \frac{(m-1) \cdot m \cdot (m+1)}{3} = \frac{m^3 - m}{3}`,

for `m ≥ 2`.

The proof assembles:
1. `isMedianMin_sum_min` (`Median.lean`) — `disp(φ) = Σ_ℓ min(c_ℓ, N − c_ℓ)`.
2. `acell_prefix_sum` (`DSquarePrefix.lean`) — prefix half reduction.
3. `acell_suffix_sum` (`DSquareSuffix.lean`) — suffix half reduction.
4. `triple_pyramid_sum` (`MedianSumHelpers.lean`) — `3·Σ(k+1)(k+2) = n(n+1)(n+2)`.

On `Cell m m` with `L = 0`, `U = 2(m − 1)`, the level range splits exactly at
`ℓ = m − 1` (no middle band).  The prefix and suffix halves both reduce to
`∑ j ∈ range (m − 1), (j+1)(j+2)/2`, so the median sum is **twice** that.
Doubled gives `∑ j (j+1)(j+2)`, evaluated by the pyramid identity to
`(m−1)·m·(m+1)/3`.

Results:
* `D_mm` — `IsMedianMin acell ((m−1)·m·(m+1)/3)` on `Cell m m` for `m ≥ 2`.

No `sorry`.
-/

namespace OrigamiCone

variable {m : ℕ}

/-- Range bound for `acell` on `Cell m m` (private helper).  Vacuous when
`Cell m m = ∅` (i.e. `m = 0`); otherwise gives `0 ≤ acell v ≤ 2(m - 1)`. -/
private lemma acell_range_mm (v : Cell m m) :
    (0 : ℤ) ≤ acell v ∧ acell v ≤ (2 * ((m : ℤ) - 1)) := by
  have h1 : v.1.val < m := v.1.isLt
  have h2 : v.2.val < m := v.2.isLt
  unfold acell
  refine ⟨by positivity, ?_⟩
  have h1' : (v.1.val : ℤ) ≤ (m : ℤ) - 1 := by
    have : (v.1.val : ℤ) + 1 ≤ (m : ℤ) := by exact_mod_cast h1
    linarith
  have h2' : (v.2.val : ℤ) ≤ (m : ℤ) - 1 := by
    have : (v.2.val : ℤ) + 1 ≤ (m : ℤ) := by exact_mod_cast h2
    linarith
  linarith

/-- **Median sum on the square**, evaluated.  For `m ≥ 2`, the
`Σ_ℓ min(c_ℓ, m·m − c_ℓ)` evaluation over `[0, 2(m − 1))` equals
`((m − 1) · m · (m + 1) / 3 : ℕ)` (cast to ℤ).  The arithmetic core of
`D_mm`, assembled from the prefix/suffix halves and the pyramid identity. -/
private lemma medianSum_acell_mm (hm : 2 ≤ m) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) (2 * ((m : ℤ) - 1)),
        min (cLeq (acell (m := m) (n := m)) ℓ)
            (((m * m : ℕ) : ℤ) - cLeq (acell (m := m) (n := m)) ℓ)
      = (((m - 1) * m * (m + 1) / 3 : ℕ) : ℤ) := by
  -- Split the Ico at ℓ = m - 1.
  have h0_le : (0 : ℤ) ≤ ((m : ℤ) - 1) := by
    have : (1 : ℤ) ≤ m := by exact_mod_cast (by omega : 1 ≤ m)
    linarith
  have hmid_le : ((m : ℤ) - 1) ≤ 2 * ((m : ℤ) - 1) := by linarith
  rw [show (Finset.Ico (0 : ℤ) (2 * ((m : ℤ) - 1))) =
        Finset.Ico (0 : ℤ) ((m : ℤ) - 1)
          ∪ Finset.Ico ((m : ℤ) - 1) (2 * ((m : ℤ) - 1)) from
        (Finset.Ico_union_Ico_eq_Ico h0_le hmid_le).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 0 ((m : ℤ) - 1) (2 * ((m : ℤ) - 1)))]
  -- Apply the prefix and suffix reductions.
  rw [acell_prefix_sum hm, acell_suffix_sum hm]
  -- Both halves are now ∑ j ∈ range (m - 1), (j+1)(j+2)/2.  Combine via doubling.
  rw [show ∑ j ∈ Finset.range (m - 1), (((j : ℤ) + 1) * ((j : ℤ) + 2) / 2)
        + ∑ j ∈ Finset.range (m - 1), (((j : ℤ) + 1) * ((j : ℤ) + 2) / 2)
      = 2 * ∑ j ∈ Finset.range (m - 1),
            (((j : ℤ) + 1) * ((j : ℤ) + 2) / 2) from by ring]
  -- Pull 2 inside: 2 · ((j+1)(j+2)/2) = (j+1)(j+2) (since (j+1)(j+2) is even).
  rw [Finset.mul_sum]
  rw [show (∑ j ∈ Finset.range (m - 1),
            2 * (((j : ℤ) + 1) * ((j : ℤ) + 2) / 2))
        = ∑ j ∈ Finset.range (m - 1), ((j : ℤ) + 1) * ((j : ℤ) + 2) from
        Finset.sum_congr rfl (fun j _ => int_two_mul_div_consec _)]
  -- Apply the pyramid identity.
  have hpyr := triple_pyramid_sum (m - 1)
  -- hpyr : 3 * Σ (k+1)(k+2) = (m-1) * m * (m+1) (in ℤ).
  -- Goal: Σ (j+1)(j+2) = ((m-1) * m * (m+1) / 3 : ℕ) cast to ℤ.
  have hcast_step : (((m - 1) * m * (m + 1) / 3 : ℕ) : ℤ)
                  = ((m - 1 : ℤ)) * m * (m + 1) / 3 := by
    rw [Int.natCast_ediv]
    push_cast
    have h : (1 : ℕ) ≤ m := by omega
    rw [Nat.cast_sub h]; push_cast; ring
  -- Reformulate hpyr with the cast targets matching our goal.
  have hpyr_target : (((m - 1 : ℕ) : ℤ)) * (((m - 1 : ℕ) + 1)) * (((m - 1 : ℕ) + 2))
                  = ((m - 1 : ℤ)) * m * (m + 1) := by
    have h : (1 : ℕ) ≤ m := by omega
    rw [Nat.cast_sub h]; push_cast; ring
  rw [hcast_step]
  -- The hpyr LHS is `3 * Σ ...`; rearrange to `Σ ... = (... / 3)` using divisibility.
  have h3_dvd : (3 : ℤ) ∣ ((m - 1 : ℤ)) * m * (m + 1) := by
    refine ⟨∑ k ∈ Finset.range (m - 1), ((k : ℤ) + 1) * ((k : ℤ) + 2), ?_⟩
    linarith [hpyr_target, hpyr]
  -- From `3 * X = Y` and `3 ∣ Y`, get `X = Y / 3`.
  have hX_eq : ∑ k ∈ Finset.range (m - 1), ((k : ℤ) + 1) * ((k : ℤ) + 2)
             = ((m - 1 : ℤ)) * m * (m + 1) / 3 := by
    obtain ⟨q, hq⟩ := h3_dvd
    have h3X : (3 : ℤ) * (∑ k ∈ Finset.range (m - 1), ((k : ℤ) + 1) * ((k : ℤ) + 2))
             = ((m - 1 : ℤ)) * m * (m + 1) := by
      linarith [hpyr_target, hpyr]
    have hX : ∑ k ∈ Finset.range (m - 1), ((k : ℤ) + 1) * ((k : ℤ) + 2) = q := by
      linarith [hq]
    rw [hq, hX]
    rw [Int.mul_ediv_cancel_left]
    decide
  exact hX_eq

/-! ## The main theorem -/

/-- **`D(m, m) = (m³ − m)/3`** (the headline closed form of Section 4).

The minimised dispersion of the antidiagonal `acell : Cell m m → ℤ` on the
square `m × m` grid (`m ≥ 2`) equals `(m-1) · m · (m+1) / 3 = (m³ − m)/3`.

Proof: combine `isMedianMin_sum_min` (the median characterisation,
`Median.lean`) with `medianSum_acell_mm` (the closed-form evaluation, this
file).  The latter routes through `acell_prefix_sum` (`DSquarePrefix.lean`),
`acell_suffix_sum` (`DSquareSuffix.lean`), and `triple_pyramid_sum`
(`MedianSumHelpers.lean`). -/
theorem D_mm (hm : 2 ≤ m) :
    IsMedianMin (acell (m := m) (n := m))
                (((m - 1) * m * (m + 1) / 3 : ℕ) : ℤ) := by
  have hLU : (0 : ℤ) ≤ 2 * ((m : ℤ) - 1) := by
    have : (1 : ℤ) ≤ m := by exact_mod_cast (by omega : 1 ≤ m)
    linarith
  have hφ : ∀ v : Cell m m, (0 : ℤ) ≤ acell v ∧ acell v ≤ (2 * ((m : ℤ) - 1)) :=
    acell_range_mm
  have hmid := isMedianMin_sum_min
    (acell (m := m) (n := m)) 0 (2 * ((m : ℤ) - 1)) hLU hφ
  have hcard : (Fintype.card (Cell m m) : ℤ) = ((m * m : ℕ) : ℤ) := by
    simp [Cell, Fintype.card_prod, Fintype.card_fin]
  rw [hcard] at hmid
  rw [medianSum_acell_mm hm] at hmid
  exact hmid

end OrigamiCone

import OrigamiCone.MedianSumHelpers

/-!
# `D(3, n)`: parity-cased triangle-sum identity (Section 4)

The arithmetic core of `D(3, n) = ⌊3n²/4⌋ + 2` (Section 4 of the paper):

  `∑_{ℓ ∈ Ico 1 n} min(ℓ, n − ℓ) = ((n * n : ℕ) : ℤ) / 4`  (in `ℤ`, truncating).

This is a **pure integer-arithmetic identity** (no reference to `cLeq` or
the grid).  Combined with `cLeq_acell_three_mid` / `cLeq_acell_three_zero` /
`cLeq_acell_three_top` and the median characterisation, it gives
`D(3, n) = ⌊3n²/4⌋ + 2` in the sibling `D3n.lean` module (next session).

Proof strategy: parity case-split on `n` via `Nat.even_or_odd`.

- **Even** `n = 2k`: split `Ico 1 (k + k)` at `k + 1`.  Lower half
  `[1, k+1)` has `min = ℓ`; sum = `k(k+1)/2`.  Upper half `[k+1, 2k)` has
  `min = 2k − ℓ`; sum = `(k−1)k/2`.  Total = `k²`.
- **Odd** `n = 2k + 1`: split at `k + 1`.  Lower `[1, k+1)`: `Σ ℓ = k(k+1)/2`.
  Upper `[k+1, 2k+1)`: `Σ (2k+1 − ℓ) = k(k+1)/2`.  Total = `k(k+1)`.

Results:
* `sum_one_to_k_two_mul` — `2 · Σ_{j < k} (j + 1) = k(k + 1)` (Gauss);
* `min_triangle_sum` — the main parity-cased identity.

No `sorry`.
-/

namespace OrigamiCone

/-! ## Gauss-sum doubled helper -/

/-- `2 · ∑_{j < k} (j + 1) = k · (k + 1)` in `ℤ`.  Doubled form to avoid Nat
division.  Proved by induction on `k`. -/
lemma sum_one_to_k_two_mul (k : ℕ) :
    2 * (∑ j ∈ Finset.range k, ((j : ℤ) + 1)) = (k : ℤ) * (k + 1) := by
  induction k with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]; push_cast; linarith [ih]

/-! ## Reflected Gauss-sum (private helper for the upper half) -/

/-- `2 · Σ_{j < m} (c - 1 - j) = (2c - m) · m - m` in `ℤ`.  Useful form: with
`c = k`, `m = k - 1`, gives `2 · Σ_{j < k-1} (k - 1 - j) = (k-1)·k`.  With
`c = k`, `m = k`, gives `2 · Σ_{j < k} (k - 1 - j) = (k-1)·k`.

Equivalent statement, written as `Σ_{j < m} (c - 1 - j)` with `c` decoupled
from `m`: doubled = `m · (2c - 1 - m)` (via `Σ(c-1) - Σj = m(c-1) - m(m-1)/2`
multiplied by 2 = `2m(c-1) - m(m-1) = m(2c - 1 - m)`). -/
private lemma sum_c_minus_one_minus_j_two_mul (c : ℤ) (m : ℕ) :
    2 * (∑ j ∈ Finset.range m, (c - 1 - (j : ℤ)))
      = (m : ℤ) * (2 * c - 1 - m) := by
  induction m with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]; push_cast; linarith [ih]

/-! ## Parity cases (private helpers) -/

/-- **Even case**: for `n = 2k = k + k`, the inner triangle sum equals `k²`. -/
private lemma min_triangle_sum_even (k : ℕ) :
    ∑ ℓ ∈ Finset.Ico (1 : ℤ) ((k + k : ℕ) : ℤ), min ℓ (((k + k : ℕ) : ℤ) - ℓ)
      = (k : ℤ) * k := by
  have hcast : ((k + k : ℕ) : ℤ) = (k : ℤ) + k := by push_cast; ring
  rw [hcast]
  by_cases hk0 : k = 0
  · subst hk0; simp
  have hk1 : 1 ≤ (k : ℤ) := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hk0
  have h_lo_bd : (1 : ℤ) ≤ (k : ℤ) + 1 := by linarith
  have h_mid_bd : ((k : ℤ) + 1) ≤ (k : ℤ) + k := by linarith
  -- Split Ico at k+1.
  rw [show Finset.Ico (1 : ℤ) ((k : ℤ) + k)
        = Finset.Ico (1 : ℤ) ((k : ℤ) + 1) ∪ Finset.Ico ((k : ℤ) + 1) ((k : ℤ) + k) from
      (Finset.Ico_union_Ico_eq_Ico h_lo_bd h_mid_bd).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 1 ((k : ℤ) + 1) ((k : ℤ) + k))]
  -- Lower half: ℓ ∈ [1, k+1), min = ℓ.
  have h_lo_term : ∀ ℓ ∈ Finset.Ico (1 : ℤ) ((k : ℤ) + 1),
      min ℓ ((k : ℤ) + k - ℓ) = ℓ := by
    intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    apply min_eq_left
    linarith
  -- Upper half: ℓ ∈ [k+1, k+k), min = (k+k) - ℓ.
  have h_hi_term : ∀ ℓ ∈ Finset.Ico ((k : ℤ) + 1) ((k : ℤ) + k),
      min ℓ ((k : ℤ) + k - ℓ) = (k : ℤ) + k - ℓ := by
    intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    apply min_eq_right
    linarith
  rw [Finset.sum_congr rfl h_lo_term, Finset.sum_congr rfl h_hi_term]
  -- Reindex lower: Ico 1 (k+1) → range k via j → j + 1.
  rw [sum_Ico_int_shift (1 : ℤ) ((k : ℤ) + 1) h_lo_bd]
  have htoNat_lo : (((k : ℤ) + 1) - 1).toNat = k := by
    have : ((k : ℤ) + 1) - 1 = (k : ℤ) := by ring
    rw [this, Int.toNat_natCast]
  rw [htoNat_lo]
  -- Reindex upper: Ico (k+1) (k+k) → range (k-1) via j → j + (k+1).
  rw [sum_Ico_int_shift ((k : ℤ) + 1) ((k : ℤ) + k) h_mid_bd]
  have htoNat_hi : (((k : ℤ) + k) - ((k : ℤ) + 1)).toNat = k - 1 := by
    have hcast' : (((k : ℤ) + k) - ((k : ℤ) + 1)) = ((k - 1 : ℕ) : ℤ) := by
      have h1 : (1 : ℕ) ≤ k := Nat.one_le_iff_ne_zero.mpr hk0
      push_cast [Nat.cast_sub h1]; ring
    rw [hcast', Int.toNat_natCast]
  rw [htoNat_hi]
  -- Simplify each summand:
  -- Lower: f((j : ℤ) + 1) where f ℓ = ℓ.  So summand = j + 1.
  -- Upper: (k+k) - ((j : ℤ) + (k+1)) = k - 1 - j.
  rw [show ∑ j ∈ Finset.range k, ((j : ℤ) + 1)
        = ∑ j ∈ Finset.range k, ((j : ℤ) + 1) from rfl]
  rw [show ∑ j ∈ Finset.range (k - 1), ((k : ℤ) + k - ((j : ℤ) + ((k : ℤ) + 1)))
        = ∑ j ∈ Finset.range (k - 1), ((k : ℤ) - 1 - (j : ℤ)) from by
      apply Finset.sum_congr rfl
      intros; ring]
  -- Combine via 2·LHS = k(k+1) + (k-1)k = 2k².
  have h_lower_2x := sum_one_to_k_two_mul k
  -- 2 · ∑ (k - 1 - j) over range (k-1) = (k-1) · (2k - 1 - (k-1)) = (k-1) · k.
  have h_upper_2x := sum_c_minus_one_minus_j_two_mul (k : ℤ) (k - 1)
  -- h_upper_2x : 2 * (∑ j ∈ range (k-1), ((k : ℤ) - 1 - j)) = ((k-1 : ℕ) : ℤ) * (2k - 1 - (k-1 : ℕ))
  -- Simplify the right-hand side: ((k - 1 : ℕ) : ℤ) = (k : ℤ) - 1, so RHS = (k - 1) * (2k - 1 - (k - 1)) = (k - 1) * k.
  have hcast_km1 : ((k - 1 : ℕ) : ℤ) = (k : ℤ) - 1 := by
    have : (1 : ℕ) ≤ k := Nat.one_le_iff_ne_zero.mpr hk0
    push_cast [Nat.cast_sub this]; ring
  rw [hcast_km1] at h_upper_2x
  -- h_upper_2x : 2 * (∑ j ∈ range (k - 1), ((k : ℤ) - 1 - j)) = ((k : ℤ) - 1) * (2k - 1 - ((k : ℤ) - 1))
  --            = (k - 1) * k.
  linarith

/-- **Odd case**: for `n = 2k + 1`, the inner triangle sum equals `k(k+1)`. -/
private lemma min_triangle_sum_odd (k : ℕ) :
    ∑ ℓ ∈ Finset.Ico (1 : ℤ) ((2 * k + 1 : ℕ) : ℤ),
        min ℓ (((2 * k + 1 : ℕ) : ℤ) - ℓ)
      = (k : ℤ) * (k + 1) := by
  have hcast : ((2 * k + 1 : ℕ) : ℤ) = 2 * (k : ℤ) + 1 := by push_cast; ring
  rw [hcast]
  by_cases hk0 : k = 0
  · subst hk0; simp
  have hk1 : 1 ≤ (k : ℤ) := by exact_mod_cast Nat.one_le_iff_ne_zero.mpr hk0
  have h_lo_bd : (1 : ℤ) ≤ (k : ℤ) + 1 := by linarith
  have h_mid_bd : ((k : ℤ) + 1) ≤ 2 * (k : ℤ) + 1 := by linarith
  -- Split Ico at k+1.
  rw [show Finset.Ico (1 : ℤ) (2 * (k : ℤ) + 1)
        = Finset.Ico (1 : ℤ) ((k : ℤ) + 1) ∪ Finset.Ico ((k : ℤ) + 1) (2 * (k : ℤ) + 1) from
      (Finset.Ico_union_Ico_eq_Ico h_lo_bd h_mid_bd).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 1 ((k : ℤ) + 1) (2 * (k : ℤ) + 1))]
  -- Lower half: ℓ ∈ [1, k+1), min = ℓ.  At ℓ=k, 2ℓ = 2k < 2k+1, so ℓ < n - ℓ.
  have h_lo_term : ∀ ℓ ∈ Finset.Ico (1 : ℤ) ((k : ℤ) + 1),
      min ℓ (2 * (k : ℤ) + 1 - ℓ) = ℓ := by
    intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    apply min_eq_left
    linarith
  -- Upper half: ℓ ∈ [k+1, 2k+1), min = (2k+1) - ℓ.
  have h_hi_term : ∀ ℓ ∈ Finset.Ico ((k : ℤ) + 1) (2 * (k : ℤ) + 1),
      min ℓ (2 * (k : ℤ) + 1 - ℓ) = 2 * (k : ℤ) + 1 - ℓ := by
    intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    apply min_eq_right
    linarith
  rw [Finset.sum_congr rfl h_lo_term, Finset.sum_congr rfl h_hi_term]
  -- Reindex lower: Ico 1 (k+1) → range k via j → j+1.
  rw [sum_Ico_int_shift (1 : ℤ) ((k : ℤ) + 1) h_lo_bd]
  have htoNat_lo : (((k : ℤ) + 1) - 1).toNat = k := by
    have : ((k : ℤ) + 1) - 1 = (k : ℤ) := by ring
    rw [this, Int.toNat_natCast]
  rw [htoNat_lo]
  -- Reindex upper: Ico (k+1) (2k+1) → range k via j → j + (k+1).
  rw [sum_Ico_int_shift ((k : ℤ) + 1) (2 * (k : ℤ) + 1) h_mid_bd]
  have htoNat_hi : (2 * (k : ℤ) + 1 - ((k : ℤ) + 1)).toNat = k := by
    have : 2 * (k : ℤ) + 1 - ((k : ℤ) + 1) = (k : ℤ) := by ring
    rw [this, Int.toNat_natCast]
  rw [htoNat_hi]
  -- Simplify upper summand: 2k+1 - (j + k+1) = k - j.
  rw [show ∑ j ∈ Finset.range k, (2 * (k : ℤ) + 1 - ((j : ℤ) + ((k : ℤ) + 1)))
        = ∑ j ∈ Finset.range k, ((k : ℤ) - (j : ℤ)) from by
      apply Finset.sum_congr rfl
      intros; ring]
  -- Convert ∑ (k - j) to ∑ (k - 1 - j) + ∑ 1, then use h_upper_2x.
  -- Actually simpler: (k - j) = ((k - 1 - j) + 1).  But (k - 1 - j) here is ℤ.
  -- Σ (k - j) = Σ ((k - 1 - j) + 1) = Σ (k - 1 - j) + k.
  -- And 2·(Σ (k - 1 - j)) = k·(k - 1) by sum_k_minus_one_minus_j_two_mul (k).
  -- So Σ (k - j) = (k·(k-1)/2) + k.  And combined with lower Σ (j+1) = k(k+1)/2:
  -- Total = k(k+1)/2 + k(k-1)/2 + k = k² + k = k(k+1).  ✓
  have h_lower_2x := sum_one_to_k_two_mul k
  -- For upper, ∑ (k - j) = ∑ ((k - 1 - j) + 1) = ∑ (k - 1 - j) + k.
  have h_upper_step : ∑ j ∈ Finset.range k, ((k : ℤ) - (j : ℤ))
                   = (∑ j ∈ Finset.range k, ((k : ℤ) - 1 - (j : ℤ))) + k := by
    have h_rew : ∑ j ∈ Finset.range k, ((k : ℤ) - (j : ℤ))
              = ∑ j ∈ Finset.range k, (((k : ℤ) - 1 - (j : ℤ)) + 1) := by
      apply Finset.sum_congr rfl; intros; ring
    rw [h_rew, Finset.sum_add_distrib]
    simp [Finset.sum_const, Finset.card_range]
  rw [h_upper_step]
  -- 2 · ∑ (k - 1 - j) over range k = k · (2k - 1 - k) = k · (k - 1).
  have h_upper_2x := sum_c_minus_one_minus_j_two_mul (k : ℤ) k
  -- Now: 2·(∑(j+1) + (∑(k-1-j) + k)) = k(k+1) + (k(k-1) + 2k) = 2k² + 2k = 2k(k+1).
  linarith

/-! ## Main parity-cased identity -/

/-- **The inner triangle sum identity** for `D(3, n)`.

`∑_{ℓ ∈ Ico 1 n} min(ℓ, n − ℓ) = ((n * n : ℕ) : ℤ) / 4`  (in `ℤ`, truncating).

The arithmetic core of `D(3, n) = ⌊3n²/4⌋ + 2`.  For even `n = 2k` the inner
sum equals `k²`; for odd `n = 2k + 1` it equals `k(k + 1)`. -/
theorem min_triangle_sum (n : ℕ) :
    ∑ ℓ ∈ Finset.Ico (1 : ℤ) (n : ℤ), min ℓ ((n : ℤ) - ℓ)
      = ((n * n : ℕ) : ℤ) / 4 := by
  rcases Nat.even_or_odd n with ⟨k, hk⟩ | ⟨k, hk⟩
  · subst hk
    rw [min_triangle_sum_even]
    -- Goal: (k : ℤ) * k = ((k + k) * (k + k) : ℕ : ℤ) / 4
    -- (k+k)*(k+k) = 4k².  4k² / 4 = k² (exact ℤ ediv).
    have h : (((k + k) * (k + k) : ℕ) : ℤ) = 4 * ((k : ℤ) * k) := by push_cast; ring
    rw [h]; omega
  · subst hk
    rw [min_triangle_sum_odd]
    -- Goal: (k : ℤ) * (k + 1) = ((2k + 1)*(2k + 1) : ℕ : ℤ) / 4
    -- (2k+1)² = 4k² + 4k + 1 = 4·(k² + k) + 1.  Truncating /4 = k² + k = k(k+1).
    have h : (((2 * k + 1) * (2 * k + 1) : ℕ) : ℤ) = 4 * ((k : ℤ) * (k + 1)) + 1 := by
      push_cast; ring
    rw [h]; omega

end OrigamiCone

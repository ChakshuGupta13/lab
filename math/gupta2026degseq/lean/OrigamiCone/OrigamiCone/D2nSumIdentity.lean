import OrigamiCone.MedianSumHelpers

/-!
# `D(2, n)`: parity-cased sum identity (Section 4)

The arithmetic core of `D(2, n) = ⌈n²/2⌉` (Section 4 of the paper):

  `∑_{ℓ ∈ Ico 0 (n : ℤ)} min (2ℓ + 1) (2n − 2ℓ − 1) = (n² + 1) / 2` (in ℤ).

This is a **pure integer-arithmetic identity** (no reference to `cLeq` or the
grid).  Combined with `cLeq_acell_two` (`D2nCLeq.lean`) and the median
characterisation (`Median.lean`), it gives `D(2, n) = ⌈n²/2⌉` in the
sibling `D2n.lean` module (next session).

Proof strategy: case-split on the parity of `n` via `Nat.even_or_odd`.

- **Even `n = 2k`**: split the Ico at `ℓ = k`.  Lower half `ℓ ∈ [0, k)`
  has `min = 2ℓ + 1` (by Gauss-odd sum: `Σ = k²`).  Upper half
  `ℓ ∈ [k, 2k)` has `min = 2n − 2ℓ − 1`; after shift `ℓ = k + j` and
  reflection `j ↦ k − 1 − j`, also `Σ = k²`.  Total = `2k² = (4k² + 1)/2`.

- **Odd `n = 2k + 1`**: split at `ℓ = k + 1`.  Lower `[0, k + 1)`:
  `Σ (2ℓ + 1) = (k + 1)²`.  Upper `[k + 1, 2k + 1)`: after shift +
  reflection, `Σ = k²`.  Total = `(k + 1)² + k² = 2k² + 2k + 1 = (n² + 1)/2`.

Results:
* `sum_odd_int` — Gauss's `Σ_{j<k} (2j + 1) = k²` (in `ℤ`);
* `sum_odd_reflect_int` — reflected variant `Σ_{j<k} (2k − 1 − 2j) = k²`;
* `min_2n_sum` — the main parity-cased identity.

No `sorry`.
-/

namespace OrigamiCone

/-! ## Gauss-style odd-sum helpers -/

/-- **Gauss's odd-sum identity** (in `ℤ`): `Σ_{j<k} (2j + 1) = k²`.
The integer version of the classical school identity; proved by induction. -/
lemma sum_odd_int (k : ℕ) :
    ∑ j ∈ Finset.range k, (2 * (j : ℤ) + 1) = (k : ℤ) * k := by
  induction k with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ, ih]
    push_cast; ring

/-- **Reflected odd-sum identity** (in `ℤ`): `Σ_{j<k} (2k − 1 − 2j) = k²`.
The reflected variant of `sum_odd_int`; proved by `Finset.sum_range_reflect`
applied to `j ↦ 2j + 1`. -/
lemma sum_odd_reflect_int (k : ℕ) :
    ∑ j ∈ Finset.range k, (2 * (k : ℤ) - 1 - 2 * (j : ℤ)) = (k : ℤ) * k := by
  have h := Finset.sum_range_reflect (fun j => 2 * (j : ℤ) + 1) k
  rw [sum_odd_int] at h
  -- h : (∑ j ∈ range k, (2 * ((k - 1 - j : ℕ) : ℤ) + 1)) = (k : ℤ) * k
  rw [show (∑ j ∈ Finset.range k, (2 * (k : ℤ) - 1 - 2 * (j : ℤ)))
        = ∑ j ∈ Finset.range k, (2 * (((k - 1 - j : ℕ) : ℤ)) + 1) from ?_]
  · exact h
  apply Finset.sum_congr rfl
  intro j hj
  rw [Finset.mem_range] at hj
  have hk1 : (1 : ℕ) ≤ k := by omega
  have hjk : j ≤ k - 1 := by omega
  have hcast : (((k - 1 - j : ℕ) : ℤ)) = (k : ℤ) - 1 - (j : ℤ) := by
    rw [Nat.cast_sub hjk, Nat.cast_sub hk1]; push_cast; ring
  rw [hcast]; ring

/-! ## Parity cases -/

/-- **Even case**: for `n = 2k`, the median sum equals `2k²`. -/
private lemma min_2n_sum_even (k : ℕ) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) ((k + k : ℕ) : ℤ),
        min (2 * ℓ + 1) (2 * ((k + k : ℕ) : ℤ) - 2 * ℓ - 1)
      = 2 * (k : ℤ) * k := by
  -- Cast (k+k : ℕ) : ℤ = (k : ℤ) + k.
  have hcast : ((k + k : ℕ) : ℤ) = (k : ℤ) + k := by push_cast; ring
  rw [hcast]
  -- Split Ico at k.
  have h0k : (0 : ℤ) ≤ (k : ℤ) := Int.natCast_nonneg _
  have hkk : (k : ℤ) ≤ (k : ℤ) + k := by linarith
  rw [show Finset.Ico (0 : ℤ) ((k : ℤ) + k)
        = Finset.Ico (0 : ℤ) (k : ℤ) ∪ Finset.Ico (k : ℤ) ((k : ℤ) + k) from
        (Finset.Ico_union_Ico_eq_Ico h0k hkk).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 0 (k : ℤ) ((k : ℤ) + k))]
  -- Lower half.
  have h_lo : ∑ ℓ ∈ Finset.Ico (0 : ℤ) (k : ℤ),
      min (2 * ℓ + 1) (2 * ((k : ℤ) + k) - 2 * ℓ - 1) = (k : ℤ) * k := by
    rw [sum_Ico_int_eq_sum_range (k : ℤ) h0k]
    rw [Int.toNat_natCast k]
    rw [show ∑ j ∈ Finset.range k,
            min (2 * ((j : ℤ)) + 1) (2 * ((k : ℤ) + k) - 2 * ((j : ℤ)) - 1)
          = ∑ j ∈ Finset.range k, (2 * ((j : ℤ)) + 1) from ?_]
    · exact sum_odd_int k
    apply Finset.sum_congr rfl
    intro j hj
    rw [Finset.mem_range] at hj
    have hjk : (j : ℤ) ≤ (k : ℤ) - 1 := by
      have : (j : ℤ) + 1 ≤ (k : ℤ) := by exact_mod_cast hj
      linarith
    rw [min_eq_left]; linarith
  -- Upper half: ℓ ∈ [k, k+k), via shift ℓ = k + j.
  have h_hi : ∑ ℓ ∈ Finset.Ico (k : ℤ) ((k : ℤ) + k),
      min (2 * ℓ + 1) (2 * ((k : ℤ) + k) - 2 * ℓ - 1) = (k : ℤ) * k := by
    rw [sum_Ico_int_shift (k : ℤ) ((k : ℤ) + k) hkk]
    have hdiff : ((k : ℤ) + k - k).toNat = k := by
      have : ((k : ℤ) + k - k) = (k : ℤ) := by ring
      rw [this, Int.toNat_natCast]
    rw [hdiff]
    rw [show ∑ j ∈ Finset.range k,
            min (2 * ((j : ℤ) + k) + 1) (2 * ((k : ℤ) + k) - 2 * ((j : ℤ) + k) - 1)
          = ∑ j ∈ Finset.range k, (2 * (k : ℤ) - 1 - 2 * (j : ℤ)) from ?_]
    · exact sum_odd_reflect_int k
    apply Finset.sum_congr rfl
    intro j hj
    rw [Finset.mem_range] at hj
    have hj_nn : (0 : ℤ) ≤ (j : ℤ) := Int.natCast_nonneg _
    rw [min_eq_right]
    · ring
    nlinarith
  rw [h_lo, h_hi]; ring

/-- **Odd case**: for `n = 2k + 1`, the median sum equals `2k² + 2k + 1`. -/
private lemma min_2n_sum_odd (k : ℕ) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) ((2 * k + 1 : ℕ) : ℤ),
        min (2 * ℓ + 1) (2 * ((2 * k + 1 : ℕ) : ℤ) - 2 * ℓ - 1)
      = 2 * (k : ℤ) * k + 2 * k + 1 := by
  have hcast : ((2 * k + 1 : ℕ) : ℤ) = 2 * (k : ℤ) + 1 := by push_cast; ring
  rw [hcast]
  -- Split Ico 0 (2k+1) at k+1.
  have h0 : (0 : ℤ) ≤ (k : ℤ) + 1 := by linarith [Int.natCast_nonneg k]
  have hk1 : (k : ℤ) + 1 ≤ 2 * (k : ℤ) + 1 := by linarith [Int.natCast_nonneg k]
  rw [show Finset.Ico (0 : ℤ) (2 * (k : ℤ) + 1)
        = Finset.Ico (0 : ℤ) ((k : ℤ) + 1) ∪ Finset.Ico ((k : ℤ) + 1) (2 * (k : ℤ) + 1) from
        (Finset.Ico_union_Ico_eq_Ico h0 hk1).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 0 ((k : ℤ) + 1) (2 * (k : ℤ) + 1))]
  -- Lower half: ℓ ∈ [0, k+1), min = 2ℓ+1. Σ = (k+1)².
  have h_lo : ∑ ℓ ∈ Finset.Ico (0 : ℤ) ((k : ℤ) + 1),
      min (2 * ℓ + 1) (2 * (2 * (k : ℤ) + 1) - 2 * ℓ - 1)
        = ((k : ℤ) + 1) * ((k : ℤ) + 1) := by
    rw [sum_Ico_int_eq_sum_range ((k : ℤ) + 1) h0]
    have htoNat : ((k : ℤ) + 1).toNat = k + 1 := by
      have : (((k + 1 : ℕ) : ℤ)) = (k : ℤ) + 1 := by push_cast; ring
      rw [← this, Int.toNat_natCast]
    rw [htoNat]
    rw [show ∑ j ∈ Finset.range (k + 1),
            min (2 * ((j : ℤ)) + 1) (2 * (2 * (k : ℤ) + 1) - 2 * ((j : ℤ)) - 1)
          = ∑ j ∈ Finset.range (k + 1), (2 * ((j : ℤ)) + 1) from ?_]
    · have h := sum_odd_int (k + 1)
      rw [h]; push_cast; ring
    apply Finset.sum_congr rfl
    intro j hj
    rw [Finset.mem_range] at hj
    have hjk : (j : ℤ) ≤ (k : ℤ) := by exact_mod_cast (by omega : j ≤ k)
    rw [min_eq_left]
    nlinarith [Int.natCast_nonneg j]
  -- Upper half: ℓ ∈ [k+1, 2k+1), via shift ℓ = (k+1) + j.
  have h_hi : ∑ ℓ ∈ Finset.Ico ((k : ℤ) + 1) (2 * (k : ℤ) + 1),
      min (2 * ℓ + 1) (2 * (2 * (k : ℤ) + 1) - 2 * ℓ - 1) = (k : ℤ) * k := by
    rw [sum_Ico_int_shift ((k : ℤ) + 1) (2 * (k : ℤ) + 1) hk1]
    have hdiff : (2 * (k : ℤ) + 1 - ((k : ℤ) + 1)).toNat = k := by
      have : (2 * (k : ℤ) + 1 - ((k : ℤ) + 1)) = (k : ℤ) := by ring
      rw [this, Int.toNat_natCast]
    rw [hdiff]
    rw [show ∑ j ∈ Finset.range k,
            min (2 * ((j : ℤ) + ((k : ℤ) + 1)) + 1)
                (2 * (2 * (k : ℤ) + 1) - 2 * ((j : ℤ) + ((k : ℤ) + 1)) - 1)
          = ∑ j ∈ Finset.range k, (2 * (k : ℤ) - 1 - 2 * (j : ℤ)) from ?_]
    · exact sum_odd_reflect_int k
    apply Finset.sum_congr rfl
    intro j hj
    rw [Finset.mem_range] at hj
    have hj_nn : (0 : ℤ) ≤ (j : ℤ) := Int.natCast_nonneg _
    rw [min_eq_right]
    · ring
    nlinarith
  rw [h_lo, h_hi]; ring

/-! ## Main parity-cased dispatch -/

/-- **The parity-cased sum identity** for `D(2, n)`.

`∑_{ℓ ∈ Ico 0 n} min (2ℓ + 1) (2n − 2ℓ − 1) = (n² + 1) / 2`  (in `ℤ`,
truncating division).

Equivalently the integer reformulation of `⌈n²/2⌉` (since `n² + 1` truncating-
divided by `2` equals `⌈n²/2⌉` for both parities of `n`).  The arithmetic core
of `D(2, n) = ⌈n²/2⌉`. -/
theorem min_2n_sum (n : ℕ) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) (n : ℤ),
        min (2 * ℓ + 1) (2 * (n : ℤ) - 2 * ℓ - 1)
      = ((n * n + 1 : ℕ) : ℤ) / 2 := by
  rcases Nat.even_or_odd n with ⟨k, hk⟩ | ⟨k, hk⟩
  · -- Even: n = k + k.
    subst hk
    have hev := min_2n_sum_even k
    -- LHS matches hev modulo (k+k : ℕ) ↔ (n : ℤ) cast normalisation.
    rw [hev]
    -- Goal: 2 * k * k = ((k+k) * (k+k) + 1 : ℕ : ℤ) / 2.
    have hN : (((k + k) * (k + k) + 1 : ℕ) : ℤ) = 2 * (2 * (k : ℤ) * k) + 1 := by
      push_cast; ring
    rw [hN]
    -- (2 * X + 1) / 2 = X for any X (omega handles).
    omega
  · -- Odd: n = 2k + 1.
    subst hk
    have hod := min_2n_sum_odd k
    rw [hod]
    -- Goal: 2 * k * k + 2 * k + 1 = ((2*k+1)*(2*k+1) + 1 : ℕ : ℤ) / 2.
    have hN : (((2 * k + 1) * (2 * k + 1) + 1 : ℕ) : ℤ)
            = 2 * (2 * (k : ℤ) * k + 2 * k + 1) := by
      push_cast; ring
    rw [hN]
    omega

end OrigamiCone

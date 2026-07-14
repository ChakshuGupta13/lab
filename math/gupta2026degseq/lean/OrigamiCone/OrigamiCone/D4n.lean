import OrigamiCone.D4nCLeq
import OrigamiCone.D4nSumIdentity

/-!
# `D(4, n) = n² + 4 + [n odd]` (Section 4)

The fourth paper closed form (after `D(m, m)`, `D(2, n)`, `D(3, n)`):

  `D(4, n) := \min_K \sum_{v \in \mathrm{Cell}\,4\,n} |\mathrm{acell}\,v - K|
            = n^2 + 4 + [\,n \text{ odd}\,]
              \quad (\text{in } \mathbb{Z})`,

for every `n ≥ 2`.

The proof assembles eight previously formalised pieces:
1. `isMedianMin_sum_min` (`Median.lean`).
2. Six per-term `cLeq` formulas for `m = 4` (`D4nCLeq.lean`, 875b2ee):
   `cLeq_acell_four_{zero, one, two, mid, top_minus_one, top}`.
3. The middle-band sum identity `min_4n_middle_sum` (`D4nSumIdentity.lean`,
   3e7dd0e) for the Ico `[3, n)` slice.

The level range for `Cell 4 n` is `[0, 4 + n − 2)` = `[0, n + 2)` (max
`acell` is `n + 1`, attained at cell `(3, n − 1)`).  The Ico split is

  `[0, n + 2) = [0, 2) ∪ [2, n) ∪ [n, n + 2)`,

with constant boundary contributions `1 + 3 + 3 + 1 = 8` and middle
`Ico [2, n)` aggregated by an auxiliary lemma that uses
`min_4n_middle_sum` and absorbs the `ℓ = 2` entry (which has `c = 6`
under `four_two` for `n ≥ 3`, and is empty for `n = 2`).

The Iverson bracket `[n odd]` is encoded as the `Nat`-modular `n % 2`.
The parity is hidden inside a truncating `ℤ` ediv via
`four_mul_sq_div_four`:

  `4 · ((n − 1)² / 4) = n² − 2 n + (n % 2)`.

Results:
* `D_4n` — `IsMedianMin acell ((n² + 4 + n % 2 : ℕ) : ℤ)` on
  `Cell 4 n` for `n ≥ 2`.

No `sorry`.
-/

namespace OrigamiCone

variable {n : ℕ}

/-- Range bound for `acell` on `Cell 4 n` (private helper).  For every `v`,
`0 ≤ acell v ≤ n + 2` (max attained at `(3, n − 1)`).  Vacuous when
`Cell 4 n = ∅` (`n = 0`). -/
private lemma acell_range_4n (v : Cell 4 n) :
    (0 : ℤ) ≤ acell v ∧ acell v ≤ ((n + 2 : ℕ) : ℤ) := by
  have h1 : v.1.val < 4 := v.1.isLt
  have h2 : v.2.val < n := v.2.isLt
  unfold acell
  refine ⟨by positivity, ?_⟩
  have h1' : (v.1.val : ℤ) ≤ 3 := by
    have : (v.1.val : ℤ) + 1 ≤ 4 := by exact_mod_cast h1
    linarith
  have h2' : (v.2.val : ℤ) ≤ (n : ℤ) - 1 := by
    have : (v.2.val : ℤ) + 1 ≤ (n : ℤ) := by exact_mod_cast h2
    linarith
  push_cast; linarith

/-! ## Parity-cased ediv -/

/-- **Parity-cased ediv** `4 · ((n − 1)² / 4) = n² − 2 n + (n % 2)`
(in `ℤ`).  Proof by parity case-split on `n`.

For `n = 2 k` (even): `(2 k − 1)² = 4 k² − 4 k + 1`, so
`((n − 1)² / 4) = k² − k` and `4 · (k² − k) = 4 k² − 4 k = n² − 2 n`
(matches `n % 2 = 0`).

For `n = 2 k + 1` (odd): `(2 k)² = 4 k²`, so
`((n − 1)² / 4) = k²` and `4 · k² = (n − 1)² = n² − 2 n + 1`
(matches `n % 2 = 1`). -/
private lemma four_mul_sq_div_four (n : ℕ) (hn : 1 ≤ n) :
    4 * ((((n - 1) * (n - 1) : ℕ) : ℤ) / 4)
      = (n : ℤ) * n - 2 * n + ((n % 2 : ℕ) : ℤ) := by
  rcases Nat.even_or_odd n with ⟨k, hk⟩ | ⟨k, hk⟩
  · -- n = k + k = 2k (even), so n % 2 = 0.
    subst hk
    have hk1 : 1 ≤ k := by omega
    have h_mod : (k + k) % 2 = 0 := by omega
    rw [h_mod]
    have h_prod_cast : (((k + k - 1) * (k + k - 1) : ℕ) : ℤ)
                     = 4 * ((k : ℤ) * k - k) + 1 := by
      have h1 : 1 ≤ k + k := by omega
      push_cast [Nat.cast_sub h1]; ring
    rw [h_prod_cast]
    -- Use Int.add_mul_ediv_left to evaluate (4x + 1) / 4 = x + 1/4 = x.
    have h_div : (4 * ((k : ℤ) * k - k) + 1) / 4 = (k : ℤ) * k - k := by
      have h_rearrange : 4 * ((k : ℤ) * k - k) + 1 = 1 + 4 * ((k : ℤ) * k - k) := by
        ring
      rw [h_rearrange,
          Int.add_mul_ediv_left 1 ((k : ℤ) * k - k) (by norm_num : (4 : ℤ) ≠ 0),
          show (1 : ℤ) / 4 = 0 from by decide, zero_add]
    rw [h_div]
    push_cast; ring
  · -- n = 2k + 1 (odd), so n % 2 = 1.
    subst hk
    have h_mod : (2 * k + 1) % 2 = 1 := by omega
    rw [h_mod]
    have h_prod_cast : (((2 * k + 1 - 1) * (2 * k + 1 - 1) : ℕ) : ℤ)
                     = 4 * ((k : ℤ) * k) := by
      have h1 : 1 ≤ 2 * k + 1 := by omega
      push_cast [Nat.cast_sub h1]; ring
    rw [h_prod_cast]
    -- Use Int.mul_ediv_cancel_left to evaluate (4 * x) / 4 = x.
    have h_div : (4 * ((k : ℤ) * k)) / 4 = (k : ℤ) * k := by
      rw [Int.mul_ediv_cancel_left _ (by norm_num : (4 : ℤ) ≠ 0)]
    rw [h_div]
    push_cast; ring

/-! ## Middle-band sum over `Ico [2, n)` -/

/-- **Aggregated middle sum** over `Ico [2, n)` for `m = 4`.

For `n ≥ 2`:
`Σ_{ℓ ∈ Ico 2 n} min(cLeq acell ℓ, 4 n − cLeq acell ℓ)
  = 4 · ((n − 1)² / 4) + 2 n − 4`.

This absorbs the `ℓ = 2` entry (which uses `cLeq_acell_four_two` for
`n ≥ 3` and is absent for `n = 2`) into a single closed form valid
for all `n ≥ 2`.

For `n = 2`, the Ico is empty and both sides are `0`.
For `n ≥ 3`, split `Ico [2, n) = {2} ∪ Ico [3, n)`, evaluate `{2}` via
`cLeq_acell_four_two` (giving `min(6, 4 n − 6) = 6` for `n ≥ 3`), and
reduce `Ico [3, n)` via `min_4n_middle_sum`. -/
private lemma middle_sum_2n_acell (hn : 2 ≤ n) :
    ∑ ℓ ∈ Finset.Ico (2 : ℤ) (n : ℤ),
        min (cLeq (acell (m := 4) (n := n)) ℓ)
            (4 * (n : ℤ) - cLeq (acell (m := 4) (n := n)) ℓ)
      = 4 * ((((n - 1) * (n - 1) : ℕ) : ℤ) / 4) + 2 * (n : ℤ) - 4 := by
  by_cases hn2 : n = 2
  · -- Base case: n = 2.  Ico 2 2 = ∅, so LHS = 0.
    subst hn2
    rw [show ((2 : ℕ) : ℤ) = (2 : ℤ) from rfl, Finset.Ico_self, Finset.sum_empty]
    -- RHS at n = 2: 4 * (((2 - 1) * (2 - 1) : ℕ) : ℤ) / 4 + 4 - 4
    --             = 4 * (1 / 4) + 0 = 4 * 0 + 0 = 0.
    decide
  · -- General case: n ≥ 3.  Split Ico [2, n) = {2} ∪ Ico [3, n).
    have hn3 : 3 ≤ n := by omega
    have h_3le : (3 : ℤ) ≤ (n : ℤ) := by exact_mod_cast hn3
    have h_2le3 : (2 : ℤ) ≤ 3 := by norm_num
    rw [show Finset.Ico (2 : ℤ) (n : ℤ)
          = Finset.Ico (2 : ℤ) 3 ∪ Finset.Ico (3 : ℤ) (n : ℤ) from
        (Finset.Ico_union_Ico_eq_Ico h_2le3 h_3le).symm,
        Finset.sum_union
          (Finset.Ico_disjoint_Ico_consecutive 2 3 (n : ℤ))]
    -- Singleton {2}: cLeq acell 2 = 6 (via four_two for n ≥ 3),
    -- min(6, 4n − 6) = 6 for n ≥ 3 (4n − 6 ≥ 6).
    rw [sum_Ico_int_shift (2 : ℤ) 3 h_2le3]
    rw [show ((3 : ℤ) - 2).toNat = 1 from rfl, Finset.sum_range_one]
    rw [show ((0 : ℕ) : ℤ) + 2 = 2 from by norm_num]
    rw [cLeq_acell_four_two hn3]
    rw [show min (6 : ℤ) (4 * (n : ℤ) - 6) = 6 from by
          apply min_eq_left; linarith]
    -- Middle band Ico [3, n): apply cLeq_acell_four_mid pointwise then min_4n_middle_sum.
    have h_mid_pointwise :
        ∀ ℓ ∈ Finset.Ico (3 : ℤ) (n : ℤ),
          min (cLeq (acell (m := 4) (n := n)) ℓ)
              (4 * (n : ℤ) - cLeq (acell (m := 4) (n := n)) ℓ)
            = min (4 * ℓ - 2) (4 * (n : ℤ) - 4 * ℓ + 2) := by
      intro ℓ hℓmem
      rw [Finset.mem_Ico] at hℓmem
      obtain ⟨h_3le_ℓ, h_ℓ_lt_n⟩ := hℓmem
      have h_ℓ_nn : (0 : ℤ) ≤ ℓ := by linarith
      set k := ℓ.toNat with hk_def
      have hk_cast : (k : ℤ) = ℓ := Int.toNat_of_nonneg h_ℓ_nn
      have h_3_lek : (3 : ℕ) ≤ k := by
        have : (3 : ℤ) ≤ (k : ℤ) := by rw [hk_cast]; exact h_3le_ℓ
        exact_mod_cast this
      have h_k1_len : k + 1 ≤ n := by
        have h : (k : ℤ) + 1 ≤ (n : ℤ) := by rw [hk_cast]; linarith
        exact_mod_cast h
      have h_cLeq : cLeq (acell (m := 4) (n := n)) ℓ = 4 * ℓ - 2 := by
        have := cLeq_acell_four_mid k h_3_lek h_k1_len
        rw [hk_cast] at this; exact this
      rw [h_cLeq]
      congr 1; ring
    rw [Finset.sum_congr rfl h_mid_pointwise]
    -- The middle sum after rewrite is exactly min_4n_middle_sum's LHS.
    rw [min_4n_middle_sum n hn3]
    -- Combine: 6 + (4·((n-1)²/4) + 2n - 10) = 4·((n-1)²/4) + 2n - 4.
    ring

/-! ## Full median sum -/

/-- **Median sum on the `4 × n` antidiagonal**, evaluated.  For `n ≥ 2`,
`Σ_ℓ min(c_ℓ, 4 n − c_ℓ)` over `[0, n + 2)` equals `n² + 4 + (n % 2)`
(in `ℤ`).  The arithmetic core of `D_4n`. -/
private lemma medianSum_acell_4n (hn : 2 ≤ n) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) ((n + 2 : ℕ) : ℤ),
        min (cLeq (acell (m := 4) (n := n)) ℓ)
            (((4 * n : ℕ) : ℤ) - cLeq (acell (m := 4) (n := n)) ℓ)
      = ((n * n + 4 + n % 2 : ℕ) : ℤ) := by
  have hn1 : 1 ≤ n := by omega
  have h_1len : (1 : ℤ) ≤ (n : ℤ) := by exact_mod_cast hn1
  have h_2len : (2 : ℤ) ≤ (n : ℤ) := by exact_mod_cast hn
  have hcast_np2 : ((n + 2 : ℕ) : ℤ) = (n : ℤ) + 2 := by push_cast; ring
  have h_4n : ((4 * n : ℕ) : ℤ) = 4 * (n : ℤ) := by push_cast; ring
  rw [hcast_np2]
  -- Split [0, n+2) = [0, 1) ∪ [1, 2) ∪ [2, n) ∪ [n, n+1) ∪ [n+1, n+2).
  have h_0le1 : (0 : ℤ) ≤ 1 := by norm_num
  have h_1le2 : (1 : ℤ) ≤ 2 := by norm_num
  have h_nlen1 : (n : ℤ) ≤ (n : ℤ) + 1 := by linarith
  have h_np1le_np2 : (n : ℤ) + 1 ≤ (n : ℤ) + 2 := by linarith
  have h_1len_np2 : (1 : ℤ) ≤ (n : ℤ) + 2 := by linarith
  have h_2len_np1 : (2 : ℤ) ≤ (n : ℤ) + 1 := by linarith
  have h_2len_np2 : (2 : ℤ) ≤ (n : ℤ) + 2 := by linarith
  -- First split: [0, n+2) = [0, 1) ∪ [1, n+2).
  rw [show Finset.Ico (0 : ℤ) ((n : ℤ) + 2)
        = Finset.Ico (0 : ℤ) 1 ∪ Finset.Ico (1 : ℤ) ((n : ℤ) + 2) from
        (Finset.Ico_union_Ico_eq_Ico h_0le1 h_1len_np2).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 0 1 ((n : ℤ) + 2))]
  -- Second split: [1, n+2) = [1, 2) ∪ [2, n+2).
  rw [show Finset.Ico (1 : ℤ) ((n : ℤ) + 2)
        = Finset.Ico (1 : ℤ) 2 ∪ Finset.Ico (2 : ℤ) ((n : ℤ) + 2) from
        (Finset.Ico_union_Ico_eq_Ico h_1le2 h_2len_np2).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 1 2 ((n : ℤ) + 2))]
  have h_nlen2 : (n : ℤ) ≤ (n : ℤ) + 2 := by linarith
  -- Third split: [2, n+2) = [2, n) ∪ [n, n+2).
  rw [show Finset.Ico (2 : ℤ) ((n : ℤ) + 2)
        = Finset.Ico (2 : ℤ) (n : ℤ) ∪ Finset.Ico (n : ℤ) ((n : ℤ) + 2) from
        (Finset.Ico_union_Ico_eq_Ico h_2len h_nlen2).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 2 (n : ℤ) ((n : ℤ) + 2))]
  -- Fourth split: [n, n+2) = [n, n+1) ∪ [n+1, n+2).
  rw [show Finset.Ico (n : ℤ) ((n : ℤ) + 2)
        = Finset.Ico (n : ℤ) ((n : ℤ) + 1) ∪ Finset.Ico ((n : ℤ) + 1) ((n : ℤ) + 2) from
        (Finset.Ico_union_Ico_eq_Ico h_nlen1 h_np1le_np2).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive (n : ℤ) ((n : ℤ) + 1) ((n : ℤ) + 2))]
  -- Reduce the four singleton sums via sum_Ico_int_shift + sum_range_one.
  rw [sum_Ico_int_shift (0 : ℤ) 1 h_0le1,
      show ((1 : ℤ) - 0).toNat = 1 from rfl, Finset.sum_range_one,
      show ((0 : ℕ) : ℤ) + 0 = 0 from by norm_num]
  rw [sum_Ico_int_shift (1 : ℤ) 2 h_1le2,
      show ((2 : ℤ) - 1).toNat = 1 from rfl, Finset.sum_range_one,
      show ((0 : ℕ) : ℤ) + 1 = 1 from by norm_num]
  rw [sum_Ico_int_shift (n : ℤ) ((n : ℤ) + 1) h_nlen1]
  rw [show ((n : ℤ) + 1 - (n : ℤ)).toNat = 1 from by
        rw [show ((n : ℤ) + 1 - (n : ℤ)) = 1 from by ring]; rfl,
      Finset.sum_range_one,
      show ((0 : ℕ) : ℤ) + (n : ℤ) = (n : ℤ) from by norm_num]
  rw [sum_Ico_int_shift ((n : ℤ) + 1) ((n : ℤ) + 2) h_np1le_np2]
  rw [show ((n : ℤ) + 2 - ((n : ℤ) + 1)).toNat = 1 from by
        rw [show ((n : ℤ) + 2 - ((n : ℤ) + 1)) = 1 from by ring]; rfl,
      Finset.sum_range_one,
      show ((0 : ℕ) : ℤ) + ((n : ℤ) + 1) = (n : ℤ) + 1 from by ring]
  -- Apply boundary cLeq lemmas.
  rw [cLeq_acell_four_zero hn1, cLeq_acell_four_one hn,
      cLeq_acell_four_top_minus_one hn]
  -- For ℓ=n+1 cLeq: cLeq_acell_four_top hn1 has type cLeq acell ↑(n+1) = 4n - 1.
  -- The current term is cLeq acell ((n : ℤ) + 1); we need to bridge the cast.
  have h_cLeq_top_cast :
      cLeq (acell (m := 4) (n := n)) ((n : ℤ) + 1) = 4 * (n : ℤ) - 1 := by
    have h := cLeq_acell_four_top hn1
    -- h : cLeq acell ↑(n + 1) = 4 * ↑n - 1
    have hcast : ((n + 1 : ℕ) : ℤ) = (n : ℤ) + 1 := by push_cast; ring
    rw [hcast] at h
    exact h
  rw [h_cLeq_top_cast, h_4n]
  -- Simplify the four boundary min's:
  -- min(1, 4n - 1) = 1 for n ≥ 1.
  rw [show min (1 : ℤ) (4 * (n : ℤ) - 1) = 1 from by
        apply min_eq_left; linarith]
  -- min(3, 4n - 3) = 3 for n ≥ 2.
  rw [show min (3 : ℤ) (4 * (n : ℤ) - 3) = 3 from by
        apply min_eq_left; linarith]
  -- min(4n - 3, 4n - (4n - 3)) = min(4n - 3, 3) = 3 for n ≥ 2.
  rw [show (4 * (n : ℤ) - (4 * (n : ℤ) - 3)) = 3 from by ring]
  rw [show min (4 * (n : ℤ) - 3) 3 = 3 from by
        apply min_eq_right; linarith]
  -- min(4n - 1, 4n - (4n - 1)) = min(4n - 1, 1) = 1 for n ≥ 1.
  rw [show (4 * (n : ℤ) - (4 * (n : ℤ) - 1)) = 1 from by ring]
  rw [show min (4 * (n : ℤ) - 1) 1 = 1 from by
        apply min_eq_right; linarith]
  -- Apply middle_sum_2n_acell.
  rw [middle_sum_2n_acell hn]
  -- Combine: 1 + 3 + (4·((n-1)²/4) + 2n - 4) + 3 + 1 = n² + 4 + (n % 2).
  -- Using four_mul_sq_div_four: 4·((n-1)²/4) = n² - 2n + (n % 2).
  have h_parity := four_mul_sq_div_four n hn1
  push_cast [Nat.cast_sub hn1] at h_parity ⊢
  linarith

/-! ## The main theorem -/

/-- **`D(4, n) = n² + 4 + [n odd]`** (Section 4 paper closed form, for
`n ≥ 2`).

The minimised dispersion of the antidiagonal `acell : Cell 4 n → ℤ` on
the `4 × n` grid equals `n² + 4 + (n % 2)`, the integer reformulation of
`n² + 4 + [n odd]` (Iverson bracket = `n % 2` in `Nat`).  Tight at
`n ≥ 2`; at `n = 0, 1` the empty / two-row grids give different values.

Proof: combine `isMedianMin_sum_min` (the median characterisation) with
`medianSum_acell_4n` (the closed-form evaluation, this file).  The latter
splits `Ico [0, n + 2)` into four corner singletons + middle `[2, n)`,
applies the six per-term `cLeq` lemmas from `D4nCLeq.lean`, the
middle-band sum from `D4nSumIdentity.lean`, and the parity-cased
ediv lemma `four_mul_sq_div_four`. -/
theorem D_4n (n : ℕ) (hn : 2 ≤ n) :
    IsMedianMin (acell (m := 4) (n := n)) ((n * n + 4 + n % 2 : ℕ) : ℤ) := by
  have hLU : (0 : ℤ) ≤ ((n + 2 : ℕ) : ℤ) := by positivity
  have hφ : ∀ v : Cell 4 n, (0 : ℤ) ≤ acell v ∧ acell v ≤ ((n + 2 : ℕ) : ℤ) :=
    acell_range_4n
  have hmid := isMedianMin_sum_min
    (acell (m := 4) (n := n)) 0 ((n + 2 : ℕ) : ℤ) hLU hφ
  have hcard : (Fintype.card (Cell 4 n) : ℤ) = ((4 * n : ℕ) : ℤ) := by
    simp [Cell, Fintype.card_prod, Fintype.card_fin]
  rw [hcard] at hmid
  rw [medianSum_acell_4n hn] at hmid
  exact hmid

end OrigamiCone

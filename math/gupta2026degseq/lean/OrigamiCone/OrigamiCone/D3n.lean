import OrigamiCone.D3nCLeq
import OrigamiCone.D3nBoundary
import OrigamiCone.D3nSumIdentity

/-!
# `D(3, n) = ⌊3n²/4⌋ + 2` (Section 4)

The third paper closed form (after `D(m, m) = (m³ − m)/3` and
`D(2, n) = ⌈n²/2⌉`):

  `D(3, n) := \min_K \sum_{v \in \mathrm{Cell}\,3\,n} |\mathrm{acell}\,v - K|
            = \lfloor 3 n^2 / 4 \rfloor + 2 = (3 n^2) / 4 + 2
              \quad (\text{in } \mathbb{Z})`,

for every `n ≥ 1` (statement is `False` at `n = 0` because the formula gives
`2 ≠ 0` while `disp ≡ 0` on the empty grid).

The proof assembles four previously formalised pieces:
1. `isMedianMin_sum_min` (`Median.lean`) — `disp(φ) = Σ_ℓ min(c_ℓ, N − c_ℓ)`.
2. `cLeq_acell_three_zero` / `_three_top` (`D3nBoundary.lean`) — boundary
   per-term `c_0 = 1` and `c_n = 3n − 1`.
3. `cLeq_acell_three_mid` (`D3nCLeq.lean`) — middle per-term `c_ℓ = 3ℓ`
   for `ℓ ∈ [1, n − 1]`.
4. `min_triangle_sum` (`D3nSumIdentity.lean`) — pure arithmetic identity
   `Σ_{ℓ ∈ [1, n)} min(ℓ, n − ℓ) = ((n*n : ℕ) : ℤ) / 4`.

The level range for `Cell 3 n` is `[0, 3 + n − 2)` = `[0, n + 1)`.  Split
the median sum into three pieces:
- `ℓ = 0`: `min(1, 3n − 1) = 1` (for `n ≥ 1`).
- `ℓ ∈ [1, n)`: `min(3ℓ, 3n − 3ℓ) = 3·min(ℓ, n − ℓ)`; sum is `3·(n²/4)`.
- `ℓ = n`: `min(3n − 1, 1) = 1` (for `n ≥ 1`).

Total: `1 + 3·(n²/4) + 1 = (3 n²)/4 + 2` via the parity-cased ediv identity
`3·(n²/4) = (3n²)/4` (true because `n² mod 4 ∈ {0, 1}`).

Results:
* `D_3n` — `IsMedianMin acell ((3*n*n : ℕ : ℤ) / 4 + 2)` on `Cell 3 n`
  for `n ≥ 1`.

No `sorry`.
-/

namespace OrigamiCone

variable {n : ℕ}

/-- Range bound for `acell` on `Cell 3 n` (private helper).  For every `v`,
`0 ≤ acell v ≤ n + 1` (max value attained at `(2, n − 1)`).  Vacuous when
`Cell 3 n = ∅` (i.e. `n = 0`). -/
private lemma acell_range_3n (v : Cell 3 n) :
    (0 : ℤ) ≤ acell v ∧ acell v ≤ ((n + 1 : ℕ) : ℤ) := by
  have h1 : v.1.val < 3 := v.1.isLt
  have h2 : v.2.val < n := v.2.isLt
  unfold acell
  refine ⟨by positivity, ?_⟩
  have h1' : (v.1.val : ℤ) ≤ 2 := by
    have : (v.1.val : ℤ) + 1 ≤ 3 := by exact_mod_cast h1
    linarith
  have h2' : (v.2.val : ℤ) ≤ (n : ℤ) - 1 := by
    have : (v.2.val : ℤ) + 1 ≤ (n : ℤ) := by exact_mod_cast h2
    linarith
  push_cast; linarith

/-- **Ediv identity** `3 · (n² / 4) = (3 n²) / 4` (in `ℤ`).  Holds because
`n² mod 4 ∈ {0, 1}`, so `3 · (n² mod 4) < 4` and the truncation distributes.
Proof by parity case-split on `n`. -/
private lemma three_mul_sq_div_four (n : ℕ) :
    3 * (((n * n : ℕ) : ℤ) / 4) = ((3 * n * n : ℕ) : ℤ) / 4 := by
  rcases Nat.even_or_odd n with ⟨k, hk⟩ | ⟨k, hk⟩
  · subst hk
    -- n = k + k.  n² = 4k².  3·(4k²)/4 = 3k².  (12k²)/4 = 3k².
    have hX : (((k + k) * (k + k) : ℕ) : ℤ) = 4 * ((k : ℤ) * k) := by
      push_cast; ring
    have hY : ((3 * (k + k) * (k + k) : ℕ) : ℤ) = 4 * (3 * ((k : ℤ) * k)) := by
      push_cast; ring
    rw [hX, hY]; omega
  · subst hk
    -- n = 2k + 1.  n² = 4k² + 4k + 1 = 4(k² + k) + 1.
    -- 3·((4(k²+k) + 1)/4) = 3·(k² + k).
    -- 3n² = 12k² + 12k + 3 = 4(3(k² + k)) + 3.  (...)/4 = 3(k² + k).
    have hX : (((2 * k + 1) * (2 * k + 1) : ℕ) : ℤ)
            = 4 * ((k : ℤ) * k + k) + 1 := by push_cast; ring
    have hY : ((3 * (2 * k + 1) * (2 * k + 1) : ℕ) : ℤ)
            = 4 * (3 * ((k : ℤ) * k + k)) + 3 := by push_cast; ring
    rw [hX, hY]; omega

/-! ## Median sum evaluation -/

/-- **Median sum on the `3 × n` antidiagonal**, evaluated.  For `n ≥ 1`,
`Σ_ℓ min(c_ℓ, 3n − c_ℓ)` over `[0, n + 1)` equals `(3 n²)/4 + 2` (in `ℤ`).
The arithmetic core of `D_3n`, assembled from the boundary lemmas, the
unified middle formula, and the parity-cased inner triangle sum. -/
private lemma medianSum_acell_3n (hn : 1 ≤ n) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) ((n + 1 : ℕ) : ℤ),
        min (cLeq (acell (m := 3) (n := n)) ℓ)
            (((3 * n : ℕ) : ℤ) - cLeq (acell (m := 3) (n := n)) ℓ)
      = ((3 * n * n : ℕ) : ℤ) / 4 + 2 := by
  -- Cast simplification.
  have hcast_np1 : ((n + 1 : ℕ) : ℤ) = (n : ℤ) + 1 := by push_cast; ring
  have h_1len : (1 : ℤ) ≤ (n : ℤ) := by exact_mod_cast hn
  have h3n : ((3 * n : ℕ) : ℤ) = 3 * (n : ℤ) := by push_cast; ring
  rw [hcast_np1]
  -- Step 1: Split Ico 0 (n+1) = Ico 0 1 ∪ Ico 1 (n+1).
  have h_0le1 : (0 : ℤ) ≤ 1 := by norm_num
  have h_1len1 : (1 : ℤ) ≤ (n : ℤ) + 1 := by linarith
  rw [show Finset.Ico (0 : ℤ) ((n : ℤ) + 1)
        = Finset.Ico (0 : ℤ) 1 ∪ Finset.Ico 1 ((n : ℤ) + 1) from
        (Finset.Ico_union_Ico_eq_Ico h_0le1 h_1len1).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 0 1 ((n : ℤ) + 1))]
  -- Step 2: Split Ico 1 (n+1) = Ico 1 n ∪ Ico n (n+1).
  have h_nlen1 : (n : ℤ) ≤ (n : ℤ) + 1 := by linarith
  rw [show Finset.Ico (1 : ℤ) ((n : ℤ) + 1)
        = Finset.Ico (1 : ℤ) (n : ℤ) ∪ Finset.Ico (n : ℤ) ((n : ℤ) + 1) from
        (Finset.Ico_union_Ico_eq_Ico h_1len h_nlen1).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 1 (n : ℤ) ((n : ℤ) + 1))]
  -- Step 3: Evaluate the two singletons via sum_Ico_int_shift.
  rw [sum_Ico_int_shift (0 : ℤ) 1 h_0le1]
  rw [show ((1 : ℤ) - 0).toNat = 1 from rfl, Finset.sum_range_one]
  rw [sum_Ico_int_shift (n : ℤ) ((n : ℤ) + 1) h_nlen1]
  rw [show ((n : ℤ) + 1 - (n : ℤ)).toNat = 1 from by
        have h : ((n : ℤ) + 1 - (n : ℤ)) = 1 := by ring
        rw [h]; rfl, Finset.sum_range_one]
  -- Step 4: Apply boundary lemmas at ℓ = 0 and ℓ = n.
  -- (j:ℤ) + 0 = 0, (j:ℤ) + n = n at j = 0.
  rw [show ((0 : ℕ) : ℤ) + 0 = 0 from by norm_num]
  rw [show ((0 : ℕ) : ℤ) + (n : ℤ) = (n : ℤ) from by norm_num]
  rw [cLeq_acell_three_zero hn, cLeq_acell_three_top hn]
  rw [h3n]
  -- Step 5: Simplify boundary summands.
  -- min(1, 3n - 1) = 1 for n ≥ 1 (since 3n - 1 ≥ 2).
  rw [show min (1 : ℤ) (3 * (n : ℤ) - 1) = 1 from by
        apply min_eq_left; linarith]
  -- min(3n - 1, 3n - (3n - 1)) = min(3n - 1, 1) = 1 for n ≥ 1.
  rw [show (3 * (n : ℤ) - (3 * (n : ℤ) - 1)) = 1 from by ring]
  rw [show min (3 * (n : ℤ) - 1) 1 = 1 from by
        apply min_eq_right; linarith]
  -- Step 6: Reduce the middle sum via cLeq_acell_three_mid pointwise.
  have h_mid_pointwise :
      ∀ ℓ ∈ Finset.Ico (1 : ℤ) (n : ℤ),
        min (cLeq (acell (m := 3) (n := n)) ℓ)
            (3 * (n : ℤ) - cLeq (acell (m := 3) (n := n)) ℓ)
          = 3 * min ℓ ((n : ℤ) - ℓ) := by
    intro ℓ hℓmem
    rw [Finset.mem_Ico] at hℓmem
    obtain ⟨h_1le_ℓ, h_ℓ_lt_n⟩ := hℓmem
    have h_ℓ_nn : (0 : ℤ) ≤ ℓ := by linarith
    set k := ℓ.toNat with hk_def
    have hk_cast : (k : ℤ) = ℓ := Int.toNat_of_nonneg h_ℓ_nn
    have h_1_lek : (1 : ℕ) ≤ k := by
      have : (1 : ℤ) ≤ (k : ℤ) := by rw [hk_cast]; exact h_1le_ℓ
      exact_mod_cast this
    have h_k1_len : k + 1 ≤ n := by
      have h : (k : ℤ) + 1 ≤ (n : ℤ) := by rw [hk_cast]; linarith
      exact_mod_cast h
    have h_cLeq : cLeq (acell (m := 3) (n := n)) ℓ = 3 * ℓ := by
      have := cLeq_acell_three_mid k h_1_lek h_k1_len
      rw [hk_cast] at this; exact this
    rw [h_cLeq]
    -- min(3ℓ, 3n - 3ℓ) = 3 * min(ℓ, n - ℓ).
    rcases le_total ℓ ((n : ℤ) - ℓ) with hℓ_le | hℓ_ge
    · rw [min_eq_left hℓ_le, min_eq_left (by linarith)]
    · rw [min_eq_right hℓ_ge, min_eq_right (by linarith)]; ring
  -- Wrap the (3*n*n:ℕ:ℤ) - cLeq in the middle into the form (3*n - cLeq).
  have h_mid_sum_eq :
      ∑ ℓ ∈ Finset.Ico (1 : ℤ) (n : ℤ),
          min (cLeq (acell (m := 3) (n := n)) ℓ)
              (3 * (n : ℤ) - cLeq (acell (m := 3) (n := n)) ℓ)
        = 3 * (((n * n : ℕ) : ℤ) / 4) := by
    rw [Finset.sum_congr rfl h_mid_pointwise]
    rw [← Finset.mul_sum]
    rw [min_triangle_sum n]
  rw [h_mid_sum_eq]
  -- Step 7: Combine with the ediv identity.
  -- Goal: 1 + 3 * ((n*n : ℕ : ℤ) / 4) + 1 = ((3*n*n : ℕ : ℤ) / 4) + 2.
  have h_ediv := three_mul_sq_div_four n
  linarith

/-! ## The main theorem -/

/-- **`D(3, n) = ⌊3n²/4⌋ + 2`** (Section 4 paper closed form, for `n ≥ 1`).

The minimised dispersion of the antidiagonal `acell : Cell 3 n → ℤ` on the
`3 × n` grid equals `(3 n² + 8) / 4 = ⌊3n²/4⌋ + 2`, expressed here in the
truncating-integer form `((3*n*n : ℕ) : ℤ) / 4 + 2`.  Requires `1 ≤ n`;
the statement is `False` at `n = 0` because the empty grid has
`disp ≡ 0 ≠ 2`.

Proof: combine `isMedianMin_sum_min` (the median characterisation,
`Median.lean`) with `medianSum_acell_3n` (the closed-form evaluation, this
file).  The latter routes through `cLeq_acell_three_{zero, mid, top}`
(boundary + middle per-term primitives) and `min_triangle_sum` (the
parity-cased inner triangle sum identity). -/
theorem D_3n (n : ℕ) (hn : 1 ≤ n) :
    IsMedianMin (acell (m := 3) (n := n)) (((3 * n * n : ℕ) : ℤ) / 4 + 2) := by
  have hLU : (0 : ℤ) ≤ ((n + 1 : ℕ) : ℤ) := by positivity
  have hφ : ∀ v : Cell 3 n, (0 : ℤ) ≤ acell v ∧ acell v ≤ ((n + 1 : ℕ) : ℤ) :=
    acell_range_3n
  have hmid := isMedianMin_sum_min
    (acell (m := 3) (n := n)) 0 ((n + 1 : ℕ) : ℤ) hLU hφ
  have hcard : (Fintype.card (Cell 3 n) : ℤ) = ((3 * n : ℕ) : ℤ) := by
    simp [Cell, Fintype.card_prod, Fintype.card_fin]
  rw [hcard] at hmid
  rw [medianSum_acell_3n hn] at hmid
  exact hmid

end OrigamiCone

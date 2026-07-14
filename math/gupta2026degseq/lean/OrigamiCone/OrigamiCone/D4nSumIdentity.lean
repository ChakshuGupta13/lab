import OrigamiCone.MedianSumHelpers
import OrigamiCone.D3nSumIdentity

/-!
# `D(4, n)`: the middle-band sum identity (Section 4)

The "middle band" for `D(4, n)` covers levels `ℓ ∈ [3, n − 1]` (= `Ico 3 n`
in Lean's half-open form).  Per `D4nCLeq.lean`, the per-term `cLeq` on this
band is `c_ℓ = 4 ℓ − 2`, so each level contributes
`min(4 ℓ − 2, 4 n − 4 ℓ + 2)` to the median sum.

This module evaluates the **middle-band sum** in closed form:

  `Σ_{ℓ ∈ Ico 3 n} min(4 ℓ − 2, 4 n − 4 ℓ + 2) =
   4 · ((n − 1)² / 4) + 2 n − 10`        (in `ℤ`, ediv truncating)

for every `n ≥ 3`.  The parity case-split of the paper formula
`D(4, n) = n² + 4 + [n odd]` is encoded **inside the truncating ediv**
`((n − 1)² / 4)`:

- for `n = 2 k` (even): `(2 k − 1)² / 4 = (4 k² − 4 k + 1) / 4 = k² − k`,
- for `n = 2 k + 1` (odd): `(2 k)² / 4 = k²`.

The sibling `D4n.lean` (next session) will evaluate this ediv via a
parity-dispatch helper to land on the headline `D(4, n) = n² + 4 + [n odd]`.

Proof strategy:
1. Pointwise factoring `min(4 ℓ − 2, 4 n − 4 ℓ + 2) = 4 · min(ℓ − 1, n − ℓ) + 2`.
2. Reindex `Ico 3 n → Ico 2 (n − 1)` via `ℓ ↔ ℓ − 1` (`Finset.sum_bij`).
3. Split `Ico 1 (n − 1) = {1} ∪ Ico 2 (n − 1)`, apply
   `min_triangle_sum (n − 1)` from `D3nSumIdentity`, subtract the `ℓ = 1`
   term `min(1, n − 2) = 1` for `n ≥ 3`.

Results:
* `min_4n_factor` — pointwise factoring (any `n`, any `ℓ`).
* `min_4n_middle_sum` — the closed-form decomposition (`n ≥ 3`).

No `sorry`.
-/

namespace OrigamiCone

/-! ## Pointwise factoring -/

/-- **Pointwise factoring** of the `D(4, n)` middle-band summand:
`min(4 ℓ − 2, 4 n − 4 ℓ + 2) = 4 · min(ℓ − 1, n − ℓ) + 2`.

Holds for any `n : ℕ` and any `ℓ : ℤ`, since `4 · min(a, b) + c =
min(4a + c, 4b + c)`. -/
lemma min_4n_factor (n : ℕ) (ℓ : ℤ) :
    min (4 * ℓ - 2) (4 * (n : ℤ) - 4 * ℓ + 2)
      = 4 * min (ℓ - 1) ((n : ℤ) - ℓ) + 2 := by
  rcases le_total (ℓ - 1) ((n : ℤ) - ℓ) with h | h
  · rw [min_eq_left h,
        min_eq_left (by linarith : 4 * ℓ - 2 ≤ 4 * (n : ℤ) - 4 * ℓ + 2)]
    ring
  · rw [min_eq_right h,
        min_eq_right (by linarith : 4 * (n : ℤ) - 4 * ℓ + 2 ≤ 4 * ℓ - 2)]
    ring

/-! ## Reindex `Ico 3 n → Ico 2 (n − 1)` -/

/-- Reindexing shift: `Σ_{ℓ ∈ Ico 3 n} g(ℓ − 1) = Σ_{j ∈ Ico 2 (n − 1)} g(j)`.

The bijection `ℓ ↦ ℓ − 1` (with inverse `j ↦ j + 1`) maps
`Ico 3 (n : ℤ)` onto `Ico 2 ((n : ℤ) − 1)`. -/
private lemma sum_Ico_3_shift_down {β : Type*} [AddCommMonoid β]
    (n : ℕ) (g : ℤ → β) :
    ∑ ℓ ∈ Finset.Ico (3 : ℤ) (n : ℤ), g (ℓ - 1)
      = ∑ j ∈ Finset.Ico (2 : ℤ) ((n : ℤ) - 1), g j := by
  apply Finset.sum_bij (fun ℓ _ => ℓ - 1)
  · -- Membership preservation.
    intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ ⊢
    omega
  · -- Injectivity.
    intros a _ b _ hab
    linarith
  · -- Surjectivity.
    intro j hj
    rw [Finset.mem_Ico] at hj
    refine ⟨j + 1, ?_, ?_⟩
    · rw [Finset.mem_Ico]; omega
    · ring
  · -- Function-equality.
    intros ℓ _
    rfl

/-! ## Main: closed-form middle-band sum -/

/-- **D(4, n) middle-band sum** in closed form: for `n ≥ 3`,

  `Σ_{ℓ ∈ Ico 3 n} min(4 ℓ − 2, 4 n − 4 ℓ + 2)
    = 4 · (((n − 1)·(n − 1) : ℕ) : ℤ) / 4 + 2 n − 10`.

The parity case-split of the paper formula `D(4, n) = n² + 4 + [n odd]`
is encoded entirely in the truncating ediv `(((n − 1)·(n − 1)) / 4)`. -/
theorem min_4n_middle_sum (n : ℕ) (hn : 3 ≤ n) :
    ∑ ℓ ∈ Finset.Ico (3 : ℤ) (n : ℤ),
        min (4 * ℓ - 2) (4 * (n : ℤ) - 4 * ℓ + 2)
      = 4 * ((((n - 1) * (n - 1) : ℕ) : ℤ) / 4) + 2 * (n : ℤ) - 10 := by
  -- Step 1: pointwise factor.
  rw [Finset.sum_congr rfl (fun ℓ _ => min_4n_factor n ℓ)]
  -- Goal: Σ_{ℓ ∈ Ico 3 n} (4 · min(ℓ - 1, n - ℓ) + 2) = ...
  rw [Finset.sum_add_distrib, ← Finset.mul_sum, Finset.sum_const]
  -- Goal: 4 · Σ min(ℓ - 1, n - ℓ) + (Ico 3 n).card • 2 = ...
  -- Step 2: card(Ico 3 n) = n - 3 (Nat).
  have hcard : (Finset.Ico (3 : ℤ) (n : ℤ)).card = n - 3 := by
    rw [Int.card_Ico]
    have : ((n : ℤ) - 3) = ((n - 3 : ℕ) : ℤ) := by
      push_cast [Nat.cast_sub hn]; ring
    rw [this, Int.toNat_natCast]
  rw [hcard]
  -- Step 3: rewrite the summand to use (n - 1) - (ℓ - 1) form.
  rw [Finset.sum_congr rfl (fun ℓ _ =>
    show min (ℓ - 1) ((n : ℤ) - ℓ) = min (ℓ - 1) (((n : ℤ) - 1) - (ℓ - 1)) by
      congr 1; ring)]
  -- Step 4: reindex Ico 3 n → Ico 2 (n - 1) via ℓ ↔ ℓ - 1.
  rw [sum_Ico_3_shift_down n (fun j => min j ((n : ℤ) - 1 - j))]
  -- Goal: 4 · Σ_{j ∈ Ico 2 (n - 1)} min(j, (n - 1) - j) + (n - 3) • 2 = ...
  -- Step 5: apply min_triangle_sum (n - 1), split Ico 1 (n - 1) = {1} ∪ Ico 2 (n - 1).
  have hcast_nm1 : ((n - 1 : ℕ) : ℤ) = (n : ℤ) - 1 := by
    have hn1 : 1 ≤ n := by omega
    push_cast [Nat.cast_sub hn1]; ring
  have h_tri := min_triangle_sum (n - 1)
  rw [hcast_nm1] at h_tri
  -- h_tri : Σ_{ℓ ∈ Ico 1 ((n : ℤ) - 1)} min(ℓ, ((n : ℤ) - 1) - ℓ)
  --       = (((n - 1) * (n - 1) : ℕ) : ℤ) / 4.
  have h_le12 : (1 : ℤ) ≤ 2 := by norm_num
  have h_le_2_n1 : (2 : ℤ) ≤ (n : ℤ) - 1 := by
    have : (3 : ℤ) ≤ (n : ℤ) := by exact_mod_cast hn
    linarith
  rw [show Finset.Ico (1 : ℤ) ((n : ℤ) - 1)
        = Finset.Ico (1 : ℤ) 2 ∪ Finset.Ico (2 : ℤ) ((n : ℤ) - 1) from
      (Finset.Ico_union_Ico_eq_Ico h_le12 h_le_2_n1).symm,
      Finset.sum_union
        (Finset.Ico_disjoint_Ico_consecutive 1 2 ((n : ℤ) - 1))] at h_tri
  -- h_tri : (Σ_{Ico 1 2} ...) + (Σ_{Ico 2 (n - 1)} ...) = (((n - 1) * (n - 1) : ℕ) : ℤ) / 4.
  -- Evaluate the {ℓ = 1} term.
  have hIco12 : Finset.Ico (1 : ℤ) 2 = {1} := by
    ext x; simp [Finset.mem_Ico, Finset.mem_singleton]; omega
  rw [hIco12, Finset.sum_singleton] at h_tri
  have h_min1 : min (1 : ℤ) ((n : ℤ) - 1 - 1) = 1 := by
    apply min_eq_left
    have : (3 : ℤ) ≤ (n : ℤ) := by exact_mod_cast hn
    linarith
  rw [h_min1] at h_tri
  -- h_tri : 1 + (Σ_{Ico 2 (n - 1)} ...) = (((n - 1) * (n - 1) : ℕ) : ℤ) / 4.
  -- Step 6: combine via linarith.
  -- Σ_{Ico 2 (n - 1)} ... = (((n - 1) * (n - 1) : ℕ) : ℤ) / 4 - 1 [from h_tri].
  -- Goal: 4 · (Σ_{Ico 2 (n - 1)} ...) + (n - 3) • 2 = 4 · (((n - 1) * (n - 1) : ℕ) : ℤ) / 4 + 2 * n - 10.
  -- After substituting the value of the sum: 4 · (Q/4 - 1) + (n - 3)·2 = 4·(Q/4) - 4 + 2n - 6 = 4·(Q/4) + 2n - 10.
  rw [nsmul_eq_mul]
  -- Normalize both h_tri and goal so the (n - 1) ediv term has matching form.
  push_cast [Nat.cast_sub hn] at h_tri ⊢
  linarith

end OrigamiCone

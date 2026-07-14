import Mathlib

/-!
# Median-sum arithmetic helpers (Section 4)

Reusable arithmetic + reindexing lemmas for evaluating the
`Σ_ℓ min(c_ℓ, N − c_ℓ)` median sum for the `D(m, n)` closed-form
specialisations of Section 4 of the paper:

* `int_two_dvd_consec` — `2 ∣ (k + 1)(k + 2)` in `ℤ`;
* `int_two_mul_div_consec` — `2 · ((k+1)(k+2) / 2) = (k+1)(k+2)` in `ℤ`;
* `triple_pyramid_sum` — `3 · Σ_{k<n} (k+1)(k+2) = n(n+1)(n+2)` in `ℤ`;
* `sum_Ico_int_eq_sum_range` — reindex `∑ ℓ ∈ Ico (0:ℤ) k, f ℓ` as
  `∑ j ∈ range k.toNat, f j` (for `0 ≤ k`);
* `sum_Ico_int_shift` — reindex `∑ ℓ ∈ Ico (a:ℤ) b, f ℓ` as
  `∑ j ∈ range (b−a).toNat, f (j + a)` (for `a ≤ b`).

These are domain-agnostic arithmetic and not specific to the grid.

No `sorry`.
-/

namespace OrigamiCone

/-! ## Divisibility of consecutive products -/

/-- `2 ∣ (k + 1)(k + 2)` in `ℤ`.  Two consecutive integers contain an even
number. -/
lemma int_two_dvd_consec (k : ℤ) : (2 : ℤ) ∣ (k + 1) * (k + 2) := by
  have h : Even ((k + 1) * ((k + 1) + 1)) := Int.even_mul_succ_self (k + 1)
  rw [show (k + 1) + 1 = k + 2 from by ring] at h
  exact h.two_dvd

/-- `2 · ((k+1)(k+2) / 2) = (k+1)(k+2)` in `ℤ`. -/
lemma int_two_mul_div_consec (k : ℤ) :
    2 * ((k + 1) * (k + 2) / 2) = (k + 1) * (k + 2) :=
  Int.mul_ediv_cancel' (int_two_dvd_consec k)

/-! ## The triple-pyramid identity -/

/-- **The triple-pyramid identity** (in `ℤ`):
`3 · Σ_{k<n} (k+1)(k+2) = n(n+1)(n+2)`.  Multiplied-out form keeps the
arithmetic on the integers (no division).  Proved by induction on `n`. -/
lemma triple_pyramid_sum (n : ℕ) :
    3 * (∑ k ∈ Finset.range n, ((k : ℤ) + 1) * ((k : ℤ) + 2))
      = (n : ℤ) * (n + 1) * (n + 2) := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [Finset.sum_range_succ]; push_cast; linarith [ih]

/-! ## Reindexing `ℤ`-`Ico` as `ℕ`-`range` -/

/-- Reindex `∑ ℓ ∈ Ico (0:ℤ) k, f ℓ` (with `0 ≤ k`) as `∑ j ∈ range k.toNat,
f j`.  The bijection `ℓ ↦ ℓ.toNat` is the identity on nonneg integers. -/
lemma sum_Ico_int_eq_sum_range
    {β : Type*} [AddCommMonoid β] (k : ℤ) (hk : 0 ≤ k) (f : ℤ → β) :
    ∑ ℓ ∈ Finset.Ico (0 : ℤ) k, f ℓ = ∑ j ∈ Finset.range k.toNat, f (j : ℤ) := by
  apply Finset.sum_bij (fun (ℓ : ℤ) (_ : ℓ ∈ Finset.Ico (0 : ℤ) k) => ℓ.toNat)
  · intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    rw [Finset.mem_range]
    have : (ℓ.toNat : ℤ) < (k.toNat : ℤ) := by
      rw [Int.toNat_of_nonneg hℓ.1, Int.toNat_of_nonneg hk]; exact hℓ.2
    exact_mod_cast this
  · intros ℓ₁ hℓ₁ ℓ₂ hℓ₂ heq
    rw [Finset.mem_Ico] at hℓ₁ hℓ₂
    have h1 : (ℓ₁.toNat : ℤ) = (ℓ₂.toNat : ℤ) := by exact_mod_cast heq
    rw [Int.toNat_of_nonneg hℓ₁.1, Int.toNat_of_nonneg hℓ₂.1] at h1
    exact h1
  · intro j hj
    rw [Finset.mem_range] at hj
    refine ⟨(j : ℤ), ?_, Int.toNat_natCast j⟩
    rw [Finset.mem_Ico]
    refine ⟨Int.natCast_nonneg _, ?_⟩
    have : ((j : ℤ) : ℤ) < ((k.toNat : ℕ) : ℤ) := by exact_mod_cast hj
    rwa [Int.toNat_of_nonneg hk] at this
  · intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    congr 1
    exact (Int.toNat_of_nonneg hℓ.1).symm

/-- Reindex `∑ ℓ ∈ Ico (a:ℤ) b, f ℓ` (with `a ≤ b`) as `∑ j ∈ range (b−a).toNat,
f (j + a)`.  The bijection `ℓ ↦ (ℓ − a).toNat` shifts the lower endpoint to 0. -/
lemma sum_Ico_int_shift
    {β : Type*} [AddCommMonoid β] (a b : ℤ) (hab : a ≤ b) (f : ℤ → β) :
    ∑ ℓ ∈ Finset.Ico a b, f ℓ
      = ∑ j ∈ Finset.range (b - a).toNat, f ((j : ℤ) + a) := by
  apply Finset.sum_bij (fun (ℓ : ℤ) (_ : ℓ ∈ Finset.Ico a b) => (ℓ - a).toNat)
  · intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    rw [Finset.mem_range]
    have hsub_nn : 0 ≤ ℓ - a := by linarith
    have : ((ℓ - a).toNat : ℤ) < ((b - a).toNat : ℤ) := by
      rw [Int.toNat_of_nonneg hsub_nn,
          Int.toNat_of_nonneg (by linarith : (0:ℤ) ≤ b - a)]
      linarith [hℓ.2]
    exact_mod_cast this
  · intros ℓ₁ hℓ₁ ℓ₂ hℓ₂ heq
    rw [Finset.mem_Ico] at hℓ₁ hℓ₂
    have h1 : ((ℓ₁ - a).toNat : ℤ) = ((ℓ₂ - a).toNat : ℤ) := by exact_mod_cast heq
    rw [Int.toNat_of_nonneg (by linarith : (0:ℤ) ≤ ℓ₁ - a),
        Int.toNat_of_nonneg (by linarith : (0:ℤ) ≤ ℓ₂ - a)] at h1
    linarith
  · intro j hj
    rw [Finset.mem_range] at hj
    refine ⟨(j : ℤ) + a, ?_, ?_⟩
    · rw [Finset.mem_Ico]
      have : ((j : ℤ) : ℤ) < ((b - a).toNat : ℤ) := by exact_mod_cast hj
      rw [Int.toNat_of_nonneg (by linarith : (0:ℤ) ≤ b - a)] at this
      exact ⟨by linarith, by linarith⟩
    · have : ((j : ℤ) + a) - a = (j : ℤ) := by ring
      rw [this, Int.toNat_natCast]
  · intro ℓ hℓ
    rw [Finset.mem_Ico] at hℓ
    have hsub_nn : (0 : ℤ) ≤ ℓ - a := by linarith
    congr 1
    rw [Int.toNat_of_nonneg hsub_nn]; ring

end OrigamiCone

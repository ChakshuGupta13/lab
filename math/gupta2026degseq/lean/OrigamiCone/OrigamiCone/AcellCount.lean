import OrigamiCone.Median
import OrigamiCone.DiameterLower

/-!
# `cLeq` of the antidiagonal on the small-triangle range (Section 4)

For the antidiagonal `acell : Cell m n → ℤ`, `acell v = v.1.val + v.2.val`, the
cumulative sublevel count satisfies a clean closed form on the "small triangle"
range `ℓ ∈ [0, min(m, n))`:

  `cLeq acell ℓ = T(ℓ + 1) = (ℓ + 1)(ℓ + 2) / 2`.

This is the cornerstone for the closed-form `D(m, n)` diameter specialisations
in §4 of the paper (in particular `D(m, m) = (m³ − m)/3`): combined with the
median characterisation (`Median.lean`), the diameter is obtained by summing
the per-level `min(c_ℓ, N − c_ℓ)` contributions, and the present file supplies
`c_ℓ` for the lower half of the antidiagonal range.

Results:
* `cLeq_acell_triangle` — the small-triangle closed form
  `cLeq acell ℓ = (ℓ+1)(ℓ+2)/2` for `0 ≤ ℓ < min(m, n)`.

Helper machinery (private): a doubled triangle-number identity (avoiding
`Nat`-division pitfalls), a per-row count derived via `Fin.castLE`, and an
empty-row case.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Doubled triangle-number identity** (private helper).  Stating the sum in
`2·Σ = (ℓ+1)(ℓ+2)` form sidesteps the `Nat`-division pitfalls that block
`omega` from reasoning about `(ℓ+1)(ℓ+2)/2` directly. -/
private lemma triangle_sum_two_mul (ℓ : ℕ) :
    2 * ∑ i ∈ Finset.range (ℓ + 1), (ℓ + 1 - i) = (ℓ + 1) * (ℓ + 2) := by
  induction ℓ with
  | zero => decide
  | succ k ih =>
    rw [Finset.sum_range_succ' (fun i => k + 1 + 1 - i) (k + 1)]
    have hrw : ∀ i ∈ Finset.range (k + 1),
        (fun i => k + 1 + 1 - i) (i + 1) = k + 1 - i := by
      intro i hi; rw [Finset.mem_range] at hi; dsimp; omega
    rw [Finset.sum_congr rfl hrw]
    change 2 * ((∑ i ∈ Finset.range (k + 1), (k + 1 - i)) + (k + 1 + 1 - 0))
      = (k + 1 + 1) * (k + 1 + 2)
    linear_combination (norm := skip) ih
    ring_nf
    omega

/-- **Per-row card on the antidiagonal, low case.**  For `i ≤ ℓ < n`, the
number of `j : Fin n` with `(i : ℤ) + j.val ≤ ℓ` is `ℓ + 1 − i`.  Exposed
publicly because the middle-band closed form (`DSquareMiddle.lean`) reuses
this row decomposition. -/
lemma acell_row_card_le {n : ℕ} (i ℓ : ℕ) (hℓn : ℓ < n) (hiℓ : i ≤ ℓ) :
    ((Finset.univ : Finset (Fin n)).filter
        (fun j => (i : ℤ) + (j.val : ℤ) ≤ (ℓ : ℤ))).card = ℓ + 1 - i := by
  have hrw : (Finset.univ : Finset (Fin n)).filter
      (fun j => (i : ℤ) + (j.val : ℤ) ≤ (ℓ : ℤ)) =
        (Finset.univ : Finset (Fin n)).filter (fun j => j.val ≤ ℓ - i) := by
    apply Finset.filter_congr
    intro j _
    constructor
    · intro h
      have : (j.val : ℤ) ≤ (ℓ : ℤ) - i := by linarith
      have hZ : (j.val : ℤ) ≤ ((ℓ - i : ℕ) : ℤ) := by
        have hsub : ((ℓ - i : ℕ) : ℤ) = (ℓ : ℤ) - i := by omega
        omega
      exact_mod_cast hZ
    · intro h
      have hZ : (j.val : ℤ) ≤ ((ℓ - i : ℕ) : ℤ) := by exact_mod_cast h
      have : (j.val : ℤ) ≤ (ℓ : ℤ) - i := by push_cast at hZ; omega
      linarith
  rw [hrw]
  have h1 : (Finset.univ : Finset (Fin n)).filter (fun j => j.val ≤ ℓ - i) =
      (Finset.univ : Finset (Fin (ℓ - i + 1))).image (Fin.castLE (by omega)) := by
    ext j
    simp [Fin.castLE]
    constructor
    · intro hj
      exact ⟨⟨j.val, by omega⟩, by ext; rfl⟩
    · rintro ⟨⟨k, hk⟩, hj⟩
      rw [← hj]; simp; omega
  rw [h1, Finset.card_image_of_injective _ (Fin.castLE_injective _),
      Finset.card_univ, Fintype.card_fin]
  omega

/-- **Per-row card on the antidiagonal, empty case** (private helper).  For
`i > ℓ`, the row is empty. -/
private lemma acell_row_card_gt {n : ℕ} (i ℓ : ℕ) (hiℓ : ℓ < i) :
    ((Finset.univ : Finset (Fin n)).filter
        (fun j => (i : ℤ) + (j.val : ℤ) ≤ (ℓ : ℤ))).card = 0 := by
  rw [Finset.card_eq_zero, Finset.eq_empty_iff_forall_notMem]
  intro j hj
  rw [Finset.mem_filter] at hj
  have h1 : (0 : ℤ) ≤ (j.val : ℤ) := Int.natCast_nonneg _
  have h2 : (ℓ : ℤ) < i := by exact_mod_cast hiℓ
  linarith [hj.2]

/-- **`cLeq` of the antidiagonal on the small-triangle range.**

For `ℓ ∈ [0, min(m, n))`, the cumulative sublevel count of the antidiagonal
`acell : Cell m n → ℤ` (defined by `acell v = v.1.val + v.2.val`) equals the
triangular number `T(ℓ + 1) = (ℓ + 1)(ℓ + 2)/2`.

This is the cornerstone of the closed-form `D(m, n)` specialisations in §4 of
the paper (e.g. `D(m, m) = (m³ − m)/3`).  Combined with
`isMedianMin_sum_min` (in `Median.lean`) it determines the contribution of the
"small-triangle" half of the antidiagonal range. -/
theorem cLeq_acell_triangle (ℓ : ℕ) (hℓm : ℓ < m) (hℓn : ℓ < n) :
    cLeq (acell (m := m) (n := n)) (ℓ : ℤ) = ((ℓ + 1) * (ℓ + 2) / 2 : ℤ) := by
  -- Step 1: convert the filter card to a ℕ row sum.
  have h_card_nat :
      ((Finset.univ : Finset (Cell m n)).filter
          (fun v => (v.1.val : ℤ) + v.2.val ≤ (ℓ : ℤ))).card
        = ∑ i ∈ Finset.range (ℓ + 1), (ℓ + 1 - i) := by
    -- Row decomposition.
    rw [Finset.card_filter, ← Finset.univ_product_univ, Finset.sum_product]
    simp_rw [← Finset.card_filter]
    -- Replace the per-row card with its closed form.
    have hrow : ∀ i : Fin m,
        ((Finset.univ : Finset (Fin n)).filter
            (fun j => (i.val : ℤ) + j.val ≤ (ℓ : ℤ))).card =
          if i.val ≤ ℓ then ℓ + 1 - i.val else 0 := by
      intro i
      split_ifs with hi
      · exact acell_row_card_le i.val ℓ hℓn hi
      · push_neg at hi
        exact acell_row_card_gt i.val ℓ hi
    rw [Finset.sum_congr rfl (fun i _ => hrow i)]
    -- Convert Fin sum → Finset.range sum, then truncate.
    rw [show (∑ i : Fin m, if (i.val : ℕ) ≤ ℓ then ℓ + 1 - i.val else 0)
          = ∑ i ∈ Finset.range m, (if i ≤ ℓ then ℓ + 1 - i else 0) from
        Fin.sum_univ_eq_sum_range (fun k => if k ≤ ℓ then ℓ + 1 - k else 0) m,
        ← Finset.sum_filter]
    congr 1
    ext i
    simp [Finset.mem_filter, Finset.mem_range]
    omega
  unfold cLeq acell
  rw [h_card_nat]
  -- Step 2: cast the ℕ closed form into ℤ via the doubled identity.
  have h2 := triangle_sum_two_mul ℓ
  have h2_z :
      2 * ((∑ i ∈ Finset.range (ℓ + 1), (ℓ + 1 - i) : ℕ) : ℤ)
        = ((ℓ + 1) * (ℓ + 2) : ℤ) := by exact_mod_cast h2
  push_cast at h2_z ⊢
  omega

end OrigamiCone

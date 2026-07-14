import OrigamiCone.AcellReflect

/-!
# Middle-band closed form for `cLeq acell` (Section 4)

For asymmetric grids `m ≤ n` and levels `ℓ` in the middle band
`[m − 1, n − 1]`, the cumulative sublevel count of the antidiagonal `acell`
satisfies the **trapezoidal closed form**

  `cLeq acell ℓ = m · (ℓ + 1) − m · (m − 1) / 2`.

Geometrically: in this regime every row `i ∈ Fin m` is nonempty (since
`i ≤ m − 1 ≤ ℓ`) and no row is capped (since `ℓ − i + 1 ≤ ℓ + 1 ≤ n`), so
each row contributes `ℓ + 1 − i` cells.  The total is

$$\sum_{i=0}^{m-1} (\ell + 1 - i) = m(\ell + 1) - \binom{m}{2}.$$

Combined with `cLeq_acell_triangle` (small triangle, `ℓ < min(m, n)`) and
`cLeq_acell_suffix` (upper suffix, `ℓ ∈ [max(m, n) − 2, m + n − 3]`), this
closes the cLeq picture for the full antidiagonal range on any `m ≤ n`
grid.  The `n ≤ m` case follows by the cell-reflection swap
(`Equiv.prodComm`) — not formalised here.

The middle band degenerates to a single point `ℓ = m − 1` for `m = n`
(covered by `D_mm`); it is genuinely nontrivial when `m < n`.

Results:
* `cLeq_acell_middle` — the trapezoidal closed form for `m ≤ n`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- Gauss-sum identity in `ℤ` (doubled form to avoid division):
`2 · ∑_{i < n} (i : ℤ) = n · (n − 1)`.  Private helper. -/
private lemma sum_range_id_two_mul_int (k : ℕ) :
    2 * (∑ i ∈ Finset.range k, (i : ℤ)) = (k : ℤ) * (k - 1) := by
  induction k with
  | zero => simp
  | succ j ih => rw [Finset.sum_range_succ]; push_cast; linarith [ih]

/-- **Middle-band closed form for `cLeq acell`** (assumes `m ≤ n`).

For `m ≤ n` and `ℓ` in the middle band — encoded as `m ≤ ℓ + 1` (i.e.
`ℓ ≥ m − 1`) and `ℓ + 1 ≤ n` (i.e. `ℓ ≤ n − 1`) — the cumulative sublevel
count satisfies the trapezoidal closed form

  `cLeq acell ℓ = m · (ℓ + 1) − m · (m − 1) / 2`.

Specialises to the triangle endpoint at `ℓ = m − 1` (giving `m(m + 1)/2`)
and the suffix endpoint at `ℓ = n − 1` (giving `m · n − m(m − 1)/2`),
matching `cLeq_acell_triangle` and `cLeq_acell_suffix` at the seams.

The `m ≤ n` constraint is implied by `m ≤ ℓ + 1 ≤ n`, so it is omitted from
the signature.  For `n ≤ m`, swap `m ↔ n` via `Equiv.prodComm` (the
antidiagonal is symmetric in coordinates); not formalised here. -/
theorem cLeq_acell_middle (ℓ : ℕ) (hℓ_lo : m ≤ ℓ + 1)
    (hℓ_hi : ℓ + 1 ≤ n) :
    cLeq (acell (m := m) (n := n)) (ℓ : ℤ)
      = (m : ℤ) * ((ℓ : ℤ) + 1) - (m : ℤ) * ((m : ℤ) - 1) / 2 := by
  have hℓ_lt_n : ℓ < n := by omega
  -- Step 1: row decomposition. cLeq = Σ_{i : Fin m} ((ℓ + 1 - i.val : ℕ) : ℤ).
  have h_card_int :
      cLeq (acell (m := m) (n := n)) (ℓ : ℤ)
        = ∑ i : Fin m, ((ℓ + 1 - i.val : ℕ) : ℤ) := by
    unfold cLeq acell
    rw [Finset.card_filter, ← Finset.univ_product_univ, Finset.sum_product]
    simp_rw [← Finset.card_filter]
    push_cast
    apply Finset.sum_congr rfl
    intro i _
    have hi_le_ℓ : i.val ≤ ℓ := by
      have : i.val < m := i.isLt
      omega
    rw [acell_row_card_le i.val ℓ hℓ_lt_n hi_le_ℓ]
  rw [h_card_int]
  -- Step 2: convert each Nat-cast summand to a ℤ expression.
  have h_summand : ∀ i : Fin m,
      ((ℓ + 1 - i.val : ℕ) : ℤ) = (ℓ : ℤ) + 1 - (i.val : ℤ) := by
    intro i
    have hi_le_ℓ_plus_1 : i.val ≤ ℓ + 1 := by
      have : i.val < m := i.isLt
      omega
    rw [Nat.cast_sub hi_le_ℓ_plus_1]
    push_cast; ring
  rw [Finset.sum_congr rfl (fun i _ => h_summand i)]
  -- Step 3: distribute, apply Gauss sum.
  rw [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  rw [Fin.sum_univ_eq_sum_range (fun k => (k : ℤ))]
  simp only [nsmul_eq_mul]
  have h_gauss : (∑ i ∈ Finset.range m, (i : ℤ)) = (m : ℤ) * ((m : ℤ) - 1) / 2 := by
    have h := sum_range_id_two_mul_int m
    have hne : (2 : ℤ) ≠ 0 := by decide
    have h' : (m : ℤ) * ((m : ℤ) - 1) = 2 * (∑ i ∈ Finset.range m, (i : ℤ)) := h.symm
    rw [h', Int.mul_ediv_cancel_left _ hne]
  rw [h_gauss]

end OrigamiCone

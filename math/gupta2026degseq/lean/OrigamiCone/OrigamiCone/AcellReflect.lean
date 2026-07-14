import OrigamiCone.AcellCount

/-!
# Cell-reflection symmetry of `cLeq acell` (Section 4)

For the antidiagonal `acell : Cell m n → ℤ`, `acell v = v.1.val + v.2.val`, the
involution `cellRefl : (i, j) ↦ (m − 1 − i, n − 1 − j)` reflects the grid
through its centre and sends `acell v ↦ (m + n − 2) − acell v`.  Counting
cardinalities through this bijection yields the **reflection symmetry**

  `cLeq acell ℓ + cLeq acell (m + n − 3 − ℓ) = m · n`,

which, combined with `cLeq_acell_triangle` (small-triangle closed form on
`ℓ ∈ [0, min(m, n))`), determines `cLeq acell` on the **upper-suffix** range
`ℓ ∈ [max(m, n) − 2, m + n − 3)` as well:

  `cLeq acell ℓ = m · n − T((m + n − 2) − ℓ)`,

closing the cumulative count over the full antidiagonal range.

Results:
* `cellRefl` — the involution `Cell m n ≃ Cell m n` reflecting through the centre;
* `cellRefl_involutive` — `cellRefl (cellRefl v) = v`;
* `acell_cellRefl` — `acell (cellRefl v) = (m + n − 2) − acell v`;
* `cLeq_acell_reflect` — the symmetry `cLeq ℓ + cLeq (m + n − 3 − ℓ) = m · n`;
* `cLeq_acell_suffix` — the upper-suffix closed form.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Cell reflection through the centre of the grid.** Reflects each
coordinate via `Fin.rev`; an involution on `Cell m n`. -/
def cellRefl : Cell m n ≃ Cell m n where
  toFun v := (Fin.rev v.1, Fin.rev v.2)
  invFun v := (Fin.rev v.1, Fin.rev v.2)
  left_inv v := by ext <;> simp [Fin.rev_rev]
  right_inv v := by ext <;> simp [Fin.rev_rev]

/-- `cellRefl` is an involution. -/
lemma cellRefl_involutive (v : Cell m n) : cellRefl (cellRefl v) = v :=
  cellRefl.left_inv v

/-- **Reflection of `acell`.** The involution `cellRefl` sends `acell v` to
`(m + n − 2) − acell v`.  When `Cell m n` is empty (`m = 0` or `n = 0`) this
statement is vacuous; otherwise the bounds `v.1.val < m` and `v.2.val < n`
give the algebra. -/
lemma acell_cellRefl (v : Cell m n) :
    acell (cellRefl v) = ((m : ℤ) + n - 2) - acell v := by
  have h1 : v.1.val < m := v.1.isLt
  have h2 : v.2.val < n := v.2.isLt
  simp only [cellRefl, Equiv.coe_fn_mk, acell, Fin.val_rev]
  have hm_cast : ((m - (v.1.val + 1) : ℕ) : ℤ) = (m : ℤ) - v.1.val - 1 := by
    have h : v.1.val + 1 ≤ m := h1
    rw [Nat.cast_sub h]; push_cast; ring
  have hn_cast : ((n - (v.2.val + 1) : ℕ) : ℤ) = (n : ℤ) - v.2.val - 1 := by
    have h : v.2.val + 1 ≤ n := h2
    rw [Nat.cast_sub h]; push_cast; ring
  rw [hm_cast, hn_cast]; ring

/-- **Reflection symmetry of `cLeq acell`.**

The cumulative sublevel count of the antidiagonal satisfies
`cLeq acell ℓ + cLeq acell (m + n − 3 − ℓ) = m · n` for every integer offset
`ℓ`.  This is the key identity behind the closed-form `D(m, n)` diameter
specialisations in §4 of the paper. -/
theorem cLeq_acell_reflect (ℓ : ℤ) :
    cLeq (acell (m := m) (n := n)) ℓ
        + cLeq (acell (m := m) (n := n)) ((m + n : ℤ) - 3 - ℓ)
      = ((m * n : ℕ) : ℤ) := by
  set ℓ' : ℤ := (m + n : ℤ) - 3 - ℓ with hℓ'def
  -- Step 1: `cellRefl` bijection rewrites `cLeq ℓ` as #{v : acell v ≥ ℓ' + 1}.
  have hbij_card :
      cLeq (acell (m := m) (n := n)) ℓ
        = (((Finset.univ : Finset (Cell m n)).filter
            (fun v => ℓ' + 1 ≤ acell v)).card : ℤ) := by
    unfold cLeq
    congr 1
    apply Finset.card_bij (fun v _ => cellRefl v)
    · -- maps the filter on `acell ≤ ℓ` into the filter on `acell ≥ ℓ' + 1`
      intro v hv
      rw [Finset.mem_filter] at hv
      rw [Finset.mem_filter]
      refine ⟨Finset.mem_univ _, ?_⟩
      rw [acell_cellRefl]; linarith [hv.2]
    · -- injective
      intros v₁ _ v₂ _ h; exact cellRefl.injective h
    · -- surjective: pre-image of `w` is `cellRefl w`
      intro w hw
      rw [Finset.mem_filter] at hw
      refine ⟨cellRefl w, ?_, cellRefl_involutive w⟩
      rw [Finset.mem_filter]
      refine ⟨Finset.mem_univ _, ?_⟩
      rw [acell_cellRefl]; linarith [hw.2]
  -- Step 2: `acell v ≥ ℓ' + 1 ↔ ¬ acell v ≤ ℓ'`.
  have heq :
      ((Finset.univ : Finset (Cell m n)).filter (fun v => ℓ' + 1 ≤ acell v))
        = (Finset.univ : Finset (Cell m n)).filter (fun v => ¬ acell v ≤ ℓ') := by
    apply Finset.filter_congr
    intro v _
    constructor
    · intro h hcontra; linarith
    · intro h; push_neg at h; linarith
  -- Step 3: complement count.
  rw [hbij_card, heq]
  have hcomp := Finset.card_filter_add_card_filter_not
      (s := (Finset.univ : Finset (Cell m n))) (p := fun v => acell v ≤ ℓ')
  rw [Finset.card_univ] at hcomp
  have hN : (Fintype.card (Cell m n) : ℕ) = m * n := by
    simp [Cell, Fintype.card_prod, Fintype.card_fin]
  rw [hN] at hcomp
  unfold cLeq
  have hZ : (((Finset.univ : Finset (Cell m n)).filter (fun v => acell v ≤ ℓ')).card : ℤ)
          + (((Finset.univ : Finset (Cell m n)).filter (fun v => ¬ acell v ≤ ℓ')).card : ℤ)
          = ((m * n : ℕ) : ℤ) := by exact_mod_cast hcomp
  linarith

/-- **Upper-suffix closed form for `cLeq acell`.**

For `ℓ ∈ [max(m, n) − 2, m + n − 3]`, the cumulative sublevel count equals
`m · n − T((m + n − 2) − ℓ)`, where `T(k) = k(k + 1)/2` is the triangular
number.  This is the dual of the small-triangle closed form
(`cLeq_acell_triangle`) obtained by composing it with the reflection
symmetry.  At the boundary `ℓ = m + n − 2` the formula trivialises
(`cLeq = m · n` directly), so it is excluded here. -/
theorem cLeq_acell_suffix (ℓ : ℕ) (hℓlo : m - 2 ≤ ℓ) (hℓlo' : n - 2 ≤ ℓ)
    (hℓhi : ℓ + 3 ≤ m + n) :
    cLeq (acell (m := m) (n := n)) (ℓ : ℤ)
      = ((m * n : ℕ) : ℤ) - ((m + n - 2 - ℓ) * (m + n - 1 - ℓ) / 2 : ℤ) := by
  -- Apply reflection symmetry, then small-triangle closed form to the small ℓ'.
  set ℓ' : ℕ := m + n - 3 - ℓ with hℓ'def
  have hℓ'lt_m : ℓ' < m := by rw [hℓ'def]; omega
  have hℓ'lt_n : ℓ' < n := by rw [hℓ'def]; omega
  have hsym := cLeq_acell_reflect (m := m) (n := n) (ℓ : ℤ)
  have htri := cLeq_acell_triangle (m := m) (n := n) ℓ' hℓ'lt_m hℓ'lt_n
  -- Translate (m + n : ℤ) - 3 - ℓ to (ℓ' : ℤ).
  have hℓ'_cast : ((m + n : ℤ) - 3 - ℓ) = (ℓ' : ℤ) := by
    rw [hℓ'def]; omega
  rw [hℓ'_cast] at hsym
  rw [htri] at hsym
  -- Now hsym : cLeq ℓ + (↑ℓ' + 1) * (↑ℓ' + 2) / 2 = ↑(m * n).
  -- Show (↑ℓ' + 1) * (↑ℓ' + 2) / 2 = (m + n - 2 - ℓ) * (m + n - 1 - ℓ) / 2
  -- via the two linear pieces (omega is linear, so we split off the product).
  have hℓ'1 : ((ℓ' : ℤ) + 1) = ((m : ℤ) + n - 2 - ℓ) := by
    rw [hℓ'def]; omega
  have hℓ'2 : ((ℓ' : ℤ) + 2) = ((m : ℤ) + n - 1 - ℓ) := by
    rw [hℓ'def]; omega
  have heq : ((ℓ' : ℤ) + 1) * ((ℓ' : ℤ) + 2) / 2 =
             ((m + n - 2 - ℓ) * (m + n - 1 - ℓ) / 2 : ℤ) := by
    rw [hℓ'1, hℓ'2]
  linarith [heq]

end OrigamiCone

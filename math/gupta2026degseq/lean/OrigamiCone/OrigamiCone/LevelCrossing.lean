import OrigamiCone.Diameter

/-!
# The level-crossing identity (Section 4)

Formalisation of the **K-parametric level-crossing identity** used twice in
Section 4 of the paper: once in the proof of Theorem `thm:diam` (for the
closed-form specialisations of `D(m,n)`), and once in the reduction strategy
("writing `c_ℓ = #{φ ≤ ℓ}`, the level-crossing identity gives `disp(φ) = Σ_ℓ
min(c_ℓ, N − c_ℓ)`").

The cleanest combinatorial form is the **wedge identity** for a single distance:
`|x − K| = #{ℓ : min(x,K) ≤ ℓ < max(x,K)}` — the number of integer levels
strictly between `x` and `K`.  Summed over a finite indexed family and
exchanging the order of summation gives the dispersion in level-crossing form.

Both forms hold over an arbitrary finite index type, but we state them in the
shape needed by Section 4, on `Cell m n`.

The specialisation `min_K disp φ K = Σ_ℓ min(c_ℓ, N − c_ℓ)` (matching the
paper's `c_ℓ = #{φ ≤ ℓ}` form) follows from the K-parametric identity by
evaluating at `K = median`, where the per-level crossing count collapses to
`min(c_ℓ, N − c_ℓ)`.  That specialisation is formalised in `Median.lean`.

Results:
* `abs_eq_Ico_card` — the wedge identity `|x − K| = #(Ico (min x K) (max x K))`;
* `disp_eq_sum_wedge` — the per-cell wedge form `disp φ K = Σ_v #(wedge v)`;
* `sum_Ico_card_swap` — Fubini swap of an indexed family of Ico-cards;
* `disp_eq_levelCrossing` — the **level-crossing identity** for `disp φ K`,
  expressing the dispersion as a sum over a uniform level range of "crossing
  counts" `#{v : min(φv,K) ≤ ℓ < max(φv,K)}`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Wedge identity** for a single integer distance.  The absolute difference
`|x − K|` equals the number of integer "levels" strictly between `x` and `K` (or
between `K` and `x`): `|x − K| = card (Ico (min x K) (max x K))`. -/
lemma abs_eq_Ico_card (x K : ℤ) :
    |x - K| = ((Finset.Ico (min x K) (max x K)).card : ℤ) := by
  rw [Int.card_Ico, Int.toNat_of_nonneg (by rcases le_total x K with h | h <;>
    simp [min_eq_left h, max_eq_right h, min_eq_right h, max_eq_left h] <;> omega)]
  rcases le_total x K with h | h
  · rw [min_eq_left h, max_eq_right h, abs_of_nonpos (by omega)]; ring
  · rw [min_eq_right h, max_eq_left h, abs_of_nonneg (by omega)]

/-- **Per-cell wedge form of `disp`.**  The dispersion of `φ` at offset `K`
equals the total number of "crossings" — pairs `(v, ℓ)` of a cell and an integer
level strictly between `φ v` and `K`:
`disp φ K = Σ_v card (Ico (min (φ v) K) (max (φ v) K))`. -/
theorem disp_eq_sum_wedge (φ : Cell m n → ℤ) (K : ℤ) :
    disp φ K =
      ∑ v : Cell m n, ((Finset.Ico (min (φ v) K) (max (φ v) K)).card : ℤ) := by
  unfold disp
  refine Finset.sum_congr rfl ?_
  intro v _
  exact abs_eq_Ico_card (φ v) K

/-- **Fubini swap for cards of `Ico` subsumed in a uniform range.**  If every
`Ico (a v) (b v)` is contained in a uniform `Ico L U`, the sum of their cards
equals a sum over `Ico L U` of the number of indices `v` whose wedge covers `ℓ`.
This is the standard "swap summation order" step used in the level-crossing
identity. -/
lemma sum_Ico_card_swap {α : Type*} [Fintype α]
    (a b : α → ℤ) (L U : ℤ)
    (hab : ∀ v, Finset.Ico (a v) (b v) ⊆ Finset.Ico L U) :
    ∑ v, (Finset.Ico (a v) (b v)).card =
      ∑ ℓ ∈ Finset.Ico L U,
        ((Finset.univ : Finset α).filter (fun v => a v ≤ ℓ ∧ ℓ < b v)).card := by
  have hcell : ∀ v, (Finset.Ico (a v) (b v)).card =
      ((Finset.Ico L U).filter (fun ℓ => a v ≤ ℓ ∧ ℓ < b v)).card := by
    intro v
    apply Finset.card_bij (fun ℓ _ => ℓ)
    · intro ℓ hℓ
      rw [Finset.mem_filter, Finset.mem_Ico]
      rw [Finset.mem_Ico] at hℓ
      refine ⟨?_, hℓ⟩
      have := hab v (by rw [Finset.mem_Ico]; exact hℓ)
      rw [Finset.mem_Ico] at this; exact this
    · intro ℓ₁ _ ℓ₂ _ h; exact h
    · intro ℓ hℓ
      rw [Finset.mem_filter, Finset.mem_Ico] at hℓ
      exact ⟨ℓ, by rw [Finset.mem_Ico]; exact hℓ.2, rfl⟩
  simp_rw [hcell, Finset.card_filter]
  rw [Finset.sum_comm]

/-- **Level-crossing identity for `disp`.**  For any uniform integer range
`[L, U)` containing all values `φ v` and the offset `K`, the dispersion equals
the sum over `ℓ ∈ [L, U)` of the number of cells whose wedge `[min(φv,K),
max(φv,K))` covers `ℓ`.  This is the form used twice in Section 4 of the paper
(`thm:diam` closed forms and `prop:reduction`). -/
theorem disp_eq_levelCrossing (φ : Cell m n → ℤ) (K : ℤ) (L U : ℤ)
    (hK : L ≤ K ∧ K ≤ U) (hφ : ∀ v, L ≤ φ v ∧ φ v ≤ U) :
    disp φ K =
      ∑ ℓ ∈ Finset.Ico L U,
        (((Finset.univ : Finset (Cell m n)).filter
            (fun v => min (φ v) K ≤ ℓ ∧ ℓ < max (φ v) K)).card : ℤ) := by
  rw [disp_eq_sum_wedge]
  have hsub : ∀ v, Finset.Ico (min (φ v) K) (max (φ v) K) ⊆ Finset.Ico L U := by
    intro v ℓ hℓ
    rw [Finset.mem_Ico] at hℓ ⊢
    have hv := hφ v
    refine ⟨?_, ?_⟩
    · have : L ≤ min (φ v) K := le_min hv.1 hK.1
      omega
    · have : max (φ v) K ≤ U := max_le hv.2 hK.2
      omega
  have h := sum_Ico_card_swap
    (α := Cell m n) (fun v => min (φ v) K) (fun v => max (φ v) K) L U hsub
  exact_mod_cast h

end OrigamiCone

import OrigamiCone.SequelEdPaperReducedForm

/-!
# Finiteness of `PaperReducedForm m d`

Extends `SequelEdPaperReducedForm` with:

* `PaperReducedForm.h_bounded`: values of a paper-reduced form are
  bounded by `m + 2d + 3` via the Lipschitz property of height functions
  and the canonical `h(0,0) = 0` normalization.

* `PaperReducedForm.paddedValueFin` and `encFin`: an encoding of a
  paper-reduced form as a pair `(W, padded_h)` into a Fintype target
  `Fin (2*d+4) × (Cell m (2*d+3) → Fin (2*(m+2*d+3)+1))`.

* `PaperReducedForm.encFin_injective`: the encoding is injective —
  paper-reduced forms with the same `(W, padded_h)` are equal.

* `PaperReducedForm.finite_paperReducedForm`: the type
  `PaperReducedForm m d` is `Finite` (for `1 ≤ m`).  Note: this is a
  `lemma`, not a global `instance`, because `Finite` cannot depend on a
  hypothesis; callers use `haveI := finite_paperReducedForm hm` to bring
  it into scope.

This is a foundational step toward proving `Ed_decomposition_of_ge_two`
without axioms: the sum in the decomposition is over a `Finset` obtained
by summing over the finite type `PaperReducedForm m d`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone

namespace PaperReducedForm

variable {m d : ℕ}

/-- **Value bound**: `|t.h p| ≤ m + 2d + 3`.  Follows from Lipschitz
(one-step changes are ±1) applied on the L¹ path from the origin `(0,0)`
(where `t.h = 0` by canonicalness) to `p`, using `|p.1| ≤ m - 1`,
`|p.2| ≤ W - 1`, and `W ≤ 2d + 3`. -/
lemma h_bounded (t : PaperReducedForm m d) (hm : 1 ≤ m)
    (p : Cell m t.W) : |t.h p| ≤ (m + 2 * d + 3 : ℤ) := by
  have hW_pos : 1 ≤ t.W := by have := t.hW; omega
  let origin : Cell m t.W := (⟨0, hm⟩, ⟨0, hW_pos⟩)
  have h_orig_zero : t.h origin = 0 := t.isCanonical.2 origin rfl rfl
  have h_lip : |t.h origin - t.h p| ≤ gdist origin p :=
    height_lipschitz t.isHeight (gdist origin p).toNat origin p (by
      have h := gdist_nonneg origin p
      omega)
  have hp1 : p.1.val ≤ m - 1 := Nat.le_sub_one_of_lt p.1.isLt
  have hp2 : p.2.val ≤ t.W - 1 := Nat.le_sub_one_of_lt p.2.isLt
  have hW_ub : t.W ≤ 2 * d + 3 := t.hW_upper
  have h_gd_eq : gdist origin p = ((p.1.val + p.2.val : ℕ) : ℤ) := by
    unfold gdist origin
    push_cast
    have h1 : |(0 : ℤ) - (p.1.val : ℤ)| = (p.1.val : ℤ) := by
      rw [zero_sub, abs_neg, abs_of_nonneg (Int.natCast_nonneg _)]
    have h2 : |(0 : ℤ) - (p.2.val : ℤ)| = (p.2.val : ℤ) := by
      rw [zero_sub, abs_neg, abs_of_nonneg (Int.natCast_nonneg _)]
    linarith
  rw [h_orig_zero, zero_sub, abs_neg] at h_lip
  rw [h_gd_eq] at h_lip
  have h_sum : p.1.val + p.2.val ≤ m + 2 * d + 1 := by omega
  push_cast at h_lip
  linarith

/-- **Padded value encoding**: shift `t.h` at cell `q` (for `q.2.val < t.W`)
by `m + 2d + 3` to make it nonneg, then pack into `Fin (2*(m+2d+3)+1)`.
For `q.2.val ≥ t.W`, the padded value is 0. -/
noncomputable def paddedValueFin (t : PaperReducedForm m d) (hm : 1 ≤ m)
    (q : Cell m (2 * d + 3)) : Fin (2 * (m + 2 * d + 3) + 1) :=
  if h_lt : q.2.val < t.W then
    let v := t.h (q.1, ⟨q.2.val, h_lt⟩)
    ⟨(v + (m + 2 * d + 3)).toNat, by
      have h_bd := t.h_bounded hm (q.1, ⟨q.2.val, h_lt⟩)
      have hv_ge : v ≥ -(m + 2 * d + 3 : ℤ) := (abs_le.mp h_bd).1
      have hv_le : v ≤ (m + 2 * d + 3 : ℤ) := (abs_le.mp h_bd).2
      have h_ub : (v + (m + 2 * d + 3 : ℤ)) ≤ 2 * (m + 2 * d + 3) := by linarith
      have h_toNat_bound : (v + (m + 2 * d + 3 : ℤ)).toNat ≤ 2 * (m + 2 * d + 3) := by
        rw [Int.toNat_le]; push_cast; exact h_ub
      omega⟩
  else 0

/-- Fintype target for the encoding: `(W, padded_h)`. -/
abbrev EncTarget (m d : ℕ) : Type :=
  Fin (2 * d + 4) × (Cell m (2 * d + 3) → Fin (2 * (m + 2 * d + 3) + 1))

/-- **Bounded encoding** of a paper-reduced form. -/
noncomputable def encFin (m d : ℕ) (hm : 1 ≤ m) :
    PaperReducedForm m d → EncTarget m d :=
  fun t => (⟨t.W, by have := t.hW_upper; omega⟩, t.paddedValueFin hm)

/-- **Injectivity** of `encFin`.  Two paper-reduced forms with the same
`(W, padded_h)` are equal.  Recovers `h` from `paddedValueFin` on the
restricted domain, then uses proof irrelevance for the propositional
fields. -/
lemma encFin_injective (hm : 1 ≤ m) :
    Function.Injective (encFin m d hm) := by
  intro t1 t2 h_eq
  have h_W_fin_eq : (⟨t1.W, _⟩ : Fin (2*d+4)) = ⟨t2.W, _⟩ := congrArg Prod.fst h_eq
  have h_W_eq : t1.W = t2.W := Fin.mk.inj_iff.mp h_W_fin_eq
  have h_pad_eq : t1.paddedValueFin hm = t2.paddedValueFin hm :=
    congrArg Prod.snd h_eq
  obtain ⟨W1, hW1, hW1_up, h1, iH1, iC1, red1, ne1⟩ := t1
  obtain ⟨W2, hW2, hW2_up, h2, iH2, iC2, red2, ne2⟩ := t2
  simp only at h_W_eq
  subst h_W_eq
  have h_h_eq : h1 = h2 := by
    funext p
    have h_lt_upper : p.2.val < 2 * d + 3 := lt_of_lt_of_le p.2.isLt hW1_up
    let q : Cell m (2*d+3) := (p.1, ⟨p.2.val, h_lt_upper⟩)
    have h_q_pad := congrFun h_pad_eq q
    unfold paddedValueFin at h_q_pad
    have h_lt_W : q.2.val < W1 := p.2.isLt
    simp only [h_lt_W, dif_pos] at h_q_pad
    have h_val_eq : (h1 (q.1, ⟨q.2.val, h_lt_W⟩) + (m + 2 * d + 3 : ℤ)).toNat =
                    (h2 (q.1, ⟨q.2.val, h_lt_W⟩) + (m + 2 * d + 3 : ℤ)).toNat :=
      Fin.mk.inj_iff.mp h_q_pad
    have hbd1 : |h1 (q.1, ⟨q.2.val, h_lt_W⟩)| ≤ (m + 2 * d + 3 : ℤ) :=
      (⟨W1, hW1, hW1_up, h1, iH1, iC1, red1, ne1⟩ : PaperReducedForm m d).h_bounded hm _
    have hbd2 : |h2 (q.1, ⟨q.2.val, h_lt_W⟩)| ≤ (m + 2 * d + 3 : ℤ) :=
      (⟨W1, hW2, hW1_up, h2, iH2, iC2, red2, ne2⟩ : PaperReducedForm m d).h_bounded hm _
    have hnn1 : (0 : ℤ) ≤ h1 (q.1, ⟨q.2.val, h_lt_W⟩) + (m + 2 * d + 3 : ℤ) := by
      have := (abs_le.mp hbd1).1; linarith
    have hnn2 : (0 : ℤ) ≤ h2 (q.1, ⟨q.2.val, h_lt_W⟩) + (m + 2 * d + 3 : ℤ) := by
      have := (abs_le.mp hbd2).1; linarith
    have h_int_eq : h1 (q.1, ⟨q.2.val, h_lt_W⟩) + (m + 2 * d + 3 : ℤ) =
                    h2 (q.1, ⟨q.2.val, h_lt_W⟩) + (m + 2 * d + 3 : ℤ) := by
      have e1 := Int.toNat_of_nonneg hnn1
      have e2 := Int.toNat_of_nonneg hnn2
      rw [← e1, ← e2, h_val_eq]
    have h_final : h1 (q.1, ⟨q.2.val, h_lt_W⟩) = h2 (q.1, ⟨q.2.val, h_lt_W⟩) := by linarith
    have h_p_eq : (q.1, (⟨q.2.val, h_lt_W⟩ : Fin W1)) = p := by ext <;> rfl
    rw [h_p_eq] at h_final
    exact h_final
  subst h_h_eq
  rfl

/-- **`PaperReducedForm m d` is Finite** (for `1 ≤ m`). -/
lemma finite_paperReducedForm (hm : 1 ≤ m) : Finite (PaperReducedForm m d) :=
  Finite.of_injective _ (encFin_injective hm)

end PaperReducedForm

end OrigamiCone.Sequel

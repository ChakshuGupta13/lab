/-
Phase F6 — Bit-indexed subspaces `V ℓ` (paper Lemma `lem:Vsubspaces`,
main.tex lines ~318–360).

For a `KyberLike spec` and field `K`, define for each layer `ℓ : Fin spec.N`

  bitOf ℓ i := (i.val / spec.L ℓ) % 2
  V ℓ := span K {basis i | bitOf ℓ i = 1}

The paper's `β_ℓ = log_2 n - 1 - ℓ` corresponds to the bit position
`log_2 (spec.L ℓ)` here (concretely for Kyber: spec.L = (128, 64, …, 2),
so the bits touched are 7, 6, …, 1).

The headline equality (eq:Vsum):

  ⨆ ℓ, V ℓ  =  span K {basis i | spec.L ⟨spec.N - 1, _⟩ ≤ i.val}.

For Kyber this is `span {e_2, …, e_{n-1}}` (since `spec.L ⟨6, _⟩ = 2`),
and the complement `{i.val < L_{N-1}} = {0, 1}` is the kernel from F1.
The abstract form generalises cleanly: the kernel is exactly the
indices below `L_{N-1}`.

Strategy:
  1. `bitOf_eq_zero_of_lt_L` — small index ⇒ bit zero.
  2. `lt_L_of_bit_one` — bit = 1 forces `i.val ≥ spec.L ℓ`.
  3. `lt_L_last_of_all_bits_zero` — main inductive arithmetic claim:
     all bits β_ℓ zero ⇒ i.val < L_{N-1}. Proved by induction down
     through ℓ via `KyberLike.L_succ` halving.
  4. Containment ⊇ via Step 3 contrapositive: `L_{N-1} ≤ i.val` ⇒
     some ℓ has `bitOf ℓ i = 1`.
  5. Containment ⊆ via Step 2 + `spec.L_antitone` (from Fellsupport).
  6. Headline by `le_antisymm`.
-/

import NttFaultRank.Fellsupport
import Mathlib.LinearAlgebra.Span.Defs

namespace NttFaultRank

namespace LayerSpec
variable {K : Type*} [Field K]
variable (spec : LayerSpec)

/-! ### Bit extraction -/

/-- Bit at position `β_ℓ` of `i.val`, written as `(i.val / spec.L ℓ) % 2`. -/
def bitOf (ℓ : Fin spec.N) (i : Fin spec.n) : ℕ := (i.val / spec.L ℓ) % 2

/-- If `i.val < spec.L ℓ`, then bit `β_ℓ` of `i` is `0`. -/
lemma bitOf_eq_zero_of_lt_L (ℓ : Fin spec.N) (i : Fin spec.n)
    (h : i.val < spec.L ℓ) : spec.bitOf ℓ i = 0 := by
  unfold bitOf
  rw [Nat.div_eq_of_lt h]

/-- If bit `β_ℓ` of `i` is `1`, then `spec.L ℓ ≤ i.val`. (Contrapositive of
    `bitOf_eq_zero_of_lt_L` plus the fact that `bitOf` only takes values 0/1.) -/
lemma L_le_of_bitOf_eq_one (ℓ : Fin spec.N) (i : Fin spec.n)
    (h : spec.bitOf ℓ i = 1) : spec.L ℓ ≤ i.val := by
  by_contra hlt
  push_neg at hlt
  rw [spec.bitOf_eq_zero_of_lt_L ℓ i hlt] at h
  exact absurd h (by decide)

/-! ### Main arithmetic: all bits zero ⇒ small -/

/-- **Core induction**: if every `bitOf ℓ i` is zero, then `i.val < spec.L ⟨k, _⟩`
    for every `k < spec.N`. Proved by induction on `k`. -/
lemma lt_L_of_all_bits_zero [KyberLike spec] (i : Fin spec.n)
    (h : ∀ ℓ : Fin spec.N, spec.bitOf ℓ i = 0) :
    ∀ (k : ℕ) (hk : k < spec.N), i.val < spec.L ⟨k, hk⟩ := by
  intro k hk
  induction k with
  | zero =>
    -- Base: i.val < n = 2 * spec.L ⟨0, hk⟩, and bit at L_0 is zero,
    -- so i.val / L_0 = 0, i.e. i.val < L_0.
    have h0 : (i.val / spec.L ⟨0, hk⟩) % 2 = 0 := h ⟨0, hk⟩
    have hbig : i.val < 2 * spec.L ⟨0, hk⟩ := by
      rw [KyberLike.L_zero_doubled hk]; exact i.isLt
    have hL_pos : 0 < spec.L ⟨0, hk⟩ := by
      rcases Nat.eq_zero_or_pos (spec.L ⟨0, hk⟩) with hz | hp
      · exfalso; rw [hz] at hbig; omega
      · exact hp
    have hquot : i.val / spec.L ⟨0, hk⟩ < 2 := by
      rw [Nat.div_lt_iff_lt_mul hL_pos]
      omega
    -- bitOf = (i / L_0) % 2 = 0, and (i / L_0) < 2, so i / L_0 = 0.
    have hzero : i.val / spec.L ⟨0, hk⟩ = 0 := by
      -- Generalize the divison so omega sees a fresh variable.
      generalize hq : i.val / spec.L ⟨0, hk⟩ = q at h0 hquot
      omega
    exact Nat.lt_of_div_eq_zero hL_pos hzero
  | succ k ih =>
    have hk_prev : k < spec.N := Nat.lt_of_succ_lt hk
    have ih' := ih hk_prev
    -- ih' : i.val < spec.L ⟨k, hk_prev⟩
    -- L_succ: spec.L ⟨k+1, hk⟩ * 2 = spec.L ⟨k, hk_prev⟩
    have hLs := KyberLike.L_succ (spec := spec) k hk
    have hbig : i.val < 2 * spec.L ⟨k + 1, hk⟩ := by
      have : spec.L ⟨k, hk_prev⟩ = 2 * spec.L ⟨k + 1, hk⟩ := by
        omega
      omega
    have hbit : (i.val / spec.L ⟨k + 1, hk⟩) % 2 = 0 := h ⟨k + 1, hk⟩
    have hL_pos : 0 < spec.L ⟨k + 1, hk⟩ := by
      rcases Nat.eq_zero_or_pos (spec.L ⟨k + 1, hk⟩) with hz | hp
      · exfalso; rw [hz] at hbig; omega
      · exact hp
    have hquot : i.val / spec.L ⟨k + 1, hk⟩ < 2 := by
      rw [Nat.div_lt_iff_lt_mul hL_pos]
      omega
    have hzero : i.val / spec.L ⟨k + 1, hk⟩ = 0 := by
      generalize hq : i.val / spec.L ⟨k + 1, hk⟩ = q at hbit hquot
      omega
    exact Nat.lt_of_div_eq_zero hL_pos hzero

/-- **Specialisation to last layer**: if every bit is zero, then
    `i.val < spec.L ⟨spec.N - 1, _⟩`. -/
lemma lt_L_last_of_all_bits_zero [KyberLike spec] (i : Fin spec.n)
    (hN : 0 < spec.N)
    (h : ∀ ℓ : Fin spec.N, spec.bitOf ℓ i = 0) :
    i.val < spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ :=
  spec.lt_L_of_all_bits_zero i h (spec.N - 1) _

/-- **Converse**: if `i.val < spec.L ⟨spec.N - 1, _⟩`, every bit is zero. -/
lemma all_bits_zero_of_lt_L_last [KyberLike spec] (i : Fin spec.n)
    (hN : 0 < spec.N)
    (h : i.val < spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩) :
    ∀ ℓ : Fin spec.N, spec.bitOf ℓ i = 0 := by
  intro ℓ
  -- L is antitone in ℓ, so L_{N-1} ≤ L_ℓ.
  have hℓ_le : ℓ.val ≤ spec.N - 1 := by have := ℓ.isLt; omega
  have hN1 : spec.N - 1 < spec.N := Nat.sub_lt hN Nat.one_pos
  have hanti : spec.L ⟨spec.N - 1, hN1⟩ ≤ spec.L ⟨ℓ.val, ℓ.isLt⟩ :=
    spec.L_antitone hN1 ℓ.isLt hℓ_le
  have hℓ_eta : (⟨ℓ.val, ℓ.isLt⟩ : Fin spec.N) = ℓ := Fin.eta ℓ ℓ.isLt
  rw [hℓ_eta] at hanti
  apply spec.bitOf_eq_zero_of_lt_L
  omega

/-! ### V subspaces -/

/-- The bit-`β_ℓ` coordinate subspace at layer `ℓ`. -/
def V (ℓ : Fin spec.N) : Submodule K (Fin spec.n → K) :=
  Submodule.span K
    { v | ∃ i : Fin spec.n, spec.bitOf ℓ i = 1 ∧ v = spec.basis (K := K) i }

/-- The complementary "big" span: indices `i` with `i.val ≥ spec.L ⟨N-1, _⟩`. -/
def spanBigBasis [KyberLike spec] (hN : 0 < spec.N) :
    Submodule K (Fin spec.n → K) :=
  Submodule.span K
    { v | ∃ i : Fin spec.n,
        spec.L ⟨spec.N - 1, Nat.sub_lt hN Nat.one_pos⟩ ≤ i.val
        ∧ v = spec.basis (K := K) i }

/-! ### Headline -/

/-- **F6 main result.** The sum of the `V ℓ` covers exactly the basis vectors
    at indices `≥ spec.L ⟨spec.N - 1, _⟩`. For Kyber (`spec.L ⟨6, _⟩ = 2`)
    this is `span {e_2, …, e_{n-1}}`, complementary to the `{e_0, e_1}` kernel. -/
theorem V_sum_eq_spanBigBasis [KyberLike spec] (hN : 0 < spec.N) :
    (⨆ ℓ : Fin spec.N, spec.V (K := K) ℓ) = spec.spanBigBasis (K := K) hN := by
  apply le_antisymm
  · -- ⊆: each V ℓ generator is basis i with bitOf ℓ i = 1; that forces
    --   spec.L ℓ ≤ i.val, and spec.L ⟨N-1,_⟩ ≤ spec.L ℓ (antitone), done.
    refine iSup_le ?_
    intro ℓ
    unfold V spanBigBasis
    apply Submodule.span_mono
    rintro v ⟨i, hbit, rfl⟩
    refine ⟨i, ?_, rfl⟩
    -- spec.L ⟨N-1,_⟩ ≤ spec.L ℓ ≤ i.val.
    have hN1 : spec.N - 1 < spec.N := Nat.sub_lt hN Nat.one_pos
    have hℓ_le : ℓ.val ≤ spec.N - 1 := by have := ℓ.isLt; omega
    have hanti : spec.L ⟨spec.N - 1, hN1⟩ ≤ spec.L ⟨ℓ.val, ℓ.isLt⟩ :=
      spec.L_antitone hN1 ℓ.isLt hℓ_le
    have hℓ_eta : (⟨ℓ.val, ℓ.isLt⟩ : Fin spec.N) = ℓ := Fin.eta ℓ ℓ.isLt
    rw [hℓ_eta] at hanti
    have hLi : spec.L ℓ ≤ i.val := spec.L_le_of_bitOf_eq_one ℓ i hbit
    omega
  · -- ⊇: each basis i with spec.L ⟨N-1,_⟩ ≤ i.val belongs to some V ℓ
    --     (because if all bits were zero, i.val < L_{N-1}, contradicting hyp).
    unfold spanBigBasis
    rw [Submodule.span_le]
    rintro v ⟨i, hi_ge, rfl⟩
    -- Find some ℓ with bitOf ℓ i = 1.
    by_contra hne
    -- hne : basis i ∉ ⨆ ℓ, V ℓ. From this, derive that bitOf ℓ i = 0 for all ℓ.
    have hall : ∀ ℓ : Fin spec.N, spec.bitOf ℓ i = 0 := by
      intro ℓ
      by_contra hbne
      -- bitOf ℓ i ∈ {0, 1}; ¬= 0 means = 1 (since (·) % 2 < 2).
      have hb1 : spec.bitOf ℓ i = 1 := by
        have : spec.bitOf ℓ i < 2 := Nat.mod_lt _ (by decide)
        omega
      -- Then basis i ∈ V ℓ ⊆ ⨆ V ℓ, contradicting hne.
      apply hne
      refine SetLike.le_def.mp (le_iSup spec.V ℓ) ?_
      unfold V
      apply Submodule.subset_span
      exact ⟨i, hb1, rfl⟩
    -- All bits zero ⇒ i.val < L_{N-1}, contradicting hi_ge.
    have := spec.lt_L_last_of_all_bits_zero i hN hall
    omega

end LayerSpec
end NttFaultRank

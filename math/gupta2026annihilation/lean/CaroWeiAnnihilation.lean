/-
  Theorem 1 of "An annihilation-number Caro-Wei bound" — Lean formalization.

  Statement (paper):  For every graphic degree sequence with maximum degree
  Δ ≥ 1, the annihilation number a satisfies a ≤ (Δ+1)/2 · W, where
  W = Σ 1/(d_i+1) is the Caro-Wei sum.

  This file formalizes the **algebraic core**.  The graph-theoretic content
  reduces to a single hypothesis on the head subset of size a:
      Σ_{i ∈ H} d_i ≤ Δ · (n − |H|).
  This is the bound the paper derives from "Σ_H d ≤ m" (definition of a) and
  the handshake lemma "Σ d_i = 2m" (Theorem 1, "Head degree-sum bound").
  Everything downstream of that bound — the pointwise inequality, the
  head/tail split, the cancellation that yields W ≥ 2a/(Δ+1) — is formalized
  here against the rationals.

  Axiom guarantee: this file uses only `propext`, `Classical.choice`,
  `Quot.sound` (the Lean 4 / Mathlib defaults). No `sorry`, no `admit`,
  no `native_decide`.
-/
import Mathlib

namespace TxGraffitiC1

open Finset BigOperators

/--
**Pointwise Caro-Wei estimate.** For naturals `k` and `Δ` with `1 ≤ Δ` and
`k ≤ Δ`,
  `1/(k+1) + k/(Δ(Δ+1)) ≥ 2/(Δ+1)`  (over ℚ),
with equality at `k = Δ-1` and `k = Δ`.

Proof: clear denominators (legitimate: every factor is positive).  The
polynomial inequality `Δ(Δ+1) + k(k+1) ≥ 2Δ(k+1)` factors as
`(Δ-k)(Δ-k-1) ≥ 0`.  Since `k ≤ Δ` in ℕ, the integer `Δ-k` is
nonnegative, so either it is `0` (product `0·(-1) = 0`) or `Δ-k ≥ 1`
(product of two nonnegative numbers).
-/
theorem pointwise (Δ k : ℕ) (hΔ : 1 ≤ Δ) (hk : k ≤ Δ) :
    (2 : ℚ) / ((Δ : ℚ) + 1)
      ≤ 1 / ((k : ℚ) + 1) + (k : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1)) := by
  have hk1_pos : (0 : ℚ) < (k : ℚ) + 1 := by positivity
  have hΔ_pos  : (0 : ℚ) < (Δ : ℚ) := by exact_mod_cast hΔ
  have hΔ1_pos : (0 : ℚ) < (Δ : ℚ) + 1 := by linarith
  -- (Δ - k)(Δ - k - 1) ≥ 0 because k ≤ Δ in ℕ.
  have h_core : (0 : ℚ) ≤ ((Δ : ℚ) - k) * ((Δ : ℚ) - k - 1) := by
    rcases eq_or_lt_of_le hk with heq | hlt
    · subst heq
      have : ((k : ℚ) - k) * ((k : ℚ) - k - 1) = 0 := by ring
      linarith
    · have h1 : ((k : ℚ)) + 1 ≤ ((Δ : ℚ)) := by exact_mod_cast hlt
      have h2 : (0 : ℚ) ≤ (Δ : ℚ) - k - 1 := by linarith
      have h3 : (0 : ℚ) ≤ (Δ : ℚ) - k := by linarith
      exact mul_nonneg h3 h2
  -- Algebraic identity: the difference IS the factored numerator over a
  -- positive denominator.
  have h_id :
      1 / ((k : ℚ) + 1) + (k : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        - 2 / ((Δ : ℚ) + 1)
      = ((Δ : ℚ) - k) * ((Δ : ℚ) - k - 1)
          / ((Δ : ℚ) * ((Δ : ℚ) + 1) * ((k : ℚ) + 1)) := by
    field_simp
    ring
  have h_denom :
      (0 : ℚ) < (Δ : ℚ) * ((Δ : ℚ) + 1) * ((k : ℚ) + 1) := by positivity
  have h_diff_nn :
      (0 : ℚ) ≤ 1 / ((k : ℚ) + 1) + (k : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
                - 2 / ((Δ : ℚ) + 1) := by
    rw [h_id]; exact div_nonneg h_core h_denom.le
  linarith

/--
**Annihilation Caro-Wei bound (algebraic core).** Given a degree vector
`d : Fin n → ℕ` with every degree at most `Δ` (and `1 ≤ Δ`), and a head
subset `H ⊆ Fin n` whose degree sum is bounded by `Δ · (n − |H|)`, one
concludes
    `|H| ≤ (Δ+1)/2 · Σ_i 1/(d i + 1)`  (over ℚ).

The hypothesis `Σ_{i ∈ H} d i ≤ Δ · (n − |H|)` is what the paper derives
from "Σ_H d ≤ m" (definition of the annihilation number) and the
handshake lemma "Σ d_i = 2m"; everything past that point is the
algebraic content formalized here.
-/
theorem annihilation_caroWei
    {n : ℕ} (Δ : ℕ) (d : Fin n → ℕ)
    (hΔ : 1 ≤ Δ) (hd_le : ∀ i, d i ≤ Δ)
    (H : Finset (Fin n))
    (h_head_sum : (∑ i ∈ H, d i) ≤ Δ * (n - H.card)) :
    (H.card : ℚ)
      ≤ ((Δ : ℚ) + 1) / 2 * ∑ i : Fin n, 1 / ((d i : ℚ) + 1) := by
  set a := H.card with ha_def
  have hΔ_pos  : (0 : ℚ) < (Δ : ℚ) := by exact_mod_cast hΔ
  have hΔ1_pos : (0 : ℚ) < (Δ : ℚ) + 1 := by linarith
  have hΔΔ1    : (0 : ℚ) < (Δ : ℚ) * ((Δ : ℚ) + 1) := mul_pos hΔ_pos hΔ1_pos
  have hHsub   : H ⊆ Finset.univ := H.subset_univ
  have h_tcard : (Finset.univ \ H).card = n - a := by
    rw [← Finset.compl_eq_univ_sdiff, Finset.card_compl, Fintype.card_fin]
  -- Tail bound: each tail term ≥ 1/(Δ+1), so the tail sums to ≥ (n-a)/(Δ+1).
  have h_tail :
      ((n - a : ℕ) : ℚ) / ((Δ : ℚ) + 1)
        ≤ ∑ i ∈ Finset.univ \ H, 1 / ((d i : ℚ) + 1) := by
    have h_ptw : ∀ i ∈ Finset.univ \ H,
        (1 : ℚ) / ((Δ : ℚ) + 1) ≤ 1 / ((d i : ℚ) + 1) := by
      intro i _
      have hdi_le : (d i : ℚ) ≤ (Δ : ℚ) := by exact_mod_cast hd_le i
      have hdi1_pos : (0 : ℚ) < (d i : ℚ) + 1 := by positivity
      exact one_div_le_one_div_of_le hdi1_pos (by linarith)
    have h_sum := Finset.sum_le_sum h_ptw
    have h_const :
        (∑ _ ∈ Finset.univ \ H, (1 : ℚ) / ((Δ : ℚ) + 1))
          = ((n - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := by
      rw [Finset.sum_const, h_tcard, nsmul_eq_mul]; ring
    linarith
  -- Head bound: for each i ∈ H, 1/(d_i+1) ≥ 2/(Δ+1) − d_i/(Δ(Δ+1)).
  have h_head :
      (a : ℚ) * (2 / ((Δ : ℚ) + 1))
          - (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        ≤ ∑ i ∈ H, 1 / ((d i : ℚ) + 1) := by
    have h_ptw : ∀ i ∈ H,
        2 / ((Δ : ℚ) + 1) - (d i : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
          ≤ 1 / ((d i : ℚ) + 1) := by
      intro i _
      have := pointwise Δ (d i) hΔ (hd_le i)
      linarith
    have h_sum := Finset.sum_le_sum h_ptw
    have h_eq :
        (∑ i ∈ H, (2 / ((Δ : ℚ) + 1) - (d i : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))))
          = (a : ℚ) * (2 / ((Δ : ℚ) + 1))
            - (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1)) := by
      rw [Finset.sum_sub_distrib, Finset.sum_const, nsmul_eq_mul,
          ← Finset.sum_div, ha_def]
    linarith
  -- Cancellation: (Σ_H d) / (Δ(Δ+1)) ≤ (n-a) / (Δ+1).
  have h_sumQ : (∑ i ∈ H, (d i : ℚ)) ≤ (Δ : ℚ) * ((n - a : ℕ) : ℚ) := by
    have : ((∑ i ∈ H, d i : ℕ) : ℚ) ≤ ((Δ : ℕ) : ℚ) * (((n - a : ℕ)) : ℚ) := by
      exact_mod_cast h_head_sum
    simpa using this
  have h_mid :
      (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        ≤ ((n - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := by
    have h_simp :
        (Δ : ℚ) * ((n - a : ℕ) : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
          = ((n - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := by
      rw [mul_div_mul_left _ _ hΔ_pos.ne']
    calc (∑ i ∈ H, (d i : ℚ)) / ((Δ : ℚ) * ((Δ : ℚ) + 1))
        ≤ (Δ : ℚ) * ((n - a : ℕ) : ℚ) / ((Δ : ℚ) * ((Δ : ℚ) + 1)) := by
          gcongr
      _ = ((n - a : ℕ) : ℚ) / ((Δ : ℚ) + 1) := h_simp
  -- Total split: W = sum over H + sum over tail.
  have h_split :
      (∑ i : Fin n, 1 / ((d i : ℚ) + 1))
        = (∑ i ∈ H, 1 / ((d i : ℚ) + 1))
          + (∑ i ∈ Finset.univ \ H, 1 / ((d i : ℚ) + 1)) := by
    have h := Finset.sum_sdiff hHsub
                (f := fun i => (1 : ℚ) / ((d i : ℚ) + 1))
    linarith
  -- Combine: W ≥ a · 2/(Δ+1) − (Σ_H d)/(Δ(Δ+1)) + (n-a)/(Δ+1)
  --              ≥ 2a/(Δ+1)
  -- (using h_mid to cancel the middle term against the tail contribution).
  have W_lb :
      2 * (a : ℚ) / ((Δ : ℚ) + 1)
        ≤ ∑ i : Fin n, 1 / ((d i : ℚ) + 1) := by
    rw [h_split]
    have h_rw :
        (a : ℚ) * (2 / ((Δ : ℚ) + 1)) = 2 * (a : ℚ) / ((Δ : ℚ) + 1) := by ring
    linarith
  -- Final rearrangement: 2a/(Δ+1) ≤ W  ⟹  a ≤ (Δ+1)/2 · W.
  have h_mul : 2 * (a : ℚ) ≤ ((Δ : ℚ) + 1) * ∑ i : Fin n, 1 / ((d i : ℚ) + 1) := by
    have := (div_le_iff₀ hΔ1_pos).mp W_lb
    linarith
  linarith

end TxGraffitiC1

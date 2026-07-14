import Mathlib

/-!
# Sequel: stars-and-bars for run-length compositions (Task E.δ.h primitive)

Substrate-independent combinatorial primitive for the paper `lem:uniform`
per-type contribution.  In the `lem:uniform` proof, a *type* is a contracted
height function with `r ≤ d - 1` frozen runs.  Recovering the original
`n`-column height function amounts to choosing a positive `r`-tuple of run
lengths summing to `n - (W - r)`, where `W` is the contracted width.  This
module counts those tuples and packages the result in the `(n - c).choose k`
shape that feeds `SequelEdRunCount.degreeBound_assembly`.

## Theorems

* `antidiagonalTuple_succ_card` — the fundamental recursion
  `|adt_{r+1}(m)| = ∑_{a ≤ m} |adt_r(m - a)|` on the first coordinate.
* `antidiagonalTuple_card` — closed form `|adt_{r+1}(m)| = (m + r).choose r`
  (stars and bars).  Proved by induction using the recursion + Mathlib's
  hockey-stick `Nat.sum_range_add_choose`.
* `positiveComp_card` — positive `r`-compositions of `S`:
  `|{f ∈ adt_r(S) | ∀ i, 1 ≤ f i}| = (S - 1).choose (r - 1)`.  Proved by the
  standard `f ↦ f - 1` shift-bijection onto `adt_r(S - r)`, then closed form.
* **`runExtension_card`** — per-type run-length count for `lem:uniform`:
  an `r`-run type of contracted width `W` extends to `n` columns in
  `(n - (W - r + 1)).choose (r - 1)` ways.  This is the `(n - c).choose k`
  shape with `c = W - r + 1` and `k = r - 1`, matching
  `SequelEdRunCount.runCount_eventually_polynomial`.

## Substrate

Imports `Mathlib` only.  No grid, no `Cell`.  Standalone.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open Finset

/-- **First-coordinate recursion for `antidiagonalTuple`.**
`|adt_{r+1}(m)| = ∑_{a ≤ m} |adt_r(m - a)|`: an `(r+1)`-tuple summing to `m`
is uniquely determined by its first coordinate `a` and the remaining `r`-tuple
summing to `m - a`. -/
theorem antidiagonalTuple_succ_card (r m : ℕ) :
    (Finset.Nat.antidiagonalTuple (r+1) m).card
      = ∑ a ∈ range (m+1), (Finset.Nat.antidiagonalTuple r (m - a)).card := by
  rw [← Finset.card_sigma]
  apply Finset.card_nbij'
    (fun f => (⟨f 0, Fin.tail f⟩ : Σ _ : ℕ, (Fin r → ℕ)))
    (fun p => Fin.cons p.1 p.2)
  · intro f hf
    simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple] at hf
    simp only [Finset.mem_coe, Finset.mem_sigma, Finset.mem_range,
      Finset.Nat.mem_antidiagonalTuple]
    have hsum : f 0 + ∑ i, Fin.tail f i = m := by
      have h := hf; rw [Fin.sum_univ_succ] at h; exact h
    refine ⟨by omega, by omega⟩
  · intro p hp
    simp only [Finset.mem_coe, Finset.mem_sigma, Finset.mem_range,
      Finset.Nat.mem_antidiagonalTuple] at hp
    simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple, Fin.sum_univ_succ,
      Fin.cons_zero, Fin.cons_succ]
    omega
  · intro f _; exact Fin.cons_self_tail f
  · intro p _; simp only [Fin.cons_zero, Fin.tail_cons]

/-- **Stars and bars** (closed form).  The number of `(r+1)`-tuples of natural
numbers summing to `m` is `(m + r).choose r`.  Proved by induction on `r` using
`antidiagonalTuple_succ_card` and Mathlib's hockey-stick identity
`Nat.sum_range_add_choose`. -/
theorem antidiagonalTuple_card (r m : ℕ) :
    (Finset.Nat.antidiagonalTuple (r+1) m).card = (m + r).choose r := by
  induction r generalizing m with
  | zero => simp
  | succ r ih =>
    rw [antidiagonalTuple_succ_card, Finset.sum_congr rfl (fun a _ => ih (m - a))]
    have hrefl : ∑ a ∈ range (m+1), ((m - a) + r).choose r
        = ∑ i ∈ range (m+1), (i + r).choose r := by
      rw [← Finset.sum_range_reflect (fun i => (i + r).choose r) (m+1)]
      apply Finset.sum_congr rfl
      intro a ha
      simp only [Finset.mem_range] at ha
      congr 1
    rw [hrefl, Nat.sum_range_add_choose m r]
    congr 1 <;> omega

/-- **Positive `r`-compositions of `S`.**  The number of `r`-tuples of *positive*
natural numbers summing to `S` is `(S - 1).choose (r - 1)`.  Proved by the
`f ↦ f - 1` shift-bijection onto `antidiagonalTuple r (S - r)`, then closed
form via `antidiagonalTuple_card`. -/
theorem positiveComp_card (r S : ℕ) (hr : 1 ≤ r) (hS : r ≤ S) :
    ((Finset.Nat.antidiagonalTuple r S).filter (fun f => ∀ i, 1 ≤ f i)).card
      = (S - 1).choose (r - 1) := by
  have hbij : ((Finset.Nat.antidiagonalTuple r S).filter (fun f => ∀ i, 1 ≤ f i)).card
      = (Finset.Nat.antidiagonalTuple r (S - r)).card := by
    apply Finset.card_nbij' (fun f => (fun i => f i - 1)) (fun g => (fun i => g i + 1))
    · intro f hf
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple] at hf
      simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple]
      obtain ⟨hsum, hpos⟩ := hf
      have hh : (∑ i, (f i - 1)) + ∑ _i : Fin r, 1 = ∑ i, f i := by
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro i _; have := hpos i; omega
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
        mul_one, hsum] at hh
      omega
    · intro g hg
      simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple] at hg
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple]
      refine ⟨?_, fun i => by omega⟩
      rw [Finset.sum_add_distrib, hg, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        smul_eq_mul, mul_one]
      omega
    · intro f hf
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple] at hf
      funext i; have := hf.2 i; dsimp only; omega
    · intro g _; funext i; dsimp only; omega
  rw [hbij]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  rw [antidiagonalTuple_card r' (S - (r' + 1))]
  congr 1 <;> omega

/-- **Per-type run-length extension count** (paper `lem:uniform`).
A contracted type has `r` frozen runs and width `W`.  Recovering the original
height function on `n` columns amounts to choosing an `r`-tuple of run lengths,
each `≥ 1`, summing to `n - (W - r)` (the `W - r` non-run columns are
unmodified).  There are `(n - (W - r + 1)).choose (r - 1)` such extensions,
matching the `(n - c).choose k` shape of `runCount_eventually_polynomial` with
`c = W - r + 1` and `k = r - 1`.

Since `r ≤ d - 1`, `k = r - 1 ≤ d - 2`, so each per-type contribution is
eventually a polynomial in `n` of degree at most `d - 2` — the natDegree bound
that `degreeBound_assembly` then propagates through the finite sum over
types. -/
theorem runExtension_card (r W n : ℕ) (hr : 1 ≤ r) (hW : r ≤ W) (hn : W ≤ n) :
    ((Finset.Nat.antidiagonalTuple r (n - W + r)).filter (fun f => ∀ i, 1 ≤ f i)).card
      = (n - (W - r + 1)).choose (r - 1) := by
  rw [positiveComp_card r (n - W + r) hr (by omega)]
  congr 1 <;> omega

end OrigamiCone.Sequel

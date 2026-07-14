import Mathlib
import OrigamiCone.SequelPolesLeibnizStep

/-!
# Sequel: PowerSeries Leibniz identification — Leibniz-side recursive identity

Third installment (turn 2b/3) of the multi-turn campaign formalising step 3
of the paper's `lem:poles` chain.

* **Turn 1 (`SequelPolesLeibnizBase`)** : definitions + `d = 0` and `N < d`
  cases.
* **Turn 2a (`SequelPolesLeibnizStep`)** : `leibnizFactor_split`,
  `leibnizFactor_cons_zero`, `leibnizFactor_cons_succ`,
  `Tmx_pow_succ_coeff_succ` (polynomial-side recurrence).
* **This module (turn 2b)** : the **Leibniz-side recurrence**

  ```
  LeibnizForm T A (e+1) (N+1)
    = T · LeibnizForm T A (e+1) N + A · LeibnizForm T A e N        (for e ≤ N)
  ```

  Proved by partitioning `Finset.Nat.antidiagonalTuple (e+2) (N+1-(e+1))` by
  the predicate `g 0 = 0` vs `g 0 ≠ 0`. The `g 0 = 0` subset bijects onto
  `antidiagonalTuple (e+1) (N-e)` via `Fin.tail` (inverse `Fin.cons 0`),
  with `leibnizFactor T A (e+1) g = A · leibnizFactor T A e (Fin.tail g)`
  (from `leibnizFactor_cons_zero` repackaged via `leibnizFactor_split`).
  The `g 0 ≠ 0` subset bijects onto `antidiagonalTuple (e+2) (N-e-1)` via
  `g ↦ Fin.cons (g 0 - 1) (Fin.tail g)`, with `leibnizFactor T A (e+1) g =
  T · leibnizFactor T A (e+1) (Fin.cons (g 0 - 1) (Fin.tail g))` (from
  `leibnizFactor_cons_succ`).
* **Turn 2c (future)** : combine the polynomial-side step
  (`Tmx_pow_succ_coeff_succ`) and the Leibniz-side step
  (`LeibnizForm_succ_succ`) via induction on `N` (parametric in `d`) to
  close `((Tmx T A)^N).coeff d = LeibnizForm T A d N` for general `d`.
* **Turn 3 (future)** : the identification `LeibnizForm T A d N
  = RseqMat T A d (N - d)` matching the abstract `d`-fold convolutional sum
  from `SequelPolesArbD`.

## Boundary case `N = e`

The recurrence holds at `N = e` (the boundary): `LeibnizForm T A (e+1) e = 0`
(vacuous, `N < e + 1`), so the RHS collapses to `A · LeibnizForm T A e e`.
Both sides equal `A^(e+1)` (the unique zero-composition term, `T^0 · A · T^0
· A · ⋯ · A · T^0 = A^(e+1)` on the LHS; `A · A^e` on the RHS).

## Theorems

* `leibnizForm_succ_zero` (`g 0 = 0` partition piece):
  `∑ g ∈ (antidiagonalTuple (e+2) M).filter (g 0 = 0), leibnizFactor T A (e+1) g
   = A · ∑ h ∈ antidiagonalTuple (e+1) M, leibnizFactor T A e h`.
  Proof via `Finset.sum_bij` with `g ↦ Fin.tail g`.
* `leibnizForm_succ_nz` (`g 0 ≠ 0` partition piece):
  `∑ g ∈ (antidiagonalTuple (e+2) (M+1)).filter (g 0 ≠ 0), leibnizFactor T A (e+1) g
   = T · ∑ k ∈ antidiagonalTuple (e+2) M, leibnizFactor T A (e+1) k`.
  Proof via `Finset.sum_bij` with `g ↦ Fin.cons (g 0 - 1) (Fin.tail g)`.
* `LeibnizForm_succ_succ` (**main turn 2b result**): for `e ≤ N`,
  `LeibnizForm T A (e+1) (N+1) = T · LeibnizForm T A (e+1) N + A
  · LeibnizForm T A e N`. Combines the two partition pieces with the
  filter-partition identity `∑ = ∑_{filter} + ∑_{filter¬}` and handles the
  `N = e` boundary case separately.

## Scope

* Imports `Mathlib` and `OrigamiCone.SequelPolesLeibnizStep`. Same
  cross-Sequel-import discipline as `SequelPolesLeibnizStep` (same-campaign,
  no parallel-session race).
* No `sorry`. Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
* NOT added to root aggregator `OrigamiCone.lean`.

Check axioms with `#print axioms OrigamiCone.Sequel.LeibnizForm_succ_succ`.
-/

namespace OrigamiCone.Sequel

open Polynomial Matrix Finset

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **`g 0 = 0` partition piece**: filtering `antidiagonalTuple (e+2) M` to
compositions with first coordinate `0`, the leibnizFactor sum bijects with
the full `antidiagonalTuple (e+1) M` sum, with an extra `A` factor pulled
out (from the `T^0 · A` block at position `0`).

Proof via `Finset.sum_bij` with map `g ↦ Fin.tail g` (inverse `Fin.cons 0`).
The value identity is `leibnizFactor T A (e+1) g = A · leibnizFactor T A e
(Fin.tail g)` for `g 0 = 0`, from `leibnizFactor_split` + `T^0 = 1`. -/
theorem leibnizForm_succ_zero (T A : Matrix ι ι R) (e M : ℕ) :
    ∑ g ∈ (Finset.Nat.antidiagonalTuple (e + 2) M).filter (fun g => g 0 = 0),
        leibnizFactor T A (e + 1) g
      = A * ∑ h ∈ Finset.Nat.antidiagonalTuple (e + 1) M, leibnizFactor T A e h := by
  rw [Finset.mul_sum]
  apply Finset.sum_bij (i := fun g _ => Fin.tail g)
  · -- maps into target
    intro g hg
    rw [Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple] at hg
    obtain ⟨hg_mem, _⟩ := hg
    rw [Finset.Nat.mem_antidiagonalTuple]
    have h : ∑ i, g i = g 0 + ∑ i, Fin.tail g i := Fin.sum_univ_succ g
    have : g 0 + ∑ i, Fin.tail g i = M := h ▸ hg_mem
    omega
  · -- injective on
    intro a ha b hb hab
    rw [Finset.mem_filter] at ha hb
    have ha0 := ha.2
    have hb0 := hb.2
    rw [← Fin.cons_self_tail a, ← Fin.cons_self_tail b, ha0, hb0, hab]
  · -- surjective onto
    intro h hh
    rw [Finset.Nat.mem_antidiagonalTuple] at hh
    refine ⟨Fin.cons 0 h, ?_, ?_⟩
    · rw [Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple]
      refine ⟨?_, ?_⟩
      · have h2 : ∑ i, (Fin.cons 0 h : Fin (e+2) → ℕ) i
                    = (Fin.cons 0 h : Fin (e+2) → ℕ) 0
                      + ∑ i, Fin.tail (Fin.cons 0 h : Fin (e+2) → ℕ) i :=
          Fin.sum_univ_succ _
        rw [Fin.cons_zero, Fin.tail_cons, zero_add] at h2
        rw [h2]; exact hh
      · rw [Fin.cons_zero]
    · rw [Fin.tail_cons]
  · -- value identity
    intro g hg
    rw [Finset.mem_filter] at hg
    have hg0 : g 0 = 0 := hg.2
    rw [leibnizFactor_split T A e g, hg0, pow_zero, one_mul]

/-- **`g 0 ≠ 0` partition piece**: filtering `antidiagonalTuple (e+2) (M+1)`
to compositions with first coordinate nonzero, the leibnizFactor sum
bijects with the full `antidiagonalTuple (e+2) M` sum, with an extra `T`
factor pulled out (from peeling one `T` off the head's `T^{g 0}` block,
since `g 0 ≥ 1`).

Proof via `Finset.sum_bij` with map `g ↦ Fin.cons (g 0 - 1) (Fin.tail g)`
(inverse `h ↦ Fin.cons (h 0 + 1) (Fin.tail h)`). The value identity is
`leibnizFactor T A (e+1) g = T · leibnizFactor T A (e+1) (Fin.cons (g 0 - 1)
(Fin.tail g))` for `g 0 ≠ 0`, from `leibnizFactor_cons_succ` after rewriting
`g = Fin.cons (g 0 - 1 + 1) (Fin.tail g)` (using `Fin.cons_self_tail`). -/
theorem leibnizForm_succ_nz (T A : Matrix ι ι R) (e M : ℕ) :
    ∑ g ∈ (Finset.Nat.antidiagonalTuple (e + 2) (M + 1)).filter
        (fun g => ¬ (g 0 = 0)),
        leibnizFactor T A (e + 1) g
      = T * ∑ k ∈ Finset.Nat.antidiagonalTuple (e + 2) M, leibnizFactor T A (e + 1) k := by
  rw [Finset.mul_sum]
  apply Finset.sum_bij (i := fun g _ => Fin.cons (g 0 - 1) (Fin.tail g))
  · -- maps into target
    intro g hg
    rw [Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple] at hg
    obtain ⟨hg_mem, hg0⟩ := hg
    rw [Finset.Nat.mem_antidiagonalTuple]
    have hs : ∑ i, g i = g 0 + ∑ i, Fin.tail g i := Fin.sum_univ_succ g
    have hsm : ∑ i, (Fin.cons (g 0 - 1) (Fin.tail g) : Fin (e+2) → ℕ) i
                = (Fin.cons (g 0 - 1) (Fin.tail g) : Fin (e+2) → ℕ) 0
                  + ∑ i, Fin.tail (Fin.cons (g 0 - 1) (Fin.tail g) : Fin (e+2) → ℕ) i :=
      Fin.sum_univ_succ _
    rw [Fin.cons_zero, Fin.tail_cons] at hsm
    rw [hsm]
    have : g 0 + ∑ i, Fin.tail g i = M + 1 := hs ▸ hg_mem
    omega
  · -- injective on
    intro a ha b hb hab
    rw [Finset.mem_filter] at ha hb
    have ha0 : a 0 ≠ 0 := ha.2
    have hb0 : b 0 ≠ 0 := hb.2
    have h_tail : Fin.tail a = Fin.tail b := by
      have := congr_arg Fin.tail hab
      rw [Fin.tail_cons, Fin.tail_cons] at this
      exact this
    have h_zero : a 0 = b 0 := by
      have := congr_arg (fun f => f 0) hab
      simp only [Fin.cons_zero] at this
      omega
    rw [← Fin.cons_self_tail a, ← Fin.cons_self_tail b, h_zero, h_tail]
  · -- surjective onto
    intro k k_mem
    rw [Finset.Nat.mem_antidiagonalTuple] at k_mem
    refine ⟨Fin.cons (k 0 + 1) (Fin.tail k), ?_, ?_⟩
    · rw [Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple]
      refine ⟨?_, ?_⟩
      · have hsm : ∑ i, (Fin.cons (k 0 + 1) (Fin.tail k) : Fin (e+2) → ℕ) i
                    = (Fin.cons (k 0 + 1) (Fin.tail k) : Fin (e+2) → ℕ) 0
                      + ∑ i, Fin.tail (Fin.cons (k 0 + 1) (Fin.tail k) : Fin (e+2) → ℕ) i :=
          Fin.sum_univ_succ _
        rw [Fin.cons_zero, Fin.tail_cons] at hsm
        have hs : ∑ i, k i = k 0 + ∑ i, Fin.tail k i := Fin.sum_univ_succ k
        have : k 0 + ∑ i, Fin.tail k i = M := hs ▸ k_mem
        rw [hsm]; omega
      · rw [Fin.cons_zero]; exact Nat.succ_ne_zero _
    · rw [Fin.cons_zero, Fin.tail_cons]
      have : k 0 + 1 - 1 = k 0 := by omega
      rw [this]
      exact Fin.cons_self_tail k
  · -- value identity
    intro g hg
    rw [Finset.mem_filter] at hg
    have hg0 : g 0 ≠ 0 := hg.2
    have hg0' : g 0 = (g 0 - 1) + 1 := by omega
    have step : g = Fin.cons (g 0 - 1 + 1) (Fin.tail g) := by
      rw [← hg0']; exact (Fin.cons_self_tail g).symm
    rw [step]
    exact leibnizFactor_cons_succ T A e (g 0 - 1) (Fin.tail g)

/-- **Leibniz-side recurrence** (main turn 2b result): for `e ≤ N`,

```
LeibnizForm T A (e+1) (N+1)
  = T · LeibnizForm T A (e+1) N + A · LeibnizForm T A e N.
```

Proof: unfold `LeibnizForm` (the `N < d` if-guard simplifies via `e ≤ N`),
then split on `N = e` vs `N ≥ e + 1`. At the boundary `N = e`, the inner
`LeibnizForm T A (e+1) N = 0` (vacuous) and the LHS is the single
zero-composition leibnizFactor = `A^(e+1)`, which equals `A · LeibnizForm
T A e e = A · A^e = A^(e+1)` via the filter-restricted form
(`leibnizForm_succ_zero`'s filter is the full singleton when `M = 0`,
because the only composition is zero). For `N ≥ e + 1`, the inner
`LeibnizForm T A (e+1) N` is non-vacuous; let `M := N - e - 1`, partition
`antidiagonalTuple (e+2) (M+1)` by `g 0 = 0` vs `g 0 ≠ 0`, apply
`leibnizForm_succ_zero` to the first part and `leibnizForm_succ_nz` to the
second. -/
theorem LeibnizForm_succ_succ (T A : Matrix ι ι R) (e N : ℕ) (hN : e ≤ N) :
    LeibnizForm T A (e + 1) (N + 1)
      = T * LeibnizForm T A (e + 1) N + A * LeibnizForm T A e N := by
  unfold LeibnizForm
  rw [if_neg (show ¬ (N + 1 < e + 1) by omega)]
  rw [if_neg (show ¬ (N < e) by omega)]
  by_cases hNe : N < e + 1
  · -- boundary: N = e (since e ≤ N < e + 1)
    rw [if_pos hNe, mul_zero, zero_add]
    rw [show N + 1 - (e + 1) = 0 from by omega, show N - e = 0 from by omega]
    have key := leibnizForm_succ_zero T A e 0
    -- The filter (g 0 = 0) on antidiagonalTuple (e+2) 0 is the full set
    -- (the only composition of 0 is the constant 0).
    have h_full :
        (Finset.Nat.antidiagonalTuple (e + 2) 0).filter (fun g => g 0 = 0)
          = Finset.Nat.antidiagonalTuple (e + 2) 0 := by
      apply Finset.filter_true_of_mem
      intro g hg
      rw [Finset.Nat.mem_antidiagonalTuple] at hg
      have h0 : g 0 ≤ ∑ i, g i := by
        rw [Fin.sum_univ_succ g]; omega
      omega
    rw [h_full] at key
    exact key
  · -- non-vacuous: N ≥ e + 1
    rw [if_neg hNe]
    rw [show N + 1 - (e + 1) = N - e from by omega]
    have hM : N - e ≥ 1 := by omega
    obtain ⟨M, hM_eq⟩ : ∃ M, N - e = M + 1 := ⟨N - e - 1, by omega⟩
    rw [hM_eq]
    rw [show N - (e + 1) = M from by omega]
    -- Partition antidiagonalTuple (e+2) (M+1) by g 0 = 0 vs g 0 ≠ 0.
    rw [← Finset.sum_filter_add_sum_filter_not _
          (fun g : Fin (e + 2) → ℕ => g 0 = 0) (leibnizFactor T A (e + 1))]
    rw [leibnizForm_succ_zero T A e (M + 1)]
    rw [leibnizForm_succ_nz T A e M]
    rw [add_comm]

end OrigamiCone.Sequel

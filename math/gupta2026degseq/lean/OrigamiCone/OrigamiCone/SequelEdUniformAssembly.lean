import OrigamiCone.SequelEdStarsBars
import OrigamiCone.SequelEdRunCount

/-!
# Sequel: assembly wrappers for `lem:uniform` (Task E.δ.h substrate interface)

The paper `lem:uniform` proves `E_d(m, ·)` agrees on `{n ≥ N}` with a
polynomial of degree at most `d - 2`.  Its argument splits into a
substrate-independent arithmetic side (the run-length composition count times
the number of *types* summed into a polynomial degree bound) and a
substrate-heavy contraction side (define types, prove the extremum-count
fiber partitions into them, prove disjointness).

The substrate-independent side is now fully banked: `SequelEdStarsBars.lean`
supplies the per-type count `runExtension_card`, `SequelEdRunCount.lean`
supplies the polynomial fit `runCount_eventually_polynomial` and the sum
assembly `degreeBound_assembly`.  This module composes them into the exact
end-to-end interface any consumer of `lem:uniform` needs.

## Theorems

* `runExtension_polynomial` — per-type polynomiality: for fixed `1 ≤ r ≤ W`,
  the count `n ↦ #{positive r-tuples summing to n - W + r}` agrees on
  `{n ≥ W}` with a polynomial of natDegree ≤ `r - 1`.
* `hdecomp_from_partition` — partition → `hdecomp` (at ℚ level): given a
  finite type of "types" whose fibers partition the height-function set with
  each fiber's card equal to a `runExtension_card` output, the total card
  decomposes as `∑ (n - c).choose k` in the `degreeBound_assembly` shape
  (with `c = W - r + 1`, `k = r - 1`).
* **`uniform_polynomial_from_partition`** — end-to-end: partition + arithmetic
  bounds ⟹ single polynomial witness of natDegree ≤ `D`, on `{n ≥ N}`.  This
  is the direct interface `lem:uniform` consumers will use once the (grid)
  contraction map produces the partition.

## Substrate

Imports `SequelEdStarsBars` (for `runExtension_card`) and `SequelEdRunCount`
(for `runCount_eventually_polynomial`, `degreeBound_assembly`).  Standalone.
Substrate-independent: no grid, no `Cell`.

## Interface to `lem:uniform`

A consumer proves `lem:uniform` per-axis degree by:

1. Defining a finite type `ι` of *types* (contracted height functions).
2. Defining fibers `fiber : ι → ℕ → Finset X` (say, height functions on `n`
   columns whose contraction is of the given type).
3. Proving fibers partition the height-function set (`hpart`) and are
   pairwise disjoint (`hdisj`).
4. Proving each fiber's card = `runExtension_card` output (`hfiber`).
5. Bounding `r t - 1 ≤ D := d - 2` (`hbound`, from `numFrozenRuns_le`) and
   `W t ≤ N := 2d - 1` (`hWN`, the contracted-width upper bound).

Steps 1–4 are the substrate-heavy work (deferred, own-session material).
Step 5 is arithmetic.  Then `uniform_polynomial_from_partition` supplies the
polynomial witness.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

open Finset

/-- **Per-type polynomiality.**  For fixed `1 ≤ r ≤ W`, the count of positive
`r`-tuples summing to `n - W + r` agrees on `{n ≥ W}` with a single polynomial
of natDegree ≤ `r - 1`.

Composes `runExtension_card` with `runCount_eventually_polynomial`.  This is the
one-type case of the paper's `lem:uniform` degree bound. -/
theorem runExtension_polynomial (r W : ℕ) (hr : 1 ≤ r) (hW : r ≤ W) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ r - 1 ∧
      ∀ n : ℕ, W ≤ n →
        (((Finset.Nat.antidiagonalTuple r (n - W + r)).filter
          (fun f => ∀ i, 1 ≤ f i)).card : ℚ) = p.eval (n : ℚ) := by
  obtain ⟨p, hp_deg, hp_eval⟩ := runCount_eventually_polynomial (W - r + 1) (r - 1)
  refine ⟨p, hp_deg, ?_⟩
  intro n hn
  rw [runExtension_card r W n hr hW hn]
  push_cast
  exact hp_eval n (by omega)

/-- **Partition to `hdecomp`.**  Given a `Finset` partition of the height-function
set into type-fibers, where each fiber's card equals the corresponding positive
`r`-composition count (i.e., `runExtension_card` output), the total card in ℚ
decomposes as `∑_t (n - (W t - r t + 1)).choose (r t - 1)` — exactly the
`hdecomp` shape (with `mult = 1`) that `degreeBound_assembly` consumes. -/
theorem hdecomp_from_partition
    {ι X : Type*} [DecidableEq X]
    (types : Finset ι) (S : ℕ → Finset X) (fiber : ι → ℕ → Finset X)
    (r W : ι → ℕ) (N : ℕ)
    (hWN : ∀ t ∈ types, W t ≤ N)
    (hfiber : ∀ t ∈ types, ∀ n, W t ≤ n →
        (fiber t n).card = (n - (W t - r t + 1)).choose (r t - 1))
    (hpart : ∀ n, N ≤ n → S n = types.biUnion (fun t => fiber t n))
    (hdisj : ∀ n, N ≤ n → (↑types : Set ι).PairwiseDisjoint (fun t => fiber t n)) :
    ∀ n, N ≤ n → ((S n).card : ℚ)
      = ∑ t ∈ types, ((n - (W t - r t + 1)).choose (r t - 1) : ℚ) := by
  intro n hn
  rw [hpart n hn, Finset.card_biUnion (hdisj n hn)]
  push_cast
  apply Finset.sum_congr rfl
  intro t ht
  have := hfiber t ht n (le_trans (hWN t ht) hn)
  exact_mod_cast this

/-- **End-to-end assembly** (paper `lem:uniform` per-axis-degree conclusion).
Given a partition of the height-function set `S n` into type-fibers indexed by
a finite type `ι`, with each fiber's card equal to a `runExtension_card` output
and each type satisfying `r t - 1 ≤ D`, the count `n ↦ |S n|` (as ℚ) agrees on
`{n ≥ N}` with a single polynomial of natDegree ≤ `D`.

Composes `hdecomp_from_partition` with `SequelEdRunCount.degreeBound_assembly`
under the substitution `c t = W t - r t + 1`, `k t = r t - 1`, `mult t = 1`.
The consumer's obligations are exactly the substrate-heavy contraction-map
pieces: `hpart`, `hdisj`, `hfiber` (see the module docstring for the interface).
-/
theorem uniform_polynomial_from_partition
    {ι X : Type*} [DecidableEq X]
    (types : Finset ι) (S : ℕ → Finset X) (fiber : ι → ℕ → Finset X)
    (r W : ι → ℕ) (D N : ℕ)
    (hr : ∀ t ∈ types, 1 ≤ r t)
    (hrW : ∀ t ∈ types, r t ≤ W t)
    (hbound : ∀ t ∈ types, r t - 1 ≤ D)
    (hWN : ∀ t ∈ types, W t ≤ N)
    (hfiber : ∀ t ∈ types, ∀ n, W t ≤ n →
        (fiber t n).card = (n - (W t - r t + 1)).choose (r t - 1))
    (hpart : ∀ n, N ≤ n → S n = types.biUnion (fun t => fiber t n))
    (hdisj : ∀ n, N ≤ n → (↑types : Set ι).PairwiseDisjoint (fun t => fiber t n)) :
    ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ n : ℕ, N ≤ n → ((S n).card : ℚ) = p.eval (n : ℚ) := by
  refine degreeBound_assembly types (fun _ => (1 : ℚ))
    (fun t => W t - r t + 1) (fun t => r t - 1) D N (fun n => ((S n).card : ℚ))
    hbound ?_ ?_
  · intro t ht
    have h1 := hr t ht; have h2 := hrW t ht; have h3 := hWN t ht
    simp only; omega
  · intro n hn
    show ((S n).card : ℚ)
        = ∑ t ∈ types, 1 * ((n - (W t - r t + 1)).choose (r t - 1) : ℚ)
    rw [hdecomp_from_partition types S fiber r W N hWN hfiber hpart hdisj n hn]
    simp

end OrigamiCone.Sequel

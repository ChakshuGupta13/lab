import Mathlib

/-!
# 1D ±1 walks: strict local max on a chain with a strict local min

Standalone reusable primitive for arguments over ±1 integer walks on a finite
chain: **given a ±1 walk `f : ℕ → ℤ` on `[0, L]` and an interior strict local
minimum at position `j` (`f (j-1) > f j`), there is an earlier index `i₀ < j`
at which `f` has a strict local maximum** — either the endpoint `i₀ = 0` with
just the right neighbour required to be smaller, or an interior index with
both neighbours smaller.

This is the 1D counterpart of the paper's "along the last column, `h` has a
strict local maximum in some row above the interior apex's row" argument in
the Boundary Lemma (`lem:boundary`) case 2a. Reusable in any origami-paper
argument that reduces a 2D grid analysis to a ±1 walk on a chain.

## Design

The walk is on `f : ℕ → ℤ` with the ±1 property enforced only on `[0, L]`
(indices beyond `L` are unconstrained). The strict local minimum hypothesis
enters only via the leftward-decrease at `j`: `f j < f (j - 1)`; the paper's
symmetric right-side hypothesis is not needed for this direction.

The proof follows the paper's own line: pick a maximizer `i₀` of `f` on
`[0, j]` — smallest index achieving the max, via `Finset.min'` on the filtered
set — then observe:

* `i₀ ≠ j` because `f (j-1) > f j` means the max is not at `j`;
* `f (i₀+1) < f i₀` because `i₀+1 ∈ [0, j]` gives `f (i₀+1) ≤ f i₀` and the
  ±1 walk forbids equality;
* if `i₀ > 0` then `f (i₀-1) < f i₀` because `i₀-1 ∈ [0, j]` gives
  `f (i₀-1) ≤ f i₀`, and by minimality of `i₀` in the filtered set the
  inequality is strict.

No `sorry`; check with
`#print axioms OrigamiCone.pm1_walk_strictMax_before_strictMin`.
-/

namespace OrigamiCone

open Finset

/-- **Strict local max on a chain with a strict local min at the endpoint.**
For a ±1 walk `f : ℕ → ℤ` on `[0, L]` with `f (j - 1) > f j` at position
`0 < j ≤ L`, there is an index `i₀ < j` at which `f` has a strict local max
on the chain: `f (i₀+1) < f i₀`, and either `i₀ = 0` (endpoint, no left
neighbour to check) or `f (i₀-1) < f i₀`. -/
theorem pm1_walk_strictMax_before_strictMin
    (L : ℕ) (f : ℕ → ℤ)
    (hwalk : ∀ i, i < L → |f (i + 1) - f i| = 1)
    (j : ℕ) (hj_pos : 0 < j) (hj_le : j ≤ L)
    (hj_lt : f j < f (j - 1)) :
    ∃ i₀ : ℕ, i₀ < j ∧
      f (i₀ + 1) < f i₀ ∧
      (i₀ = 0 ∨ f (i₀ - 1) < f i₀) := by
  classical
  -- STEP 1: pick a maximiser of `f` on the chain `[0, j]`.
  have hrange_nonempty : (Finset.range (j + 1)).Nonempty := ⟨0, by simp⟩
  obtain ⟨M, hM_mem, hM_max⟩ := Finset.exists_max_image (Finset.range (j + 1)) f hrange_nonempty
  set S := (Finset.range (j + 1)).filter (fun a => f a = f M) with hS
  have hS_nonempty : S.Nonempty := ⟨M, by simp [S, hM_mem]⟩
  -- i₀ = the smallest index in [0, j] achieving the max value f M.
  set i₀ := S.min' hS_nonempty with hi₀
  have hi₀_mem : i₀ ∈ S := S.min'_mem hS_nonempty
  have hi₀_min : ∀ k ∈ S, i₀ ≤ k := fun k hk => S.min'_le k hk
  have hi₀_range : i₀ ∈ Finset.range (j + 1) := (Finset.mem_filter.mp hi₀_mem).1
  have hi₀_val : f i₀ = f M := (Finset.mem_filter.mp hi₀_mem).2
  have hi₀_le_j : i₀ ≤ j := by
    have := Finset.mem_range.mp hi₀_range
    omega
  have hM_range : M ∈ Finset.range (j + 1) := hM_mem
  have hM_le_j : M ≤ j := by have := Finset.mem_range.mp hM_range; omega
  -- STEP 2: i₀ < j. If i₀ = j, then f j = f M = max, but f (j - 1) > f j contradicts.
  have hi₀_lt_j : i₀ < j := by
    by_contra h_eq
    push_neg at h_eq
    have hi₀_eq_j : i₀ = j := le_antisymm hi₀_le_j h_eq
    -- f j = f M, but f (j - 1) > f j and (j - 1) ∈ [0, j].
    have h_jm1_range : j - 1 ∈ Finset.range (j + 1) := by
      apply Finset.mem_range.mpr; omega
    have h_jm1_le_M : f (j - 1) ≤ f M := hM_max _ h_jm1_range
    rw [← hi₀_val, hi₀_eq_j] at h_jm1_le_M
    omega
  -- STEP 3: right neighbour i₀ + 1 has f (i₀+1) < f i₀.
  have h_right : f (i₀ + 1) < f i₀ := by
    have h_iu1_range : i₀ + 1 ∈ Finset.range (j + 1) := by
      apply Finset.mem_range.mpr; omega
    have h_iu1_le : f (i₀ + 1) ≤ f i₀ := by
      rw [hi₀_val]; exact hM_max _ h_iu1_range
    -- ±1 walk: |f (i₀+1) - f i₀| = 1, so f (i₀+1) ≠ f i₀.
    have h_walk_here : |f (i₀ + 1) - f i₀| = 1 := hwalk i₀ (by omega)
    have h_ne : f (i₀ + 1) ≠ f i₀ := by
      intro h_eq
      rw [h_eq, sub_self, abs_zero] at h_walk_here
      exact absurd h_walk_here (by norm_num)
    exact lt_of_le_of_ne h_iu1_le h_ne
  -- STEP 4: left neighbour (if i₀ > 0). By minimality of i₀ in S, f (i₀-1) < f i₀.
  refine ⟨i₀, hi₀_lt_j, h_right, ?_⟩
  by_cases h_pos : i₀ = 0
  · exact Or.inl h_pos
  · right
    push_neg at h_pos
    have h_id1_range : i₀ - 1 ∈ Finset.range (j + 1) := by
      apply Finset.mem_range.mpr; omega
    have h_id1_le : f (i₀ - 1) ≤ f i₀ := by
      rw [hi₀_val]; exact hM_max _ h_id1_range
    -- If f (i₀ - 1) = f i₀ = f M, then i₀ - 1 ∈ S with i₀ - 1 < i₀, contradicting min'.
    have h_id1_ne : f (i₀ - 1) ≠ f i₀ := by
      intro h_eq
      have h_id1_in_S : i₀ - 1 ∈ S := by
        rw [hS]; apply Finset.mem_filter.mpr
        exact ⟨h_id1_range, h_eq.trans hi₀_val⟩
      have : i₀ ≤ i₀ - 1 := hi₀_min _ h_id1_in_S
      omega
    exact lt_of_le_of_ne h_id1_le h_id1_ne

end OrigamiCone

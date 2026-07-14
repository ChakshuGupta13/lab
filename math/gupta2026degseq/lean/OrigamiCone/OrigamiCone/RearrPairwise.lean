import Mathlib.Data.Fin.Tuple.Sort
import Mathlib.Tactic.Linarith
import OrigamiCone.RearrMonotone

/-!
# Rearrangement preserves the pairwise `1`-bound (fact (ii) of `prop:monotone`)

Paper fact (ii) of the **Monotone reduction** (Section 4,
`prop:monotone`, main.tex L1043-1047): if `|x k − y k| ≤ 1` for all `k`,
then the increasing rearrangements satisfy
`|(x ∘ sort x) k − (y ∘ sort y) k| ≤ 1` for all `k`.

Strategy: derive from fact (iii) (`sort_pointwise_le`,
`RearrMonotone.lean`) applied to two shifted sequences.  The
pairwise-1 bound `|x k − y k| ≤ 1` is equivalent to the conjunction
`x k ≤ y k + 1 ∧ y k ≤ x k + 1`, i.e. pointwise comparisons against
the shifted partner.  Apply (iii) to each direction; collapse the
shifted partner's sorted form back to the original sorted form
shifted by `±1` using **shift-commutes-with-sort**:

  `(fun k => y k + c) ∘ Tuple.sort (fun k => y k + c) = fun k => (y ∘ Tuple.sort y) k + c`.

This is `Tuple.unique_monotone` applied to `f = fun k => y k + c`,
`σ = Tuple.sort y`, `τ = Tuple.sort (fun k => y k + c)` (both
permutations of `f`, both giving monotone compositions).

Results:
* `Tuple.sort_add_const` — shift commutes with sort.
* `sort_pairwise_abs_le_one` — fact (ii).

No `sorry`.
-/

namespace OrigamiCone

variable {L : ℕ}

/-- **Sort commutes with shift by a constant.**  For integer-valued
`y : Fin L → ℤ` and any `c : ℤ`, the sorted shifted sequence equals the
shifted sorted sequence.

This is the key building block: shifting by a constant is monotone, so
it does not change the sort permutation up to extensional equality of
the composed functions. -/
lemma sort_add_const (y : Fin L → ℤ) (c : ℤ) :
    (fun k : Fin L => y k + c) ∘ Tuple.sort (fun k : Fin L => y k + c)
      = fun k : Fin L => (y ∘ Tuple.sort y) k + c := by
  -- Both sides are monotone permutations of `y + c`, so equal by
  -- `Tuple.unique_monotone`.
  -- Reshape RHS as `(y + c) ∘ sort y`:
  have h_rhs_eq : (fun k : Fin L => (y ∘ Tuple.sort y) k + c)
      = (fun k : Fin L => y k + c) ∘ Tuple.sort y := by
    funext k; rfl
  rw [h_rhs_eq]
  -- Both sides are monotone (LHS by Tuple.monotone_sort; RHS because
  -- shifting by c preserves monotonicity of y ∘ sort y).
  have h_mono_sy : Monotone ((fun k : Fin L => y k + c) ∘ Tuple.sort y) :=
    fun a b hab => by
      have h : y (Tuple.sort y a) ≤ y (Tuple.sort y b) := Tuple.monotone_sort y hab
      change y (Tuple.sort y a) + c ≤ y (Tuple.sort y b) + c
      linarith
  exact Tuple.unique_monotone (Tuple.monotone_sort _) h_mono_sy

/-- **Fact (ii) of `prop:monotone`.**  If `|x k − y k| ≤ 1` for every
`k`, then the increasing rearrangements also satisfy
`|(x ∘ sort x) k − (y ∘ sort y) k| ≤ 1` for every `k`.

Proof: split the absolute bound into two pointwise inequalities,
`x k ≤ y k + 1` and `y k ≤ x k + 1`.  Apply `sort_pointwise_le` to
each (with one side shifted by `+1`), then collapse the shifted
sorted form via `sort_add_const`. -/
theorem sort_pairwise_abs_le_one {x y : Fin L → ℤ}
    (h : ∀ k, |x k - y k| ≤ 1) (k : Fin L) :
    |(x ∘ Tuple.sort x) k - (y ∘ Tuple.sort y) k| ≤ 1 := by
  -- Two pointwise inequalities from `|x − y| ≤ 1`.
  have h_xy : ∀ k, x k ≤ y k + 1 := fun k => by have := (abs_le.mp (h k)).2; linarith
  have h_yx : ∀ k, y k ≤ x k + 1 := fun k => by have := (abs_le.mp (h k)).1; linarith
  -- Direction 1: `xs k ≤ ys k + 1`.
  -- Apply `sort_pointwise_le` to `x` and `(y + 1)`; collapse using
  -- `sort_add_const`.
  have h_dir1 : (x ∘ Tuple.sort x) k
      ≤ ((fun i => y i + 1) ∘ Tuple.sort (fun i => y i + 1)) k :=
    sort_pointwise_le h_xy k
  have h_collapse_y :
      ((fun i => y i + 1) ∘ Tuple.sort (fun i => y i + 1)) k
        = (y ∘ Tuple.sort y) k + 1 := by
    have := sort_add_const y 1
    exact congr_fun this k
  rw [h_collapse_y] at h_dir1
  -- Direction 2: `ys k ≤ xs k + 1` by symmetry.
  have h_dir2 : (y ∘ Tuple.sort y) k
      ≤ ((fun i => x i + 1) ∘ Tuple.sort (fun i => x i + 1)) k :=
    sort_pointwise_le h_yx k
  have h_collapse_x :
      ((fun i => x i + 1) ∘ Tuple.sort (fun i => x i + 1)) k
        = (x ∘ Tuple.sort x) k + 1 := by
    have := sort_add_const x 1
    exact congr_fun this k
  rw [h_collapse_x] at h_dir2
  -- Combine: `|(x ∘ sort x) k - (y ∘ sort y) k| ≤ 1`.
  rw [abs_le]; constructor <;> linarith

end OrigamiCone

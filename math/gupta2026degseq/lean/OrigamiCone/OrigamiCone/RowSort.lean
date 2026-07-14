import Mathlib.Tactic.Linarith
import OrigamiCone.RearrLipschitz1
import OrigamiCone.RearrPairwise
import OrigamiCone.Reduction

/-!
# Row-sort preserves `1`-Lipschitz on the grid (Sub-4a of `prop:monotone`)

First piece of the assembly for `prop:monotone` (Section 4, main.tex
L1024-1056).  Defines the **row-sort** of a grid function `φ : Cell m n → ℤ`
and proves it preserves the `1`-Lipschitz property.

The row-sort is the first step of the paper's monotone rearrangement:
each row is independently replaced by its increasing rearrangement.
The paper's claim ("rows are sorted and `1`-Lipschitz by (i), and adjacent
rows, which are within `1` entrywise beforehand, stay within `1` by (ii),
so columns remain `1`-Lipschitz") is exactly the `IsLipschitz1` preservation
proved here.

Definitions:
* `rowOf` — row `i` of `φ` as a function `Fin n → ℤ`.
* `rowSort` — the grid function with each row sorted.

Results:
* `rowSort_isLipschitz1` — if `φ` is `1`-Lipschitz on the grid, so is `rowSort φ`.

Multiset preservation, column-sort, and the final assembly come in
subsequent modules.  No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- Row `i` of `φ` viewed as a function `Fin n → ℤ`. -/
def rowOf (φ : Cell m n → ℤ) (i : Fin m) : Fin n → ℤ :=
  fun j => φ (i, j)

/-- The row-sort of `φ`: each row replaced by its increasing rearrangement.

Concretely, `rowSort φ (i, j)` is the `j`-th smallest value of row `i` of
`φ`.  Implementing this directly through `Tuple.sort` and the per-row
function `rowOf φ i`. -/
def rowSort (φ : Cell m n → ℤ) : Cell m n → ℤ :=
  fun v => (rowOf φ v.1 ∘ Tuple.sort (rowOf φ v.1)) v.2

/-- **Adjacency dichotomy on the grid.**  Two adjacent cells either share a
row (and differ by one in column) or share a column (and differ by one in
row).  This is the standard horizontal/vertical edge case-split. -/
private lemma adj_cases {p q : Cell m n} (h : adj p q) :
    (p.1 = q.1 ∧ ((p.2.val : ℤ) - q.2.val).natAbs = 1) ∨
      (((p.1.val : ℤ) - q.1.val).natAbs = 1 ∧ p.2 = q.2) := by
  unfold adj gdist at h
  -- h : (|p.1.val - q.1.val|.natAbs + |p.2.val - q.2.val|.natAbs : ℕ) = 1 (cast to ℤ)
  have h_sum : ((p.1.val : ℤ) - q.1.val).natAbs
                + ((p.2.val : ℤ) - q.2.val).natAbs = 1 := by exact_mod_cast h
  rcases (Nat.eq_zero_or_pos ((p.1.val : ℤ) - q.1.val).natAbs) with h1 | h1
  · -- |p.1 - q.1|.natAbs = 0 ⟹ p.1 = q.1 (as Fin m, via Fin.ext)
    have hp1q1 : p.1.val = q.1.val := by
      have : ((p.1.val : ℤ) - q.1.val) = 0 := by omega
      omega
    refine Or.inl ⟨Fin.ext hp1q1, ?_⟩
    omega
  · -- |p.1 - q.1|.natAbs ≥ 1; from the sum bound, = 1 and the other is 0.
    have hp1q1 : ((p.1.val : ℤ) - q.1.val).natAbs = 1 := by omega
    have hp2q2 : ((p.2.val : ℤ) - q.2.val).natAbs = 0 := by omega
    have hp2eq : p.2.val = q.2.val := by
      have : ((p.2.val : ℤ) - q.2.val) = 0 := by omega
      omega
    exact Or.inr ⟨hp1q1, Fin.ext hp2eq⟩

/-- **Rows of a `1`-Lipschitz grid function are path-`1`-Lipschitz.**  Adjacent
column indices `j, j+1` give cells `(i, j), (i, j+1)` adjacent in the grid,
whence `|φ (i, j) - φ (i, j+1)| ≤ 1`. -/
private lemma rowOf_pathLipschitz1 {φ : Cell m n → ℤ}
    (hφ : IsLipschitz1 φ) (i : Fin m) : PathLipschitz1 (rowOf φ i) := by
  intro j j' hjj'
  -- (i, j) and (i, j') are grid-adjacent (same row, columns differ by 1).
  have h_adj : adj ((i, j) : Cell m n) ((i, j') : Cell m n) := by
    unfold adj gdist
    have hi : ((i.val : ℤ) - i.val).natAbs = 0 := by omega
    have hj : ((j.val : ℤ) - j'.val).natAbs = 1 := by
      have h_succ : j.val + 1 = j'.val := hjj'
      omega
    show (((((i.val : ℤ) - i.val).natAbs + ((j.val : ℤ) - j'.val).natAbs) : ℕ) : ℤ) = 1
    rw [hi, hj]; rfl
  exact hφ _ _ h_adj

/-- **Adjacent rows of a `1`-Lipschitz grid function are pairwise within `1`.**
Cells `(i, k), (i', k)` with `|i.val - i'.val| = 1` are grid-adjacent, so the
two rows differ by at most one entrywise. -/
private lemma rowOf_pairwise_le {φ : Cell m n → ℤ} (hφ : IsLipschitz1 φ)
    {i i' : Fin m} (hii' : ((i.val : ℤ) - i'.val).natAbs = 1) (k : Fin n) :
    |rowOf φ i k - rowOf φ i' k| ≤ 1 := by
  -- (i, k) and (i', k) are grid-adjacent.
  have h_adj : adj ((i, k) : Cell m n) ((i', k) : Cell m n) := by
    unfold adj gdist
    have hk : ((k.val : ℤ) - k.val).natAbs = 0 := by omega
    show (((((i.val : ℤ) - i'.val).natAbs + ((k.val : ℤ) - k.val).natAbs) : ℕ) : ℤ) = 1
    rw [hii', hk]; rfl
  exact hφ _ _ h_adj

/-- **The row-sort preserves the `1`-Lipschitz property.**

Adjacency case-split via `adj_cases`:
* Horizontal edge (same row, adjacent columns): apply fact (i)
  (`sort_pathLipschitz1`) to the sorted row.
* Vertical edge (adjacent rows, same column): apply fact (ii)
  (`sort_pairwise_abs_le_one`) to the two sorted rows. -/
theorem rowSort_isLipschitz1 {φ : Cell m n → ℤ} (hφ : IsLipschitz1 φ) :
    IsLipschitz1 (rowSort φ) := by
  intro p q hpq
  rcases adj_cases hpq with ⟨hrow, hcol⟩ | ⟨hrow, hcol⟩
  · -- Horizontal edge: p.1 = q.1, |p.2.val - q.2.val|.natAbs = 1.
    -- WLOG p.2.val + 1 = q.2.val (or vice versa); apply fact (i) to row p.1.
    have h_path : PathLipschitz1 (rowOf φ p.1 ∘ Tuple.sort (rowOf φ p.1)) :=
      sort_pathLipschitz1 (rowOf_pathLipschitz1 hφ p.1)
    -- Need: |rowSort φ p - rowSort φ q| ≤ 1.
    -- rowSort φ p = (rowOf φ p.1 ∘ sort (rowOf φ p.1)) p.2.
    -- rowSort φ q = (rowOf φ q.1 ∘ sort (rowOf φ q.1)) q.2 = same sort (since p.1 = q.1).
    show |(rowOf φ p.1 ∘ Tuple.sort (rowOf φ p.1)) p.2
            - (rowOf φ q.1 ∘ Tuple.sort (rowOf φ q.1)) q.2| ≤ 1
    rw [show q.1 = p.1 from hrow.symm]
    -- Now both use the SAME sort permutation; apply PathLipschitz1.
    -- |q.2.val - p.2.val|.natAbs = 1 ⟹ p.2.val + 1 = q.2.val or q.2.val + 1 = p.2.val.
    have h_adj_idx : (p.2.val + 1 = q.2.val) ∨ (q.2.val + 1 = p.2.val) := by omega
    rcases h_adj_idx with h_left | h_right
    · exact h_path p.2 q.2 h_left
    · rw [abs_sub_comm]
      exact h_path q.2 p.2 h_right
  · -- Vertical edge: |p.1.val - q.1.val|.natAbs = 1, p.2 = q.2.
    -- Apply fact (ii) to rows p.1 and q.1.
    have h_pair : ∀ k, |rowOf φ p.1 k - rowOf φ q.1 k| ≤ 1 :=
      rowOf_pairwise_le hφ hrow
    have h_sort := sort_pairwise_abs_le_one h_pair p.2
    -- h_sort : |(rowOf φ p.1 ∘ sort (rowOf φ p.1)) p.2
    --          - (rowOf φ q.1 ∘ sort (rowOf φ q.1)) p.2| ≤ 1.
    -- Goal: |rowSort φ p - rowSort φ q| ≤ 1, which after unfolding rowSort
    -- and using p.2 = q.2 (hcol) is exactly h_sort.
    show |(rowOf φ p.1 ∘ Tuple.sort (rowOf φ p.1)) p.2
            - (rowOf φ q.1 ∘ Tuple.sort (rowOf φ q.1)) q.2| ≤ 1
    rw [show q.2 = p.2 from hcol.symm]
    exact h_sort

end OrigamiCone

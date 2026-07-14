import OrigamiCone.AcellCount

/-!
# Antidiagonal cardinality (Sub-5 of `thm:deg4count`)

The **antidiagonal** at level `s` in `Cell m n` is the set
`{v : Cell m n | acell v = s}` where `acell v = v.1.val + v.2.val`.
This module pins down the four cardinalities that the
opposite-corner CC|EE family analysis in `thm:deg4count` needs:
the corner antidiagonals (size 1) and the corner-adjacent
antidiagonals (size 2).

The full closed form
  `|{v | acell v = s}| = min (s + 1) (m + n - 1 - s)`  (clamped to `min m n`)
is not formalised here; only its boundary values are needed by the
opposite-corner analysis.  The downstream argument is:
* the BL–TR tent peaks at level `s* = (δ + m + n − 2) / 2`;
* a degree-4 vertex requires the peak antidiagonal to contain ≥ 2 cells;
* the only `s*` with `|antidiag s*| ≥ 2` adjacent to corners are
  `s* ∈ {1, m + n − 3}`, which both have exactly 2 cells.

Results:
* `card_acell_eq_zero` — corner `s = 0`: singleton `{(0, 0)}`.
* `card_acell_eq_top` — corner `s = m + n − 2`: singleton `{(m−1, n−1)}`.
* `card_acell_eq_one_level` — `s = 1` with `m, n ≥ 2`: pair `{(0,1), (1,0)}`.
* `card_acell_eq_top_minus_one_level` — `s = m + n − 3` with `m, n ≥ 2`:
  pair `{(m−2, n−1), (m−1, n−2)}`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-! ## Antidiagonal cardinality at the corner level `s = 0` -/

/-- **Corner antidiagonal at `s = 0` is the singleton `(0, 0)`.**
For any `m, n ≥ 1`, `|{v : Cell m n | acell v = 0}| = 1`. -/
theorem card_acell_eq_zero (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (Finset.univ.filter (fun v : Cell m n => acell v = 0)).card = 1 := by
  -- The unique element is `(⟨0, _⟩, ⟨0, _⟩)`.
  rw [Finset.card_eq_one]
  refine ⟨(⟨0, by omega⟩, ⟨0, by omega⟩), ?_⟩
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  unfold acell
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- acell v = 0 ⟹ v = (0, 0)
    have hv1 : v.1.val = 0 := by
      have h2 : 0 ≤ (v.2.val : ℤ) := by positivity
      have h1 : 0 ≤ (v.1.val : ℤ) := by positivity
      omega
    have hv2 : v.2.val = 0 := by
      have h2 : 0 ≤ (v.2.val : ℤ) := by positivity
      have h1 : 0 ≤ (v.1.val : ℤ) := by positivity
      omega
    exact Prod.ext (Fin.ext hv1) (Fin.ext hv2)
  · -- v = (0, 0) ⟹ acell v = 0
    subst h
    rfl

/-- **Top-corner antidiagonal at `s = m + n − 2` is the singleton `(m−1, n−1)`.**
For any `m, n ≥ 1`, `|{v : Cell m n | acell v = m + n − 2}| = 1`. -/
theorem card_acell_eq_top (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (Finset.univ.filter
        (fun v : Cell m n => acell v = ((m + n - 2 : ℕ) : ℤ))).card = 1 := by
  rw [Finset.card_eq_one]
  refine ⟨(⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩), ?_⟩
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
  unfold acell
  refine ⟨fun h => ?_, fun h => ?_⟩
  · -- acell v = m + n − 2 ⟹ v = (m-1, n-1)
    have hv1_lt : v.1.val < m := v.1.isLt
    have hv2_lt : v.2.val < n := v.2.isLt
    have hv1 : v.1.val = m - 1 := by omega
    have hv2 : v.2.val = n - 1 := by omega
    exact Prod.ext (Fin.ext hv1) (Fin.ext hv2)
  · -- v = (m-1, n-1) ⟹ acell v = m + n − 2
    subst h
    push_cast
    omega

/-! ## Two-element antidiagonals (the deg-4 case) -/

/-- **Antidiagonal at `s = 1` has exactly two cells** when both `m, n ≥ 2`.
The two cells are `(0, 1)` and `(1, 0)`. -/
theorem card_acell_eq_one_level (hm : 2 ≤ m) (hn : 2 ≤ n) :
    (Finset.univ.filter (fun v : Cell m n => acell v = 1)).card = 2 := by
  -- Show the filter equals the pair `{(0, 1), (1, 0)}`.
  have h_eq : Finset.univ.filter (fun v : Cell m n => acell v = 1)
      = {(⟨0, by omega⟩, ⟨1, by omega⟩), (⟨1, by omega⟩, ⟨0, by omega⟩)} := by
    ext v
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
      Finset.mem_singleton]
    unfold acell
    refine ⟨fun h => ?_, fun h => ?_⟩
    · -- acell v = 1 ⟹ v ∈ {(0,1), (1,0)}.
      have hv1_lt : v.1.val < m := v.1.isLt
      have hv2_lt : v.2.val < n := v.2.isLt
      -- (v.1.val : ℤ) + v.2.val = 1, with v.1.val, v.2.val ≥ 0.
      rcases (show v.1.val = 0 ∨ v.1.val = 1 by omega) with h1 | h1
      · -- v.1.val = 0 ⟹ v.2.val = 1, so v = (⟨0, _⟩, ⟨1, _⟩) = first elt.
        left
        have hv2 : v.2.val = 1 := by omega
        exact Prod.ext (Fin.ext h1) (Fin.ext hv2)
      · -- v.1.val = 1 ⟹ v.2.val = 0, so v = (⟨1, _⟩, ⟨0, _⟩) = second elt.
        right
        have hv2 : v.2.val = 0 := by omega
        exact Prod.ext (Fin.ext h1) (Fin.ext hv2)
    · rcases h with h | h <;> subst h <;> simp
  rw [h_eq]
  -- Two distinct cells.
  apply Finset.card_pair
  intro h_mem
  simp only [Prod.mk.injEq, Fin.mk.injEq] at h_mem
  omega

/-- **Antidiagonal at `s = m + n − 3` has exactly two cells** when both
`m, n ≥ 2`.  The two cells are `(m−2, n−1)` and `(m−1, n−2)`. -/
theorem card_acell_eq_top_minus_one_level (hm : 2 ≤ m) (hn : 2 ≤ n) :
    (Finset.univ.filter
        (fun v : Cell m n => acell v = ((m + n - 3 : ℕ) : ℤ))).card = 2 := by
  have h_eq : Finset.univ.filter (fun v : Cell m n => acell v = ((m + n - 3 : ℕ) : ℤ))
      = {(⟨m - 2, by omega⟩, ⟨n - 1, by omega⟩),
         (⟨m - 1, by omega⟩, ⟨n - 2, by omega⟩)} := by
    ext v
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_insert,
      Finset.mem_singleton]
    unfold acell
    refine ⟨fun h => ?_, fun h => ?_⟩
    · have hv1_lt : v.1.val < m := v.1.isLt
      have hv2_lt : v.2.val < n := v.2.isLt
      have hcast : ((m + n - 3 : ℕ) : ℤ) = (m : ℤ) + n - 3 := by omega
      rcases (show v.1.val = m - 2 ∨ v.1.val = m - 1 by omega) with h1 | h1
      · -- v.1.val = m-2 ⟹ v.2.val = n-1, so v = first elt.
        left
        have hv2 : v.2.val = n - 1 := by omega
        exact Prod.ext (Fin.ext h1) (Fin.ext hv2)
      · -- v.1.val = m-1 ⟹ v.2.val = n-2, so v = second elt.
        right
        have hv2 : v.2.val = n - 2 := by omega
        exact Prod.ext (Fin.ext h1) (Fin.ext hv2)
    · rcases h with h | h <;> subst h <;> push_cast <;> omega
  rw [h_eq]
  apply Finset.card_pair
  intro h_mem
  simp only [Prod.mk.injEq, Fin.mk.injEq] at h_mem
  omega

end OrigamiCone

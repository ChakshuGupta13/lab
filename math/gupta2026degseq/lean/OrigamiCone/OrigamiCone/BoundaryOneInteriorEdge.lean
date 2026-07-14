import OrigamiCone.BoundaryOneInterior
import OrigamiCone.Boundary
import OrigamiCone.RidgeMax

/-!
# Case 2b: `p_B` on the top edge — two bottom-corner strict local maxima

Paper: `lem:boundary` case 2b (paper §, in prose after the corner case).
With `p_B = (0, cB)` an EDGE cell on the top side (`1 ≤ cB ≤ n - 2` in
0-indexed convention, matching the paper's 1-indexed `2 ≤ c ≤ n - 1`) and
`p_I` interior, the two bottom corners `(m - 1, 0)` and `(m - 1, n - 1)`
are each a strict local maximum of the cone-pair envelope `cpe p_B p_I δ`.

## Argument

At each bottom corner, BOTH cones strictly peak on the corner's two grid
neighbours by exactly one:

* `gdist p_B` peaks at `(m - 1, 0)`: `p_B = (0, cB)` differs from the
  corner in row (`0 ≠ m - 1`, using `m ≥ 2`) AND column (`cB ≠ 0`, using
  `cB ≥ 1`). The paper's `c ≥ 2` (1-indexed) is this `cB ≥ 1` condition.
* `gdist p_B` peaks at `(m - 1, n - 1)`: similarly with `cB ≠ n - 1`
  (using `cB ≤ n - 2`, the paper's `c ≤ n - 1`).
* `gdist p_I` peaks at either corner: `p_I` interior differs from any
  corner in both coordinates (`interior_corner_cone_strictMax`).

Both cones strictly peak ⟹ `min` (= `cpe`) strictly peaks. No parity
condition needed: the drop across each edge is exactly `-1` for each
cone, so it is exactly `-1` for `min`.

## What this module does NOT cover

The paper's case 2b requires a THIRD max — established by the dual-
envelope contradiction ("if only these two, then `h` rises down every
column, contradicting the interior minimum `p_I`"). That argument needs
the dual envelope construction and the anti-cone framework, and lives in
a separate follow-up module. This file gives the FIRST TWO maxima only.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Case 2b, FIRST maximum.** With `p_B = (0, cB)` on the top edge
(`1 ≤ cB ≤ n - 2`), `p_I` interior, the bottom-left corner `(m - 1, 0)`
is a strict local maximum of `cpe p_B p_I δ`. -/
theorem oneInterior_TopEdge_bottomLeft_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
  have hcBL : IsCorner
      ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) :=
    ⟨Or.inr rfl, Or.inl rfl⟩
  apply cpe_strictMax_of_both
  · -- `d(p_B, ·)` strict-max at (m-1, 0): p_B = (0, cB) differs in row (0 ≠ m-1
    -- from m ≥ 2) and column (cB ≠ 0 from cB ≥ 1).
    apply (corner_strictMax_iff hm hn hcBL).mpr
    refine ⟨?_, ?_⟩
    · dsimp only; omega
    · dsimp only; omega
  · -- `d(p_I, ·)` strict-max at any corner: interior apex.
    exact interior_corner_cone_strictMax hm hn h_I hcBL

/-- **Case 2b, SECOND maximum.** With `p_B = (0, cB)` on the top edge
(`1 ≤ cB ≤ n - 2`), `p_I` interior, the bottom-right corner
`(m - 1, n - 1)` is a strict local maximum of `cpe p_B p_I δ`. The
`cB ≥ 1` hypothesis is kept for API uniformity with the paper's case 2b
framing but is not used by this proof (only `cB + 1 < n` — i.e., paper's
`c ≤ n - 1` — is needed to place `p_B` off the BR column). -/
theorem oneInterior_TopEdge_bottomRight_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {cB : ℕ}
    (_hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  have hcBR : IsCorner
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) :=
    ⟨Or.inr rfl, Or.inr rfl⟩
  apply cpe_strictMax_of_both
  · -- `d(p_B, ·)` strict-max at (m-1, n-1): p_B = (0, cB) differs in row (0 ≠ m-1)
    -- and column (cB ≠ n-1 from cB + 1 < n, i.e., cB ≤ n - 2).
    apply (corner_strictMax_iff hm hn hcBR).mpr
    refine ⟨?_, ?_⟩
    · dsimp only; omega
    · dsimp only; omega
  · exact interior_corner_cone_strictMax hm hn h_I hcBR

/-- **Case 2b: both bottom corners are pairwise-distinct strict local
maxima.** Packaging theorem: with `p_B = (0, cB)` on the top edge
(`1 ≤ cB ≤ n - 2`) and `p_I` interior, `(m - 1, 0)` and `(m - 1, n - 1)`
are both `IsStrictLocalMax` of `cpe p_B p_I δ`, and they are distinct
(`n ≥ 3` from `cB + 1 < n ∧ 1 ≤ cB`). This is the paper's first-two-
maxima observation in case 2b. The third max (via dual envelope
contradiction) is separate work. -/
theorem oneInterior_TopEdge_two_bottom_max
    (hm : 2 ≤ m) {cB : ℕ}
    (hcB_pos : 1 ≤ cB) (hcB_lt : cB + 1 < n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ∧
    IsStrictLocalMax
      (cpe ((⟨0, by omega⟩ : Fin m), (⟨cB, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ∧
    ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ≠
      (((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) :
        Cell m n) := by
  have hn : 2 ≤ n := by omega
  refine ⟨oneInterior_TopEdge_bottomLeft_max hm hn hcB_pos hcB_lt h_I δ,
          oneInterior_TopEdge_bottomRight_max hm hn hcB_pos hcB_lt h_I δ,
          ?_⟩
  intro heq
  have := congrArg (fun c : Cell m n => c.2.val) heq
  dsimp at this
  omega

end OrigamiCone

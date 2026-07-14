import OrigamiCone.BoundaryOneInterior
import OrigamiCone.BoundaryOneInteriorEdge
import OrigamiCone.BoundaryOneInteriorBotEdge
import OrigamiCone.Boundary
import OrigamiCone.RidgeMax

/-!
# Case 2b for left and right edges — direct proofs

Left edge: `p_B = (rB, 0)` with `1 ≤ rB ≤ m - 2`. The two right corners
`(0, n - 1)` and `(m - 1, n - 1)` are strict local maxima.

Right edge: `p_B = (rB, n - 1)` with `1 ≤ rB ≤ m - 2`. The two left corners
`(0, 0)` and `(m - 1, 0)` are strict local maxima.

The proofs follow the same pattern as `BoundaryOneInteriorEdge`:
`corner_strictMax_iff` on the `p_B` cone (the corner differs from `p_B`
in each coord) + `interior_corner_cone_strictMax` on the `p_I` cone,
then `cpe_strictMax_of_both`. A transpose substrate `Cell m n → Cell n m`
is NOT needed — the proof pattern is symmetric under swapping which of
`p_B`'s coordinates is fixed.

## Abstract lemma

* `oneInterior_farCorner_max`: for any `p_B`, any corner `v` differing
  from `p_B` in each coordinate, and any interior `p_I`, `v` is a strict
  local max of `cpe p_B p_I δ`. Direct one-line composition of
  `corner_strictMax_iff` + `interior_corner_cone_strictMax` +
  `cpe_strictMax_of_both`.

The 8 edge-corner-max theorems (top / bottom / left / right × two far
corners each) all follow as one-line specialisations of this lemma
(only 4 written out here — top / bottom already landed in
BoundaryOneInteriorEdge / BoundaryOneInteriorBotEdge).

## Results

* `oneInterior_LeftEdge_topRight_max`, `_bottomRight_max`.
* `oneInterior_RightEdge_topLeft_max`, `_bottomLeft_max`.
* `oneInterior_LeftEdge_two_right_max`, `oneInterior_RightEdge_two_left_max`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Abstract case 2b lemma.** For any `p_B` (edge or corner), any grid
corner `v` differing from `p_B` in EACH coordinate, and any interior
`p_I`, `v` is a strict local maximum of `cpe p_B p_I δ`. The proof
matches the `oneInterior_TLcorner_opposite_max` idiom exactly. -/
theorem oneInterior_farCorner_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {p_B v : Cell m n}
    (hv_corner : IsCorner v)
    (hne_row : p_B.1.val ≠ v.1.val) (hne_col : p_B.2.val ≠ v.2.val)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax (cpe p_B p_I δ) v := by
  apply cpe_strictMax_of_both
  · exact (corner_strictMax_iff hm hn hv_corner).mpr ⟨hne_row, hne_col⟩
  · exact interior_corner_cone_strictMax hm hn h_I hv_corner

/-- **Left-edge FIRST maximum:** top-right corner. -/
theorem oneInterior_LeftEdge_topRight_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  have := hrB_pos  -- omega uses this to close rB ≠ 0 (linter misses omega deps)
  apply oneInterior_farCorner_max hm hn (v := ((⟨0, by omega⟩ : Fin m),
                                                (⟨n - 1, by omega⟩ : Fin n)))
  · exact ⟨Or.inl rfl, Or.inr rfl⟩
  · dsimp only; omega
  · dsimp only; omega
  · exact h_I

/-- **Left-edge SECOND maximum:** bottom-right corner. -/
theorem oneInterior_LeftEdge_bottomRight_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  apply oneInterior_farCorner_max hm hn (v := ((⟨m - 1, by omega⟩ : Fin m),
                                                (⟨n - 1, by omega⟩ : Fin n)))
  · exact ⟨Or.inr rfl, Or.inr rfl⟩
  · dsimp only; omega
  · dsimp only; omega
  · exact h_I

/-- **Right-edge FIRST maximum:** top-left corner. -/
theorem oneInterior_RightEdge_topLeft_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
  have := hrB_pos  -- omega uses this to close rB ≠ 0 (linter misses omega deps)
  apply oneInterior_farCorner_max hm hn (v := ((⟨0, by omega⟩ : Fin m),
                                                (⟨0, by omega⟩ : Fin n)))
  · exact ⟨Or.inl rfl, Or.inl rfl⟩
  · dsimp only; omega
  · dsimp only; omega
  · exact h_I

/-- **Right-edge SECOND maximum:** bottom-left corner. -/
theorem oneInterior_RightEdge_bottomLeft_max
    (hm : 2 ≤ m) (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) := by
  have := hrB_pos  -- kept for API uniformity across the four side-edge theorems
  apply oneInterior_farCorner_max hm hn (v := ((⟨m - 1, by omega⟩ : Fin m),
                                                (⟨0, by omega⟩ : Fin n)))
  · exact ⟨Or.inr rfl, Or.inl rfl⟩
  · dsimp only; omega
  · dsimp only; omega
  · exact h_I

/-- **Left-edge two right corners packaged with distinctness.** -/
theorem oneInterior_LeftEdge_two_right_max
    (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ∧
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ∧
    ((⟨0, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) ≠
      (((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) :
        Cell m n) := by
  have hm : 2 ≤ m := by omega
  refine ⟨oneInterior_LeftEdge_topRight_max hm hn hrB_pos hrB_lt h_I δ,
          oneInterior_LeftEdge_bottomRight_max hm hn hrB_pos hrB_lt h_I δ,
          ?_⟩
  intro heq
  have := congrArg (fun c : Cell m n => c.1.val) heq
  dsimp at this
  omega

/-- **Right-edge two left corners packaged with distinctness.** -/
theorem oneInterior_RightEdge_two_left_max
    (hn : 2 ≤ n) {rB : ℕ}
    (hrB_pos : 1 ≤ rB) (hrB_lt : rB + 1 < m)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I δ)
      ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ∧
    IsStrictLocalMax
      (cpe ((⟨rB, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ∧
    ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) ≠
      (((⟨m - 1, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) :
        Cell m n) := by
  have hm : 2 ≤ m := by omega
  refine ⟨oneInterior_RightEdge_topLeft_max hm hn hrB_pos hrB_lt h_I δ,
          oneInterior_RightEdge_bottomLeft_max hm hn hrB_pos hrB_lt h_I δ,
          ?_⟩
  intro heq
  have := congrArg (fun c : Cell m n => c.1.val) heq
  dsimp at this
  omega

end OrigamiCone

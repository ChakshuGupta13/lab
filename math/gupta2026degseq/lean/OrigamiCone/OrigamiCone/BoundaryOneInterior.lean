import OrigamiCone.Boundary

/-!
# The Boundary Lemma (Section 3, `lem:boundary`), one-interior-one-boundary case

Formalisation of the first ingredients of the **one-interior-one-boundary**
case of the Boundary Lemma (`lem:boundary`): with one apex at a corner and
the other interior, the paper argues (§3, `lem:boundary` case 2) that the
envelope has at least THREE strict local maxima, contradicting the two-maxima
constraint of a degree-4 vertex. This module supplies two of the three
building blocks for the WLOG `p_B = (0, 0)` (top-left) sub-case:

1. **OPPOSITE-corner max**: the corner `(m-1, n-1)` is a strict local max of
   `cpe p_B p_I δ` — the "both cones peak" argument, mirroring the both-
   interior case's `bothInterior_corner_strictMax`.
2. **Step-inward-lowers-cpe** (last column + last row): stepping from the
   last column `n - 1` inward to `n - 2` strictly lowers `cpe` (at any row
   `i`), and symmetrically for the last row. This is the primitive the
   paper uses to lift a strict local max ALONG the last column to a strict
   local max IN THE GRID: since the inward neighbour is strictly smaller,
   a column-strict-max cell has its third grid-neighbour (the inward one)
   also smaller.

Remaining for the full case 2a: closing the last-column analysis by
combining the step-inward-lowers lemma with `PmOneWalk.pm1_walk_
strictMax_before_strictMin` (the 1D `±1`-walk local-max primitive) applied
to `cpe` restricted to the last column, plus the identification of
`(p_I.1, n - 1)` as a strict local minimum in that column. The paper's
symmetric last-row argument gives the third max. Case 2b (edge apex) uses
the dual Envelope Lemma; separate work.

## Mechanism (summary)

* **Opposite corner**: `p_B = (0, 0)` differs from `(m-1, n-1)` in both
  coordinates (`m, n ≥ 2`), so `corner_strictMax_iff` places `gdist p_B` at
  a strict local max; `interior_corner_cone_strictMax` does the same for
  `gdist p_I`. The min-plus-shift `cpe` inherits both maxima via
  `cpe_strictMax_of_both`.
* **Step-inward-lowers**: both `gdist p_B` and `gdist p_I` decrease by 1
  stepping from `(i, n-1)` to `(i, n-2)` — the former because `p_B` is at
  column `0` and the step moves toward it, the latter because `p_I` is at
  column `s ≤ n-2 < n-1` (from `IsInterior`, which forces `n ≥ 3`) and the
  step also moves toward it. So `min` of the two shifted cones decreases by
  exactly 1. `omega` closes the `natAbs`-arithmetic after `unfold cpe gdist`.

Scope: this module covers ONLY the WLOG `p_B = TL` sub-case's opposite-corner
maximum and the step-inward primitives. Other placements (`p_B` at TR / BL /
BR) follow by grid reflections `i ↦ m-1-i`, `j ↦ n-1-j` (formal transport
across those reflections is separate work). The 1D `±1`-walk primitive that
locates the column-strict-max above `p_I`'s row lives in
`OrigamiCone.PmOneWalk` and is not applied here.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **`lem:boundary` case (2a), FIRST maximum.** With `p_B` at the top-left
corner `(0, 0)` and `p_I` interior, the opposite corner `(m-1, n-1)` is a
strict local maximum of the cone-pair envelope `cpe p_B p_I δ`. This is one of
the three strict local maxima the paper argues for; the remaining two — a
strict local max above `p_I`'s row on the last column, and a strict local max
left of `p_I`'s column on the last row — are established by 1D ±1-walk
arguments not formalised here. -/
theorem oneInterior_TLcorner_opposite_max (hm : 2 ≤ m) (hn : 2 ≤ n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) :
    IsStrictLocalMax
      (cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ)
      ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) := by
  -- The opposite corner is a strict local maximum of both cones separately;
  -- the min of two strict maxima (up to a constant shift) is a strict maximum.
  have hcBR : IsCorner ((⟨m - 1, by omega⟩ : Fin m), (⟨n - 1, by omega⟩ : Fin n)) :=
    ⟨Or.inr rfl, Or.inr rfl⟩
  apply cpe_strictMax_of_both
  · -- `d(p_B, ·)` strict-max at (m-1, n-1): p_B = (0, 0) differs from opposite corner
    -- in both coordinates (since m, n ≥ 2).
    apply (corner_strictMax_iff hm hn hcBR).mpr
    refine ⟨?_, ?_⟩
    · dsimp only; omega
    · dsimp only; omega
  · -- `d(p_I, ·)` strict-max at (m-1, n-1): interior apex ⟹ every corner is strict-max.
    exact interior_corner_cone_strictMax hm hn h_I hcBR

/-- **Step-inward-lowers-cpe (last column).** With `p_B` at the top-left
corner `(0, 0)` and `p_I` interior (which forces `n ≥ 3`), for any row `i`
the envelope `cpe p_B p_I δ` is strictly smaller at `(i, n - 2)` than at
`(i, n - 1)`: stepping inward from the last column lowers `cpe` by exactly
one. Both cones drop by one on that step because `p_B` is at column `0` and
`p_I` is at column `s ≤ n - 2` (`< n - 1`), so the step moves toward the
apex-column-direction for both. -/
theorem oneInterior_TLcorner_lastCol_stepIn_lower (hm : 2 ≤ m) (hn : 3 ≤ n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) (i : Fin m) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          (i, (⟨n - 2, by omega⟩ : Fin n)) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          (i, (⟨n - 1, by omega⟩ : Fin n)) := by
  obtain ⟨_, _, hs_pos, hs_bd⟩ := h_I
  unfold cpe gdist
  dsimp only
  -- Both `gdist p_B` and `gdist p_I` decrease by exactly 1 stepping from column
  -- n - 1 to column n - 2. So `min (gdist p_B v) (δ + gdist p_I v)` decreases by 1.
  omega

/-- **Step-inward-lowers-cpe (last row).** With `p_B` at the top-left corner
`(0, 0)` and `p_I` interior (which forces `m ≥ 3`), for any column `j` the
envelope `cpe p_B p_I δ` is strictly smaller at `(m - 2, j)` than at
`(m - 1, j)`: stepping inward from the last row lowers `cpe` by exactly one.
Symmetric to `oneInterior_TLcorner_lastCol_stepIn_lower` with row and
column axes swapped. -/
theorem oneInterior_TLcorner_lastRow_stepIn_lower (hm : 3 ≤ m) (hn : 2 ≤ n)
    {p_I : Cell m n} (h_I : IsInterior p_I) (δ : ℤ) (j : Fin n) :
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          ((⟨m - 2, by omega⟩ : Fin m), j) <
    cpe ((⟨0, by omega⟩ : Fin m), (⟨0, by omega⟩ : Fin n)) p_I δ
          ((⟨m - 1, by omega⟩ : Fin m), j) := by
  obtain ⟨hr_pos, hr_bd, _, _⟩ := h_I
  unfold cpe gdist
  dsimp only
  omega

end OrigamiCone

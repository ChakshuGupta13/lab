import OrigamiCone.CCEEAdmissibleDelta
import OrigamiCone.Deg4FamilySum

/-!
# CC|EE family count assembly (Sub-6b of `thm:deg4count`)

This module assembles the per-pair admissible-δ count
(`ccee_admissible_delta_count` from Sub-6a) into the closed-form CC|EE family
count `familyCCEE = 4` declared in `Deg4FamilySum`.

The decomposition is

  `family CC|EE  =  (2 opposite-corner pairs) × (admissible-δ per pair)
                 =  2 × 2  =  4`.

The two opposite-corner pairs are `(BL, TR) = ((0, 0), (m-1, n-1))` and
`(BR, TL) = ((0, n-1), (m-1, 0))`.  Both pairs have Manhattan distance
`D = m + n - 2 = L`, so both yield identical per-pair admissible-δ sets
`{2 - L, L - 2}`.  The per-pair count function
`ccee_admissible_delta_count` (Sub-6a) is therefore **pair-agnostic**:
its statement depends only on `L`, not on which corners are picked.
The factor `2` in `family_CCEE_decomposition` is the cardinality of the
opposite-corner pair set `{(BL, TR), (BR, TL)}`, treated here as the
integer literal `2`; the structural fact that the BR-TL per-pair count
equals the BL-TR per-pair count is captured by reusing the same
Sub-6a lemma (no separate symmetry lemma needed, since the lemma
statement is invariant under the symmetry).

The geometric witnesses for each pair are:
* the BL-TR pair via `cpe_BL_TR_eq_acell_tent` (Sub-3) +
  `card_acell_eq_one_level` / `card_acell_eq_top_minus_one_level` (Sub-5);
* the BR-TL pair via `cpe_BR_TL_eq_diag_tent` (Sub-3, dual tent shape) +
  the same Sub-5 cardinalities under the column-reflection bijection
  `(i, j) ↔ (i, n - 1 - j)` (an OFG automorphism).

Adjacent-corner CC|EE candidates are killed by Sub-4: shared-coord apexes
have strict maxima confined to a single boundary row/column, giving a cone
(not a degree-4 vertex).  Sub-4 is invoked downstream by the family-count
assembly in `thm:deg4count` (Sub-7), not directly here.

This module performs only the **integer arithmetic** of the assembly
(`2 × 2 = 4`); the underlying *geometric* fact that the cone-pair admissible
offsets bijection-count degree-4 OFG vertices in this family is the
content of the Cone-pair Bijection (`prop:conepair`) and the Ridge Lemma
(`lem:ridge`), already formalised in `ConePair.lean` / `RidgeMax.lean`.

Result:
* `family_CCEE_decomposition` — `2 × (per-pair admissible-δ count) = familyCCEE`
  (cast to `ℤ`).

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **CC|EE family count assembly.**  Two opposite-corner pairs, each with
the per-pair admissible-δ count of 2 (Sub-6a), give the closed-form family
count `familyCCEE = 4`. -/
theorem family_CCEE_decomposition (hm : 3 ≤ m) (hn : 3 ≤ n) :
    2 * (((Finset.Icc (-((m + n - 2 : ℕ) : ℤ)) ((m + n - 2 : ℕ) : ℤ)).filter
        (fun δ : ℤ => δ = 2 - ((m + n - 2 : ℕ) : ℤ)
                  ∨ δ = ((m + n - 2 : ℕ) : ℤ) - 2)).card : ℤ) = familyCCEE := by
  rw [ccee_admissible_delta_count hm hn]
  norm_num [familyCCEE]

end OrigamiCone

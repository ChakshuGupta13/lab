import OrigamiCone.CCEEFamilyCount
import OrigamiCone.CEEEFamilyCount
import OrigamiCone.CECEFamilyCount
import OrigamiCone.EEEEFamilyCount

/-!
# Degree-4 count final assembly (`thm:deg4count`)

The paper's **Theorem 3.6 / `thm:deg4count`** ("Degree-4 count") states that
for `min(m, n) ≥ 3`, the number of degree-4 OFG vertices of `OFG(M_{m,n})` is
`2 m² + 2 n² + 6 m n − 10 (m + n) − 4`.

This module holds two complementary arithmetic-layer assemblies:

* `deg4_count_closed_form_sum` — the raw polynomial identity
  `2·familyCCEE + 2·familyCEEE + familyCECE + familyEEEE = deg4Headline`
  applied at natural `m, n` (a direct `ℕ → ℤ` re-cast of `deg4_family_sum`).
* `deg4_count_from_decompositions` — the SAME sum, but with each `family*` on
  the LHS explicitly replaced by the corresponding
  `family_*_decomposition` LHS (per-pair, per-bucket, per-corner-total, and
  per-orientation-count).  This form makes visible in the proof term that
  the four decomposition lemmas feed into the final assembly, closing the
  wiring gap between the family-level arithmetic and the headline.

The remaining geometric content — that the per-family LHSes are correct
cardinalities of the corresponding degree-4 OFG-vertex families — depends on
`lem:boundary` (formalised in `BoundaryFinal.lean`, `k_maxima_not_deg4`) and
`lem:ridge` (formalised in `RidgeMax.lean`); those substrate lemmas are
established, so the algebraic assemblies here are the last steps needed to
close `thm:deg4count`'s arithmetic backbone.

Results:
* `deg4_count_closed_form_sum` — closed-form sum → headline (via
  `deg4_family_sum`).
* `deg4_count_from_decompositions` — the four `family_*_decomposition`
  LHSes → headline, explicitly composing the arithmetic backbone.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Closed-form arithmetic sum for `thm:deg4count`.**  A direct ℕ → ℤ
re-cast of `deg4_family_sum`: the six family closed forms sum to the
headline polynomial `deg4Headline`. -/
theorem deg4_count_closed_form_sum (_hm : 3 ≤ m) (_hn : 3 ≤ n) :
    (2 * familyCCEE + 2 * familyCEEE (m : ℤ) n
        + familyCECE (m : ℤ) n + familyEEEE (m : ℤ) n)
      = deg4Headline (m : ℤ) n :=
  deg4_family_sum (m : ℤ) (n : ℤ)

/-- **Wired final assembly for `thm:deg4count`.**  Sums the four
`family_*_decomposition` LHSes (a per-pair filter cardinality for CC|EE, a
per-bucket `Icc`-sum for CE|EE, and the per-corner / per-orientation
polynomial forms for CE|CE and EE|EE) directly to the headline
`deg4Headline m n`.

The proof rewrites each geometric LHS via the corresponding
`family_*_decomposition` (turning it into the closed-form `family*`) and
then invokes `deg4_family_sum`.  This makes the four decomposition lemmas
demonstrably on the path from the geometric-count LHSes to the headline. -/
theorem deg4_count_from_decompositions (hm : 3 ≤ m) (hn : 3 ≤ n) :
    2 * (2 * (((Finset.Icc
                  (-((m + n - 2 : ℕ) : ℤ)) ((m + n - 2 : ℕ) : ℤ)).filter
                (fun δ : ℤ => δ = 2 - ((m + n - 2 : ℕ) : ℤ)
                            ∨ δ = ((m + n - 2 : ℕ) : ℤ) - 2)).card : ℤ))
      + 2 * (8 * (((Finset.Icc 2 (m - 2)).card
                     + (Finset.Icc 3 (n - 1)).card : ℕ) : ℤ))
      + ((2 * ((n - 2) * (n - 3) : ℤ)) + (2 * ((m - 2) * (m - 3) : ℤ)) + 16)
      + ((2 : ℤ) * (((m - 2) * (n - 2) : ℤ) + 2 * ((m - 3) * (n - 3) : ℤ)))
      = deg4Headline (m : ℤ) n := by
  rw [family_CCEE_decomposition hm hn, family_CEEE_decomposition hm hn,
      family_CECE_decomposition hm hn, family_EEEE_decomposition hm hn]
  exact deg4_family_sum (m : ℤ) (n : ℤ)

end OrigamiCone


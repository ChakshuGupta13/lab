import Mathlib.Tactic.Ring

/-!
# Degree-4 family-count algebraic glue (`thm:deg4count`)

The paper's **Theorem 3.6 / `thm:deg4count`** ("Degree-4 count") computes, for
`min(m, n) ≥ 3`, the number of degree-4 OFG vertices as

  `2 m² + 2 n² + 6 m n − 10 (m + n) − 4`.

The proof partitions degree-4 vertices by the kinds (corner `C`, edge `E`) of
their two cone-pair minima and the two resulting maxima, into six families:

  `CC|EE` and its dual `EE|CC` (each = `4`),
  `CE|EE` and its dual `EE|CE` (each = `8(m + n − 6)`),
  `CE|CE` (self-dual, `= 2(m − 2)(m − 3) + 2(n − 2)(n − 3) + 16`),
  `EE|EE` (self-dual, `= 2(m − 2)(n − 2) + 4(m − 3)(n − 3)`).

This module is the **algebraic glue** — verifying that the six closed-form
family counts sum to the headline polynomial.  The closed-form counts
themselves (each a geometric lattice count over cone-pairs and their ridge
maxima, building on Lemma 3.5 / `lem:ridge` already formalised in
`RidgeMax.lean`) are deferred.

This isolates the polynomial bookkeeping of `thm:deg4count` from the four
geometric per-family arguments: once each `family*` is shown to be the
cardinality of its kind-pair bucket, `deg4_family_sum` combines them into the
headline.

Results:
* `familyCCEE`, `familyCEEE m n`, `familyCECE m n`, `familyEEEE m n` —
  the four closed-form family counts (in `ℤ`, the paper's signed-arithmetic
  form; downstream count theorems cast `ℕ`-cardinalities into `ℤ`).
* `deg4Headline m n` — the paper's headline polynomial
  `2 m² + 2 n² + 6 m n − 10 (m + n) − 4`.
* `deg4_family_sum` — the algebraic identity
  `2 · CC|EE + 2 · CE|EE + CE|CE + EE|EE = deg4Headline`.

No `sorry`.
-/

namespace OrigamiCone

/-! ## Family counts -/

/-- **Corner-corner family `CC | EE`**: closed-form count is `4`.

Two opposite-corner cone-pair apexes with admissible `δ ∈ {+1, −1}` produce
exactly two edge maxima each; the two opposite-corner pairs give `4` degree-4
vertices total.  See `main.tex` lines 553–566. -/
def familyCCEE : ℤ := 4

/-- **Corner-edge / edge-edge family `CE | EE`**: closed-form count is
`8 (m + n − 6)`.

A corner minimum with an edge minimum on a non-incident side yields an L-shaped
ridge; the `ℓ = 2` configurations give `(m − 3) + (n − 3) = m + n − 6` degree-4
vertices per corner-and-side pair.  Four corners times two non-incident sides
gives `8 (m + n − 6)`.  See `main.tex` lines 568–582. -/
def familyCEEE (m n : ℤ) : ℤ := 8 * (m + n - 6)

/-- **Corner-edge / corner-edge family `CE | CE`** (self-dual): closed-form
count is `2 (m − 2)(m − 3) + 2 (n − 2)(n − 3) + 16`.

A corner minimum with an edge minimum on an *incident* side yields a vee-and-line
ridge with one interior peak; the `δ` parity classes give `(c − 2)` peak
positions, summing to `\binom{n − 2}{2}` per top side and `\binom{m − 2}{2}` per
left side; four corners and the non-incident `c = 2` configurations add `16`.
See `main.tex` lines 625–655. -/
def familyCECE (m n : ℤ) : ℤ :=
  2 * (m - 2) * (m - 3) + 2 * (n - 2) * (n - 3) + 16

/-- **Edge-edge / edge-edge family `EE | EE`** (self-dual): closed-form count is
`2 (m − 2)(n − 2) + 4 (m − 3)(n − 3)`.

Two opposite-side edge minima yield a tridiagonal ridge: the diagonal entries
`|c₁ − c₂| = 0` contribute `(n − 2)(m − 2)` and the two off-diagonals
`|c₁ − c₂| = 1` contribute `2 (n − 3)(m − 3)`, doubled for left–right
orientation.  See `main.tex` lines 658–696. -/
def familyEEEE (m n : ℤ) : ℤ :=
  2 * (m - 2) * (n - 2) + 4 * (m - 3) * (n - 3)

/-! ## Headline polynomial -/

/-- **Paper headline polynomial** for the degree-4 count (`thm:deg4count`):
`2 m² + 2 n² + 6 m n − 10 (m + n) − 4`. -/
def deg4Headline (m n : ℤ) : ℤ :=
  2 * m * m + 2 * n * n + 6 * m * n - 10 * (m + n) - 4

/-! ## The algebraic glue -/

/-- **Theorem 3.6 algebraic glue**: the six family counts sum to the headline.

The four self-dual / dual-paired family counts combine, with multiplicity `2`
on the dual-paired families (`CC|EE` and `CE|EE`), into the paper's headline
polynomial `2m² + 2n² + 6mn − 10(m+n) − 4`.

This isolates the polynomial bookkeeping from the four geometric per-family
counts.  Once each `family*` is shown to be the cardinality of its kind-pair
bucket, `deg4_family_sum` is the final assembly step. -/
theorem deg4_family_sum (m n : ℤ) :
    2 * familyCCEE + 2 * familyCEEE m n
        + familyCECE m n + familyEEEE m n
      = deg4Headline m n := by
  unfold familyCCEE familyCEEE familyCECE familyEEEE deg4Headline
  ring

end OrigamiCone

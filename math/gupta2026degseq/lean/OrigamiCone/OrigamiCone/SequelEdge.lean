import Mathlib

/-!
# Sequel meta-theorem: the edge reduction (`lem:edgered`)

Standalone formalisation of the edge-reduction lemma of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:edgered` isolates the top-degree behaviour of the count by collapsing a
**single-edge** configuration — all `a` apexes on one side of the grid — to a
one-dimensional envelope. If the apexes lie on the top row `i = 1` at columns
`c_1 < … < c_a` with offsets `o_s`, then

> `E_{A,c}(i,j) = (i-1) + τ(j)`,  where  `τ(j) = min_s (o_s + |j - c_s|)`

is the one-dimensional lower envelope on the path `P_n`. Consequently `E_{A,c}`
increases strictly down every column, its minima are the row-`1` apexes, its
maxima lie in the bottom row at the strict local maxima of `τ`, the configuration
count is independent of `m`, and the number of maxima of `E_{A,c}` equals the
number of local maxima of `τ`. This is what reduces the single-edge count to the
`±1`-walk count of `lem:binom` and supplies the `4/(d-2)!` leading coefficient of
`thm:leading`.

This module proves the algebraic core:

* `dgrid_toprow` : the cone distance from a top-row apex `(1,c)` factors as
  `|1-i| + |c-j| = (i-1) + |c-j|` for `i ≥ 1`;
* `edge_reduction` : the envelope itself factors, `E2D i j = (i-1) + τ j` for
  `i ≥ 1` (the lemma's headline identity, a constant pulled out of `Finset.inf'`);
* `edge_strict_down` : `E2D (i+1) j = E2D i j + 1` — `E` increases by exactly one
  per downward step;
* `edge_col_strictMono` : hence `E2D · j` is strictly increasing down each column,
  placing the minima in row `1` and the maxima in the bottom row.

Scope: the apexes are an abstract nonempty `Finset` of indices with a column map;
this module proves the envelope factorisation and its column monotonicity. The
identification of the maxima with the strict local maxima of `τ`, and the
resulting `m`-independence of the *extremum count*, need the extremum machinery of
the Maxima Criterion (`lem:maxima`) and are **not** formalised here.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.edge_reduction`.
-/

namespace OrigamiCone.Sequel

open Finset

variable {ι : Type*} (o col : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

/-- The one-dimensional lower envelope `τ(j) = min_s (o_s + |c_s - j|)` on the
path, with apex columns `col` and offsets `o`. -/
def tau (j : ℤ) : ℤ := S.inf' hS (fun s => o s + |col s - j|)

/-- The two-dimensional envelope `E_{A,c}(i,j) = min_s (o_s + d((1,c_s),(i,j)))`
of a single-edge (top-row) configuration: apexes `(1, col s)`, offsets `o s`. -/
def E2D (i j : ℤ) : ℤ := S.inf' hS (fun s => o s + (|(1 : ℤ) - i| + |col s - j|))

/-- **Top-row cone distance.** For `i ≥ 1` the `L¹` distance from a top-row apex
`(1,c)` to `(i,j)` is `(i-1) + |c-j|`. -/
theorem dgrid_toprow (c i j : ℤ) (hi : 1 ≤ i) :
    |(1 : ℤ) - i| + |c - j| = (i - 1) + |c - j| := by
  rw [abs_of_nonpos (by omega)]; ring

/-- **Edge reduction** (algebraic core of `lem:edgered`). For `i ≥ 1` the top-row
envelope factors as `E2D i j = (i-1) + τ j`: the constant row-offset `i-1` pulls
out of the minimum. -/
theorem edge_reduction (i j : ℤ) (hi : 1 ≤ i) :
    E2D o col S hS i j = (i - 1) + tau o col S hS j := by
  unfold E2D tau
  have hrw : (fun s => o s + (|(1 : ℤ) - i| + |col s - j|))
           = (fun s => (i - 1) + (o s + |col s - j|)) := by
    funext s; rw [abs_of_nonpos (by omega : (1 : ℤ) - i ≤ 0)]; ring
  rw [hrw]
  apply le_antisymm
  · obtain ⟨s, hsS, hs⟩ := S.exists_mem_eq_inf' hS (fun s => o s + |col s - j|)
    calc S.inf' hS (fun s => (i - 1) + (o s + |col s - j|))
          ≤ (i - 1) + (o s + |col s - j|) := Finset.inf'_le _ hsS
      _ = (i - 1) + S.inf' hS (fun s => o s + |col s - j|) := by rw [hs]
  · apply Finset.le_inf'
    intro s hsS
    have : S.inf' hS (fun s => o s + |col s - j|) ≤ o s + |col s - j| :=
      Finset.inf'_le _ hsS
    omega

/-- **Strict descent.** Each downward step increases the top-row envelope by
exactly one: `E2D (i+1) j = E2D i j + 1` for `i ≥ 1`. -/
theorem edge_strict_down (i j : ℤ) (hi : 1 ≤ i) :
    E2D o col S hS (i + 1) j = E2D o col S hS i j + 1 := by
  rw [edge_reduction o col S hS (i + 1) j (by omega),
      edge_reduction o col S hS i j hi]
  ring

/-- **Column monotonicity.** The top-row envelope is strictly increasing down each
column on `{i ≥ 1}`; hence its minima lie in row `1` and its maxima in the bottom
row. -/
theorem edge_col_strictMono (j a b : ℤ) (ha : 1 ≤ a) (hab : a < b) :
    E2D o col S hS a j < E2D o col S hS b j := by
  rw [edge_reduction o col S hS a j ha, edge_reduction o col S hS b j (by omega)]
  omega

end OrigamiCone.Sequel

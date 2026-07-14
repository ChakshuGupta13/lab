import Mathlib

/-!
# Sequel meta-theorem: parameter count of a degree-`d` height function (`cor:Odparams`)

Standalone formalisation of the parameter-count corollary of the sequel paper

> *Degree-`d` vertex counts of the `m × n` origami flip graph:
> structure and a polynomial conjecture.*

`Corollary cor:Odparams` states: a height function `h` on `G_{m,n}` with `d`
extrema is determined by at most `3⌊d/2⌋ − 1` integer parameters, an `O(d)`
count depending only on `d` — neither on `m` nor on `n`.

The mechanism is a book-keeping over the Envelope Structure Theorem
(`thm:envelope`, `SequelEnvThm.envelope_structure_reverse`). Write `a` for
the number of strict local minima and `b` for the number of strict local
maxima, so `a + b = d`. By `thm:envelope`, `h = E_{A,c}` with `A` the `a`
minima and `c_s = h(p_s)`; the configuration `(A, c)` uses `3a` integers
(`2a` apex coordinates plus `a` offsets). Adding a constant to every
offset shifts `E_{A,c}` by that constant, so the offsets carry one
redundant degree of freedom; the normalisation `h(1,1) = 0` removes it,
leaving `3a − 1` integers to determine `h`. The negation `h ↔ −h` preserves
the defining conditions and swaps minima with maxima, so `h` is equally
determined by its `b` maxima, hence by `3b − 1` integers. The smaller
count `3·min(a, b) − 1` applies, and `min(a, b) ≤ ⌊d/2⌋` gives the bound
`3⌊d/2⌋ − 1`.

Contents:

* `paramDim a := 3 * a − 1` : parameter dimension of a size-`a`
  configuration after the shift-invariance gauge is fixed.
* `paramDim_dual (a b : ℕ) : min (paramDim a) (paramDim b) = paramDim (min a b)`
  : book-keeping identity that combines the two encoding routes into the
  smaller one.
* `min_le_half_of_add (a b d : ℕ) (h : a + b = d) : min a b ≤ d / 2`
  : the pure-arithmetic core.
* `cor_Odparams (a b d : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b) (h : a + b = d) :`
  `paramDim (min a b) ≤ 3 * (d / 2) − 1` : the corollary, in the reduced
  arithmetic form used downstream by any counting argument that consumes it.

Scope: the corollary reduces to an arithmetic bound on `min(a, b)` under
`a + b = d`. The encoding of a configuration into a `ℤ`-tuple of size
`3a − 1` is described in the paper (and here in `paramDim`) but its
formalisation as an injection needs the full configuration type of
`SequelEnvThm` and is not the mathematical content of the corollary; the
arithmetic bound below IS.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.cor_Odparams`.
-/

namespace OrigamiCone.Sequel

/-- **Parameter dimension of a size-`a` configuration.** After the normalisation
`E_{A,c}(1,1) = 0` removes the additive-shift gauge on the offsets, a
configuration with `a` apexes has `3a − 1` integer parameters: `2a` apex
coordinates plus `a − 1` independent offset differences. -/
def paramDim (a : ℕ) : ℕ := 3 * a - 1

/-- **Duality book-keeping.** The parameter dimension `3·min(a,b) − 1` of the
two-route encoding of `cor:Odparams` equals `paramDim` at the smaller side.
This is the identity `min(3a − 1, 3b − 1) = 3·min(a,b) − 1`. -/
theorem paramDim_min (a b : ℕ) : min (paramDim a) (paramDim b) = paramDim (min a b) := by
  unfold paramDim
  rcases le_total a b with hab | hab
  · rw [min_eq_left hab, min_eq_left (by omega : 3 * a - 1 ≤ 3 * b - 1)]
  · rw [min_eq_right hab, min_eq_right (by omega : 3 * b - 1 ≤ 3 * a - 1)]

/-- **Arithmetic core of `cor:Odparams`.** For non-negative integers with
`a + b = d`, `min(a, b) ≤ ⌊d / 2⌋`. Trivially derived from the fact that at
least one of `a, b` is at most `d / 2`. -/
theorem min_le_half_of_add (a b d : ℕ) (h : a + b = d) : min a b ≤ d / 2 := by
  rcases le_total a b with hab | hab
  · rw [min_eq_left hab]; omega
  · rw [min_eq_right hab]; omega

/-- **`cor:Odparams` (arithmetic form).** A height function with `d = a + b`
extrema — `a` strict local minima and `b` strict local maxima, each at
least one — is determined by at most `3⌊d/2⌋ − 1` integer parameters. -/
theorem cor_Odparams (a b d : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b) (h : a + b = d) :
    paramDim (min a b) ≤ 3 * (d / 2) - 1 := by
  have hmin : min a b ≤ d / 2 := min_le_half_of_add a b d h
  have hmin_pos : 1 ≤ min a b := by
    rcases le_total a b with hab | hab
    · rw [min_eq_left hab]; exact ha
    · rw [min_eq_right hab]; exact hb
  unfold paramDim; omega

/-- **Dual formulation of `cor:Odparams`.** Equivalent to `cor_Odparams`,
foregrounding the two-route encoding: `min` of the min-side count `3a − 1`
and the max-side count `3b − 1` is at most `3⌊d/2⌋ − 1`. -/
theorem cor_Odparams_dual (a b d : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b) (h : a + b = d) :
    min (paramDim a) (paramDim b) ≤ 3 * (d / 2) - 1 := by
  rw [paramDim_min]; exact cor_Odparams a b d ha hb h

end OrigamiCone.Sequel

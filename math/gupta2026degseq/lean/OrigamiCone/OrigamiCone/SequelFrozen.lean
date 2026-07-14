import Mathlib

/-!
# Sequel meta-theorem: the Frozen Classification (`lem:frozen`), clean directions

Standalone formalisation of the directions of the Frozen Classification of the
sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:frozen` is the combinatorial heart of the period-`1` (polynomiality)
mechanism: it classifies which column transitions of a height function carry no
strict local extremum. Read column by column, a height function is a sequence of
proper `3`-colourings of the path `P_m` (`Fin m → ℤ/3`, adjacent entries
distinct). For three consecutive columns `u, v, w` with `u_i ≠ v_i` and
`w_i ≠ v_i`, the middle column `v` carries **no** strict local extremum iff the
triple is **frozen**: there is a slope `k ∈ {1,2}` with `u_i = v_i - k` and
`w_i = v_i + k` for all `i`. The "extremum-free ⟺ frozen" dichotomy is what makes
the extremum-free transfer matrix `T_0` collapse, on the colour-rotation quotient,
to the identity (`lem:quotient`), giving peripheral spectrum `{1}` and genuine
polynomiality.

This module formalises the two **clean** directions, working in `ℤ/3` with cells
of a column indexed by `ℕ` (the height is a parameter `m`):

* `frozen_imp_extremumFree` (the `⟸` direction, **complete**): a frozen triple has
  no strict local extremum, because the two horizontal neighbours `u_i = v_i - k`
  and `w_i = v_i + k` differ (`k ≠ -k` in `ℤ/3`), so the present neighbours never
  all share one colour;
* `extremum_imp_aligned` (the reformulation): a strict local extremum forces the
  horizontal neighbours to agree, i.e. the row is **aligned** (`u_i = w_i`) —
  extrema occur only at aligned rows;
* `rainbow_imp_const_slope` (the algebraic core of the `⟹` direction): if every
  row is **rainbow** (`w_i - v_i = -(u_i - v_i)`), then properness of `u` and `w`
  forces the offset `u_i - v_i` to be **constant** from each row to the next — the
  paper's "Constant slope" step, which turns rainbow-everywhere into a single
  frozen slope.

Scope: the full `⟹` direction also needs the *Cascade* — that an extremum-free
middle column has **no** aligned row at all (every row is rainbow). The cascade is
a maximal-aligned-block induction with grid-boundary base cases; its local steps
(the up/down propagation rules) are `ℤ/3`-decidable, but the global induction is a
substantial finite-induction argument over the column and is **not** formalised
here. With the cascade granted, `rainbow_imp_const_slope` closes the `⟹`
direction; this module supplies the `⟸` direction in full and the `⟹` direction's
algebraic engine.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.frozen_imp_extremumFree`.
-/

namespace OrigamiCone.Sequel

/-- A strict local extremum of the middle column `v` at row `i` (grid height `m`):
the present neighbours all share one colour. The horizontal neighbours `u i`, `w i`
are always present; the vertical neighbours `v (i-1)`, `v (i+1)` are present at the
interior. "Share one colour" is encoded as all present neighbours equal to `u i`. -/
def isExtremum (u v w : ℕ → ZMod 3) (m i : ℕ) : Prop :=
  u i = w i ∧ (0 < i → v (i - 1) = u i) ∧ (i + 1 < m → v (i + 1) = u i)

/-- The column triple `(u, v, w)` is **frozen**: there is a nonzero slope `k`
(necessarily in `{1,2}`) with `u = v - k` and `w = v + k` everywhere. -/
def IsFrozen (u v w : ℕ → ZMod 3) : Prop :=
  ∃ k : ZMod 3, k ≠ 0 ∧ (∀ i, u i = v i - k) ∧ (∀ i, w i = v i + k)

/-- **Frozen ⟹ extremum-free** (the `⟸` direction of `lem:frozen`, complete). A
frozen triple carries no strict local extremum: the horizontal neighbours
`u_i = v_i - k` and `w_i = v_i + k` differ (`k ≠ -k` in `ℤ/3` for `k ≠ 0`), so the
present neighbours never all share one colour. -/
theorem frozen_imp_extremumFree (u v w : ℕ → ZMod 3) (m : ℕ)
    (hf : IsFrozen u v w) : ∀ i, ¬ isExtremum u v w m i := by
  obtain ⟨k, hk, hu, hw⟩ := hf
  intro i hext
  have heq : u i = w i := hext.1
  rw [hu i, hw i] at heq
  have hne : ∀ x y : ZMod 3, y ≠ 0 → x - y ≠ x + y := by decide
  exact hne (v i) k hk heq

/-- **Extrema occur only at aligned rows** (the reformulation underlying the `⟹`
direction). A strict local extremum forces the horizontal neighbours to agree,
`u_i = w_i` — the row is *aligned*. -/
theorem extremum_imp_aligned (u v w : ℕ → ZMod 3) (m i : ℕ)
    (hext : isExtremum u v w m i) : u i = w i :=
  hext.1

/-- **Rainbow ⟹ constant slope** (algebraic core of the `⟹` direction of
`lem:frozen`, the paper's "Constant slope" step). If every row is rainbow
(`w_j - v_j = -(u_j - v_j)`), then properness of `u`, `v`, `w` forces the offset
`u_i - v_i` to be constant from row `i` to row `i+1`. Iterated, this collapses a
rainbow-everywhere column to a single frozen slope. -/
theorem rainbow_imp_const_slope (u v w : ℕ → ZMod 3) (i : ℕ)
    (hsv : v (i + 1) ≠ v i) (hu : u (i + 1) ≠ u i) (hw : w (i + 1) ≠ w i)
    (hrain : ∀ j, w j - v j = -(u j - v j)) :
    u (i + 1) - v (i + 1) = u i - v i := by
  set sv := v (i + 1) - v i with hsvdef
  set a := u i - v i with hadef
  set x := u (i + 1) - v (i + 1) with hxdef
  have hs : sv ≠ 0 := sub_ne_zero.mpr hsv
  have h1 : x ≠ a - sv := by
    intro h; apply hu
    have hux : u (i + 1) = x + v (i + 1) := by rw [hxdef]; ring
    rw [hux, h, hsvdef, hadef]; ring
  have h2 : x ≠ a + sv := by
    intro h; apply hw
    have hwi1 : w (i + 1) = -(u (i + 1) - v (i + 1)) + v (i + 1) := by
      rw [← hrain (i + 1)]; ring
    have hwi : w i = -(u i - v i) + v i := by rw [← hrain i]; ring
    rw [hwi1, hwi, ← hxdef, ← hadef, h, hsvdef]; ring
  have key : ∀ p q r : ZMod 3, r ≠ 0 → p ≠ q - r → p ≠ q + r → p = q := by decide
  exact key x a sv hs h1 h2

end OrigamiCone.Sequel

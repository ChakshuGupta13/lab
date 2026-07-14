import Mathlib

/-!
# Sequel meta-theorem: parity backbone of the envelope section

Standalone formalisation of the parity arithmetic underlying two lemmas of the
sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

A lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))` of `L¹` distance cones on
the grid is a height function precisely when the offsets obey the **parity
condition** `(PAR)`: `c_s - c_t ≡ d(p_s, p_t) (mod 2)` for all `s, t`. Two lemmas
of the paper rest on the same arithmetic fact — that `(PAR)` forces every cone to
share one parity at each cell:

* `Lemma lem:parityvalid` (`(PAR) ⟹ E_{A,c}` is a height function): "each cone has
  value `≡ (i+j) + (c_s + p_{s,1} + p_{s,2}) (mod 2)`; `(PAR)` makes the
  parenthesised quantity independent of `s`, so all cones share one parity at
  every cell."
* `Lemma lem:paritygap` (an inactive cone exceeds `E(v)` by at least `2`): "two
  integers of equal parity that are unequal differ by at least `2`, and an
  inactive cone exceeds `E(v)`."

This module proves that shared-parity backbone and its two arithmetic corollaries:

* `dgrid_parity` : the `L¹` cone distance has parity `d(p,v) ≡ (p₁+p₂)+(v₁+v₂)`;
* `cones_share_parity` : `(PAR)` ⟹ any two cone values agree mod `2` at every
  cell (the core of `lem:parityvalid`);
* `parity_gap` : two integers of equal parity that are unequal differ by `≥ 2`;
* `inactive_cone_gap` : combining the two, a strictly larger cone exceeds a
  smaller one by `≥ 2` (arithmetic core of `lem:paritygap`);
* `lipschitz_edge` : a `1`-Lipschitz integer map that is unequal across an edge
  differs by exactly `1` (the step turning a `1`-Lipschitz `E` that is never
  equal across an edge into a height function, the conclusion of
  `lem:parityvalid`).

Scope: the cones are taken abstractly as points of `ℤ × ℤ` with the `L¹` metric;
this module proves the parity arithmetic, not the envelope geometry. The
`min`-over-apexes assembly (that `E_{A,c}` is itself `1`-Lipschitz as a minimum of
`1`-Lipschitz cones, and that its minima are the apexes) is the geometric layer
of the Envelope Structure Theorem and is **not** formalised here.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.cones_share_parity`.
-/

namespace OrigamiCone.Sequel

open scoped Int

/-- The `L¹` (grid) distance between two cells, the value carried by a distance
cone seated at `p` and evaluated at `v`. -/
def dgrid (p v : ℤ × ℤ) : ℤ := |p.1 - v.1| + |p.2 - v.2|

/-- The value at `v` of the distance cone with apex `p` and offset `c`. -/
def coneVal (c : ℤ) (p v : ℤ × ℤ) : ℤ := c + dgrid p v

/-- **Cone parity.** A distance cone has parity `d(p,v) ≡ (p₁+p₂)+(v₁+v₂) (mod 2)`
at every cell, because `|x| ≡ x (mod 2)`. This is the per-cone parity statement of
`lem:parityvalid`. -/
theorem dgrid_parity (p v : ℤ × ℤ) :
    dgrid p v ≡ (p.1 + p.2) + (v.1 + v.2) [ZMOD 2] := by
  unfold dgrid Int.ModEq
  rcases abs_cases (p.1 - v.1) with ⟨e1, _⟩ | ⟨e1, _⟩ <;>
  rcases abs_cases (p.2 - v.2) with ⟨e2, _⟩ | ⟨e2, _⟩ <;>
  rw [e1, e2] <;> omega

/-- **Shared parity under (PAR)** (core of `lem:parityvalid`). If the offsets obey
the parity condition `c_s - c_t ≡ d(p_s, p_t) (mod 2)`, then the two cones take
values of equal parity at *every* cell `v`. -/
theorem cones_share_parity (cs ct : ℤ) (ps pt v : ℤ × ℤ)
    (PAR : cs - ct ≡ dgrid ps pt [ZMOD 2]) :
    coneVal cs ps v ≡ coneVal ct pt v [ZMOD 2] := by
  have hs := dgrid_parity ps v
  have ht := dgrid_parity pt v
  have hst := dgrid_parity ps pt
  unfold coneVal Int.ModEq at *
  omega

/-- **Parity gap** (arithmetic core of `lem:paritygap`). Two integers of equal
parity that are unequal differ by at least `2`. -/
theorem parity_gap (a b : ℤ) (h : a ≡ b [ZMOD 2]) (hne : a ≠ b) : 2 ≤ |a - b| := by
  unfold Int.ModEq at h
  rcases abs_cases (a - b) with ⟨e, _⟩ | ⟨e, _⟩ <;> rw [e] <;> omega

/-- **Inactive-cone gap** (arithmetic core of `lem:paritygap`). Under `(PAR)`, a
cone whose value at `v` strictly exceeds another's exceeds it by at least `2` —
there is no parity-forbidden gap of exactly `1`. The full lemma additionally
needs that the envelope `E(v)` is attained as a cone value (the geometric layer
disclaimed below). -/
theorem inactive_cone_gap (cs ct : ℤ) (ps pt v : ℤ × ℤ)
    (PAR : cs - ct ≡ dgrid ps pt [ZMOD 2])
    (hlt : coneVal cs ps v < coneVal ct pt v) :
    coneVal cs ps v + 2 ≤ coneVal ct pt v := by
  have hpar := cones_share_parity cs ct ps pt v PAR
  have hne : coneVal cs ps v ≠ coneVal ct pt v := ne_of_lt hlt
  have := parity_gap _ _ hpar hne
  rcases abs_cases (coneVal cs ps v - coneVal ct pt v) with ⟨e, _⟩ | ⟨e, _⟩ <;>
    rw [e] at this <;> omega

/-- **Edge step** (conclusion of `lem:parityvalid`). A `1`-Lipschitz integer map
whose values are unequal across an edge differs by exactly `1`. Applied to a
`1`-Lipschitz envelope `E` that, under `(PAR)`, is never equal across an edge,
this is what makes `E` a height function. -/
theorem lipschitz_edge (a b : ℤ) (hlip : |a - b| ≤ 1) (hne : a ≠ b) :
    |a - b| = 1 := by
  rcases abs_cases (a - b) with ⟨e, _⟩ | ⟨e, _⟩ <;> rw [e] at hlip ⊢ <;> omega

end OrigamiCone.Sequel

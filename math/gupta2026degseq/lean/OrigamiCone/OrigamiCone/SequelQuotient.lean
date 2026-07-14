import Mathlib

/-!
# Sequel meta-theorem: the colour-rotation quotient (`lem:quotient`)

Standalone formalisation of the structural content of `Lemma lem:quotient` of
the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:quotient` is the third ingredient of the period-`1` (polynomiality)
mechanism (after `lem:ratGF` and `lem:poles`). It identifies the mechanism that
collapses the period from `3` (apparent in the colour transfer matrix `T_0`'s
unit-circle spectrum `{1, ω, ω²}`) to `1` (the actual `OFG` period):

> The colour rotation `ρ : c ↦ c + 1 (mod 3)` commutes with `T_0` and acts
> freely on admissible pairs. The `OFG` count lives in the `ρ`-invariant
> (trivial-character) block `T_0^{triv}`. The states on each cycle of `T_0` are
> the frozen pairs `(c-k, c)` for some slope `k ∈ {1,2}`; the rotation
> identifies the three columns of each cycle, so on the trivial-character block
> each `3`-cycle collapses to a single self-loop: `T_0^{triv}` is the identity
> on the `2^m` frozen orbits and zero on the transient (non-frozen) ones. Hence
> its spectrum is `{0, 1}`, peripheral part `{1}`, and the GF (via `lem:poles`)
> has a single pole at `z = 1`.

The paper's argument has three components: (i) `ρ` commutes with `T_0` and acts
freely; (ii) on frozen pairs, the `T_0`-successor lies in the same `ρ`-orbit;
(iii) on transient (non-frozen) pairs, `T_0` has a zero row. Combining (i)–(iii)
with `ρ`-invariance of the `OFG` count gives `T_0^{triv} = identity on frozen
orbits + zero on transient` — which is what `lem:quotient` proves.

This module formalises the **abstract structural core** of the argument: for
any matrix `T : Matrix X X R` over a commutative ring whose rows are either
zero (the transient case) or single-column indicators at a column lying in the
`ρ`-orbit of the row index (the frozen-cycle case), the action of `T` on any
`ρ`-invariant function reduces to the indicator of the "active" rows:

* `iterate_invariant` : a `ρ`-invariant function `f` is invariant under every
  power of `ρ` — the orbit-invariance primitive;
* `quotient_action` (`lem:quotient`, **abstract content**): for a row-structure
  matrix `T` (each row either zero on transients or a single-column indicator
  at `σ(i)` with `σ(i) ∈ ρ`-orbit of `i`) and a `ρ`-invariant `f`, we have
  `(T *ᵥ f) i = if active i then f i else 0` — `T` acts as `1 ↦ 1` on
  the active block and as `1 ↦ 0` on the transient block. When `ρ` additionally
  preserves `active` (true in the paper's setting since `ρ` preserves frozen
  slope), the `ρ`-invariant subspace `V` decomposes as `V_active ⊕ V_transient`
  and `T|_V` is the diagonal operator with spectrum `{0, 1}` (peripheral `{1}`).
  The `ρ`-preservation hypothesis is not stated in the theorem itself: only the
  pointwise action `T *ᵥ f` is proved here, not the subspace decomposition.

The conclusion `quotient_action` is the structural content of `lem:quotient`'s
"`T_0^{triv}` is the identity on the `2^m` frozen orbits and zero on the
transient ones": specialising to the colour transfer setting (`X` = admissible
column pairs of `P_m`, `ρ` = the diagonal-`+1`-mod-`3` rotation, `active i` =
"`i` is a frozen pair", `σ i` = its frozen successor) recovers the paper's
claim. The actual transfer matrix's row structure (each row either zero or
single-column at the frozen successor) is the combinatorial input — supplied by
`SequelFrozen.frozen_imp_extremumFree` + `SequelCascade.cascade` (the Frozen
Classification).

The spectral consequence (`{0, 1}` spectrum ⟹ peripheral spectrum `{1}` ⟹
single pole at `z = 1` for the GF via `SequelPoles.poles_at_x_zero`) is the
*conclusion* that the abstract structure (`T = diag(0, 1)` on the invariant
subspace) implies. The diagonal-operator-has-spectrum-`{0,1}` step is
standard linear algebra and is not re-formalised in this module.

Scope: the abstract row-structure theorem is proved end-to-end. The
instantiation to the paper's transfer matrix `T_0`(colour-rotation, admissible
pairs, frozen classification) is the structural application disclosed in the
docstring but **not** glued in this module; it requires assembling the Frozen
Classification (`SequelFrozen` + `SequelCascade`) into a concrete row-structure
hypothesis. The standard diagonal-operator-spectrum step from `T = diag(0,1)`
to "peripheral spectrum `{1}`" is also disclosed but not re-derived.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.quotient_action`.
-/

namespace OrigamiCone.Sequel

open Matrix

/-- **Orbit invariance.** A function invariant under a permutation `ρ` is
invariant under every iterate of `ρ`. -/
theorem iterate_invariant {X R : Type*} (ρ : X → X) (f : X → R)
    (hinv : ∀ x, f (ρ x) = f x) (k : ℕ) :
    ∀ x, f (ρ^[k] x) = f x := by
  induction k with
  | zero => intro x; rfl
  | succ k IH =>
    intro x
    rw [Function.iterate_succ', Function.comp_apply, hinv]
    exact IH x

/-- **Quotient action** (`lem:quotient`, abstract content). For a matrix `T`
each of whose rows is either entirely zero (transient case) or a single-column
indicator at `σ i` with `σ i` in the `ρ`-orbit of `i` (frozen-cycle case), the
action of `T` on any `ρ`-invariant function `f` is the diagonal operator
`(T *ᵥ f) i = if active i then f i else 0`. This is the paper's
"`T_0^{triv} = identity on frozen orbits + zero on transient ones`". -/
theorem quotient_action {X : Type*} [Fintype X] [DecidableEq X] {R : Type*}
    [CommRing R] (T : Matrix X X R) (ρ : X → X) (f : X → R)
    (active : X → Prop) [DecidablePred active]
    (hinv : ∀ x, f (ρ x) = f x)
    (htrans : ∀ i, ¬ active i → ∀ j, T i j = 0)
    (hsucc : ∀ i, active i → ∃ σi : X, (∃ k : ℕ, σi = ρ^[k] i) ∧
        ∀ j, T i j = if j = σi then 1 else 0) :
    ∀ i, (T *ᵥ f) i = if active i then f i else 0 := by
  intro i
  by_cases hi : active i
  · simp only [hi, if_true]
    obtain ⟨σi, ⟨k, hk⟩, hT⟩ := hsucc i hi
    show ∑ j, T i j * f j = f i
    simp only [hT]
    rw [Finset.sum_eq_single σi]
    · simp
      rw [hk]
      exact iterate_invariant ρ f hinv k i
    · intro j _ hj; simp [hj]
    · intro h; exact absurd (Finset.mem_univ _) h
  · simp only [hi, if_false]
    show ∑ j, T i j * f j = 0
    simp [htrans i hi]

end OrigamiCone.Sequel

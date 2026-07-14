import Mathlib

/-!
# Sequel: `lem:quotient` instantiated at the colour-rotation transfer matrix

Standalone formalisation of the **concrete** content of `Lemma lem:quotient` of
the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`SequelQuotient.quotient_action` proves the *abstract* row-structure form:
for a matrix `T` whose rows are either entirely zero (transient case) or a
single-column indicator at `σ i` with `σ i` in the `ρ`-orbit of `i`
(frozen-cycle case), and any `ρ`-invariant function `f`, the action
`T *ᵥ f` collapses to the indicator of "active" rows. The instantiation to the
paper's concrete setting was explicitly disclaimed as out of scope in that
module's docstring:

> "Scope: the abstract row-structure theorem is proved end-to-end. The
> instantiation to the paper's transfer matrix `T_0` (colour-rotation,
> admissible pairs, frozen classification) is the structural application
> disclosed in the docstring but **not** glued in this module; it requires
> assembling the Frozen Classification (`SequelFrozen` + `SequelCascade`) into a
> concrete row-structure hypothesis."

This module provides the **algebraic half** of that abstract-to-concrete
gap: it builds a concrete column-pair model, defines a transfer matrix `T0`
with the row-structure property *by construction* (frozen rows are
single-column indicators; transient rows are zero), and proves
`T0_quotient_action` for this `T0`. The remaining *combinatorial half* — that
the paper's actual transfer matrix on admissible column pairs equals this
`T0` — is the content of the Frozen Classification
(`SequelFrozen.frozen_imp_extremumFree` + `SequelCascade.cascade`) and is not
re-derived here. The two halves together establish `lem:quotient` for the
paper's matrix.

We model the paper's column-pair state space, colour rotation, and frozen-pair
successor concretely:

* `Col m := Fin m → ZMod 3` — a single column of length `m`.
* `ρcol (u, v) := (u + 1, v + 1)` — the colour rotation acting componentwise.
* `σpair (u, v) := (v, 2v - u)` — the frozen-pair successor, defined as a
  total function on column pairs (matches the paper's recurrence; on frozen
  pairs `v = u + k` it specialises to `(u + k, v + k)`).
* `frozenPair (u, v) := ∃ k ≠ 0, ∀ i, v i = u i + k` — the paper's
  "frozen column pair with non-trivial constant slope" predicate.
* `T0 m R s s' := if frozenPair s ∧ s' = σpair s then 1 else 0` — the concrete
  colour transfer matrix in row-structure form.

The two key algebraic facts are:

* `σpair_eq_ρcol_iterate` : on a frozen pair with slope `k : ZMod 3`, the
  successor `σpair` coincides with the `k.val`-fold iterate of `ρcol`. This is
  exactly the row-structure hypothesis required by `quotient_action`: the
  frozen-cycle successor lies in the `ρ`-orbit of the row index.
* `frozenPair_ρcol` : `ρcol` preserves frozenness with the same slope, so the
  "active set" `frozenPair` is `ρcol`-invariant. This is the additional
  hypothesis that promotes the pointwise `T0 *ᵥ f` formula to a clean
  `V_frozen ⊕ V_transient` decomposition of the `ρcol`-invariant subspace.

The main theorem `T0_quotient_action` then specialises `quotient_action` to the
concrete `(T0, ρcol, frozenPair, σpair)` quadruple: for any `ρcol`-invariant
`f`, the action of `T0` is the diagonal `(T0 *ᵥ f) s = if frozenPair s then f s
else 0`. This is precisely the paper's

> "`T_0^{triv}` is the identity on the `2^m` frozen orbits and zero on the
> transient ones"

at the level of the pointwise action on `ρcol`-invariant functions.

Scope:

* The concrete row-structure instantiation is proved end-to-end (no `sorry`).
* The cardinality claim "`2^m` frozen orbits" is a separate orbit-counting
  fact: frozen pairs are parametrised by `(u : Col m, k : {1, 2})`, so the
  `ρcol`-orbits of frozen pairs are in bijection with column-difference classes
  modulo translation by `(1, 1)`. This module proves the *structural* claim
  (`T0` collapses to a diagonal on the `ρcol`-invariant block); the
  *enumerative* claim (cardinality `2^m`) is downstream and not formalised
  here.
* The spectral consequence (`{0, 1}` spectrum ⟹ peripheral spectrum `{1}` ⟹
  single pole at `z = 1` of the GF via `SequelPoles.poles_at_x_zero`) is again
  the standard diagonal-operator-spectrum step and is not re-derived.
* Per the discipline, this module only imports `Mathlib` and inlines a
  copy of the orbit-invariance primitive (`iterate_invariant_local`) and of the
  abstract quotient action's proof skeleton, rather than importing
  `OrigamiCone.SequelQuotient`. The inlined `iterate_invariant_local` matches
  `SequelQuotient.iterate_invariant` definitionally; the proof of
  `T0_quotient_action` replicates the proof skeleton of
  `SequelQuotient.quotient_action`.

The non-vacuity of the concrete model is witnessed by
`exists_frozenPair`: for every `m`, the pair `(0, 1)` (constant zero column and
constant one column) is frozen with slope `k = 1`, so the frozen block is
non-empty and the theorem is not over the empty set.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.T0_quotient_action`.
-/

namespace OrigamiCone.Sequel

open Matrix

/-- A column of length `m`: a function from `Fin m` to `ZMod 3`. -/
abbrev Col (m : ℕ) := Fin m → ZMod 3

/-- **Colour rotation** `ρ`. Acts componentwise: `(u, v) ↦ (u + 1, v + 1)`. -/
def ρcol {m : ℕ} : Col m × Col m → Col m × Col m :=
  fun s => (fun i => s.1 i + 1, fun i => s.2 i + 1)

/-- **Frozen-pair successor** `σ`. Defined as a total function on column pairs
by `(u, v) ↦ (v, 2v - u)`. On a frozen pair `v = u + k` this specialises to
`(u + k, v + k) = ρcol^[k.val] (u, v)` (see `σpair_eq_ρcol_iterate`). -/
def σpair {m : ℕ} : Col m × Col m → Col m × Col m :=
  fun s => (s.2, fun i => 2 * s.2 i - s.1 i)

/-- **Frozen column pair**: `v` equals `u` plus a non-zero constant slope `k`
componentwise. -/
def frozenPair {m : ℕ} (s : Col m × Col m) : Prop :=
  ∃ k : ZMod 3, k ≠ 0 ∧ ∀ i, s.2 i = s.1 i + k

instance frozenPair_decidable {m : ℕ} : DecidablePred (@frozenPair m) := by
  intro s; unfold frozenPair; infer_instance

/-- **Non-vacuity.** For every `m`, the pair `(0, 1)` of constant columns is
frozen with slope `1`. So `frozenPair` is satisfied for at least one state and
the main theorem is not vacuous. -/
theorem exists_frozenPair (m : ℕ) :
    ∃ s : Col m × Col m, frozenPair s := by
  refine ⟨(fun _ => 0, fun _ => 1), 1, ?_, ?_⟩
  · decide
  · intro _; simp

/-- **Iteration formula for the colour rotation.** The `n`-fold iterate of
`ρcol` shifts both columns by `n` (cast to `ZMod 3`). -/
lemma ρcol_iterate {m : ℕ} (s : Col m × Col m) (n : ℕ) :
    ρcol^[n] s = (fun i => s.1 i + n, fun i => s.2 i + n) := by
  induction n with
  | zero => ext <;> simp
  | succ n IH =>
    rw [Function.iterate_succ', Function.comp_apply, IH]
    ext i
    · show s.1 i + (n : ZMod 3) + 1 = s.1 i + ((n + 1 : ℕ) : ZMod 3)
      push_cast; ring
    · show s.2 i + (n : ZMod 3) + 1 = s.2 i + ((n + 1 : ℕ) : ZMod 3)
      push_cast; ring

/-- **Key algebraic identity.** On a frozen pair with slope `k`, the successor
`σpair` equals the `k.val`-fold iterate of the colour rotation `ρcol`. This is
the row-structure hypothesis required by `quotient_action`: `σpair s` lies in
the `ρcol`-orbit of `s`. -/
lemma σpair_eq_ρcol_iterate {m : ℕ} (s : Col m × Col m) (k : ZMod 3)
    (hv : ∀ i, s.2 i = s.1 i + k) :
    σpair s = ρcol^[k.val] s := by
  rw [ρcol_iterate]
  ext i
  · show s.2 i = s.1 i + (k.val : ZMod 3)
    rw [hv]; congr 1; exact (ZMod.natCast_zmod_val k).symm
  · show 2 * s.2 i - s.1 i = s.2 i + (k.val : ZMod 3)
    rw [hv, show ((k.val : ZMod 3) : ZMod 3) = k from ZMod.natCast_zmod_val k]
    ring

/-- The colour rotation preserves frozenness with the same slope. So the
"active set" `frozenPair` is `ρcol`-invariant, which is the hypothesis that
promotes the pointwise `quotient_action` formula to a `V_frozen ⊕ V_transient`
subspace decomposition. -/
lemma frozenPair_ρcol {m : ℕ} (s : Col m × Col m) :
    frozenPair (ρcol s) ↔ frozenPair s := by
  unfold frozenPair ρcol
  refine ⟨?_, ?_⟩
  · rintro ⟨k, hk, hv⟩
    refine ⟨k, hk, ?_⟩
    intro i
    have := hv i
    show s.2 i = s.1 i + k
    linear_combination this
  · rintro ⟨k, hk, hv⟩
    refine ⟨k, hk, ?_⟩
    intro i
    show s.2 i + 1 = s.1 i + 1 + k
    linear_combination hv i

/-- Inline copy of `SequelQuotient.iterate_invariant` (orbit-invariance
primitive). Stated locally to keep this module standalone (per the discipline
of importing only `Mathlib`); the statement matches
`SequelQuotient.iterate_invariant` verbatim. -/
private theorem iterate_invariant_local {X R : Type*} (ρ : X → X) (f : X → R)
    (hinv : ∀ x, f (ρ x) = f x) (k : ℕ) :
    ∀ x, f (ρ^[k] x) = f x := by
  induction k with
  | zero => intro x; rfl
  | succ k IH =>
    intro x
    rw [Function.iterate_succ', Function.comp_apply, hinv]
    exact IH x

/-- **Concrete transfer matrix on column pairs.** `T0 m R s s' = 1` iff `s` is
a frozen pair and `s'` is its successor `σpair s`; zero elsewhere. This is the
row-structure form required to apply `quotient_action`: frozen rows are
single-column indicators at `σpair s`; non-frozen rows are entirely zero. -/
noncomputable def T0 (m : ℕ) (R : Type*) [CommRing R] :
    Matrix (Col m × Col m) (Col m × Col m) R :=
  fun s s' => if frozenPair s ∧ s' = σpair s then 1 else 0

/-- **`lem:quotient` instantiated at the colour-rotation transfer matrix.** For
any `ρcol`-invariant function `f`, the action of `T0` on `f` is the diagonal
indicator of frozen pairs:
`(T0 *ᵥ f) s = if frozenPair s then f s else 0`.

This is the concrete content of the paper's

> "`T_0^{triv}` is the identity on the `2^m` frozen orbits and zero on the
> transient ones"

at the level of the pointwise action on `ρcol`-invariant functions. The
spectral conclusion (`{0, 1}` spectrum ⟹ peripheral spectrum `{1}` ⟹ single
pole at `z = 1` of the generating function via `SequelPoles.poles_at_x_zero`)
is the standard diagonal-operator-spectrum step and is not re-formalised here.

The proof inlines the structure of the abstract `quotient_action` rather than
importing `OrigamiCone.SequelQuotient` (per the discipline of importing only
`Mathlib`). The two halves of the case-split correspond to the two row types of
`T0`: the frozen case (indicator row at `σpair s = ρcol^[k.val] s`) collapses
the sum to `f (σpair s) = f s` via `iterate_invariant_local`; the transient
case (zero row) collapses the sum to `0`.

Non-vacuity is witnessed by `exists_frozenPair`: a frozen pair exists for every
`m`, so the `f s` branch of the conclusion is attained on a non-empty set. -/
theorem T0_quotient_action {m : ℕ} {R : Type*} [CommRing R]
    (f : (Col m × Col m) → R)
    (hinv : ∀ s, f (ρcol s) = f s) :
    ∀ s, (T0 m R *ᵥ f) s = if frozenPair s then f s else 0 := by
  classical
  intro s
  by_cases hi : frozenPair s
  · simp only [hi, if_true]
    show ∑ s', T0 m R s s' * f s' = f s
    have hT : ∀ s', T0 m R s s' = if s' = σpair s then 1 else 0 := by
      intro s'; simp [T0, hi]
    simp only [hT]
    rw [Finset.sum_eq_single (σpair s)]
    · simp
      obtain ⟨k, _, hv⟩ := hi
      have hiter : σpair s = ρcol^[k.val] s := σpair_eq_ρcol_iterate s k hv
      rw [hiter]
      exact iterate_invariant_local ρcol f hinv k.val s
    · intro j _ hj; simp [hj]
    · intro h; exact absurd (Finset.mem_univ _) h
  · simp only [hi, if_false]
    show ∑ s', T0 m R s s' * f s' = 0
    have hzero : ∀ s', T0 m R s s' = 0 := by intro s'; simp [T0, hi]
    simp [hzero]

end OrigamiCone.Sequel

import OrigamiCone.SequelEdPaperReducedFormFinite

/-!
# Finset representation of `PaperReducedForm m d`

Extends `SequelEdPaperReducedFormFinite`.  Given `Finite (PaperReducedForm m d)`
(from the previous module), upgrade to `Fintype` (via `Fintype.ofFinite`) and
expose the elements as a `Finset`.  This is the target `S`-construction for
`Ed_decomposition_of_ge_two`: the residual axiom's `S : Finset ContractionType`
becomes the image of this Finset under `(W, runCount)`.

## Contents

* `PaperReducedForm.instFintype`: `Fintype` instance (noncomputable, uses
  `Fintype.ofFinite`).
* `PRFinset m d`: `Finset (PaperReducedForm m d)` of all paper-reduced forms
  (via `(Fintype.ofFinite _).elems`).
* `PRFinset_forall`: universal quantification `∀ t : PaperReducedForm m d,
  t ∈ PRFinset m d`.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline
(no new).
-/

namespace OrigamiCone.Sequel

open OrigamiCone

namespace PaperReducedForm

variable {m d : ℕ}

/-- **`Fintype` instance** for `PaperReducedForm m d` (given `1 ≤ m`).
Noncomputable — uses `Fintype.ofFinite` which invokes `Classical.choice` /
decidable equality on the fintype target of `encFin`.

Not a global `instance` (requires the `hm` hypothesis). -/
noncomputable def toFintype (hm : 1 ≤ m) : Fintype (PaperReducedForm m d) :=
  have := finite_paperReducedForm (m := m) (d := d) hm
  Fintype.ofFinite _

/-- The `Finset` of all paper-reduced forms on `Cell m · → ℤ` with exactly
`d` extrema, up to width `≤ 2d + 3`.  Noncomputable. -/
noncomputable def PRFinset (m d : ℕ) (hm : 1 ≤ m) : Finset (PaperReducedForm m d) :=
  (toFintype hm).elems

/-- **Universality of `PRFinset`**: every paper-reduced form is in the
Finset. -/
lemma mem_PRFinset (hm : 1 ≤ m) (t : PaperReducedForm m d) :
    t ∈ PRFinset m d hm :=
  (toFintype hm).complete t

end PaperReducedForm

end OrigamiCone.Sequel

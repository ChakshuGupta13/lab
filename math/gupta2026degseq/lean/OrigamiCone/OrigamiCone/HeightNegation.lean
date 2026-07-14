import OrigamiCone.AcellHeight

/-!
# Negation primitives for heights and strict-local extrema

Three short primitives currently used implicitly in proofs but never
named:

* `IsHeight.neg` — negation preserves the height-function property.
* `isStrictLocalMax_neg_iff` / `isStrictLocalMin_neg_iff` — `IsStrictLocal{Max,Min} (−h) q ↔ IsStrictLocal{Min,Max} h q`.
* `negAcell_isHeight` — `IsHeight (fun v => −acell v)` (one-line corollary).

The strict-extremum negation iffs are the abstraction underlying the
sign-flip step in `acell_strictMin_origin` (see `AcellExtremaWitness.lean`,
`f54669d`); having them as named primitives keeps future "min via −max"
proofs short.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **Negation preserves height functions.** If `h` is a height function
(values change by exactly one across each edge), so is `-h`. -/
lemma IsHeight.neg {h : Cell m n → ℤ} (hh : IsHeight h) :
    IsHeight (fun v => -h v) := by
  intro p q hpq
  show |-h p - -h q| = 1
  rw [show -h p - -h q = -(h p - h q) by ring, abs_neg]
  exact hh p q hpq

/-- **Strict-local maximum of `−h` ↔ strict-local minimum of `h`.**
Since the strict-local-max condition is `∀ u, adj q u → h u = h q − 1`,
applying it to `−h` and negating gives the strict-local-min condition for
`h`. -/
lemma isStrictLocalMax_neg_iff {h : Cell m n → ℤ} (q : Cell m n) :
    IsStrictLocalMax (fun v => -h v) q ↔ IsStrictLocalMin h q := by
  refine ⟨fun hMax u hu => ?_, fun hMin u hu => ?_⟩
  · have := hMax u hu; linarith
  · have := hMin u hu; linarith

/-- **Strict-local minimum of `−h` ↔ strict-local maximum of `h`.**
Symmetric to `isStrictLocalMax_neg_iff`. -/
lemma isStrictLocalMin_neg_iff {h : Cell m n → ℤ} (q : Cell m n) :
    IsStrictLocalMin (fun v => -h v) q ↔ IsStrictLocalMax h q := by
  refine ⟨fun hMin u hu => ?_, fun hMax u hu => ?_⟩
  · have := hMin u hu; linarith
  · have := hMax u hu; linarith

/-- **`−acell` is a height function.**  Immediate from `acell_isHeight`
and `IsHeight.neg`. -/
lemma negAcell_isHeight : IsHeight (fun v : Cell m n => -acell v) :=
  acell_isHeight.neg

end OrigamiCone

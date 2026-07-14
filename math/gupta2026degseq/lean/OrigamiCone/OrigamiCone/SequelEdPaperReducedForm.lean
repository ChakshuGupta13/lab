import OrigamiCone.SequelEdPaperReduced
import OrigamiCone.SequelEd
import OrigamiCone.SequelEdColumnPartition
import OrigamiCone.SequelEdReducedHF
namespace OrigamiCone.Sequel

open OrigamiCone Finset

/-- **Paper-reduced form on `Cell m W`**: an `IsCanonical`, `IsHeight` function
with no adjacent frozen columns, with `2 ≤ W ≤ 2d+3` (paper's bound), and
exactly `d` strict local extrema.  This is the target type for the paper's
contraction map (`lem:uniform` proof body).

The `W` is data (not a hypothesis) — different reduced forms can have different
widths.  The structure bundles all invariants a fully-reduced form satisfies. -/
structure PaperReducedForm (m d : ℕ) where
  /-- Width of the reduced form.  Bounded above by `2d + 3` (paper's width bound). -/
  W : ℕ
  /-- Width lower bound: two boundary columns are required. -/
  hW : 2 ≤ W
  /-- Width upper bound: reduced forms fit within `Cell m (2d+3)`. -/
  hW_upper : W ≤ 2 * d + 3
  /-- The reduced height function. -/
  h : Cell m W → ℤ
  /-- Height property: adjacent-cell differences are ±1. -/
  isHeight : IsHeight h
  /-- Canonical: the origin cell has height 0. -/
  isCanonical : IsCanonicalHeight h
  /-- Paper-reduced: no two adjacent columns are both frozen (isolated frozen
  columns are the "run markers" of the contraction picture). -/
  reduced : ∀ j : Fin W, ∀ (hj1 : j.val + 1 < W),
    ¬ (frozenColumn h j ∧ frozenColumn h ⟨j.val + 1, hj1⟩)
  /-- Extremum count matches the target `d`. -/
  numExtremaEq : (Finset.univ.filter (IsStrictLocalExtremum h)).card = d

namespace PaperReducedForm

variable {m d : ℕ}

/-- The number of frozen columns in a paper-reduced form (which equals the
number of "runs" in any preimage under the paper's contraction map).  -/
def runCount (t : PaperReducedForm m d) : ℕ :=
  (Finset.univ.filter (frozenColumn t.h)).card

/-- Coerce a `PaperReducedForm m d` to a `ReducedHF m` — every paper-reduced
form is a reduced height function; the `PaperReducedForm` structure adds the
width upper bound and extremum count. -/
def toReducedHF (t : PaperReducedForm m d) : ReducedHF m where
  W := t.W
  hW := t.hW
  h := t.h
  isHeight := t.isHeight
  isCanonical := t.isCanonical.2
  reduced := t.reduced

/-- `runCount t = numFrozenColumns t.h` by definition. -/
lemma runCount_eq_numFrozenColumns (t : PaperReducedForm m d) :
    t.runCount = numFrozenColumns t.h :=
  rfl

/-- **`runCount ≤ d + 1`** (paper's bound: at most `d + 1` frozen runs in a
paper-reduced form with `d` extrema).  Follows from
`numFrozenRuns_lt_of_numExtrema_eq` (which gives `numFrozenRuns + 1 ≤ d`,
i.e., `numFrozenRuns ≤ d - 1 ≤ d + 1`) combined with
`ReducedHF.numFrozenColumns_eq_numFrozenRuns`. -/
theorem runCount_le_d_add_one (t : PaperReducedForm m d) (hm : 0 < m) :
    t.runCount ≤ d + 1 := by
  show numFrozenColumns t.h ≤ d + 1
  have h_eq : numFrozenColumns t.h = numFrozenRuns t.h :=
    t.toReducedHF.numFrozenColumns_eq_numFrozenRuns hm
  rw [h_eq]
  by_cases hd : d = 0
  · subst hd
    have h1 : numFrozenRuns t.h + 1 ≤ 0 :=
      numFrozenRuns_lt_of_numExtrema_eq t.h t.isHeight hm
        (by have := t.hW; omega) 0 t.numExtremaEq
    omega
  · have h1 : numFrozenRuns t.h + 1 ≤ d :=
      numFrozenRuns_lt_of_numExtrema_eq t.h t.isHeight hm
        (by have := t.hW; omega) d t.numExtremaEq
    omega

/-- **Encoding injection**: `PaperReducedForm m d` injects into
`Σ W : Fin (2d + 4), (Cell m W.val → ℤ)`.  Two paper-reduced forms are equal
iff their (W, h) data match; the proof-fields are irrelevant (subsingleton).
Foundational step toward proving `Finite (PaperReducedForm m d)`. -/
noncomputable def embed (m d : ℕ) :
    PaperReducedForm m d ↪ Σ W : Fin (2 * d + 4), Cell m W.val → ℤ where
  toFun t := ⟨⟨t.W, by have := t.hW_upper; omega⟩, t.h⟩
  inj' := by
    intro t1 t2 h_eq
    have h_fst : t1.W = t2.W := by
      have := congrArg Sigma.fst h_eq
      simpa [Fin.mk.injEq] using this
    obtain ⟨W1, hW1, hW1_up, h1, iH1, iC1, red1, ne1⟩ := t1
    obtain ⟨W2, hW2, hW2_up, h2, iH2, iC2, red2, ne2⟩ := t2
    simp only at h_fst
    subst h_fst
    have h_snd : h1 = h2 := by
      have := (Sigma.mk.injEq _ _ _ _).mp h_eq
      exact this.2.eq
    subst h_snd
    rfl

end PaperReducedForm

end OrigamiCone.Sequel

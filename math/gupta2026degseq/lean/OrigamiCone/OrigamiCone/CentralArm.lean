import OrigamiCone.RidgeMax

/-!
# The central diagonal arm (Section 3, `cor:centralarm`)

Formalisation of the **Central diagonal arm** corollary (`cor:centralarm`) of the
Ridge Lemma.  After a reflection making `r₁ < r₂`, the four fold lines
`i = r₁, r₂`, `j = s₁, s₂` cut the grid into nine blocks; the open *central block*
`(r₁,r₂) × (s₁,s₂)` consists entirely of admissible rows and columns, so by the
Ridge Lemma every ridge cell there is a strict local maximum.  Inside the central
block the ridge is a single diagonal segment: an antidiagonal `i + j = const` when
the apexes are *aligned* (`s₁ < s₂`) and a main diagonal `i − j = const` when
*anti-aligned* (`s₁ > s₂`).

Results:
* `InCentralBlock` — the open central block (each coordinate strictly between the
  two apex coordinates);
* `InCentralBlock.apex_ne` — a non-empty central block forces the apexes to differ
  in both coordinates;
* `centralBlock_rowAdm`, `centralBlock_colAdm`, `centralBlock_doublyAdm` — central
  cells are doubly admissible;
* `centralBlock_ridge_strictMax` — every ridge cell in the central block is a
  strict local maximum of the cone-pair envelope (the operative content of the
  corollary, via `ridge_strictMax_iff_admissible`);
* `centralBlock_onRidge_iff_antidiag` — aligned case: the ridge is the
  antidiagonal `2(i+j) = δ + r₁ + r₂ + s₁ + s₂`;
* `centralBlock_onRidge_iff_diag` — anti-aligned case: the ridge is the main
  diagonal `2(i−j) = δ + r₁ + r₂ − s₁ − s₂`.

The lattice **count** `L(δ)` of the cells of this diagonal segment, and the
boundary-and-corner count `B(δ)` completing `M(δ) = L(δ) + B(δ)`, are not
formalised here.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- **The open central block** of the cone pair: cells whose row lies strictly
between the apex rows and whose column lies strictly between the apex columns. -/
def InCentralBlock (p₁ p₂ v : Cell m n) : Prop :=
  (min p₁.1.val p₂.1.val < v.1.val ∧ v.1.val < max p₁.1.val p₂.1.val) ∧
    (min p₁.2.val p₂.2.val < v.2.val ∧ v.2.val < max p₁.2.val p₂.2.val)

/-- A non-empty central block forces the two apexes to differ in both
coordinates (the standing hypothesis of the Ridge Lemma). -/
lemma InCentralBlock.apex_ne {p₁ p₂ v : Cell m n} (h : InCentralBlock p₁ p₂ v) :
    p₁.1.val ≠ p₂.1.val ∧ p₁.2.val ≠ p₂.2.val := by
  obtain ⟨⟨hi1, hi2⟩, hj1, hj2⟩ := h
  exact ⟨by omega, by omega⟩

/-- Central cells have an admissible row (they lie strictly between the apex
rows). -/
lemma centralBlock_rowAdm {p₁ p₂ v : Cell m n} (h : InCentralBlock p₁ p₂ v) :
    RowAdm p₁ p₂ v :=
  Or.inr (Or.inr h.1)

/-- Central cells have an admissible column. -/
lemma centralBlock_colAdm {p₁ p₂ v : Cell m n} (h : InCentralBlock p₁ p₂ v) :
    ColAdm p₁ p₂ v :=
  Or.inr (Or.inr h.2)

/-- Central cells are doubly admissible. -/
lemma centralBlock_doublyAdm {p₁ p₂ v : Cell m n} (h : InCentralBlock p₁ p₂ v) :
    RowAdm p₁ p₂ v ∧ ColAdm p₁ p₂ v :=
  ⟨centralBlock_rowAdm h, centralBlock_colAdm h⟩

/-- **The operative content of `cor:centralarm`.**  Every ridge cell in the
central block is a strict local maximum of the cone-pair envelope: central cells
are doubly admissible, so the Ridge Lemma's on-ridge characterisation applies. -/
theorem centralBlock_ridge_strictMax {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (hcb : InCentralBlock p₁ p₂ v) (hv : OnRidge p₁ p₂ δ v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v := by
  obtain ⟨hr, hs⟩ := hcb.apex_ne
  exact (ridge_strictMax_iff_admissible hr hs hv).mpr (centralBlock_doublyAdm hcb)

/-- **Aligned case of the diagonal arm.**  With `r₁ < r₂` and `s₁ < s₂`, a central
cell lies on the ridge iff it lies on the antidiagonal
`2(i+j) = δ + r₁ + r₂ + s₁ + s₂`. -/
lemma centralBlock_onRidge_iff_antidiag {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (hcb : InCentralBlock p₁ p₂ v)
    (hr : p₁.1.val < p₂.1.val) (hs : p₁.2.val < p₂.2.val) :
    OnRidge p₁ p₂ δ v ↔
      2 * ((v.1.val : ℤ) + v.2.val)
        = δ + (p₁.1.val + p₂.1.val + p₁.2.val + p₂.2.val) := by
  obtain ⟨⟨hi1, hi2⟩, hj1, hj2⟩ := hcb
  unfold OnRidge gdist
  omega

/-- **Anti-aligned case of the diagonal arm.**  With `r₁ < r₂` and `s₁ > s₂`, a
central cell lies on the ridge iff it lies on the main diagonal
`2(i−j) = δ + r₁ + r₂ − s₁ − s₂`. -/
lemma centralBlock_onRidge_iff_diag {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (hcb : InCentralBlock p₁ p₂ v)
    (hr : p₁.1.val < p₂.1.val) (hs : p₂.2.val < p₁.2.val) :
    OnRidge p₁ p₂ δ v ↔
      2 * ((v.1.val : ℤ) - v.2.val)
        = δ + (p₁.1.val + p₂.1.val - p₁.2.val - p₂.2.val) := by
  obtain ⟨⟨hi1, hi2⟩, hj1, hj2⟩ := hcb
  unfold OnRidge gdist
  omega

end OrigamiCone

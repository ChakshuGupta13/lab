import OrigamiCone.Parity

/-!
# Cone-pair minima (Proposition `prop:conepair`, minima characterisation)

Formalisation of the **minima characterisation** at the heart of the Cone-pair
bijection (Proposition `prop:conepair`) of Section 3.

A degree-4 vertex is, by the Envelope Lemma, the lower envelope of its two
min-cones; the Cone-pair bijection runs this in reverse, attaching to an
admissible pair `({p₁,p₂}, δ)` the height function
`h(v) = min(d(p₁,v), δ + d(p₂,v))` (a height function by the Parity Lemma) and
asking that its minima be exactly `{p₁,p₂}`.  The structural content — the part of
`prop:conepair` that is pure grid geometry, independent of the degree-4
hypothesis — is:

> with both apexes **active** (`−D < δ < D`, `D = d(p₁,p₂)`, `p₁ ≠ p₂`), the
> strict local minima of `h` are exactly `p₁` and `p₂`.

The paper's phrase "an active apex is the global minimum of its cone, while any
other vertex has, in whichever cone is smaller there, a neighbour one step nearer
that apex where `h` is strictly smaller" is exactly the two directions proved
here: the apexes are minima (`conePair_apexL_strictMin`, `conePair_apexR_strictMin`),
and no other cell is (`conePair_strictMin_imp_apex`, via the step-toward lemma).

The full bijection additionally requires the "exactly two maxima" condition and
the `h(1,1)=0` normalisation, which select the degree-4 vertices among all
admissible pairs; those are not part of this geometric kernel.

Results:
* `cpe` — the cone-pair envelope `min(d(p₁,·), δ + d(p₂,·))`;
* `conePair_apexL_strictMin`, `conePair_apexR_strictMin` — each active apex is a
  strict local minimum;
* `conePair_strictMin_imp_apex` — a strict local minimum is one of the apexes
  (for every offset `δ`);
* `conePair_strictMin_iff` — under both apexes active, the minima are exactly
  `{p₁, p₂}`.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- The cone-pair envelope `h(v) = min(d(p₁,v), δ + d(p₂,v))`. -/
def cpe (p₁ p₂ : Cell m n) (δ : ℤ) (v : Cell m n) : ℤ :=
  min (gdist p₁ v) (δ + gdist p₂ v)

/-- **Left apex is a strict local minimum** when active.  If `δ > −D` (the left
cone is strictly smaller at `p₁`, i.e. `p₁` is active) and `δ` has the parity of
`D = d(p₁,p₂)`, then `p₁` is a strict local minimum of the cone-pair envelope. -/
lemma conePair_apexL_strictMin {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) (hact : -gdist p₁ p₂ < δ) :
    IsStrictLocalMin (cpe p₁ p₂ δ) p₁ := by
  -- value at p₁: min(0, δ+D) = 0, since active (δ+D > 0).
  have hD : gdist p₂ p₁ = gdist p₁ p₂ := gdist_comm p₂ p₁
  have hp1 : cpe p₁ p₂ δ p₁ = 0 := by
    unfold cpe
    rw [gdist_self, hD]
    have : (0 : ℤ) ≤ δ + gdist p₁ p₂ := by omega
    omega
  intro u hu
  -- u is a neighbour of p₁: d(p₁,u) = 1, and d(p₂,u) ≥ D − 1.
  have hu1 : gdist p₁ u = 1 := hu
  have htri := gdist_triangle p₂ u p₁
  have huone : gdist u p₁ = 1 := by rw [gdist_comm]; exact hu
  -- δ + D is even and positive, hence ≥ 2, so δ + d(p₂,u) ≥ δ + D − 1 ≥ 1.
  have hval : cpe p₁ p₂ δ u = 1 := by
    unfold cpe
    rw [hu1]
    have : (1 : ℤ) ≤ δ + gdist p₂ u := by omega
    omega
  omega

/-- **Right apex is a strict local minimum** when active.  If `δ < D` (the right
cone is strictly smaller at `p₂`) and `δ` has the parity of `D`, then `p₂` is a
strict local minimum of the cone-pair envelope. -/
lemma conePair_apexR_strictMin {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0) (hact : δ < gdist p₁ p₂) :
    IsStrictLocalMin (cpe p₁ p₂ δ) p₂ := by
  -- value at p₂: min(D, δ) = δ, since active (δ < D).
  have hp2 : cpe p₁ p₂ δ p₂ = δ := by
    unfold cpe
    rw [gdist_self]
    have : δ + 0 ≤ gdist p₁ p₂ := by omega
    omega
  intro u hu
  have hu2 : gdist p₂ u = 1 := hu
  have htri := gdist_triangle p₁ u p₂
  have huone : gdist u p₂ = 1 := by rw [gdist_comm]; exact hu
  -- d(p₁,u) ≥ D − 1, and δ ≤ D − 2 (parity + active), so d(p₁,u) ≥ δ + 1.
  have hval : cpe p₁ p₂ δ u = δ + 1 := by
    unfold cpe
    rw [hu2]
    have : δ + 1 ≤ gdist p₁ u := by omega
    omega
  omega

/-- **A strict local minimum is an apex.**  Any strict local minimum of the
cone-pair envelope is `p₁` or `p₂` — for *every* offset `δ` (parity is not needed
in this direction).  (Any other cell has, in whichever cone is smaller there, a
neighbour one step nearer that apex where `h` is strictly smaller, so it cannot be
a minimum.) -/
lemma conePair_strictMin_imp_apex {p₁ p₂ : Cell m n} {δ : ℤ} {w : Cell m n}
    (hw : IsStrictLocalMin (cpe p₁ p₂ δ) w) : w = p₁ ∨ w = p₂ := by
  by_contra hcon
  push_neg at hcon
  obtain ⟨hw1, hw2⟩ := hcon
  -- whichever cone is the active (smaller) one at w, step toward its apex.
  rcases (by omega : gdist p₁ w ≤ δ + gdist p₂ w ∨ δ + gdist p₂ w < gdist p₁ w) with hle | hlt
  · -- left cone active: h w = d(p₁,w); step toward p₁ lowers it.
    have hval : cpe p₁ p₂ δ w = gdist p₁ w := by unfold cpe; omega
    obtain ⟨u, hadj, hstep⟩ := exists_step_toward hw1
    have hcone1 : gdist p₁ u = gdist p₁ w - 1 := by
      rw [gdist_comm p₁ u, hstep, gdist_comm w p₁]
    have hlo : cpe p₁ p₂ δ u ≤ gdist p₁ w - 1 := by
      unfold cpe; rw [← hcone1]; exact min_le_left _ _
    have hstrict := hw u hadj
    rw [hval] at hstrict
    omega
  · -- right cone active: h w = δ + d(p₂,w); step toward p₂ lowers it.
    have hval : cpe p₁ p₂ δ w = δ + gdist p₂ w := by unfold cpe; omega
    obtain ⟨u, hadj, hstep⟩ := exists_step_toward hw2
    have hcone2 : gdist p₂ u = gdist p₂ w - 1 := by
      rw [gdist_comm p₂ u, hstep, gdist_comm w p₂]
    have hlo : cpe p₁ p₂ δ u ≤ δ + gdist p₂ w - 1 := by
      unfold cpe
      have : δ + gdist p₂ u = δ + gdist p₂ w - 1 := by rw [hcone2]; ring
      rw [← this]; exact min_le_right _ _
    have hstrict := hw u hadj
    rw [hval] at hstrict
    omega

/-- **Cone-pair minima characterisation** (`prop:conepair`, geometric kernel).
With distinct apexes, parity-correct offset, and both apexes active
(`−D < δ < D`), the strict local minima of the cone-pair envelope
`h(v) = min(d(p₁,v), δ + d(p₂,v))` are exactly `p₁` and `p₂`. -/
theorem conePair_strictMin_iff {p₁ p₂ : Cell m n} {δ : ℤ}
    (hδ : (δ - gdist p₁ p₂) % 2 = 0)
    (hactL : -gdist p₁ p₂ < δ) (hactR : δ < gdist p₁ p₂) (w : Cell m n) :
    IsStrictLocalMin (cpe p₁ p₂ δ) w ↔ w = p₁ ∨ w = p₂ := by
  constructor
  · exact conePair_strictMin_imp_apex
  · rintro (rfl | rfl)
    · exact conePair_apexL_strictMin hδ hactL
    · exact conePair_apexR_strictMin hδ hactR

end OrigamiCone

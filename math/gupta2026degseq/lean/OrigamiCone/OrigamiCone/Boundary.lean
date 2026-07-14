import OrigamiCone.RidgeMax

/-!
# The Boundary Lemma (Section 3, `lem:boundary`), interior-apex case

Formalisation of the **both-apexes-interior** case of the Boundary Lemma
(`lem:boundary`): if both apexes of a cone-pair envelope are interior grid
vertices, then all four grid corners are strict local maxima, so the envelope has
at least four maxima — far more than the two a degree-4 vertex permits.  This is
the clean half of the lemma, the part that rules out two interior apexes
outright.

The mechanism is exactly the paper's: an interior apex `p` differs from every
corner in both coordinates, so by the corner characterisation
(`corner_strictMax_iff`) each corner is a strict local maximum of the single cone
`d(p,·)`; and a cell that is a strict local maximum of *both* cones is a strict
local maximum of their lower envelope (`IsStrictLocalMax.min`).

Results:
* `IsStrictLocalMax.const_add`, `IsStrictLocalMax.min` — generic facts: a
  constant shift and the lower envelope of two strict local maxima;
* `cpe_strictMax_of_both` — a cell that is a strict local maximum of both distance
  cones is one of the cone-pair envelope;
* `IsInterior` — an interior grid vertex (both coordinates strictly inside);
* `interior_corner_cone_strictMax` — an interior apex makes every corner a strict
  local maximum of its cone;
* `bothInterior_corner_strictMax` — both apexes interior ⟹ every corner is a
  strict local maximum of the cone-pair envelope;
* `bothInterior_four_corner_maxima` — the four explicit corners are all strict
  local maxima;
* `bothInterior_exists_four_maxima` — a four-element set of strict local maxima.

The remaining case of `lem:boundary` — one interior apex and one boundary apex
forcing at least three maxima via the three-branch ridge analysis — is not
formalised here.

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- Adding a constant preserves a strict local maximum (the `±1` structure is
unchanged). -/
lemma IsStrictLocalMax.const_add {f : Cell m n → ℤ} {v : Cell m n}
    (hf : IsStrictLocalMax f v) (c : ℤ) :
    IsStrictLocalMax (fun w => c + f w) v := by
  intro u hu
  have := hf u hu
  simp only
  omega

/-- **The lower envelope of two strict local maxima is a strict local maximum.**
If both `f` and `g` attain a strict local maximum at `v` (every neighbour is
exactly one lower), so does `w ↦ min (f w) (g w)`. -/
lemma IsStrictLocalMax.min {f g : Cell m n → ℤ} {v : Cell m n}
    (hf : IsStrictLocalMax f v) (hg : IsStrictLocalMax g v) :
    IsStrictLocalMax (fun w => min (f w) (g w)) v := by
  intro u hu
  have h1 := hf u hu
  have h2 := hg u hu
  simp only
  omega

/-- A cell that is a strict local maximum of **both** distance cones is a strict
local maximum of the cone-pair envelope `cpe p₁ p₂ δ`. -/
lemma cpe_strictMax_of_both {p₁ p₂ : Cell m n} {δ : ℤ} {v : Cell m n}
    (h1 : IsStrictLocalMax (gdist p₁) v) (h2 : IsStrictLocalMax (gdist p₂) v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v :=
  h1.min (h2.const_add δ)

/-- An **interior grid vertex**: both coordinates strictly inside the grid. -/
def IsInterior (p : Cell m n) : Prop :=
  0 < p.1.val ∧ p.1.val + 1 < m ∧ 0 < p.2.val ∧ p.2.val + 1 < n

/-- An interior apex makes every grid corner a strict local maximum of its
distance cone: an interior apex differs from a corner in both coordinates. -/
lemma interior_corner_cone_strictMax (hm : 2 ≤ m) (hn : 2 ≤ n) {p v : Cell m n}
    (hp : IsInterior p) (hc : IsCorner v) :
    IsStrictLocalMax (gdist p) v := by
  obtain ⟨hi0, hi1, hj0, hj1⟩ := hp
  refine (corner_strictMax_iff hm hn hc).mpr ⟨?_, ?_⟩
  · rcases hc.1 with h | h <;> omega
  · rcases hc.2 with h | h <;> omega

/-- **Both apexes interior ⟹ every corner is a maximum.**  If both apexes are
interior, every grid corner is a strict local maximum of the cone-pair envelope. -/
theorem bothInterior_corner_strictMax (hm : 2 ≤ m) (hn : 2 ≤ n)
    {p₁ p₂ : Cell m n} {δ : ℤ} (h1 : IsInterior p₁) (h2 : IsInterior p₂)
    {v : Cell m n} (hc : IsCorner v) :
    IsStrictLocalMax (cpe p₁ p₂ δ) v :=
  cpe_strictMax_of_both
    (interior_corner_cone_strictMax hm hn h1 hc)
    (interior_corner_cone_strictMax hm hn h2 hc)

/-- The four grid corners, as cells (needs `2 ≤ m`, `2 ≤ n`). -/
private def cTL (hm : 2 ≤ m) (hn : 2 ≤ n) : Cell m n := (⟨0, by omega⟩, ⟨0, by omega⟩)
private def cTR (hm : 2 ≤ m) (hn : 2 ≤ n) : Cell m n := (⟨0, by omega⟩, ⟨n - 1, by omega⟩)
private def cBL (hm : 2 ≤ m) (hn : 2 ≤ n) : Cell m n := (⟨m - 1, by omega⟩, ⟨0, by omega⟩)
private def cBR (hm : 2 ≤ m) (hn : 2 ≤ n) : Cell m n := (⟨m - 1, by omega⟩, ⟨n - 1, by omega⟩)

private lemma isCorner_cTL (hm : 2 ≤ m) (hn : 2 ≤ n) : IsCorner (cTL hm hn) :=
  ⟨Or.inl rfl, Or.inl rfl⟩
private lemma isCorner_cTR (hm : 2 ≤ m) (hn : 2 ≤ n) : IsCorner (cTR hm hn) :=
  ⟨Or.inl rfl, Or.inr rfl⟩
private lemma isCorner_cBL (hm : 2 ≤ m) (hn : 2 ≤ n) : IsCorner (cBL hm hn) :=
  ⟨Or.inr rfl, Or.inl rfl⟩
private lemma isCorner_cBR (hm : 2 ≤ m) (hn : 2 ≤ n) : IsCorner (cBR hm hn) :=
  ⟨Or.inr rfl, Or.inr rfl⟩

/-- **The four corners are strict local maxima.**  With both apexes interior, each
of the four explicit grid corners is a strict local maximum of the cone-pair
envelope. -/
theorem bothInterior_four_corner_maxima (hm : 2 ≤ m) (hn : 2 ≤ n)
    {p₁ p₂ : Cell m n} {δ : ℤ} (h1 : IsInterior p₁) (h2 : IsInterior p₂) :
    IsStrictLocalMax (cpe p₁ p₂ δ) (cTL hm hn) ∧
    IsStrictLocalMax (cpe p₁ p₂ δ) (cTR hm hn) ∧
    IsStrictLocalMax (cpe p₁ p₂ δ) (cBL hm hn) ∧
    IsStrictLocalMax (cpe p₁ p₂ δ) (cBR hm hn) :=
  ⟨bothInterior_corner_strictMax hm hn h1 h2 (isCorner_cTL hm hn),
   bothInterior_corner_strictMax hm hn h1 h2 (isCorner_cTR hm hn),
   bothInterior_corner_strictMax hm hn h1 h2 (isCorner_cBL hm hn),
   bothInterior_corner_strictMax hm hn h1 h2 (isCorner_cBR hm hn)⟩

/-- **At least four maxima.**  With both apexes interior, the cone-pair envelope
has a four-element set of strict local maxima — incompatible with the exactly two
maxima of a degree-4 vertex.  Hence two interior apexes never yield a degree-4
vertex (the both-interior case of `lem:boundary`). -/
theorem bothInterior_exists_four_maxima (hm : 2 ≤ m) (hn : 2 ≤ n)
    {p₁ p₂ : Cell m n} {δ : ℤ} (h1 : IsInterior p₁) (h2 : IsInterior p₂) :
    ∃ s : Finset (Cell m n), s.card = 4 ∧
      ∀ c ∈ s, IsStrictLocalMax (cpe p₁ p₂ δ) c := by
  obtain ⟨hTL, hTR, hBL, hBR⟩ := bothInterior_four_corner_maxima hm hn h1 h2 (δ := δ)
  refine ⟨{cTL hm hn, cTR hm hn, cBL hm hn, cBR hm hn}, ?_, ?_⟩
  · -- the four corners are pairwise distinct, so the set has four elements
    have dTLTR : cTL hm hn ≠ cTR hm hn := by
      simp only [cTL, cTR, ne_eq, Prod.mk.injEq, Fin.mk.injEq]; omega
    have dTLBL : cTL hm hn ≠ cBL hm hn := by
      simp only [cTL, cBL, ne_eq, Prod.mk.injEq, Fin.mk.injEq]; omega
    have dTLBR : cTL hm hn ≠ cBR hm hn := by
      simp only [cTL, cBR, ne_eq, Prod.mk.injEq, Fin.mk.injEq]; omega
    have dTRBL : cTR hm hn ≠ cBL hm hn := by
      simp only [cTR, cBL, ne_eq, Prod.mk.injEq, Fin.mk.injEq]; omega
    have dTRBR : cTR hm hn ≠ cBR hm hn := by
      simp only [cTR, cBR, ne_eq, Prod.mk.injEq, Fin.mk.injEq]; omega
    have dBLBR : cBL hm hn ≠ cBR hm hn := by
      simp only [cBL, cBR, ne_eq, Prod.mk.injEq, Fin.mk.injEq]; omega
    rw [Finset.card_insert_of_notMem (by simp [dTLTR, dTLBL, dTLBR]),
        Finset.card_insert_of_notMem (by simp [dTRBL, dTRBR]),
        Finset.card_insert_of_notMem (by simp [dBLBR]),
        Finset.card_singleton]
  · intro c hcmem
    simp only [Finset.mem_insert, Finset.mem_singleton] at hcmem
    rcases hcmem with h | h | h | h <;> subst h
    · exact hTL
    · exact hTR
    · exact hBL
    · exact hBR

end OrigamiCone

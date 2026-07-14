import OrigamiCone.SequelEdge
import OrigamiCone.SequelBinom

/-!
# Sequel meta-theorem: four-side disjoint decomposition (`lem:splitedge`)

Standalone formalisation of the four-side single-edge decomposition of the
sequel paper

> *Degree-`d` vertex counts of the `m × n` origami flip graph:
> structure and a polynomial conjecture.*

`Lemma lem:splitedge` decomposes the count `N_{(a,b)}(m, n)` of a-minima
b-maxima configurations into single-edge and multi-edge parts:

  N_{(a,b)}(m, n) = 2 f_{(a,b)}(n) + 2 f_{(a,b)}(m) + R_{(a,b)}(m, n),

where the four single-edge families (all apexes on the top row, bottom
row, left column, or right column of the `m × n` grid) are pairwise
disjoint for `a ≥ 2`, and `R_{(a,b)}` counts the multi-edge configurations
whose apexes are not all on one side.

The mathematically substantive content of the lemma is:

1. **Four-side disjointness** for `a ≥ 2`: no configuration lies in two of
   the four single-edge families simultaneously.
2. **Edge reduction**: on the top-row family, the 2D envelope reduces to
   `E_{A,c}(i,j) = (i-1) + τ(j)` (already proved: `SequelEdge.edge_reduction`).
3. **Balanced-split vanishing**: each family count vanishes unless `(a,b)`
   is balanced, from `SequelBinom.card_turnPatterns` via the
   `±1`-walk correspondence.

The overall decomposition formula is a definitional identity once the
four families are separated (`R` is the multi-edge residual by
definition). This module formalises the substantive (1) — the disjointness
— on the `ℤ × ℤ` substrate, along with the four side-predicates used
downstream. Parts (2) and (3) are re-exported from the imported modules.

Contents:

* `Side` : one of the four sides, `top / bottom / left / right`.
* `AllOnSide p S m n sd` : the predicate that every apex `p s` for `s ∈ S`
  lies on side `sd` of the `m × n` grid.
* `AllOnSide_top_bottom_disjoint`,
  `AllOnSide_left_right_disjoint` : contradiction between opposite sides
  for `m, n ≥ 2` (any non-empty `S`).
* `AllOnSide_top_left_disjoint`, `AllOnSide_top_right_disjoint`,
  `AllOnSide_bottom_left_disjoint`, `AllOnSide_bottom_right_disjoint` :
  contradiction between adjacent sides for `S.card ≥ 2` under `Set.InjOn p S`.
* `AllOnSide_pairwise_disjoint` : the six-pair disjointness bundled.

Scope: pure structural content of `lem:splitedge`. The full formula
`N_{(a,b)} = 2 f_{(a,b)}(n) + 2 f_{(a,b)}(m) + R_{(a,b)}` is not stated as
a Lean theorem because `N_{(a,b)}` and `R_{(a,b)}` require an ambient
formalisation of the configuration-counting set not present in the sequel
Lean modules; the DEFINITION `R := N − (single-edge parts)` makes the
formula tautologous once the disjointness proved here is in hand.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.AllOnSide_pairwise_disjoint`.
-/

namespace OrigamiCone.Sequel

/-- One of the four sides of the `m × n` grid. -/
inductive Side where
  | top | bottom | left | right
  deriving DecidableEq, Repr

open Side

variable {ι : Type*} (p : ι → ℤ × ℤ) (S : Finset ι)

/-- All apexes indexed by `S` lie on side `sd` of the `m × n` grid. -/
def AllOnSide (m n : ℤ) (sd : Side) : Prop :=
  match sd with
  | top    => ∀ s ∈ S, (p s).1 = 1
  | bottom => ∀ s ∈ S, (p s).1 = m
  | left   => ∀ s ∈ S, (p s).2 = 1
  | right  => ∀ s ∈ S, (p s).2 = n

/-- **Two-elements-in-a-Finset extractor.** For `S.card ≥ 2`, there are two
distinct elements of `S`. -/
private lemma two_mem_of_two_le_card {α : Type*} {S : Finset α} (h : 2 ≤ S.card) :
    ∃ s ∈ S, ∃ t ∈ S, s ≠ t := by
  have := Finset.one_lt_card.mp h
  exact this

/-- **Top ↔ Bottom disjointness.** Under `m ≥ 2`, no non-empty configuration is
simultaneously top-row (`.1 = 1`) and bottom-row (`.1 = m`). -/
theorem AllOnSide_top_bottom_disjoint (m n : ℤ) (hm : 2 ≤ m) (hS : S.Nonempty)
    (htop : AllOnSide p S m n top) (hbot : AllOnSide p S m n bottom) : False := by
  obtain ⟨s, hsS⟩ := hS
  have h1 := htop s hsS
  have h2 := hbot s hsS
  omega

/-- **Left ↔ Right disjointness.** Under `n ≥ 2`, no non-empty configuration is
simultaneously left-column (`.2 = 1`) and right-column (`.2 = n`). -/
theorem AllOnSide_left_right_disjoint (m n : ℤ) (hn : 2 ≤ n) (hS : S.Nonempty)
    (hlt : AllOnSide p S m n left) (hrt : AllOnSide p S m n right) : False := by
  obtain ⟨s, hsS⟩ := hS
  have h1 := hlt s hsS
  have h2 := hrt s hsS
  omega

/-- **Top ∧ Left disjointness** at `S.card ≥ 2`. Adjacent sides meet only at
the corner `(1, 1)`; forcing every apex to that single cell contradicts apex
distinctness. -/
theorem AllOnSide_top_left_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (htop : AllOnSide p S m n top) (hlt : AllOnSide p S m n left) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (1, 1) := by
    ext
    · exact htop s hsS
    · exact hlt s hsS
  have hpt : p t = (1, 1) := by
    ext
    · exact htop t htS
    · exact hlt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

/-- **Top ∧ Right disjointness** at `S.card ≥ 2`. -/
theorem AllOnSide_top_right_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (htop : AllOnSide p S m n top) (hrt : AllOnSide p S m n right) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (1, n) := by
    ext
    · exact htop s hsS
    · exact hrt s hsS
  have hpt : p t = (1, n) := by
    ext
    · exact htop t htS
    · exact hrt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

/-- **Bottom ∧ Left disjointness** at `S.card ≥ 2`. -/
theorem AllOnSide_bottom_left_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (hbot : AllOnSide p S m n bottom) (hlt : AllOnSide p S m n left) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (m, 1) := by
    ext
    · exact hbot s hsS
    · exact hlt s hsS
  have hpt : p t = (m, 1) := by
    ext
    · exact hbot t htS
    · exact hlt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

/-- **Bottom ∧ Right disjointness** at `S.card ≥ 2`. -/
theorem AllOnSide_bottom_right_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (hbot : AllOnSide p S m n bottom) (hrt : AllOnSide p S m n right) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (m, n) := by
    ext
    · exact hbot s hsS
    · exact hrt s hsS
  have hpt : p t = (m, n) := by
    ext
    · exact hbot t htS
    · exact hrt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

/-- **Pairwise four-side disjointness of the single-edge families** at `a ≥ 2`.
Given `m, n ≥ 2`, `S.card ≥ 2`, and `Set.InjOn p S`, no configuration lies in
two of the four single-edge families simultaneously. This is the `lem:splitedge`
disjointness claim in its cleanest form; the six explicit contradictions above
are the case analysis of the pair `(sd, sd')`. -/
theorem AllOnSide_pairwise_disjoint (m n : ℤ) (hm : 2 ≤ m) (hn : 2 ≤ n)
    (hcard : 2 ≤ S.card) (hinj : Set.InjOn p S) (sd sd' : Side) (hne : sd ≠ sd')
    (hsd : AllOnSide p S m n sd) (hsd' : AllOnSide p S m n sd') : False := by
  have hnonempty : S.Nonempty := Finset.card_pos.mp (by omega)
  -- Six unordered pairs; enumerate by matching on `sd, sd'`.
  cases sd <;> cases sd' <;> first
    | exact hne rfl
    | exact AllOnSide_top_bottom_disjoint p S m n hm hnonempty hsd hsd'
    | exact AllOnSide_top_bottom_disjoint p S m n hm hnonempty hsd' hsd
    | exact AllOnSide_left_right_disjoint p S m n hn hnonempty hsd hsd'
    | exact AllOnSide_left_right_disjoint p S m n hn hnonempty hsd' hsd
    | exact AllOnSide_top_left_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_top_left_disjoint p S m n hcard hinj hsd' hsd
    | exact AllOnSide_top_right_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_top_right_disjoint p S m n hcard hinj hsd' hsd
    | exact AllOnSide_bottom_left_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_bottom_left_disjoint p S m n hcard hinj hsd' hsd
    | exact AllOnSide_bottom_right_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_bottom_right_disjoint p S m n hcard hinj hsd' hsd

end OrigamiCone.Sequel

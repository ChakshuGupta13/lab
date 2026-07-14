import OrigamiCone.ColoringQuotient

/-!
# Colouring-side flippability and the bridge to height-side extrema

This module is the **option 3 spike**: a focused step toward "full Model A".
`OrigamiCone.QuotientDegree` proves Lemma 2.1 (`degree = #strict-local-extrema`)
on the height side; `OrigamiCone.ColoringQuotient` shows the height-shift
quotient is the paper's `R₃(G)/(ℤ/3)` quotient (`quotient_iso`).  What is still
purely paper-prose is the *cell-level* version of Lemma 2.1's flippability
↔ extremum step **on the colouring side directly**.

This module supplies that bridge:

* `ColFlippable c v` — the colouring-graph flippability predicate at cell `v`:
  some colour other than `c v` is absent from the neighbours of `v`.  This is
  the standard `R₃(G)` reconfiguration adjacency condition (the new colour, if
  any, is the unique missing one).
* `colFlippable_iff_extremum` — for `c = colOf h` (with `h` a height function),
  flippability at `v` is exactly "strict local extremum of `h` at `v`".

Combined with `ColoringQuotient.quotient_iso`, this lets a downstream consumer
state and use Lemma 2.1 **directly on the colouring side** (`R₃(G)/(ℤ/3)`)
without invoking the height-shift model — the substance of full Model A —
while still relying on the already-proved height-side degree count
(`QuotientDegree.ofgDegree_eq_extrema`) for the cardinality.

The full Model A endpoint (a colouring-native `OFGDegree (mkColC c)
= #(strict-local-extrema-of-any-height-lift)`) requires the Ginepro–Hull
**existence** direction (every proper colouring lifts to a height function),
which is the external interface still disclosed-and-deferred.  Without
existence, the colouring-side degree count can only be stated on the *image*
of `mkColV`, where it is by `quotient_iso` literally the same number.

No `sorry`; `#print axioms` is the standard `[propext, Classical.choice,
Quot.sound]`.
-/

namespace OrigamiCone
namespace ColoringModel

variable {m n : ℕ}

/-- A colouring `c` is **flippable at cell `v`** when some colour other than
`c v` is absent from `v`'s neighbours: then exactly one of `c v + 1`, `c v + 2`
is absent, that colour is the forced new value at `v`, and the result is again
a proper colouring (changing `c` only at `v`). -/
def ColFlippable (c : Coloring m n) (v : Cell m n) : Prop :=
  ∃ k : ZMod 3, k ≠ 0 ∧ ∀ u, adj v u → c u ≠ c v + k

/-- **The colouring-side / height-side flippability bridge.** For a proper
colouring `c = colOf h` (with `h : IsHeight`), the cell `v` is `ColFlippable`
iff it is a strict local extremum of `h`.

The forward direction is the *only* place the bijection content of Lemma 2.1
is needed at the cell level on the colouring side; the cardinality of the
flippable set then matches the cover-side count from
`QuotientDegree.ofgDegree_eq_extrema`. -/
theorem colFlippable_iff_extremum {h : Cell m n → ℤ}
    (hh : IsHeight h) (v : Cell m n) :
    ColFlippable (colOf h) v ↔ IsStrictLocalExtremum h v := by
  constructor
  · -- (⇒)  Some forbidden colour `c v + k` (k ∈ {1,2}) at every neighbour.
    --   For each neighbour u, h u = h v ± 1; the forbidden colour rules out
    --   one sign, forcing all neighbours to the OTHER sign, hence v is a
    --   strict local extremum (max if k = 1, min if k = 2).
    rintro ⟨k, hk0, hk⟩
    have hk12 : k = 1 ∨ k = 2 := by
      fin_cases k
      · exact absurd rfl hk0
      · left; rfl
      · right; rfl
    rcases hk12 with hk1 | hk2
    · -- k = 1: forbid (h v) + 1.  Then every nbr u has h u = h v - 1 ⟹ MAX.
      refine Or.inl (fun u hu => ?_)
      have hstep := hh v u hu
      have hcases : h u = h v - 1 ∨ h u = h v + 1 := by
        rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 hstep with hs | hs
        · left; linarith
        · right; linarith
      rcases hcases with hdn | hup
      · -- h u = h v - 1: as required for IsStrictLocalMax.
        exact hdn
      · -- h u = h v + 1: then c u = c v + 1 = c v + k, contradicting hk.
        exfalso
        have := hk u hu
        apply this
        show ((h u : ℤ) : ZMod 3) = ((h v : ℤ) : ZMod 3) + k
        rw [hup, hk1]; push_cast; ring
    · -- k = 2: forbid (h v) + 2 = (h v) - 1.  Then every nbr h u = h v + 1 ⟹ MIN.
      refine Or.inr (fun u hu => ?_)
      have hstep := hh v u hu
      have hcases : h u = h v - 1 ∨ h u = h v + 1 := by
        rcases (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 hstep with hs | hs
        · left; linarith
        · right; linarith
      rcases hcases with hdn | hup
      · -- h u = h v - 1: then c u = c v - 1 = c v + 2 = c v + k, contradicting hk.
        exfalso
        have := hk u hu
        apply this
        show ((h u : ℤ) : ZMod 3) = ((h v : ℤ) : ZMod 3) + k
        rw [hdn, hk2]
        push_cast
        -- goal: -1 + ↑(h v) = 2 + ↑(h v) in ZMod 3, i.e. -1 = 2.
        have h3 : ((3 : ℤ) : ZMod 3) = 0 := by decide
        linear_combination -h3
      · -- h u = h v + 1: as required for IsStrictLocalMin.
        exact hup
  · -- (⇐)  strict local extremum ⟹ forbid the colour on the "absent" side.
    rintro (hmax | hmin)
    · -- Strict local max: every nbr has height h v - 1, hence colour (h v) - 1 = (h v) + 2.
      --   The forbidden colour at v is (h v) + 1, i.e. take k = 1.
      refine ⟨1, by decide, ?_⟩
      intro u hu
      have huh : h u = h v - 1 := hmax u hu
      -- Goal: colOf h u ≠ colOf h v + 1, i.e. (h u : ZMod 3) ≠ (h v : ZMod 3) + 1.
      show ((h u : ℤ) : ZMod 3) ≠ ((h v : ℤ) : ZMod 3) + 1
      rw [huh]
      push_cast
      intro hc
      -- hc : (h v - 1 : ZMod 3) = (h v : ZMod 3) + 1.
      -- Simplify: (h v - 1) - (h v + 1) ≡ -2 ≡ 1 ≠ 0 mod 3.
      have : ((-2 : ℤ) : ZMod 3) = 0 := by linear_combination hc
      revert this; push_cast; decide
    · -- Strict local min: every nbr has height h v + 1, colour (h v) + 1.
      --   Forbidden colour at v is (h v) + 2, i.e. take k = 2.
      refine ⟨2, by decide, ?_⟩
      intro u hu
      have huh : h u = h v + 1 := hmin u hu
      show ((h u : ℤ) : ZMod 3) ≠ ((h v : ℤ) : ZMod 3) + 2
      rw [huh]
      push_cast
      intro hc
      have : ((-1 : ℤ) : ZMod 3) = 0 := by linear_combination hc
      revert this; push_cast; decide

end ColoringModel
end OrigamiCone

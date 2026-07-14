import OrigamiCone.Degree2
import OrigamiCone.Dual

/-!
# Degree-3 characterization (Theorem 3.3)

Formalisation of the characterization underlying **Theorem 3.3** (degree-3 count
`4(m+n-4)`):

> A degree-3 vertex has three extrema, so two minima and one maximum, or one
> minimum and two maxima.  Colour inversion `h ↦ -h` exchanges the two families.
> A vertex with a unique maximum `q` is the cone at `q`, with exactly two minima
> precisely when `q` is a non-corner boundary vertex.

This module proves the shift-invariant characterization core:

* `degree_three_iff` — a height function has degree 3 **iff** it is a cone
  `coneC q C`, or its colour-inversion `-coneC q C`, with `q` a **non-corner
  boundary** apex (exactly one coordinate an endpoint);
* `kappa_two_iff_boundary` — `κ(q) = 2` iff exactly one coordinate is an
  endpoint (the geometric "non-corner boundary" condition).

The "count `4(m+n-4)`" is the grid-enumeration layer (the number of non-corner
boundary cells is `2(m+n-4)`, doubled by the colour-inversion bijection); the
characterization is the mathematical core delivered by the Cone Lemma + Cone
Classification + the colour-inversion duality (`OrigamiCone.Dual`).

No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- A non-corner boundary apex: exactly one coordinate is a path endpoint. -/
def IsBoundaryNonCorner (q : Cell m n) : Prop :=
  (IsEndpoint q.1 ∧ ¬ IsEndpoint q.2) ∨ (¬ IsEndpoint q.1 ∧ IsEndpoint q.2)

/-- **κ(q) = 2 iff `q` is a non-corner boundary apex.** -/
theorem kappa_two_iff_boundary (hm : 2 ≤ m) (hn : 2 ≤ n) (q : Cell m n) :
    kappa q = 2 ↔ IsBoundaryNonCorner q := by
  constructor
  · intro h
    rw [kappa_eq_mul hm hn] at h
    -- factors are each 1 or 2; product 2 ⟹ one is 1, the other 2.
    by_cases hr : IsEndpoint q.1 <;> by_cases hc : IsEndpoint q.2
    · exfalso
      rw [pathEnd_card_endpoint hm hr, pathEnd_card_endpoint hn hc] at h; omega
    · exact Or.inl ⟨hr, hc⟩
    · exact Or.inr ⟨hr, hc⟩
    · exfalso
      rw [pathEnd_card_interior hm hr, pathEnd_card_interior hn hc] at h; omega
  · intro h
    exact kappa_boundary hm hn h

/-- **Degree-3, unique-maximum branch.** A height function with a *unique* strict
local maximum and degree 3 is exactly a cone `coneC q C` with `q` a non-corner
boundary apex. -/
theorem degree_three_unique_max (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) :
    ((neighbors h).ncard = 3 ∧ ∃ q, ∀ q', IsStrictLocalMax h q' → q' = q)
      ↔ ∃ (q : Cell m n) (C : ℤ), IsBoundaryNonCorner q ∧ h = coneC q C := by
  have hmn : 2 ≤ m * n := le_trans (by norm_num) (Nat.mul_le_mul hm hn)
  constructor
  · rintro ⟨hdeg, q, huniq⟩
    have happ : h = coneC q (h q) := by
      funext v; simp only [coneC]; exact cone_max hh huniq v
    have hk : (neighbors h).ncard = 1 + kappa q := by
      rw [happ]; exact coneC_degree hmn q (h q)
    rw [hdeg] at hk
    have hk2 : kappa q = 2 := by omega
    exact ⟨q, h q, (kappa_two_iff_boundary hm hn q).mp hk2, happ⟩
  · rintro ⟨q, C, hb, rfl⟩
    refine ⟨?_, q, coneC_unique_max q C⟩
    rw [coneC_degree hmn q C, (kappa_two_iff_boundary hm hn q).mpr hb]

/-- **Theorem 3.3 (degree-3 characterization).** For `m,n ≥ 2`, a height function
`h` has degree 3 iff it is a cone `coneC q C`, or the colour-inversion
`-coneC q C` of one, whose apex `q` is a non-corner boundary vertex.  The two
cases are the "two minima, one maximum" and "one minimum, two maxima" families
exchanged by colour inversion. -/
theorem degree_three_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) :
    (neighbors h).ncard = 3
      ↔ (∃ (q : Cell m n) (C : ℤ), IsBoundaryNonCorner q ∧ h = coneC q C)
        ∨ (∃ (q : Cell m n) (C : ℤ), IsBoundaryNonCorner q
              ∧ h = (fun v => -(coneC q C) v)) := by
  have hmn : 2 ≤ m * n := le_trans (by norm_num) (Nat.mul_le_mul hm hn)
  have hm0 : 0 < m := by omega
  have hn0 : 0 < n := by omega
  haveI : Nonempty (Cell m n) := ⟨(⟨0, hm0⟩, ⟨0, hn0⟩)⟩
  constructor
  · intro hdeg
    -- #max + #min = 3, both ≥ 1, so one of them is 1.
    have hsum : (Finset.univ.filter (IsStrictLocalMax h)).card
        + (Finset.univ.filter (IsStrictLocalMin h)).card = 3 := by
      rw [← extrema_card_split hmn, ← degree_eq_extrema hh hmn]; exact hdeg
    obtain ⟨wm, hwm⟩ := exists_strictLocalMax hh
    obtain ⟨wn, hwn⟩ := exists_strictLocalMin hh
    have hmax_pos : 0 < (Finset.univ.filter (IsStrictLocalMax h)).card :=
      Finset.card_pos.mpr ⟨wm, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hwm⟩⟩
    have hmin_pos : 0 < (Finset.univ.filter (IsStrictLocalMin h)).card :=
      Finset.card_pos.mpr ⟨wn, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hwn⟩⟩
    by_cases hmax1 : (Finset.univ.filter (IsStrictLocalMax h)).card = 1
    · -- unique maximum: cone branch
      left
      obtain ⟨q, hq⟩ := Finset.card_eq_one.mp hmax1
      have huniq : ∀ q', IsStrictLocalMax h q' → q' = q := by
        intro q' hq'
        have hmem : q' ∈ Finset.univ.filter (IsStrictLocalMax h) :=
          Finset.mem_filter.mpr ⟨Finset.mem_univ _, hq'⟩
        rw [hq, Finset.mem_singleton] at hmem; exact hmem
      exact (degree_three_unique_max hm hn hh).mp ⟨hdeg, q, huniq⟩
    · -- otherwise the minimum is unique: dual cone branch
      right
      have hmin1 : (Finset.univ.filter (IsStrictLocalMin h)).card = 1 := by omega
      -- work with -h, whose strict local maxima are h's strict local minima
      set h' := fun v => -h v with hh'def
      have hh' : IsHeight h' := isHeight_neg hh
      have hmaxh' : (Finset.univ.filter (IsStrictLocalMax h')).card = 1 := by
        rw [← hmin1]; congr 1; apply Finset.filter_congr
        intro v _; rw [strictMax_neg_iff]
      obtain ⟨q, hq⟩ := Finset.card_eq_one.mp hmaxh'
      have huniq : ∀ q', IsStrictLocalMax h' q' → q' = q := by
        intro q' hq'
        have hmem : q' ∈ Finset.univ.filter (IsStrictLocalMax h') :=
          Finset.mem_filter.mpr ⟨Finset.mem_univ _, hq'⟩
        rw [hq, Finset.mem_singleton] at hmem; exact hmem
      have hdeg' : (neighbors h').ncard = 3 := by rw [neighbors_neg]; exact hdeg
      obtain ⟨q2, C, hb, hcone⟩ := (degree_three_unique_max hm hn hh').mp ⟨hdeg', q, huniq⟩
      refine ⟨q2, C, hb, ?_⟩
      funext v
      have hv : -h v = coneC q2 C v := congrFun hcone v
      show h v = -(coneC q2 C v)
      omega
  · rintro (⟨q, C, hb, rfl⟩ | ⟨q, C, hb, rfl⟩)
    · rw [coneC_degree hmn q C, (kappa_two_iff_boundary hm hn q).mpr hb]
    · rw [neighbors_neg, coneC_degree hmn q C, (kappa_two_iff_boundary hm hn q).mpr hb]

end OrigamiCone

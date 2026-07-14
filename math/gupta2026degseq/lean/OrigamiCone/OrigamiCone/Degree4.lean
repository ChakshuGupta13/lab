import OrigamiCone.Degree3

/-!
# Degree-4 characterization (Theorem 3.4)

Formalisation of **Theorem 3.4**:

> For all `m,n ≥ 2`, the degree-4 vertices of `OFG(M_{m,n})` are exactly the
> height functions with two strict local minima and two strict local maxima.
> No degree-4 vertex is a cone.

The paper's argument: a degree-4 vertex has four extrema, at least one of each
type; a unique maximum would make it a cone, of degree 2, 3, or 5 (Cone
Classification), never 4, and dually for a unique minimum; so it has at least two
of each, forced to exactly two by the total of four.

This module proves:

* `card_max_one_degree_mem` / `card_min_one_degree_mem` — a height function with a
  *unique* strict local maximum (resp. minimum) is a cone, so has degree in
  `{2,3,5}`;
* `degree_four_iff` — degree 4 **iff** exactly two strict local maxima and two
  strict local minima;
* `degree_four_not_cone` — a degree-4 vertex is neither a cone `coneC q C` nor a
  dual cone `-coneC q C`.

This is the full content of Theorem 3.4 (it is a characterization, not a count,
so no grid-enumeration layer is deferred).  No `sorry`.
-/

namespace OrigamiCone

variable {m n : ℕ}

/-- A height function with exactly one strict local maximum is a cone, hence of
degree 2, 3, or 5 (never 4). -/
lemma card_max_one_degree_mem (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) (hmax1 : (Finset.univ.filter (IsStrictLocalMax h)).card = 1) :
    (neighbors h).ncard = 2 ∨ (neighbors h).ncard = 3 ∨ (neighbors h).ncard = 5 := by
  have hmn : 2 ≤ m * n := le_trans (by norm_num) (Nat.mul_le_mul hm hn)
  obtain ⟨q, hq⟩ := Finset.card_eq_one.mp hmax1
  have huniq : ∀ q', IsStrictLocalMax h q' → q' = q := by
    intro q' hq'
    have hmem : q' ∈ Finset.univ.filter (IsStrictLocalMax h) :=
      Finset.mem_filter.mpr ⟨Finset.mem_univ _, hq'⟩
    rw [hq, Finset.mem_singleton] at hmem; exact hmem
  have happ : h = coneC q (h q) := by
    funext v; simp only [coneC]; exact cone_max hh huniq v
  have hk : (neighbors h).ncard = 1 + kappa q := by
    rw [happ]; exact coneC_degree hmn q (h q)
  rcases kappa_mem hm hn q with h1 | h2 | h4
  · left; rw [hk, h1]
  · right; left; rw [hk, h2]
  · right; right; rw [hk, h4]

/-- Dually, a height function with exactly one strict local minimum is a dual
cone, hence of degree 2, 3, or 5. -/
lemma card_min_one_degree_mem (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) (hmin1 : (Finset.univ.filter (IsStrictLocalMin h)).card = 1) :
    (neighbors h).ncard = 2 ∨ (neighbors h).ncard = 3 ∨ (neighbors h).ncard = 5 := by
  have hh' : IsHeight (fun v => -h v) := isHeight_neg hh
  have hmaxh' : (Finset.univ.filter (IsStrictLocalMax (fun v => -h v))).card = 1 := by
    rw [← hmin1]; congr 1; apply Finset.filter_congr
    intro v _; rw [strictMax_neg_iff]
  have hres := card_max_one_degree_mem hm hn hh' hmaxh'
  rwa [neighbors_neg] at hres

/-- **Theorem 3.4 (degree-4 characterization).** For `m,n ≥ 2`, a height function
has degree 4 iff it has exactly two strict local maxima and two strict local
minima. -/
theorem degree_four_iff (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) :
    (neighbors h).ncard = 4
      ↔ (Finset.univ.filter (IsStrictLocalMax h)).card = 2
        ∧ (Finset.univ.filter (IsStrictLocalMin h)).card = 2 := by
  have hmn : 2 ≤ m * n := le_trans (by norm_num) (Nat.mul_le_mul hm hn)
  have hm0 : 0 < m := by omega
  have hn0 : 0 < n := by omega
  haveI : Nonempty (Cell m n) := ⟨(⟨0, hm0⟩, ⟨0, hn0⟩)⟩
  have hsum : (neighbors h).ncard
      = (Finset.univ.filter (IsStrictLocalMax h)).card
        + (Finset.univ.filter (IsStrictLocalMin h)).card := by
    rw [degree_eq_extrema hh hmn, extrema_card_split hmn]
  obtain ⟨wm, hwm⟩ := exists_strictLocalMax hh
  obtain ⟨wn, hwn⟩ := exists_strictLocalMin hh
  have hmax_pos : 0 < (Finset.univ.filter (IsStrictLocalMax h)).card :=
    Finset.card_pos.mpr ⟨wm, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hwm⟩⟩
  have hmin_pos : 0 < (Finset.univ.filter (IsStrictLocalMin h)).card :=
    Finset.card_pos.mpr ⟨wn, Finset.mem_filter.mpr ⟨Finset.mem_univ _, hwn⟩⟩
  constructor
  · intro hdeg
    have hmax_ne1 : (Finset.univ.filter (IsStrictLocalMax h)).card ≠ 1 := by
      intro hc; rcases card_max_one_degree_mem hm hn hh hc with h2 | h3 | h5 <;> omega
    have hmin_ne1 : (Finset.univ.filter (IsStrictLocalMin h)).card ≠ 1 := by
      intro hc; rcases card_min_one_degree_mem hm hn hh hc with h2 | h3 | h5 <;> omega
    omega
  · rintro ⟨hmax2, hmin2⟩
    rw [hsum, hmax2, hmin2]

/-- **No degree-4 vertex is a cone** (or a dual cone).  A degree-4 vertex has two
strict local maxima and two strict local minima, so it is neither `coneC q C`
(unique maximum) nor `-coneC q C` (unique minimum). -/
theorem degree_four_not_cone (hm : 2 ≤ m) (hn : 2 ≤ n) {h : Cell m n → ℤ}
    (hh : IsHeight h) (hdeg : (neighbors h).ncard = 4) :
    (¬ ∃ (q : Cell m n) (C : ℤ), h = coneC q C)
      ∧ (¬ ∃ (q : Cell m n) (C : ℤ), h = (fun v => -(coneC q C) v)) := by
  obtain ⟨hmax2, hmin2⟩ := (degree_four_iff hm hn hh).mp hdeg
  constructor
  · rintro ⟨q, C, rfl⟩
    have hsingle : (Finset.univ.filter (IsStrictLocalMax (coneC q C))).card = 1 := by
      rw [show Finset.univ.filter (IsStrictLocalMax (coneC q C)) = {q} from ?_,
          Finset.card_singleton]
      ext v
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
      exact ⟨fun hv => coneC_unique_max q C v hv, fun he => he ▸ coneC_max_at q C⟩
    omega
  · rintro ⟨q, C, rfl⟩
    have hsingle :
        (Finset.univ.filter (IsStrictLocalMin (fun v => -(coneC q C) v))).card = 1 := by
      rw [show Finset.univ.filter (IsStrictLocalMin (fun v => -(coneC q C) v)) = {q} from ?_,
          Finset.card_singleton]
      ext v
      simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_singleton]
      rw [strictMin_neg_iff]
      exact ⟨fun hv => coneC_unique_max q C v hv, fun he => he ▸ coneC_max_at q C⟩
    omega

end OrigamiCone

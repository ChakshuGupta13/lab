import OrigamiCone.DegreeExtrema

/-!
# Lemma 2.1 in the rotation quotient: degree = #extrema for `mn ≥ 3`

`OrigamiCone.DegreeExtrema` proves the degree–extrema correspondence in the
*cover* — the height-flip graph, with adjacency `OFGAdj` (two height functions
differing at a single cell) and guard `mn ≥ 2`.  The paper's **Lemma 2.1** is
about the origami flip graph *itself*, the quotient
`OFG = R₃(G_{m,n}) / (ℤ/3ℤ)` of proper `3`-colourings modulo the global colour
rotation, and (after the boundary analysis) claims the range `mn ≥ 3`.  This
module kernel-checks that quotient statement, closing the gap between the
cover-level Lean theorem and the paper's quotient-level Lemma 2.1.

## The model

Via the Ginepro–Hull bijection `(eq:iso)` (origami ↔ proper `3`-colouring) and
the bipartite height lift, a proper `3`-colouring is `h mod 3` for a height
function `h`; two height functions give the same colouring iff they differ by a
constant in `3ℤ`, and the global colour rotation `c ↦ c+1` is the height shift
`h ↦ h+1`.  Composing the `3ℤ` lift-ambiguity with the order-`3` rotation gives
the full additive group `ℤ`, so

  `OFG  ≅  {height functions} / (+ℤ shift)`,

the model used here: an OFG vertex is a shift-class `mkV h = ⟦h⟧`, and two classes
are adjacent when some representatives are flip-adjacent.  The shift `h ↦ h+k` is
a graph automorphism of the cover (it preserves `IsHeight` and commutes with the
flip, `flipAt_add_const`), so this is the standard quotient graph.

The colouring side of the bijection (`eq:iso` itself) is the project-wide
disclosed interface, not reformalised.  What is proved here is the new `mn ≥ 3`
content: passing to the quotient neither **collapses** two distinct flips
(`flip_mkV_injOn`, the step that genuinely needs a third cell, `mn ≥ 3`) nor
creates a **self-loop** (`flip_mkV_ne`, `mn ≥ 2`).  Hence the cover-neighbourhood
maps bijectively onto the OFG-neighbourhood and the degree is unchanged.

Main result: `ofgDegree_eq_extrema` — for `mn ≥ 3` and `h` a height function, the
degree of `mkV h` in the quotient equals the number of strict local extrema of
`h`.  No `sorry`; `#print axioms` is the standard `propext / Classical.choice /
Quot.sound` triple.
-/

namespace OrigamiCone
namespace QuotientModel

variable {m n : ℕ}

-- ===========================================================================
-- Phase 1: the shift quotient, and shift-invariance of the cover structure
-- ===========================================================================

/-- Two height functions represent the same OFG vertex when they differ by a
global additive constant — the height-lift form of the colour rotation. -/
def ShiftEq (h h' : Cell m n → ℤ) : Prop := ∃ k : ℤ, h' = fun v => h v + k

lemma ShiftEq.rfl' (h : Cell m n → ℤ) : ShiftEq h h := ⟨0, by funext v; ring⟩

lemma ShiftEq.symm {h h' : Cell m n → ℤ} : ShiftEq h h' → ShiftEq h' h := by
  rintro ⟨k, rfl⟩; exact ⟨-k, by funext v; ring⟩

lemma ShiftEq.trans {h h' h'' : Cell m n → ℤ} :
    ShiftEq h h' → ShiftEq h' h'' → ShiftEq h h'' := by
  rintro ⟨k, rfl⟩ ⟨l, rfl⟩; exact ⟨k + l, by funext v; ring⟩

instance shiftSetoid (m n : ℕ) : Setoid (Cell m n → ℤ) where
  r := ShiftEq
  iseqv := ⟨ShiftEq.rfl', ShiftEq.symm, ShiftEq.trans⟩

/-- An OFG vertex: a height function modulo global shift. -/
abbrev OFGVertex (m n : ℕ) := Quotient (shiftSetoid m n)

/-- The OFG vertex of a height function. -/
def mkV (h : Cell m n → ℤ) : OFGVertex m n := ⟦h⟧

lemma mkV_eq_iff {h h' : Cell m n → ℤ} : mkV h = mkV h' ↔ ShiftEq h h' :=
  Quotient.eq

/-- Two representatives differing by a constant give the same OFG vertex. -/
lemma mkV_add_const (h : Cell m n → ℤ) (k : ℤ) :
    mkV (fun v => h v + k) = mkV h :=
  Quotient.sound ⟨-k, by funext v; ring⟩

/-- Adding a constant preserves being a height function. -/
lemma isHeight_add_const {h : Cell m n → ℤ} (hh : IsHeight h) (k : ℤ) :
    IsHeight (fun v => h v + k) := by
  intro p q hpq
  have h1 := hh p q hpq
  show |(h p + k) - (h q + k)| = 1
  rw [show (h p + k) - (h q + k) = h p - h q by ring]; exact h1

/-- Shift-invariance of strict local maxima. -/
lemma strictMax_add_const {h : Cell m n → ℤ} {k : ℤ} {v : Cell m n} :
    IsStrictLocalMax (fun w => h w + k) v ↔ IsStrictLocalMax h v := by
  unfold IsStrictLocalMax
  constructor
  · intro hmax u hu; have := hmax u hu; simp only at this; omega
  · intro hmax u hu; have := hmax u hu; simp only; omega

/-- Shift-invariance of strict local minima. -/
lemma strictMin_add_const {h : Cell m n → ℤ} {k : ℤ} {v : Cell m n} :
    IsStrictLocalMin (fun w => h w + k) v ↔ IsStrictLocalMin h v := by
  unfold IsStrictLocalMin
  constructor
  · intro hmin u hu; have := hmin u hu; simp only at this; omega
  · intro hmin u hu; have := hmin u hu; simp only; omega

/-- Shift-invariance of strict local extrema. -/
lemma strictExtremum_add_const {h : Cell m n → ℤ} {k : ℤ} {v : Cell m n} :
    IsStrictLocalExtremum (fun w => h w + k) v ↔ IsStrictLocalExtremum h v := by
  unfold IsStrictLocalExtremum
  rw [strictMax_add_const, strictMin_add_const]

/-- The flip commutes with the shift, pointwise (maximum part). -/
lemma flipMax_add_const {h : Cell m n → ℤ} (k : ℤ) (v w : Cell m n) :
    flipMax (fun u => h u + k) v w = flipMax h v w + k := by
  rcases eq_or_ne w v with hw | hw
  · subst hw; simp only [flipMax_apply_self]; ring
  · rw [flipMax_apply_ne hw, flipMax_apply_ne hw]

/-- The flip commutes with the shift, pointwise (minimum part). -/
lemma flipMin_add_const {h : Cell m n → ℤ} (k : ℤ) (v w : Cell m n) :
    flipMin (fun u => h u + k) v w = flipMin h v w + k := by
  rcases eq_or_ne w v with hw | hw
  · subst hw; simp only [flipMin_apply_self]; ring
  · rw [flipMin_apply_ne hw, flipMin_apply_ne hw]

/-- **Flip–shift commutation.** Flipping `h+k` at `v` is flipping `h` at `v`,
then shifting: the shift is an automorphism of the flip graph. -/
lemma flipAt_add_const {h : Cell m n → ℤ} (k : ℤ) (v : Cell m n) :
    flipAt (fun u => h u + k) v = fun w => flipAt h v w + k := by
  unfold flipAt
  by_cases hm : IsStrictLocalMax h v
  · rw [if_pos (strictMax_add_const.mpr hm), if_pos hm]
    funext w; exact flipMax_add_const k v w
  · rw [if_neg (fun hc => hm (strictMax_add_const.mp hc)), if_neg hm]
    funext w; exact flipMin_add_const k v w

-- ===========================================================================
-- The quotient graph: adjacency and degree
-- ===========================================================================

/-- OFG adjacency on the quotient: distinct vertices with flip-adjacent
representatives. -/
def IsOFGNeighbor (C D : OFGVertex m n) : Prop :=
  C ≠ D ∧ ∃ g g', mkV g = C ∧ mkV g' = D ∧ OFGAdj g g'

/-- The OFG neighbour set of a vertex. -/
def OFGNeighbors (C : OFGVertex m n) : Set (OFGVertex m n) :=
  {D | IsOFGNeighbor C D}

/-- The OFG degree of a vertex: the number of its neighbours. -/
noncomputable def OFGDegree (C : OFGVertex m n) : ℕ := (OFGNeighbors C).ncard

-- ===========================================================================
-- Phase 2: no self-loop (mn ≥ 2) and no collapse (mn ≥ 3) — the crux
-- ===========================================================================

/-- **No self-loop** (`mn ≥ 2`). A flip never returns to the same OFG vertex:
a flip changes one cell by `±2`, a shift changes every cell equally, and with a
second cell present the two cannot agree. -/
lemma flip_mkV_ne {h : Cell m n → ℤ} (hmn : 2 ≤ m * n) {v : Cell m n}
    (hv : IsStrictLocalExtremum h v) : mkV (flipAt h v) ≠ mkV h := by
  rw [Ne, mkV_eq_iff]
  rintro ⟨k, hk⟩
  -- hk : h = fun w => flipAt h v w + k
  obtain ⟨u, hu⟩ := exists_neighbor hmn v
  have hune : u ≠ v := by rintro rfl; simp [adj, gdist_self] at hu
  have hku : h u = flipAt h v u + k := congrFun hk u
  rw [flipAt_apply_ne hune] at hku
  have hk0 : k = 0 := by omega
  have hkv : h v = flipAt h v v + k := congrFun hk v
  have heq : flipAt h v v = h v := by omega
  exact flipAt_self_ne hmn hv heq

/-- A third cell exists once the grid has at least three cells. -/
lemma exists_third (hmn : 3 ≤ m * n) (v v' : Cell m n) :
    ∃ w, w ≠ v ∧ w ≠ v' := by
  by_contra hcon
  push_neg at hcon
  have hsub : (Finset.univ : Finset (Cell m n)) ⊆ {v, v'} := by
    intro w _
    rcases eq_or_ne w v with hw | hw
    · simp [hw]
    · simp [hcon w hw]
  have h1 := Finset.card_le_card hsub
  have h2 : ({v, v'} : Finset (Cell m n)).card ≤ 2 :=
    le_trans (Finset.card_insert_le _ _) (by simp)
  have h3 : (Finset.univ : Finset (Cell m n)).card = m * n := by
    rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
  omega

/-- **No collapse** (`mn ≥ 3`). Distinct flips land in distinct OFG vertices:
the map `v ↦ mkV (flipAt h v)` is injective on the strict local extrema.  This is
the step that genuinely needs a third cell — at `M_{1,2}` (`mn = 2`) the two
flips of a single edge collapse to one neighbour. -/
lemma flip_mkV_injOn {h : Cell m n → ℤ} (hmn : 3 ≤ m * n) :
    Set.InjOn (fun v => mkV (flipAt h v))
      ↑(Finset.univ.filter (IsStrictLocalExtremum h)) := by
  intro v hv v' _ heq
  have hext : IsStrictLocalExtremum h v := by
    have := Finset.mem_coe.mp hv; exact (Finset.mem_filter.mp this).2
  by_contra hne
  -- heq : mkV (flipAt h v) = mkV (flipAt h v')
  rw [mkV_eq_iff] at heq
  obtain ⟨k, hk⟩ := heq
  -- hk : flipAt h v' = fun w => flipAt h v w + k
  obtain ⟨w, hwv, hwv'⟩ := exists_third hmn v v'
  have e1 : flipAt h v w = h w := flipAt_apply_ne hwv
  have e2 : flipAt h v' w = h w := flipAt_apply_ne hwv'
  have hkw : flipAt h v' w = flipAt h v w + k := congrFun hk w
  rw [e1, e2] at hkw
  have hk0 : k = 0 := by omega
  have eqfun : flipAt h v' = flipAt h v := by
    rw [hk]; funext u; rw [hk0]; ring
  have ev1 : flipAt h v' v = h v := flipAt_apply_ne hne
  rw [eqfun] at ev1
  exact flipAt_self_ne (by omega) hext ev1

-- ===========================================================================
-- Phase 3: the OFG-neighbourhood is the flipped extrema, and the count
-- ===========================================================================

/-- The OFG neighbourhood of `mkV h` is exactly the image of `h`'s strict local
extrema under `v ↦ mkV (flipAt h v)`. -/
lemma ofgNeighbors_eq_image {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 3 ≤ m * n) :
    OFGNeighbors (mkV h)
      = (fun v => mkV (flipAt h v)) ''
          ↑(Finset.univ.filter (IsStrictLocalExtremum h)) := by
  have hmn' : 2 ≤ m * n := by omega
  ext D
  simp only [OFGNeighbors, Set.mem_setOf_eq, IsOFGNeighbor, Set.mem_image,
             Finset.coe_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨_, g, g', hg, hg', hadj⟩
    -- g ~ h, so g = h + (-k) for some k
    rw [mkV_eq_iff] at hg
    obtain ⟨k, hk⟩ := hg
    -- hk : h = fun v => g v + k
    have hg_eq : g = fun v => h v + (-k) := by
      funext v; have := congrFun hk v; simp only at this; omega
    have hgH : IsHeight g := by rw [hg_eq]; exact isHeight_add_const hh (-k)
    obtain ⟨w, hwext, hflip⟩ := neighbor_is_flipAt hmn' hgH hadj
    refine ⟨w, ?_, ?_⟩
    · -- w is a strict local extremum of h
      rw [hg_eq] at hwext; exact strictExtremum_add_const.mp hwext
    · -- mkV (flipAt h w) = D
      rw [← hg', ← hflip, hg_eq, flipAt_add_const, mkV_add_const]
  · rintro ⟨v, hvext, rfl⟩
    exact ⟨(flip_mkV_ne hmn' hvext).symm, h, flipAt h v, rfl, rfl,
           ofgAdj_flipAt hmn' hh hvext⟩

/-- **Lemma 2.1 in the quotient (`mn ≥ 3`).** For a height function `h`, the
degree of the OFG vertex `mkV h` equals the number of strict local extrema of
`h`.  This is the paper's degree–extrema correspondence for the origami flip
graph itself (the rotation quotient), with its honest range `mn ≥ 3`. -/
theorem ofgDegree_eq_extrema {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 3 ≤ m * n) :
    OFGDegree (mkV h)
      = (Finset.univ.filter (IsStrictLocalExtremum h)).card := by
  unfold OFGDegree
  rw [ofgNeighbors_eq_image hh hmn, Set.ncard_image_of_injOn (flip_mkV_injOn hmn),
      Set.ncard_coe_finset]

end QuotientModel
end OrigamiCone

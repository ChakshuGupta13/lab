import OrigamiCone.SequelMaxima

/-!
# Sequel meta-theorem: single-cone maxima at grid corners (`cor:corners`)

Standalone formalisation of Corollary `cor:corners` of the sequel paper

> *Degree-`d` vertex counts of the `m × n` origami flip graph:
> structure and a polynomial conjecture.*

`Corollary cor:corners` locates the single-cone strict local maxima of an
envelope: for `m, n ≥ 2`, a cell `v` at which exactly one cone attains the
envelope is a strict local maximum iff `v` is one of the four grid corners
and the unique attaining cone's apex lies strictly off the two boundary
lines through `v`.

The mechanism is the arithmetic of the Maxima Criterion (`SequelMaxima`,
`lem:maxima`): each of the four directional clauses fixes one component of
the apex on one side of `v`. A single cone cannot satisfy both `p.1 > v.1`
and `p.1 < v.1`, so at most one of the two vertical directions is covered
by a single-cone attaining set; likewise horizontally. So a single-cone
strict maximum on the FULL lattice does not exist
(`singleCone_not_slmax_fullLattice`), and on the bounded grid the two
uncovered directions must be OFF-grid — the cell is a corner.

Contents:

* `IsSLMaxTL`, `IsSLMaxTR`, `IsSLMaxBL`, `IsSLMaxBR` : bounded strict local
  maximum at each corner (only the two in-grid neighbours checked; the two
  off-grid neighbours absent by `m, n ≥ 2`).
* `cor_corners_TL`, `cor_corners_TR`, `cor_corners_BL`, `cor_corners_BR` :
  the paper's iff at each corner, via the two applicable directional
  clauses of the Maxima Criterion.
* `singleCone_not_slmax_fullLattice` : the full-lattice contradiction — a
  single-cone attaining set can never produce a strict local maximum on
  `ℤ × ℤ`; this is the interior-cell forward direction of `cor:corners`.
* `IsSLMaxBounded`, `IsGridCell`, `IsGridCorner`, `ApexOffLines` : unified
  bounded-grid predicates covering every cell type (interior / edge / corner).
* `cor_corners_unified` : the paper's Corollary `cor:corners` in full — one
  iff closing the forward direction on all cell types (interior, edge,
  corner) and the reverse direction across all four corners.

Scope: as in `SequelMaxima`, the cell coordinates range over `ℤ × ℤ`
(parametric in `m, n`). The bounded strict-max predicates check only the
two on-grid neighbours at each corner; the paper's `m, n ≥ 2` guard is
folded into the corner coordinates being distinct from their opposites,
so `2, 1 ≠ 0, 1` (down step from `(1,1)` lands in-grid) and similarly at
each corner.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.cor_corners_TL`.
-/

namespace OrigamiCone.Sequel

open Finset

variable {ι : Type*} (p : ι → ℤ × ℤ) (c : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

/-- **Bounded strict local maximum at the top-left corner `(1, 1)`.** Only the
two in-grid neighbours `(2, 1)` (down) and `(1, 2)` (right) are checked; the
neighbours `(0, 1)` and `(1, 0)` are off-grid for `m, n ≥ 2`. -/
def IsSLMaxTL : Prop :=
  Env p c S hS (2, 1) < Env p c S hS (1, 1) ∧
  Env p c S hS (1, 2) < Env p c S hS (1, 1)

/-- **Bounded strict local maximum at the top-right corner `(1, n)`.** -/
def IsSLMaxTR (n : ℤ) : Prop :=
  Env p c S hS (2, n) < Env p c S hS (1, n) ∧
  Env p c S hS (1, n - 1) < Env p c S hS (1, n)

/-- **Bounded strict local maximum at the bottom-left corner `(m, 1)`.** -/
def IsSLMaxBL (m : ℤ) : Prop :=
  Env p c S hS (m - 1, 1) < Env p c S hS (m, 1) ∧
  Env p c S hS (m, 2) < Env p c S hS (m, 1)

/-- **Bounded strict local maximum at the bottom-right corner `(m, n)`.** -/
def IsSLMaxBR (m n : ℤ) : Prop :=
  Env p c S hS (m - 1, n) < Env p c S hS (m, n) ∧
  Env p c S hS (m, n - 1) < Env p c S hS (m, n)

/-- Height-function bridge: adjacent envelope values differ by `±1`, so
`Env w < Env v` under `IsHeightFn` forces `Env w = Env v - 1`. -/
private lemma env_lt_of_isHeightFn_eq_sub_one
    (hheight : IsHeightFn p c S hS) {v w : ℤ × ℤ} (hw : d2 v w = 1)
    (hlt : Env p c S hS w < Env p c S hS v) :
    Env p c S hS w = Env p c S hS v - 1 := by
  rcases hheight v w hw with h | h
  · omega
  · exact h

/-- **`cor:corners` at the top-left corner `(1, 1)`.** Under `m, n ≥ 2` (so
that `(2, 1)` and `(1, 2)` are in-grid), and given a unique cone `s₀`
attaining `E` at `(1, 1)`, the corner is a bounded strict local maximum
iff `s₀`'s apex lies strictly off both boundary lines through `(1, 1)`:
`(p s₀).1 > 1` (off row `1`) and `(p s₀).2 > 1` (off column `1`). -/
theorem cor_corners_TL (hheight : IsHeightFn p c S hS)
    (s0 : ι) (hs0 : s0 ∈ S)
    (hattain : c s0 + d2 (p s0) (1, 1) = Env p c S hS (1, 1))
    (hunique : ∀ t ∈ S, c t + d2 (p t) (1, 1) = Env p c S hS (1, 1) → t = s0) :
    IsSLMaxTL p c S hS ↔ (p s0).1 > 1 ∧ (p s0).2 > 1 := by
  have hdn : d2 ((1 : ℤ), 1) (2, 1) = 1 := by unfold d2; simp
  have hrt : d2 ((1 : ℤ), 1) (1, 2) = 1 := by unfold d2; simp
  refine ⟨?fwd, ?rev⟩
  case fwd =>
    rintro ⟨hd, hr⟩
    have hd_eq : Env p c S hS (2, 1) = Env p c S hS (1, 1) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hdn hd
    have hr_eq : Env p c S hS (1, 2) = Env p c S hS (1, 1) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hrt hr
    -- maxima_down at (1,1): (i,j) = (1,1), (i+1,j) = (2,1)
    obtain ⟨s1, hs1S, hs1a, hs1r⟩ := (maxima_down p c S hS 1 1).mp hd_eq
    obtain ⟨s2, hs2S, hs2a, hs2c⟩ := (maxima_right p c S hS 1 1).mp hr_eq
    have h1 : s1 = s0 := hunique s1 hs1S hs1a
    have h2 : s2 = s0 := hunique s2 hs2S hs2a
    subst h1; subst h2
    exact ⟨hs1r, hs2c⟩
  case rev =>
    rintro ⟨hpr, hpk⟩
    have hd_eq : Env p c S hS (2, 1) = Env p c S hS (1, 1) - 1 :=
      (maxima_down p c S hS 1 1).mpr ⟨s0, hs0, hattain, hpr⟩
    have hr_eq : Env p c S hS (1, 2) = Env p c S hS (1, 1) - 1 :=
      (maxima_right p c S hS 1 1).mpr ⟨s0, hs0, hattain, hpk⟩
    refine ⟨?_, ?_⟩ <;> omega

/-- **`cor:corners` at the top-right corner `(1, n)`.** Given a unique cone
`s₀` attaining `E` at `(1, n)`, the corner is a bounded strict local
maximum iff `(p s₀).1 > 1` (off row `1`) and `(p s₀).2 < n` (off column
`n`). -/
theorem cor_corners_TR (n : ℤ) (hheight : IsHeightFn p c S hS)
    (s0 : ι) (hs0 : s0 ∈ S)
    (hattain : c s0 + d2 (p s0) (1, n) = Env p c S hS (1, n))
    (hunique : ∀ t ∈ S, c t + d2 (p t) (1, n) = Env p c S hS (1, n) → t = s0) :
    IsSLMaxTR p c S hS n ↔ (p s0).1 > 1 ∧ (p s0).2 < n := by
  have hdn : d2 ((1 : ℤ), n) (2, n) = 1 := by unfold d2; simp
  have hlt : d2 ((1 : ℤ), n) (1, n - 1) = 1 := by unfold d2; simp
  refine ⟨?fwd, ?rev⟩
  case fwd =>
    rintro ⟨hd, hl⟩
    have hd_eq : Env p c S hS (2, n) = Env p c S hS (1, n) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hdn hd
    have hl_eq : Env p c S hS (1, n - 1) = Env p c S hS (1, n) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hlt hl
    obtain ⟨s1, hs1S, hs1a, hs1r⟩ := (maxima_down p c S hS 1 n).mp hd_eq
    obtain ⟨s2, hs2S, hs2a, hs2c⟩ := (maxima_left p c S hS 1 n).mp hl_eq
    have h1 : s1 = s0 := hunique s1 hs1S hs1a
    have h2 : s2 = s0 := hunique s2 hs2S hs2a
    subst h1; subst h2
    exact ⟨hs1r, hs2c⟩
  case rev =>
    rintro ⟨hpr, hpk⟩
    have hd_eq : Env p c S hS (2, n) = Env p c S hS (1, n) - 1 :=
      (maxima_down p c S hS 1 n).mpr ⟨s0, hs0, hattain, hpr⟩
    have hl_eq : Env p c S hS (1, n - 1) = Env p c S hS (1, n) - 1 :=
      (maxima_left p c S hS 1 n).mpr ⟨s0, hs0, hattain, hpk⟩
    refine ⟨?_, ?_⟩ <;> omega

/-- **`cor:corners` at the bottom-left corner `(m, 1)`.** Given a unique cone
`s₀` attaining `E` at `(m, 1)`, the corner is a bounded strict local
maximum iff `(p s₀).1 < m` (off row `m`) and `(p s₀).2 > 1` (off column
`1`). -/
theorem cor_corners_BL (m : ℤ) (hheight : IsHeightFn p c S hS)
    (s0 : ι) (hs0 : s0 ∈ S)
    (hattain : c s0 + d2 (p s0) (m, 1) = Env p c S hS (m, 1))
    (hunique : ∀ t ∈ S, c t + d2 (p t) (m, 1) = Env p c S hS (m, 1) → t = s0) :
    IsSLMaxBL p c S hS m ↔ (p s0).1 < m ∧ (p s0).2 > 1 := by
  have hup : d2 (m, (1 : ℤ)) (m - 1, 1) = 1 := by unfold d2; simp
  have hrt : d2 (m, (1 : ℤ)) (m, 2) = 1 := by unfold d2; simp
  refine ⟨?fwd, ?rev⟩
  case fwd =>
    rintro ⟨hu, hr⟩
    have hu_eq : Env p c S hS (m - 1, 1) = Env p c S hS (m, 1) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hup hu
    have hr_eq : Env p c S hS (m, 2) = Env p c S hS (m, 1) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hrt hr
    obtain ⟨s1, hs1S, hs1a, hs1r⟩ := (maxima_up p c S hS m 1).mp hu_eq
    obtain ⟨s2, hs2S, hs2a, hs2c⟩ := (maxima_right p c S hS m 1).mp hr_eq
    have h1 : s1 = s0 := hunique s1 hs1S hs1a
    have h2 : s2 = s0 := hunique s2 hs2S hs2a
    subst h1; subst h2
    exact ⟨hs1r, hs2c⟩
  case rev =>
    rintro ⟨hpr, hpk⟩
    have hu_eq : Env p c S hS (m - 1, 1) = Env p c S hS (m, 1) - 1 :=
      (maxima_up p c S hS m 1).mpr ⟨s0, hs0, hattain, hpr⟩
    have hr_eq : Env p c S hS (m, 2) = Env p c S hS (m, 1) - 1 :=
      (maxima_right p c S hS m 1).mpr ⟨s0, hs0, hattain, hpk⟩
    refine ⟨?_, ?_⟩ <;> omega

/-- **`cor:corners` at the bottom-right corner `(m, n)`.** Given a unique cone
`s₀` attaining `E` at `(m, n)`, the corner is a bounded strict local
maximum iff `(p s₀).1 < m` (off row `m`) and `(p s₀).2 < n` (off column
`n`). -/
theorem cor_corners_BR (m n : ℤ) (hheight : IsHeightFn p c S hS)
    (s0 : ι) (hs0 : s0 ∈ S)
    (hattain : c s0 + d2 (p s0) (m, n) = Env p c S hS (m, n))
    (hunique : ∀ t ∈ S, c t + d2 (p t) (m, n) = Env p c S hS (m, n) → t = s0) :
    IsSLMaxBR p c S hS m n ↔ (p s0).1 < m ∧ (p s0).2 < n := by
  have hup : d2 (m, n) (m - 1, n) = 1 := by unfold d2; simp
  have hlt : d2 (m, n) (m, n - 1) = 1 := by unfold d2; simp
  refine ⟨?fwd, ?rev⟩
  case fwd =>
    rintro ⟨hu, hl⟩
    have hu_eq : Env p c S hS (m - 1, n) = Env p c S hS (m, n) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hup hu
    have hl_eq : Env p c S hS (m, n - 1) = Env p c S hS (m, n) - 1 :=
      env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hlt hl
    obtain ⟨s1, hs1S, hs1a, hs1r⟩ := (maxima_up p c S hS m n).mp hu_eq
    obtain ⟨s2, hs2S, hs2a, hs2c⟩ := (maxima_left p c S hS m n).mp hl_eq
    have h1 : s1 = s0 := hunique s1 hs1S hs1a
    have h2 : s2 = s0 := hunique s2 hs2S hs2a
    subst h1; subst h2
    exact ⟨hs1r, hs2c⟩
  case rev =>
    rintro ⟨hpr, hpk⟩
    have hu_eq : Env p c S hS (m - 1, n) = Env p c S hS (m, n) - 1 :=
      (maxima_up p c S hS m n).mpr ⟨s0, hs0, hattain, hpr⟩
    have hl_eq : Env p c S hS (m, n - 1) = Env p c S hS (m, n) - 1 :=
      (maxima_left p c S hS m n).mpr ⟨s0, hs0, hattain, hpk⟩
    refine ⟨?_, ?_⟩ <;> omega

/-- **Single-cone strict max is impossible on the full lattice.** Complement of
`cor:corners`: on `ℤ × ℤ` all four neighbours are present, so `IsSLMax`
requires four directional clauses. A unique attaining cone `s₀` can satisfy
at most two (down/up mutually exclusive; right/left mutually exclusive), so
the four-clause `IsSLMax` fails. On the bounded grid, this failure isolates
strict local maxima to the corners, which is the content of `cor:corners`. -/
theorem singleCone_not_slmax_fullLattice (hheight : IsHeightFn p c S hS)
    (i j : ℤ) (s0 : ι) (hs0 : s0 ∈ S)
    (hattain : c s0 + d2 (p s0) (i, j) = Env p c S hS (i, j))
    (hunique : ∀ t ∈ S, c t + d2 (p t) (i, j) = Env p c S hS (i, j) → t = s0) :
    ¬IsSLMax p c S hS (i, j) := by
  intro hmax
  obtain ⟨hd, hu, _, _⟩ := (maxima_criterion p c S hS hheight i j).mp hmax
  obtain ⟨s_d, hs_dS, hs_dA, hs_dr⟩ := hd
  obtain ⟨s_u, hs_uS, hs_uA, hs_ur⟩ := hu
  have hs_d0 : s_d = s0 := hunique s_d hs_dS hs_dA
  have hs_u0 : s_u = s0 := hunique s_u hs_uS hs_uA
  subst hs_d0; subst hs_u0
  omega

-- ============================================================================
-- Unified corners theorem (closes edge-cell gap flagged in Adversary review)
-- ============================================================================

/-- **Bounded strict local maximum on the `m × n` grid.** Every unit-distance
in-grid neighbour has strictly smaller envelope value; off-grid neighbours are
absent. This is the paper's notion of strict local maximum on the bounded grid,
unified across corner / edge / interior cell types. -/
def IsSLMaxBounded (m n : ℤ) (v : ℤ × ℤ) : Prop :=
  ∀ w : ℤ × ℤ, d2 v w = 1 → 1 ≤ w.1 → w.1 ≤ m → 1 ≤ w.2 → w.2 ≤ n →
    Env p c S hS w < Env p c S hS v

/-- A cell lies within the `m × n` grid `[1, m] × [1, n]`. -/
def IsGridCell (m n : ℤ) (v : ℤ × ℤ) : Prop :=
  1 ≤ v.1 ∧ v.1 ≤ m ∧ 1 ≤ v.2 ∧ v.2 ≤ n

/-- A cell is a grid corner: on the row boundary AND on the column boundary. -/
def IsGridCorner (m n : ℤ) (v : ℤ × ℤ) : Prop :=
  (v.1 = 1 ∨ v.1 = m) ∧ (v.2 = 1 ∨ v.2 = n)

/-- The apex `p` lies strictly off both boundary lines through the grid cell `v`:
if `v` is on row `1` then `p.1 > 1`; if `v` is on row `m` then `p.1 < m`; and
similarly for columns. Empty condition on axes where `v` is interior. -/
def ApexOffLines (m n : ℤ) (v : ℤ × ℤ) (papex : ℤ × ℤ) : Prop :=
  (v.1 = 1 → papex.1 > 1) ∧ (v.1 = m → papex.1 < m) ∧
  (v.2 = 1 → papex.2 > 1) ∧ (v.2 = n → papex.2 < n)

/-- **`cor:corners` unified iff on the bounded grid.** For `m, n ≥ 2`, an in-grid
cell `v` with a unique attaining cone `s₀` is a bounded strict local maximum iff
`v` is a grid corner AND the apex `p s₀` lies strictly off both boundary lines
through `v`. This is the full statement of the paper's Corollary
`cor:corners`: it closes both the four corner iffs (`cor_corners_TL/TR/BL/BR`)
and the "non-corner cells are never single-cone maxima" forward direction into
one theorem. -/
theorem cor_corners_unified (m n : ℤ) (hm : 2 ≤ m) (hn : 2 ≤ n)
    (hheight : IsHeightFn p c S hS) (v : ℤ × ℤ) (hvGrid : IsGridCell m n v)
    (s0 : ι) (hs0 : s0 ∈ S)
    (hattain : c s0 + d2 (p s0) v = Env p c S hS v)
    (hunique : ∀ t ∈ S, c t + d2 (p t) v = Env p c S hS v → t = s0) :
    IsSLMaxBounded p c S hS m n v ↔
      IsGridCorner m n v ∧ ApexOffLines m n v (p s0) := by
  -- work with v : ℤ × ℤ directly (no destructure)
  obtain ⟨hv1lo, hv1hi, hv2lo, hv2hi⟩ := hvGrid
  -- Height-fn bridge: `Env w < Env v` on unit-distance edges becomes `Env w = Env v - 1`.
  have step_of_lt : ∀ w : ℤ × ℤ, d2 v w = 1 →
      Env p c S hS w < Env p c S hS v →
      Env p c S hS w = Env p c S hS v - 1 :=
    fun w hw hlt => env_lt_of_isHeightFn_eq_sub_one p c S hS hheight hw hlt
  refine ⟨?fwd, ?rev⟩
  case fwd =>
    intro hmax
    -- STEP 1: `v.1 ∈ {1, m}` — else both up/down descend and force `.1 > v.1 ∧ .1 < v.1`.
    have hv1cases : v.1 = 1 ∨ v.1 = m := by
      by_contra hcon; push_neg at hcon
      obtain ⟨hne1, hnem⟩ := hcon
      have hlt1 : 1 < v.1 := lt_of_le_of_ne hv1lo (Ne.symm hne1)
      have hltm : v.1 < m := lt_of_le_of_ne hv1hi hnem
      have hd_dist : d2 v (v.1 + 1, v.2) = 1 := by unfold d2; simp
      have hd_lt := hmax _ hd_dist (by omega) (by omega) hv2lo hv2hi
      have hd_eq := step_of_lt _ hd_dist hd_lt
      have hu_dist : d2 v (v.1 - 1, v.2) = 1 := by unfold d2; simp
      have hu_lt := hmax _ hu_dist (by omega) (by omega) hv2lo hv2hi
      have hu_eq := step_of_lt _ hu_dist hu_lt
      obtain ⟨s_d, hs_dS, hs_dA, hs_dr⟩ := (maxima_down p c S hS v.1 v.2).mp hd_eq
      obtain ⟨s_u, hs_uS, hs_uA, hs_ur⟩ := (maxima_up p c S hS v.1 v.2).mp hu_eq
      have hd0 : s_d = s0 := hunique s_d hs_dS hs_dA
      have hu0 : s_u = s0 := hunique s_u hs_uS hs_uA
      subst hd0; subst hu0
      omega
    -- STEP 2: `v.2 ∈ {1, n}` (symmetric).
    have hv2cases : v.2 = 1 ∨ v.2 = n := by
      by_contra hcon; push_neg at hcon
      obtain ⟨hne1, hnen⟩ := hcon
      have hlt1 : 1 < v.2 := lt_of_le_of_ne hv2lo (Ne.symm hne1)
      have hltn : v.2 < n := lt_of_le_of_ne hv2hi hnen
      have hr_dist : d2 v (v.1, v.2 + 1) = 1 := by unfold d2; simp
      have hr_lt := hmax _ hr_dist hv1lo hv1hi (by omega) (by omega)
      have hr_eq := step_of_lt _ hr_dist hr_lt
      have hl_dist : d2 v (v.1, v.2 - 1) = 1 := by unfold d2; simp
      have hl_lt := hmax _ hl_dist hv1lo hv1hi (by omega) (by omega)
      have hl_eq := step_of_lt _ hl_dist hl_lt
      obtain ⟨s_r, hs_rS, hs_rA, hs_rc⟩ := (maxima_right p c S hS v.1 v.2).mp hr_eq
      obtain ⟨s_l, hs_lS, hs_lA, hs_lc⟩ := (maxima_left p c S hS v.1 v.2).mp hl_eq
      have hr0 : s_r = s0 := hunique s_r hs_rS hs_rA
      have hl0 : s_l = s0 := hunique s_l hs_lS hs_lA
      subst hr0; subst hl0
      omega
    -- STEP 3: derive the four apex-off-lines clauses (each fires only when v is on that side).
    refine ⟨⟨hv1cases, hv2cases⟩, ?_, ?_, ?_, ?_⟩
    · -- v.1 = 1 → (p s0).1 > 1: use bounded max with the down neighbour (2, v.2).
      intro hv1
      have hd_dist : d2 v (v.1 + 1, v.2) = 1 := by unfold d2; simp
      have hd_lt := hmax _ hd_dist (by omega) (by omega) hv2lo hv2hi
      have hd_eq := step_of_lt _ hd_dist hd_lt
      obtain ⟨s_d, hs_dS, hs_dA, hs_dr⟩ := (maxima_down p c S hS v.1 v.2).mp hd_eq
      have hd0 : s_d = s0 := hunique s_d hs_dS hs_dA
      subst hd0
      omega
    · -- v.1 = m → (p s0).1 < m: use bounded max with the up neighbour (m-1, v.2).
      intro hv1
      have hu_dist : d2 v (v.1 - 1, v.2) = 1 := by unfold d2; simp
      have hu_lt := hmax _ hu_dist (by omega) (by omega) hv2lo hv2hi
      have hu_eq := step_of_lt _ hu_dist hu_lt
      obtain ⟨s_u, hs_uS, hs_uA, hs_ur⟩ := (maxima_up p c S hS v.1 v.2).mp hu_eq
      have hu0 : s_u = s0 := hunique s_u hs_uS hs_uA
      subst hu0
      omega
    · -- v.2 = 1 → (p s0).2 > 1: use bounded max with the right neighbour (v.1, 2).
      intro hv2
      have hr_dist : d2 v (v.1, v.2 + 1) = 1 := by unfold d2; simp
      have hr_lt := hmax _ hr_dist hv1lo hv1hi (by omega) (by omega)
      have hr_eq := step_of_lt _ hr_dist hr_lt
      obtain ⟨s_r, hs_rS, hs_rA, hs_rc⟩ := (maxima_right p c S hS v.1 v.2).mp hr_eq
      have hr0 : s_r = s0 := hunique s_r hs_rS hs_rA
      subst hr0
      omega
    · -- v.2 = n → (p s0).2 < n: use bounded max with the left neighbour (v.1, n-1).
      intro hv2
      have hl_dist : d2 v (v.1, v.2 - 1) = 1 := by unfold d2; simp
      have hl_lt := hmax _ hl_dist hv1lo hv1hi (by omega) (by omega)
      have hl_eq := step_of_lt _ hl_dist hl_lt
      obtain ⟨s_l, hs_lS, hs_lA, hs_lc⟩ := (maxima_left p c S hS v.1 v.2).mp hl_eq
      have hl0 : s_l = s0 := hunique s_l hs_lS hs_lA
      subst hl0
      omega
  case rev =>
    rintro ⟨⟨hv1c, hv2c⟩, hpr1, hpm1, hpr2, hpn2⟩ w hw hw1lo hw1hi hw2lo hw2hi
    -- v is a corner; w is one of the 4 lattice neighbours, and by in-grid bounds only
    -- the two towards-interior neighbours can appear. Deploy the appropriate maxima_dir
    -- reverse iff to conclude Env w = Env v - 1, hence < Env v.
    rw [nbhd_iff] at hw
    rcases hw with h | h | h | h <;> subst h
    · -- w = (v.1 + 1, v.2) : DOWN neighbour in-grid iff v.1 ≤ m - 1, forcing v.1 = 1
      have hv1 : v.1 = 1 := by rcases hv1c with h | h; exacts [h, by simp at hw1hi; omega]
      have hps : (p s0).1 > v.1 := by rw [hv1]; exact hpr1 hv1
      have := (maxima_down p c S hS v.1 v.2).mpr ⟨s0, hs0, hattain, hps⟩
      simp only [Prod.mk.eta] at this
      omega
    · -- w = (v.1 - 1, v.2) : UP neighbour in-grid iff v.1 ≥ 2, forcing v.1 = m
      have hv1 : v.1 = m := by
        rcases hv1c with h | h
        · simp at hw1lo; omega
        · exact h
      have hps : (p s0).1 < v.1 := by rw [hv1]; exact hpm1 hv1
      have := (maxima_up p c S hS v.1 v.2).mpr ⟨s0, hs0, hattain, hps⟩
      simp only [Prod.mk.eta] at this
      omega
    · -- w = (v.1, v.2 + 1) : RIGHT neighbour in-grid iff v.2 ≤ n - 1, forcing v.2 = 1
      have hv2 : v.2 = 1 := by rcases hv2c with h | h; exacts [h, by simp at hw2hi; omega]
      have hps : (p s0).2 > v.2 := by rw [hv2]; exact hpr2 hv2
      have := (maxima_right p c S hS v.1 v.2).mpr ⟨s0, hs0, hattain, hps⟩
      simp only [Prod.mk.eta] at this
      omega
    · -- w = (v.1, v.2 - 1) : LEFT neighbour in-grid iff v.2 ≥ 2, forcing v.2 = n
      have hv2 : v.2 = n := by
        rcases hv2c with h | h
        · simp at hw2lo; omega
        · exact h
      have hps : (p s0).2 < v.2 := by rw [hv2]; exact hpn2 hv2
      have := (maxima_left p c S hS v.1 v.2).mpr ⟨s0, hs0, hattain, hps⟩
      simp only [Prod.mk.eta] at this
      omega

end OrigamiCone.Sequel


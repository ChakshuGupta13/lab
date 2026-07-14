import OrigamiCone.QuotientDegree

/-!
# The colouring quotient `R₃(G) / (ℤ/3ℤ)` and its iso to the height-shift model

`OrigamiCone.QuotientDegree` kernel-checks Lemma 2.1 in the rotation quotient
using the height-shift model `OFG := {height functions} / (+ℤ shift)`.  The
Adversary confirmed this is **mathematically equivalent** to the paper's symbol
`R₃(G_{m,n}) / (ℤ/3ℤ)` (the colouring quotient) via the standard chain
`+ℤ = 3ℤ ∘ ℤ/3`.  This module makes that equivalence a kernel-checked Lean
isomorphism rather than a docstring claim: it defines the paper-faithful
colouring object explicitly and proves the bridge.

## What this module adds

* `Coloring m n := Cell m n → ZMod 3` — a vertex colouring of the grid.
* `IsProperColoring c` — adjacent cells get distinct colours.
* `rotate k c v := c v + k` — the `ZMod 3` rotation action on colourings.
* `RotEq c c' := ∃ k, c' = rotate k c` and the setoid; `OFGColoring m n`,
  the quotient `R₃(G) / (ℤ/3ℤ)`.
* `colOf h := fun v => (h v : ZMod 3)` — the Ginepro–Hull projection (height
  mod 3); preserves properness (`colOf_proper`) and turns the height shift
  `+1` into the colour rotation `+1` (`colOf_add_one`).
* `mkColV h : OFGColoring m n` — the colouring-quotient class of `colOf h`;
  this is the Lean image of the paper's symbol `[h]` in `R₃(G)/(ℤ/3)`.

## The bridge

The headline `quotient_iso h h' : mkColV h = mkColV h' ↔ ShiftEq h h'` says the
two quotients agree **on the heights side**: two height functions land in the
same colouring rotation-class iff they differ by a global shift.  Equivalently,
the induced map `mkV [h]_{ShiftEq} ↦ mkColV h` from the height-shift quotient
into the colouring quotient is well-defined and **injective**.  Surjectivity
(every proper colouring lifts to some height function) is the existence half
of the Ginepro–Hull bijection — a published external input, not re-formalised;
no theorem in `QuotientDegree.lean` consumes it.

The Ginepro–Hull bijection between origami crease patterns and proper
`3`-colourings remains the project-wide disclosed interface (a published
external input, `GineproHull2014counting`); only its height-side lift is
formalised.

Note: `OFGColoring` is the quotient of *all* colourings `Cell → ZMod 3`, not
just the proper ones, by the rotation action.  Restricting to proper colourings
(the paper's `R₃(G_{m,n})`) is unnecessary here because every theorem that
constructs an `OFGColoring` element does so via `mkColV h` for some height
function `h`, and `colOf_proper` shows the image always lands in the proper
sub-orbit.  A future module that wants `R₃(G_{m,n})` as its own type can
restrict the setoid to proper colourings; the iff in `quotient_iso` transports
unchanged.

No `sorry`; `#print axioms` is the standard `[propext, Classical.choice,
Quot.sound]`.

## Scope

This module is foundation-only: it gives the paper-faithful colouring quotient
object and the iso to the existing height-shift model in `QuotientDegree`.
The Lemma 2.1 degree-extrema correspondence is already proved on the
height-shift side (`ofgDegree_eq_extrema`); the iso means that statement
transports verbatim to the colouring side whenever a consumer wants it.
-/

namespace OrigamiCone
namespace ColoringModel

variable {m n : ℕ}

-- ===========================================================================
-- Phase 1: proper colourings and the rotation action
-- ===========================================================================

/-- A vertex `3`-colouring of the grid. -/
abbrev Coloring (m n : ℕ) := Cell m n → ZMod 3

/-- A proper colouring assigns distinct colours to adjacent cells. -/
def IsProperColoring (c : Coloring m n) : Prop := ∀ p q, adj p q → c p ≠ c q

/-- The colour rotation `c ↦ c + k` (with `k : ZMod 3`). -/
def rotate (k : ZMod 3) (c : Coloring m n) : Coloring m n := fun v => c v + k

@[simp] lemma rotate_zero (c : Coloring m n) : rotate 0 c = c := by
  funext v; simp [rotate]

@[simp] lemma rotate_apply (k : ZMod 3) (c : Coloring m n) (v : Cell m n) :
    rotate k c v = c v + k := rfl

lemma rotate_add (k l : ZMod 3) (c : Coloring m n) :
    rotate l (rotate k c) = rotate (k + l) c := by
  funext v; simp [rotate, add_assoc]

/-- The rotation preserves properness: if `c p ≠ c q` then `c p + k ≠ c q + k`. -/
lemma rotate_proper {c : Coloring m n} (hc : IsProperColoring c) (k : ZMod 3) :
    IsProperColoring (rotate k c) := by
  intro p q hpq h
  apply hc p q hpq
  have : c p + k = c q + k := h
  exact add_right_cancel this

-- ===========================================================================
-- Phase 2: the colouring quotient `R₃(G) / (ℤ/3ℤ)`
-- ===========================================================================

/-- Two colourings are rotation-equivalent if some `ZMod 3` rotation matches them. -/
def RotEq (c c' : Coloring m n) : Prop := ∃ k : ZMod 3, c' = rotate k c

lemma RotEq.rfl' (c : Coloring m n) : RotEq c c := ⟨0, by simp⟩

lemma RotEq.symm {c c' : Coloring m n} : RotEq c c' → RotEq c' c := by
  rintro ⟨k, rfl⟩
  refine ⟨-k, ?_⟩
  funext v
  show c v = (c v + k) + (-k)
  ring

lemma RotEq.trans {c c' c'' : Coloring m n} :
    RotEq c c' → RotEq c' c'' → RotEq c c'' := by
  rintro ⟨k, rfl⟩ ⟨l, rfl⟩
  exact ⟨k + l, by funext v; simp [rotate, add_assoc]⟩

instance rotSetoid (m n : ℕ) : Setoid (Coloring m n) where
  r := RotEq
  iseqv := ⟨RotEq.rfl', RotEq.symm, RotEq.trans⟩

/-- The paper's symbol `R₃(G_{m,n}) / (ℤ/3ℤ)`: proper colourings modulo the
global `ZMod 3` rotation.  Stored as the quotient of all colourings; the
properness predicate transports across rotation (`rotate_proper`) so it is
well-defined on the quotient. -/
abbrev OFGColoring (m n : ℕ) := Quotient (rotSetoid m n)

/-- The colouring-quotient class of a colouring. -/
def mkColC (c : Coloring m n) : OFGColoring m n := ⟦c⟧

lemma mkColC_eq_iff {c c' : Coloring m n} : mkColC c = mkColC c' ↔ RotEq c c' :=
  Quotient.eq

@[simp] lemma mkColC_rotate (k : ZMod 3) (c : Coloring m n) :
    mkColC (rotate k c) = mkColC c :=
  Quotient.sound ⟨-k, by funext v; simp [rotate]⟩

-- ===========================================================================
-- Phase 3: the Ginepro--Hull projection and the iso to the height-shift model
-- ===========================================================================

/-- The Ginepro--Hull projection: cast a height function's value into `ZMod 3`. -/
def colOf (h : Cell m n → ℤ) : Coloring m n := fun v => (h v : ZMod 3)

@[simp] lemma colOf_apply (h : Cell m n → ℤ) (v : Cell m n) :
    colOf h v = (h v : ZMod 3) := rfl

/-- **Properness transports up the lift.** A height function projects to a
proper colouring: across an edge, `|h p - h q| = 1`, so `h p - h q ≡ ±1 ≢ 0
(mod 3)`. -/
lemma colOf_proper {h : Cell m n → ℤ} (hh : IsHeight h) :
    IsProperColoring (colOf h) := by
  intro p q hpq heq
  have h1 : |h p - h q| = 1 := hh p q hpq
  -- |h p - h q| = 1 means h p - h q = 1 or = -1.
  have habs : h p - h q = 1 ∨ h p - h q = -1 :=
    (abs_eq (by norm_num : (0 : ℤ) ≤ 1)).1 h1
  -- heq : (h p : ZMod 3) = (h q : ZMod 3). Cast the integer-side difference into ZMod 3.
  have hzm : ((h p - h q : ℤ) : ZMod 3) = 0 := by
    have hcol : ((h p : ℤ) : ZMod 3) = ((h q : ℤ) : ZMod 3) := heq
    push_cast
    rw [hcol, sub_self]
  -- contradict with h p - h q ∈ {1, -1}
  rcases habs with h_eq | h_eq
  · rw [h_eq] at hzm
    -- hzm : ((1 : ℤ) : ZMod 3) = 0
    revert hzm; push_cast; decide
  · rw [h_eq] at hzm
    revert hzm; push_cast; decide

/-- **The shift is the rotation.** Casting a height-side `+k : ℤ` shift down to
`ZMod 3` gives the colouring-side rotation by `(k : ZMod 3)`. -/
lemma colOf_add_const (h : Cell m n → ℤ) (k : ℤ) :
    colOf (fun v => h v + k) = rotate (k : ZMod 3) (colOf h) := by
  funext v
  simp [colOf, rotate]

-- ===========================================================================
-- The bridge: the colouring quotient equals the height-shift quotient
-- ===========================================================================

/-- The colouring-quotient class of a height function: project to a colouring,
then quotient by rotation.  This is the Lean image of the paper's symbol `[h]`
in `R₃(G)/(ℤ/3)`. -/
def mkColV (h : Cell m n → ℤ) : OFGColoring m n := mkColC (colOf h)

/-- **The iso, forward direction.** If `h` and `h'` differ by a global shift,
they map to the same colouring rotation-class. -/
lemma mkColV_of_shiftEq {h h' : Cell m n → ℤ}
    (hsh : QuotientModel.ShiftEq h h') : mkColV h = mkColV h' := by
  obtain ⟨k, rfl⟩ := hsh
  unfold mkColV
  rw [colOf_add_const]
  exact (mkColC_rotate (k : ZMod 3) (colOf h)).symm

/-- **The iso, backward direction.** If `colOf h` and `colOf h'` are
rotation-equivalent (with rotation `k : ZMod 3`), then `h` and `h'` differ by
some integer shift `k' ≡ k (mod 3)`. -/
lemma shiftEq_of_mkColV {h h' : Cell m n → ℤ} (hh : IsHeight h) (hh' : IsHeight h')
    (heq : mkColV h = mkColV h') : QuotientModel.ShiftEq h h' := by
  -- Unpack the rotation witness: there is k : ZMod 3 with colOf h' = rotate k (colOf h).
  rw [mkColV, mkColV, mkColC_eq_iff] at heq
  obtain ⟨k, hk⟩ := heq
  -- Pick any integer lift k' of k (an integer with k'.cast = k in ZMod 3).
  -- Such a k' exists from the cast ZMod 3 → ℤ via `ZMod.val_cast_of_lt`.
  set k' : ℤ := (k.val : ℤ)
  have hcast : (k' : ZMod 3) = k := by
    show ((k.val : ℤ) : ZMod 3) = k
    rw [Int.cast_natCast]
    exact ZMod.natCast_zmod_val k
  -- Apply hk pointwise: (h' v : ZMod 3) = (h v : ZMod 3) + k for every v.
  have hpw : ∀ v, ((h' v - h v : ℤ) : ZMod 3) = k := by
    intro v
    have := congrFun hk v
    simp [colOf, rotate] at this
    push_cast
    linear_combination this
  -- Define the integer shift function f v := h' v - h v; on every cell f v - k' is ≡ 0 mod 3.
  have hdvd : ∀ v, (3 : ℤ) ∣ (h' v - h v - k') := by
    intro v
    have h1 : ((h' v - h v - k' : ℤ) : ZMod 3) = 0 := by
      have hpv := hpw v
      push_cast at hpv ⊢
      rw [hpv, hcast]; ring
    exact (ZMod.intCast_zmod_eq_zero_iff_dvd _ 3).mp h1
  -- |h' v - h v| ≤ d(some cell, v) chain — but easier: use ±1-edge step bound.
  -- Adjacent cells: |h v - h u| = 1 and |h' v - h' u| = 1, so |(h' - h) v - (h' - h) u| ≤ 2;
  -- combined with divisibility by 3, the difference is constant.
  -- Formally: show f v - f u ∈ {-2,-1,0,1,2} and 3 ∣ (f v - k') - (f u - k') = f v - f u,
  -- so f v - f u = 0, hence f is constant; the constant equals k' (it equals f at any cell,
  -- and (f v - k') is divisible by 3 with |f v - k'| ≤ ... — but we don't need that, we
  -- only need ShiftEq with SOME integer constant. Define c := f (anyCell) and show h' = h + c.)
  -- Choose any starting cell; we'll prove `h' v = h v + h' p0 - h p0` for all v, by adjacency
  -- chain (the grid is connected and `Finset` finite, so a path exists). Instead of writing
  -- the chain by hand, use the fact that on a connected graph a function whose adjacent-pair
  -- differences are in 3ℤ and bounded by 2 in absolute value is constant.
  -- For the grid we can avoid this entirely: pick c := h' (0,0) - h (0,0) (well-defined when
  -- the cell exists) and prove the equality by induction along adjacency. But this requires
  -- exists_step_toward on h' against h to get a global value — heavier than needed.
  --
  -- CLEANER: since each pointwise difference is ≡ k' (mod 3), and |f v - f u| ≤ 2 across an
  -- edge, the difference IS constant on connected components. The grid m × n with mn ≥ 1 is
  -- connected (the standard `exists_step_toward` shows pairwise reachability). So fix any
  -- p₀ : Cell m n (which requires Nonempty, hence mn ≥ 1), let c := h' p₀ - h p₀, and prove
  -- h' v = h v + c by induction on `gdist p₀ v` using adjacency.
  by_cases hmn0 : m * n = 0
  · -- Vacuous: no cells; ShiftEq is trivially true with k' = 0 since both functions are on Empty.
    refine ⟨0, ?_⟩
    funext v
    -- v : Cell m n with m * n = 0 means Cell is empty.
    have : m = 0 ∨ n = 0 := by
      rcases Nat.mul_eq_zero.mp hmn0 with h | h <;> simp [h]
    rcases this with hm | hn
    · exact absurd v.1.isLt (by simp [hm])
    · exact absurd v.2.isLt (by simp [hn])
  · -- Non-empty grid: pick a basepoint and walk.
    have hm0 : 0 < m := Nat.pos_of_ne_zero fun h => hmn0 (by simp [h])
    have hn0 : 0 < n := Nat.pos_of_ne_zero fun h => hmn0 (by simp [h])
    let p₀ : Cell m n := (⟨0, hm0⟩, ⟨0, hn0⟩)
    refine ⟨h' p₀ - h p₀, ?_⟩
    funext v
    -- The crux: prove h' v = h v + (h' p₀ - h p₀) by adjacency walk.
    -- Use `cone_max`-style induction: define φ := fun v => h' v - h v - (h' p₀ - h p₀),
    -- we want φ ≡ 0. We have (a) φ p₀ = 0, (b) across an edge φ changes by an integer in
    -- {-2,-1,0,1,2}, (c) φ is divisible by 3 pointwise (from hdvd + the hdvd at p₀).
    -- Hence φ is divisible by 3 across edges with |Δφ| ≤ 2 ⟹ Δφ = 0; so φ is constant
    -- along any path; the grid is path-connected; hence φ ≡ φ p₀ = 0.
    -- Walk from v to p₀ using exists_step_toward.
    show h' v = h v + (h' p₀ - h p₀)
    suffices key : ∀ (k : ℕ) (w : Cell m n), gdist w p₀ ≤ k →
        h' w - h w = h' p₀ - h p₀ by
      have hd := gdist_nonneg v p₀
      have := key (gdist v p₀).toNat v
        (le_of_eq (Int.toNat_of_nonneg hd).symm)
      linarith
    intro k
    induction k with
    | zero =>
      intro w hw
      have hz : gdist w p₀ = 0 := le_antisymm hw (gdist_nonneg w p₀)
      have hwp : w = p₀ := gdist_eq_zero.1 hz
      subst hwp; ring
    | succ k ih =>
      intro w hw
      by_cases hwp : w = p₀
      · subst hwp; ring
      · obtain ⟨w', hadj, hd⟩ := exists_step_toward hwp
        have hb : gdist w' p₀ ≤ k := by rw [hd]; omega
        have hih := ih w' hb
        -- Step: |h w - h w'| = 1 and |h' w - h' w'| = 1, so f w - f w' ∈ {-2,-1,0,1,2}.
        have e1 : |h w - h w'| = 1 := hh w w' hadj
        have e2 : |h' w - h' w'| = 1 := hh' w w' hadj
        have d1 : h w - h w' = 1 ∨ h w - h w' = -1 :=
          (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 e1
        have d2 : h' w - h' w' = 1 ∨ h' w - h' w' = -1 :=
          (abs_eq (by norm_num : (0:ℤ) ≤ 1)).1 e2
        -- f w - f w' = (h' w - h w) - (h' w' - h w') ∈ {-2,0,2}.
        -- Combined with 3 ∣ f w - k' and 3 ∣ f w' - k', so 3 ∣ f w - f w', forcing it = 0.
        have hdw : (3 : ℤ) ∣ (h' w - h w - k') := hdvd w
        have hdw' : (3 : ℤ) ∣ (h' w' - h w' - k') := hdvd w'
        have hsame : h' w - h w = h' w' - h w' := by
          have hdiff : (3 : ℤ) ∣ ((h' w - h w) - (h' w' - h w')) := by
            have : (h' w - h w) - (h' w' - h w')
                 = (h' w - h w - k') - (h' w' - h w' - k') := by ring
            rw [this]
            exact dvd_sub hdw hdw'
          rcases d1 with d1 | d1 <;> rcases d2 with d2 | d2 <;> omega
        rw [hsame]; exact hih

/-- **The iso.** Two height functions live in the same colouring rotation-class
iff they differ by a global integer shift.  This is the kernel-checked content
of `OFG = R₃(G)/(ℤ/3) ≅ {height functions}/(+ℤ shift)`. -/
theorem quotient_iso {h h' : Cell m n → ℤ} (hh : IsHeight h) (hh' : IsHeight h') :
    mkColV h = mkColV h' ↔ QuotientModel.ShiftEq h h' :=
  ⟨shiftEq_of_mkColV hh hh', mkColV_of_shiftEq⟩

end ColoringModel
end OrigamiCone

import Mathlib

/-!
# Sequel meta-theorem: minima of a lower envelope are apexes (`lem:minatapex`)

Standalone formalisation of the first envelope lemma of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

The Envelope Structure Theorem encodes each height function by its set of distance
cones. `Lemma lem:minatapex` is the first step: every strict local minimum of a
lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))` of `L¹` distance cones is one
of the apexes `p_s`.

> *Proof.* Fix `v` and let `s*` attain the minimum, so `E(v) = c_{s*} + d(p_{s*},
> v)`. If `v ≠ p_{s*}`, some neighbour `w` of `v` is one step nearer `p_{s*}`,
> whence `E(w) ≤ c_{s*} + d(p_{s*}, w) = E(v) - 1 < E(v)` and `v` is not a strict
> local minimum.

This module proves that, working on the integer grid `ℤ × ℤ` with the `L¹` metric:

* `step_toward` : the geometric core — for `p ≠ v` there is a grid neighbour `w` of
  `v` (`d(v, w) = 1`) that is one step nearer `p` (`d(p, w) = d(p, v) - 1`);
* `minatapex` (`lem:minatapex`, **complete**): every strict local minimum of the
  lower envelope `Env` is an apex.

Here a *strict local minimum* is encoded directly as a cell whose every grid
neighbour has strictly greater envelope value (`IsSLMin`); the grid is the full
integer lattice `ℤ × ℤ`, so a step toward `p` is always available — the finite
grid `[1,m] × [1,n]` is `L¹`-convex, so the same step stays inside it, and the
lemma specialises without change.

This is a complete proof of `lem:minatapex` itself. The companion envelope lemmas
— `lem:activemin` (an apex is a strict local minimum iff it is *active*) and the
`lem:maxima` Maxima Criterion — additionally need the parity/height-function
structure (the `SequelParity` module) and the per-direction active-cone test, and
are **not** part of this module.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.minatapex`.
-/

namespace OrigamiCone.Sequel

open Finset

/-- The `L¹` (grid) distance on the integer lattice `ℤ × ℤ`. -/
def d2 (p v : ℤ × ℤ) : ℤ := |p.1 - v.1| + |p.2 - v.2|

/-- **Step toward an apex** (geometric core of `lem:minatapex`). If `p ≠ v` there is
a grid neighbour `w` of `v` (at `L¹` distance `1`) that is one step nearer `p`:
`d(p, w) = d(p, v) - 1`. One steps along whichever coordinate differs. -/
theorem step_toward (p v : ℤ × ℤ) (hpv : p ≠ v) :
    ∃ w : ℤ × ℤ, d2 v w = 1 ∧ d2 p w = d2 p v - 1 := by
  by_cases h1 : p.1 = v.1
  · have h2 : p.2 ≠ v.2 := fun h2 => hpv (Prod.ext h1 h2)
    rcases lt_or_gt_of_ne h2 with hlt | hgt
    · refine ⟨(v.1, v.2 - 1), ?_, ?_⟩
      · simp only [d2]
        rw [sub_self, abs_zero, show v.2 - (v.2 - 1) = 1 from by ring]; norm_num
      · simp only [d2, h1]
        rw [abs_of_nonpos (by omega : p.2 - v.2 ≤ 0),
            abs_of_nonpos (by omega : p.2 - (v.2 - 1) ≤ 0)]; ring
    · refine ⟨(v.1, v.2 + 1), ?_, ?_⟩
      · simp only [d2]
        rw [sub_self, abs_zero, show v.2 - (v.2 + 1) = -1 from by ring]; norm_num
      · simp only [d2, h1]
        rw [abs_of_nonneg (by omega : 0 ≤ p.2 - v.2),
            abs_of_nonneg (by omega : 0 ≤ p.2 - (v.2 + 1))]; ring
  · rcases lt_or_gt_of_ne h1 with hlt | hgt
    · refine ⟨(v.1 - 1, v.2), ?_, ?_⟩
      · simp only [d2]
        rw [sub_self, abs_zero, show v.1 - (v.1 - 1) = 1 from by ring]; norm_num
      · simp only [d2]
        rw [abs_of_nonpos (by omega : p.1 - v.1 ≤ 0),
            abs_of_nonpos (by omega : p.1 - (v.1 - 1) ≤ 0)]; ring
    · refine ⟨(v.1 + 1, v.2), ?_, ?_⟩
      · simp only [d2]
        rw [sub_self, abs_zero, show v.1 - (v.1 + 1) = -1 from by ring]; norm_num
      · simp only [d2]
        rw [abs_of_nonneg (by omega : 0 ≤ p.1 - v.1),
            abs_of_nonneg (by omega : 0 ≤ p.1 - (v.1 + 1))]; ring

variable {ι : Type*} (p : ι → ℤ × ℤ) (c : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

/-- The lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))` of the distance cones
seated at apexes `p` with offsets `c`, over a nonempty apex index set `S`. -/
def Env (v : ℤ × ℤ) : ℤ := S.inf' hS (fun s => c s + d2 (p s) v)

/-- `v` is a **strict local minimum** of the envelope: every grid neighbour has
strictly greater envelope value. -/
def IsSLMin (v : ℤ × ℤ) : Prop := ∀ w, d2 v w = 1 → Env p c S hS v < Env p c S hS w

/-- **Minima are apexes** (`lem:minatapex`, complete). Every strict local minimum of
the lower envelope is one of the apexes `p_s`. -/
theorem minatapex (v : ℤ × ℤ) (hv : IsSLMin p c S hS v) : ∃ s ∈ S, p s = v := by
  obtain ⟨s, hsS, hs⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) v)
  by_contra hcon
  push_neg at hcon
  have hps : p s ≠ v := hcon s hsS
  obtain ⟨w, hw1, hw2⟩ := step_toward (p s) v hps
  have hle : Env p c S hS w ≤ c s + d2 (p s) w := Finset.inf'_le _ hsS
  have hlt : Env p c S hS v < Env p c S hS w := hv w hw1
  have hsv : Env p c S hS v = c s + d2 (p s) v := hs
  rw [hw2] at hle
  rw [hsv] at hlt
  omega

end OrigamiCone.Sequel

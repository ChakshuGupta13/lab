import Mathlib

/-!
# Sequel meta-theorem: active apexes are exactly the minima (`lem:activemin`)

Standalone formalisation of the activity criterion of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:activemin` is the second envelope lemma: once the offsets satisfy the
parity condition (so the lower envelope `E := E_{A,c}` is a height function), an
apex `p_s` is a strict local minimum of `E` **iff** it is *active*, where activity
is the linear condition

> `c_s < c_t + d(p_t, p_s)`  for all `t ≠ s`.

This is what makes the strict-local-minimum set equal to the *active* apex set in
the Envelope Structure Theorem, encoding each degree-`d` vertex by `O(d)` apex
parameters subject to linear inequalities.

This module proves **both directions**, working on the integer lattice `ℤ × ℤ`
with the `L¹` metric:

* `d2_self`, `d2_triangle` : the metric facts (`d(p,p) = 0`; the `1`-Lipschitz
  bound `d(p,v) - d(v,w) ≤ d(p,w)`);
* `step_toward` : for `p ≠ v` a grid neighbour of `v` one step nearer `p`;
* `active_imp_slmin` (the `⟸` direction): if `p_s` is active then it is a strict
  local minimum — `E(p_s) = c_s`, every neighbour has `E ≥ c_s`, and the
  height-function property upgrades `≥` to `>`;
* `slmin_imp_active` (the `⟹` direction): if `p_s` is a strict local minimum then
  it is active — contrapositive: an inactive apex has a neighbour reached by
  stepping toward a violating (or envelope-attaining) cone, lowering the value;
* `activemin_iff` : the full biconditional `IsSLMin (p s) ↔ Active s`.

The two faithful hypotheses match the paper's setting exactly:

* `IsHeightFn` (the envelope is a height function: adjacent cells differ by `±1`) —
  the paper's standing assumption "Assume (PAR), so `E` is a height function",
  established by `lem:parityvalid` (its arithmetic core is the `SequelParity`
  module); taken here as a hypothesis;
* `Function.Injective p` (the apexes are *distinct* points) — the standard envelope
  setup where `A` is a finite set of distinct cells; needed only for the `⟹`
  direction (to step from `p_s` toward a distinct violating apex).

The grid is the full lattice `ℤ × ℤ`; as in `SequelEnvelope`, the bounded grid
`[1,m] × [1,n]` is `L¹`-convex so the criterion specialises without change.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.activemin_iff`.
-/

namespace OrigamiCone.Sequel

open Finset

/-- The `L¹` (grid) distance on the integer lattice `ℤ × ℤ`. -/
def d2 (p v : ℤ × ℤ) : ℤ := |p.1 - v.1| + |p.2 - v.2|

/-- Self-distance is zero. -/
theorem d2_self (p : ℤ × ℤ) : d2 p p = 0 := by simp [d2]

/-- **`L¹` triangle inequality** (the cone is `1`-Lipschitz): stepping by `d(v,w)`
cannot decrease the distance to `p` by more than `d(v,w)`. -/
theorem d2_triangle (p v w : ℤ × ℤ) : d2 p v - d2 v w ≤ d2 p w := by
  simp only [d2]
  have t1 := abs_sub_abs_le_abs_sub (p.1 - v.1) (p.1 - w.1)
  have t2 := abs_sub_abs_le_abs_sub (p.2 - v.2) (p.2 - w.2)
  rw [show (p.1 - v.1) - (p.1 - w.1) = w.1 - v.1 from by ring] at t1
  rw [show (p.2 - v.2) - (p.2 - w.2) = w.2 - v.2 from by ring] at t2
  have c1 : |w.1 - v.1| = |v.1 - w.1| := abs_sub_comm _ _
  have c2 : |w.2 - v.2| = |v.2 - w.2| := abs_sub_comm _ _
  omega

/-- **Step toward a point.** If `p ≠ v` there is a grid neighbour `w` of `v` (at
`L¹` distance `1`) one step nearer `p`. -/
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

/-- The lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))`. -/
def Env (v : ℤ × ℤ) : ℤ := S.inf' hS (fun s => c s + d2 (p s) v)

/-- `v` is a strict local minimum of the envelope. -/
def IsSLMin (v : ℤ × ℤ) : Prop := ∀ w, d2 v w = 1 → Env p c S hS v < Env p c S hS w

/-- Apex `s` is **active**: it is an apex (`s ∈ S`) whose offset strictly undercuts
every other cone evaluated at `p_s` — `c_s < c_t + d(p_t, p_s)` for `t ≠ s`. -/
def Active (s : ι) : Prop := s ∈ S ∧ ∀ t ∈ S, t ≠ s → c s < c t + d2 (p t) (p s)

/-- The envelope is a **height function**: adjacent grid cells differ by exactly one.
This is the paper's standing consequence of the parity condition `(PAR)`
(`lem:parityvalid`), taken here as a hypothesis. -/
def IsHeightFn : Prop :=
  ∀ v w, d2 v w = 1 → Env p c S hS w = Env p c S hS v + 1 ∨ Env p c S hS w = Env p c S hS v - 1

/-- **Active ⟹ strict local minimum** (the `⟸` direction of `lem:activemin`). If the
envelope is a height function and apex `s` is active, then `p_s` is a strict local
minimum. -/
theorem active_imp_slmin (hheight : IsHeightFn p c S hS)
    (s : ι) (hact : Active p c S s) : IsSLMin p c S hS (p s) := by
  obtain ⟨hsS, hactlt⟩ := hact
  classical
  have hEnvps : Env p c S hS (p s) = c s := by
    apply le_antisymm
    · have := Finset.inf'_le (fun t => c t + d2 (p t) (p s)) hsS
      simpa [Env, d2_self] using this
    · apply Finset.le_inf'
      intro t htS
      by_cases hts : t = s
      · subst hts; simp [d2_self]
      · exact le_of_lt (hactlt t htS hts)
  intro w hw
  rw [hEnvps]
  have hge : c s ≤ Env p c S hS w := by
    apply Finset.le_inf'
    intro t htS
    by_cases hts : t = s
    · subst hts; have : d2 (p t) w = 1 := hw; omega
    · have htri : d2 (p t) (p s) - d2 (p s) w ≤ d2 (p t) w := d2_triangle _ _ _
      have hact' := hactlt t htS hts
      rw [hw] at htri
      omega
  rcases hheight (p s) w hw with h | h
  · rw [hEnvps] at h; omega
  · rw [hEnvps] at h; omega

/-- **Strict local minimum ⟹ active** (the `⟹` direction of `lem:activemin`). If the
apexes are distinct and `p_s` is a strict local minimum, then apex `s` is active.
Contrapositive: an inactive apex has either a strictly lower envelope at `p_s` than
`c_s` (cone `s` is not attained — step toward the attaining apex) or a violating cone
`t` with `c_t + d(p_t, p_s) = c_s` (step toward `p_t`); either way some neighbour does
not exceed `E(p_s)`. -/
theorem slmin_imp_active (hinj : Function.Injective p)
    (s : ι) (hsS : s ∈ S) (hmin : IsSLMin p c S hS (p s)) : Active p c S s := by
  classical
  refine ⟨hsS, ?_⟩
  by_contra hcon
  push_neg at hcon
  obtain ⟨t, htS, hts, htle⟩ := hcon
  obtain ⟨u0, hu0S, hu0e⟩ := S.exists_mem_eq_inf' hS (fun t => c t + d2 (p t) (p s))
  have hEnvle : Env p c S hS (p s) ≤ c s := by
    have := Finset.inf'_le (fun t => c t + d2 (p t) (p s)) hsS
    simpa [Env, d2_self] using this
  have hEnveq : Env p c S hS (p s) = c u0 + d2 (p u0) (p s) := hu0e
  obtain ⟨u, huS, hus, hule⟩ :
      ∃ u ∈ S, u ≠ s ∧ c u + d2 (p u) (p s) ≤ Env p c S hS (p s) := by
    by_cases hEq : Env p c S hS (p s) = c s
    · exact ⟨t, htS, hts, by rw [hEq]; exact htle⟩
    · have hlt : Env p c S hS (p s) < c s := lt_of_le_of_ne hEnvle hEq
      refine ⟨u0, hu0S, ?_, le_of_eq hEnveq.symm⟩
      intro h
      rw [h, d2_self] at hEnveq
      omega
  have hpu : p u ≠ p s := fun h => hus (hinj h)
  obtain ⟨w, hw1, hw2⟩ := step_toward (p u) (p s) hpu
  have hwle : Env p c S hS w ≤ c u + d2 (p u) w := Finset.inf'_le _ huS
  rw [hw2] at hwle
  have := hmin w hw1
  omega

/-- **Activity criterion** (`lem:activemin`, full biconditional). For a height
-function envelope with distinct apexes, an apex is a strict local minimum iff it is
active. -/
theorem activemin_iff (hheight : IsHeightFn p c S hS) (hinj : Function.Injective p)
    (s : ι) (hsS : s ∈ S) : IsSLMin p c S hS (p s) ↔ Active p c S s :=
  ⟨fun hmin => slmin_imp_active p c S hS hinj s hsS hmin,
   fun hact => active_imp_slmin p c S hS hheight s hact⟩

end OrigamiCone.Sequel

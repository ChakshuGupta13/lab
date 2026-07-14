import Mathlib

/-!
# Sequel meta-theorem: the Envelope Structure Theorem, reverse direction (`thm:envelope`)

Standalone formalisation of the reverse direction of the Envelope Structure
Theorem of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Theorem thm:envelope` is the central bijection of the envelope encoding: height
functions with `a` strict local minima correspond bijectively to apex/offset pairs
`(A, c)` of size `a` satisfying the activity condition `(ACT)` and the parity
condition `(PAR)`. The two directions are:

> **Forward.** Given a height function `h` with minimum set `P`, take `A = P`,
> `c_p = h(p)`. The companion paper's Envelope Lemma gives `h = E_{A,c}`; activity
> and parity follow from `lem:activemin` and the height-function property.
>
> **Reverse.** Given `(A, c)` satisfying `(ACT)` and `(PAR)`, the envelope `E_{A,c}`
> is a height function (`lem:parityvalid`), its strict local minima are apexes
> (`lem:minatapex`), and every apex is one (`lem:activemin`); so its minimum set is
> exactly `A`.

This module proves the **reverse direction** end-to-end:

* `step_toward`, `d2_triangle`, `dgrid_parity` : the `L¹` geometric and parity
  facts (re-derived for self-containment);
* `env_lipschitz`, `cones_par`, `env_par`, `env_neq_edge`, `env_isHeightFn` : the
  height-function chain (`lem:parityvalid`);
* `minatapex` : strict local minima are apexes (`lem:minatapex`);
* `active_imp_slmin` : active apexes are strict local minima (the `⟸` half of
  `lem:activemin`);
* `envelope_structure_reverse` (`thm:envelope`, reverse direction, **complete**):
  given `(ACT)` and `(PAR)`, the strict-local-minimum set of `E_{A,c}` equals the
  apex image `{p s : s ∈ S}`;
* `slmin_card_eq` (the `|A| = a` cardinality clause): under apex distinctness, the
  strict-local-minimum set has the same cardinality as the apex index set.

The **forward direction** additionally requires the companion paper's Envelope
Lemma `eq:env` — that an arbitrary height function `h` with minimum set `P` agrees
with the lower envelope of cones seated at `P` with offsets `h(p)` — which the
sequel paper cites as an external input. Since `eq:env` is not a theorem of the
sequel paper, the forward direction is **not** formalised here; the reverse
direction proved here is the structural content the sequel contributes (with the
external `eq:env` providing surjectivity onto the codomain).

The grid is the full lattice `ℤ × ℤ`. The bounded grid `[1,m] × [1,n]` with its
`L¹`-convexity specialises without change.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.envelope_structure_reverse`.
-/

namespace OrigamiCone.Sequel

open scoped Int
open Finset

/-- The `L¹` (grid) distance on the integer lattice. -/
def d2 (p v : ℤ × ℤ) : ℤ := |p.1 - v.1| + |p.2 - v.2|

theorem d2_self (p : ℤ × ℤ) : d2 p p = 0 := by simp [d2]

theorem d2_comm (p v : ℤ × ℤ) : d2 p v = d2 v p := by simp [d2, abs_sub_comm]

/-- `L¹` triangle inequality. -/
theorem d2_triangle (p v w : ℤ × ℤ) : d2 p v - d2 v w ≤ d2 p w := by
  simp only [d2]
  have t1 := abs_sub_abs_le_abs_sub (p.1 - v.1) (p.1 - w.1)
  have t2 := abs_sub_abs_le_abs_sub (p.2 - v.2) (p.2 - w.2)
  rw [show (p.1 - v.1) - (p.1 - w.1) = w.1 - v.1 from by ring] at t1
  rw [show (p.2 - v.2) - (p.2 - w.2) = w.2 - v.2 from by ring] at t2
  have c1 : |w.1 - v.1| = |v.1 - w.1| := abs_sub_comm _ _
  have c2 : |w.2 - v.2| = |v.2 - w.2| := abs_sub_comm _ _
  omega

/-- Step toward a point at `L¹` distance one nearer. -/
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

/-- `d(p, v) ≡ (p.1 + p.2) + (v.1 + v.2) (mod 2)`. -/
theorem dgrid_parity (p v : ℤ × ℤ) :
    d2 p v ≡ (p.1 + p.2) + (v.1 + v.2) [ZMOD 2] := by
  unfold d2 Int.ModEq
  rcases abs_cases (p.1 - v.1) with ⟨e1, _⟩ | ⟨e1, _⟩ <;>
  rcases abs_cases (p.2 - v.2) with ⟨e2, _⟩ | ⟨e2, _⟩ <;>
  rw [e1, e2] <;> omega

variable {ι : Type*} (p : ι → ℤ × ℤ) (c : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

/-- The lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))`. -/
def Env (v : ℤ × ℤ) : ℤ := S.inf' hS (fun s => c s + d2 (p s) v)

/-- The activity condition `(ACT)`: `c_s < c_t + d(p_t, p_s)` for all `t ≠ s`. -/
def Active (s : ι) : Prop := s ∈ S ∧ ∀ t ∈ S, t ≠ s → c s < c t + d2 (p t) (p s)

/-- The parity condition `(PAR)`: `c_s - c_t ≡ d(p_s, p_t) (mod 2)`. -/
def Par : Prop := ∀ s ∈ S, ∀ t ∈ S, c s - c t ≡ d2 (p s) (p t) [ZMOD 2]

/-- `v` is a strict local minimum of the envelope. -/
def IsSLMin (v : ℤ × ℤ) : Prop := ∀ w, d2 v w = 1 → Env p c S hS v < Env p c S hS w

/-- The envelope is `1`-Lipschitz (`SequelHeight`). -/
theorem env_lipschitz (v w : ℤ × ℤ) :
    Env p c S hS v - Env p c S hS w ≤ d2 v w := by
  obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) w)
  have h1 : Env p c S hS v ≤ c s0 + d2 (p s0) v := Finset.inf'_le _ hs0
  have htri : d2 (p s0) v - d2 v w ≤ d2 (p s0) w := d2_triangle _ _ _
  have hw : Env p c S hS w = c s0 + d2 (p s0) w := hs0e
  omega

/-- Under `(PAR)` two cones agree mod `2` at every cell (`SequelHeight`). -/
theorem cones_par (hpar : Par p c S) (v : ℤ × ℤ) {s t : ι} (hs : s ∈ S) (ht : t ∈ S) :
    c s + d2 (p s) v ≡ c t + d2 (p t) v [ZMOD 2] := by
  have hps := dgrid_parity (p s) v
  have hpt := dgrid_parity (p t) v
  have hst := dgrid_parity (p s) (p t)
  have hpar' := hpar s hs t ht
  unfold Int.ModEq at *
  omega

/-- Under `(PAR)` the envelope shares the common cone parity. -/
theorem env_par (hpar : Par p c S) (v : ℤ × ℤ) {r : ι} (hr : r ∈ S) :
    Env p c S hS v ≡ c r + d2 (p r) v [ZMOD 2] := by
  obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) v)
  have hEnv : Env p c S hS v = c s0 + d2 (p s0) v := hs0e
  rw [hEnv]
  exact cones_par p c S hpar v hs0 hr

/-- Across an edge under `(PAR)`, the envelope parity flips. -/
theorem env_neq_edge (hpar : Par p c S) (v w : ℤ × ℤ) (hvw : d2 v w = 1) :
    Env p c S hS v ≠ Env p c S hS w := by
  obtain ⟨r, hr⟩ := id hS
  have h1 := env_par p c S hS hpar v hr
  have h2 := env_par p c S hS hpar w hr
  have hdv := dgrid_parity (p r) v
  have hdw := dgrid_parity (p r) w
  have hodd : ¬ ((v.1 + v.2) ≡ (w.1 + w.2) [ZMOD 2]) := by
    unfold d2 at hvw
    unfold Int.ModEq
    rcases abs_cases (v.1 - w.1) with ⟨e1, _⟩ | ⟨e1, _⟩ <;>
    rcases abs_cases (v.2 - w.2) with ⟨e2, _⟩ | ⟨e2, _⟩ <;> omega
  intro hEq
  apply hodd
  unfold Int.ModEq at *
  omega

/-- Under `(PAR)` the envelope is a **height function** (`lem:parityvalid`). -/
theorem env_isHeightFn (hpar : Par p c S) (v w : ℤ × ℤ) (hvw : d2 v w = 1) :
    Env p c S hS w = Env p c S hS v + 1 ∨ Env p c S hS w = Env p c S hS v - 1 := by
  have hlvw := env_lipschitz p c S hS v w
  have hlwv := env_lipschitz p c S hS w v
  rw [d2_comm w v] at hlwv
  rw [hvw] at hlvw hlwv
  have hne := env_neq_edge p c S hS hpar v w hvw
  omega

/-- **Minima are apexes** (`lem:minatapex`). Every strict local minimum of the
envelope is the seat of some apex. -/
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

/-- **Active apexes are strict local minima** (the `⟸` half of `lem:activemin`).
Under `(PAR)` (giving the height-function property) and activity, `p_s` is a strict
local minimum. -/
theorem active_imp_slmin (hpar : Par p c S) (s : ι) (hact : Active p c S s) :
    IsSLMin p c S hS (p s) := by
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
  rcases env_isHeightFn p c S hS hpar (p s) w hw with h | h
  · rw [hEnvps] at h; omega
  · rw [hEnvps] at h; omega

/-- **Envelope Structure Theorem, reverse direction** (`thm:envelope`, reverse,
**complete**). Given `(ACT)` for every apex and `(PAR)`, the strict-local-minimum
set of the envelope is exactly the apex image — the "under-the-bijection" content
of the structure theorem on the codomain side. The `|A| = a` cardinality clause is
additionally supplied by `slmin_card_eq` under apex distinctness. -/
theorem envelope_structure_reverse (hpar : Par p c S)
    (hAct : ∀ s ∈ S, Active p c S s) (v : ℤ × ℤ) :
    IsSLMin p c S hS v ↔ ∃ s ∈ S, p s = v := by
  refine ⟨minatapex p c S hS v, ?_⟩
  rintro ⟨s, hsS, hsv⟩
  rw [← hsv]
  exact active_imp_slmin p c S hS hpar s (hAct s hsS)

/-- **Minimum-set cardinality** (the `|A| = a` clause of `thm:envelope`). Under
`(ACT)`, `(PAR)`, and apex distinctness, the strict-local-minimum set has the same
cardinality as the apex index set — the apex map is the bijection. -/
theorem slmin_card_eq (hpar : Par p c S) (hinj : Function.Injective p)
    (hAct : ∀ s ∈ S, Active p c S s) :
    (S.image p).card = S.card ∧
      ∀ v ∈ S.image p, IsSLMin p c S hS v := by
  refine ⟨Finset.card_image_of_injective _ hinj, ?_⟩
  intro v hv
  rw [Finset.mem_image] at hv
  obtain ⟨s, hsS, rfl⟩ := hv
  exact active_imp_slmin p c S hS hpar s (hAct s hsS)

end OrigamiCone.Sequel

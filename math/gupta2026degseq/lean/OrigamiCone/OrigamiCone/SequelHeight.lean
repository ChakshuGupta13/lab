import Mathlib

/-!
# Sequel meta-theorem: a parity-valid envelope is a height function (`lem:parityvalid`)

Standalone formalisation of the height-function lemma of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:parityvalid` is what makes the envelope encoding well-typed: if the
offsets satisfy the parity condition `(PAR)` — `c_s - c_t ≡ d(p_s, p_t) (mod 2)`
for all `s, t` — then the lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))` is a
**height function**: adjacent grid cells differ by exactly one.

> *Proof.* `E_{A,c}` is a minimum of `1`-Lipschitz functions, hence `1`-Lipschitz.
> Since `d(p_s, (i,j)) ≡ (i+j) + (c_s + p_{s,1} + p_{s,2}) (mod 2)`, each cone has
> value `≡ (i+j) + (c_s + p_{s,1} + p_{s,2}) (mod 2)`; `(PAR)` makes the
> parenthesised quantity independent of `s`, so all cones share one parity at every
> cell. Adjacent cells differ in `i+j` by one, so the minimum differs across every
> edge; a `1`-Lipschitz integer map that is never equal across an edge differs by
> exactly one.

This module proves the **complete** lemma, working on the integer lattice `ℤ × ℤ`
with the `L¹` metric:

* `env_lipschitz` : the envelope is `1`-Lipschitz — `|E v - E w| ≤ d(v, w)` — as a
  minimum of the `1`-Lipschitz cones (via the triangle inequality `d2_triangle`);
* `dgrid_parity`, `cones_par`, `env_par` : under `(PAR)` every cone, and hence the
  envelope, has a single parity at each cell (`E v ≡ c_r + d(p_r, v) (mod 2)` for
  any reference apex `r`);
* `env_neq_edge` : across an edge the parity flips (the coordinate sum `(i+j)`
  changes by one), so `E v ≠ E w`;
* `env_isHeightFn` (`lem:parityvalid`, **complete**): combining the `1`-Lipschitz
  bound with the parity flip, `E w = E v + 1 ∨ E w = E v - 1` across every edge —
  `E_{A,c}` is a height function.

The conclusion `env_isHeightFn` is stated exactly in the shape of the `IsHeightFn`
hypothesis assumed by the companion module `SequelActive` (the activity criterion
`lem:activemin`): under `(PAR)` it supplies that hypothesis, closing the loop
`(PAR) ⇒ height function ⇒ (active ⇔ minimum)`. Both modules are self-contained
(`import Mathlib` only), so they carry textually identical copies of the `d2` and
`Env` definitions; the discharge is therefore by definitional identity of those
copies, not a Lean-enforced shared import. The parity arithmetic backbone is shared
with `SequelParity` and likewise re-derived here.

The grid is the full lattice `ℤ × ℤ`; the bounded grid `[1,m] × [1,n]` specialises
without change (a step that leaves the grid is irrelevant to the in-grid
height-function property).

No `sorry`; check with `#print axioms OrigamiCone.Sequel.env_isHeightFn`.
-/

namespace OrigamiCone.Sequel

open scoped Int
open Finset

/-- The `L¹` (grid) distance on the integer lattice `ℤ × ℤ`. -/
def d2 (p v : ℤ × ℤ) : ℤ := |p.1 - v.1| + |p.2 - v.2|

/-- **`L¹` triangle inequality** (each cone is `1`-Lipschitz). -/
theorem d2_triangle (p v w : ℤ × ℤ) : d2 p v - d2 v w ≤ d2 p w := by
  simp only [d2]
  have t1 := abs_sub_abs_le_abs_sub (p.1 - v.1) (p.1 - w.1)
  have t2 := abs_sub_abs_le_abs_sub (p.2 - v.2) (p.2 - w.2)
  rw [show (p.1 - v.1) - (p.1 - w.1) = w.1 - v.1 from by ring] at t1
  rw [show (p.2 - v.2) - (p.2 - w.2) = w.2 - v.2 from by ring] at t2
  have c1 : |w.1 - v.1| = |v.1 - w.1| := abs_sub_comm _ _
  have c2 : |w.2 - v.2| = |v.2 - w.2| := abs_sub_comm _ _
  omega

/-- The `L¹` distance is symmetric. -/
theorem d2_comm (p v : ℤ × ℤ) : d2 p v = d2 v p := by simp [d2, abs_sub_comm]

/-- **Cone parity.** `d(p, v) ≡ (p₁+p₂) + (v₁+v₂) (mod 2)`, since `|x| ≡ x`. -/
theorem dgrid_parity (p v : ℤ × ℤ) :
    d2 p v ≡ (p.1 + p.2) + (v.1 + v.2) [ZMOD 2] := by
  unfold d2 Int.ModEq
  rcases abs_cases (p.1 - v.1) with ⟨e1, _⟩ | ⟨e1, _⟩ <;>
  rcases abs_cases (p.2 - v.2) with ⟨e2, _⟩ | ⟨e2, _⟩ <;>
  rw [e1, e2] <;> omega

variable {ι : Type*} (p : ι → ℤ × ℤ) (c : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

/-- The lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))`. -/
def Env (v : ℤ × ℤ) : ℤ := S.inf' hS (fun s => c s + d2 (p s) v)

/-- The **parity condition** `(PAR)`: `c_s - c_t ≡ d(p_s, p_t) (mod 2)` for all
apexes `s, t ∈ S`. -/
def Par : Prop := ∀ s ∈ S, ∀ t ∈ S, c s - c t ≡ d2 (p s) (p t) [ZMOD 2]

/-- **Envelope is `1`-Lipschitz** (minimum of `1`-Lipschitz cones): `E v - E w ≤
d(v, w)`. The symmetric bound follows by swapping `v, w`. -/
theorem env_lipschitz (v w : ℤ × ℤ) :
    Env p c S hS v - Env p c S hS w ≤ d2 v w := by
  obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) w)
  have h1 : Env p c S hS v ≤ c s0 + d2 (p s0) v := Finset.inf'_le _ hs0
  have htri : d2 (p s0) v - d2 v w ≤ d2 (p s0) w := d2_triangle _ _ _
  have hw : Env p c S hS w = c s0 + d2 (p s0) w := hs0e
  omega

/-- **Cones share parity at a cell** (under `(PAR)`). -/
theorem cones_par (hpar : Par p c S) (v : ℤ × ℤ) {s t : ι} (hs : s ∈ S) (ht : t ∈ S) :
    c s + d2 (p s) v ≡ c t + d2 (p t) v [ZMOD 2] := by
  have hps := dgrid_parity (p s) v
  have hpt := dgrid_parity (p t) v
  have hst := dgrid_parity (p s) (p t)
  have hpar' := hpar s hs t ht
  unfold Int.ModEq at *
  omega

/-- **Envelope parity** (under `(PAR)`): `E v` shares the common cone parity, i.e.
`E v ≡ c_r + d(p_r, v) (mod 2)` for any reference apex `r ∈ S`. -/
theorem env_par (hpar : Par p c S) (v : ℤ × ℤ) {r : ι} (hr : r ∈ S) :
    Env p c S hS v ≡ c r + d2 (p r) v [ZMOD 2] := by
  obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) v)
  have hEnv : Env p c S hS v = c s0 + d2 (p s0) v := hs0e
  rw [hEnv]
  exact cones_par p c S hpar v hs0 hr

/-- **Parity flips across an edge** (under `(PAR)`): adjacent cells `v, w` with
`d(v, w) = 1` have envelopes of opposite parity, hence `E v ≠ E w`. -/
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

/-- **The envelope is a height function** (`lem:parityvalid`, complete). Under
`(PAR)`, adjacent grid cells differ by exactly one: `E w = E v + 1 ∨ E w = E v - 1`
whenever `d(v, w) = 1`. This matches the shape of the `IsHeightFn` hypothesis assumed
by the activity criterion (`SequelActive`), supplying it under `(PAR)` (the two
modules carry identical `d2`/`Env` copies, so the match is definitional). -/
theorem env_isHeightFn (hpar : Par p c S) (v w : ℤ × ℤ) (hvw : d2 v w = 1) :
    Env p c S hS w = Env p c S hS v + 1 ∨ Env p c S hS w = Env p c S hS v - 1 := by
  have hlvw := env_lipschitz p c S hS v w
  have hlwv := env_lipschitz p c S hS w v
  rw [d2_comm w v] at hlwv
  rw [hvw] at hlvw hlwv
  have hne := env_neq_edge p c S hS hpar v w hvw
  omega

end OrigamiCone.Sequel

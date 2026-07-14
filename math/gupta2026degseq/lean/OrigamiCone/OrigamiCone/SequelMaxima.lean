import Mathlib

/-!
# Sequel meta-theorem: the k-cone Maxima Criterion (`lem:maxima`)

Standalone formalisation of the Maxima Criterion of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:maxima` locates the strict local **maxima** of a lower envelope
`E := E_{A,c}`, generalising the two-cone ridge lemma of the companion paper to any
number of cones. With apexes `p_s = (r_s, s_s)`, a cell `v = (i, j)` is a strict
local maximum iff in each present neighbour direction some **active** cone (one
attaining `E(v)`) points back toward its apex:

> down (`i<m`): `∃ s, c_s + d(p_s,v) = E(v) ∧ r_s > i`;  up (`i>1`): `… r_s < i`;
> right (`j<n`): `… s_s > j`;  left (`j>1`): `… s_s < j`.

The mechanism is purely the behaviour of a minimum under a unit grid step: stepping
from `v` to a neighbour `w`, each cone changes by `±1` — `-1` if the step moves
toward its apex, `+1` otherwise — so the minimum drops by one (`E(w) = E(v) - 1`)
exactly when some envelope-attaining cone moves toward its apex.

This module proves the criterion, working on the integer lattice `ℤ × ℤ`:

* `maxima_down`, `maxima_up`, `maxima_right`, `maxima_left` : the four directional
  steps — `E` decreases in a given direction iff some active cone points back —
  each read off the minimum directly via the directional cone-distance identities;
* `nbhd_iff` : a cell `w` is a unit grid neighbour of `(i,j)` iff it is one of the
  four lattice neighbours;
* `maxima_criterion` (`lem:maxima`, **complete on the full lattice**): given that the
  envelope is a height function (the `(PAR)` consequence supplied by `SequelHeight`),
  `v` is a strict local maximum iff all four directional active-cone conditions hold.

Scope: the grid is the full lattice `ℤ × ℤ`, so all four neighbours are present and
the criterion is the conjunction of the four directional clauses. On the bounded grid
`[1,m] × [1,n]` the paper's "present neighbour direction" guards drop the clauses at
the boundary; that boundary bookkeeping is the only specialisation and is not
modelled here. The height-function property is taken as a hypothesis (`IsHeightFn`),
matching the paper's standing assumption and discharged under `(PAR)` by
`SequelHeight`.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.maxima_criterion`.
-/

namespace OrigamiCone.Sequel

open Finset

/-- The `L¹` (grid) distance on the integer lattice `ℤ × ℤ`. -/
def d2 (p v : ℤ × ℤ) : ℤ := |p.1 - v.1| + |p.2 - v.2|

/-- Downward step (`i → i+1`) toward an apex below (`p.1 > i`) decreases the cone. -/
theorem d2_down_lt (p : ℤ × ℤ) (i j : ℤ) (h : p.1 > i) :
    d2 p (i + 1, j) = d2 p (i, j) - 1 := by
  simp only [d2]
  rw [abs_of_nonneg (by omega : (0:ℤ) ≤ p.1 - i),
      abs_of_nonneg (by omega : (0:ℤ) ≤ p.1 - (i + 1))]; ring

/-- Downward step away from an apex (`p.1 ≤ i`) increases the cone. -/
theorem d2_down_ge (p : ℤ × ℤ) (i j : ℤ) (h : p.1 ≤ i) :
    d2 p (i + 1, j) = d2 p (i, j) + 1 := by
  simp only [d2]
  rw [abs_of_nonpos (by omega : p.1 - i ≤ 0),
      abs_of_nonpos (by omega : p.1 - (i + 1) ≤ 0)]; ring

/-- Upward step (`i → i-1`) toward an apex above (`p.1 < i`) decreases the cone. -/
theorem d2_up_lt (p : ℤ × ℤ) (i j : ℤ) (h : p.1 < i) :
    d2 p (i - 1, j) = d2 p (i, j) - 1 := by
  simp only [d2]
  rw [abs_of_nonpos (by omega : p.1 - i ≤ 0),
      abs_of_nonpos (by omega : p.1 - (i - 1) ≤ 0)]; ring

/-- Upward step away from an apex (`p.1 ≥ i`) increases the cone. -/
theorem d2_up_ge (p : ℤ × ℤ) (i j : ℤ) (h : p.1 ≥ i) :
    d2 p (i - 1, j) = d2 p (i, j) + 1 := by
  simp only [d2]
  rw [abs_of_nonneg (by omega : (0:ℤ) ≤ p.1 - i),
      abs_of_nonneg (by omega : (0:ℤ) ≤ p.1 - (i - 1))]; ring

/-- Rightward step (`j → j+1`) toward an apex (`p.2 > j`) decreases the cone. -/
theorem d2_right_lt (p : ℤ × ℤ) (i j : ℤ) (h : p.2 > j) :
    d2 p (i, j + 1) = d2 p (i, j) - 1 := by
  simp only [d2]
  rw [abs_of_nonneg (by omega : (0:ℤ) ≤ p.2 - j),
      abs_of_nonneg (by omega : (0:ℤ) ≤ p.2 - (j + 1))]; ring

/-- Rightward step away from an apex (`p.2 ≤ j`) increases the cone. -/
theorem d2_right_ge (p : ℤ × ℤ) (i j : ℤ) (h : p.2 ≤ j) :
    d2 p (i, j + 1) = d2 p (i, j) + 1 := by
  simp only [d2]
  rw [abs_of_nonpos (by omega : p.2 - j ≤ 0),
      abs_of_nonpos (by omega : p.2 - (j + 1) ≤ 0)]; ring

/-- Leftward step (`j → j-1`) toward an apex (`p.2 < j`) decreases the cone. -/
theorem d2_left_lt (p : ℤ × ℤ) (i j : ℤ) (h : p.2 < j) :
    d2 p (i, j - 1) = d2 p (i, j) - 1 := by
  simp only [d2]
  rw [abs_of_nonpos (by omega : p.2 - j ≤ 0),
      abs_of_nonpos (by omega : p.2 - (j - 1) ≤ 0)]; ring

/-- Leftward step away from an apex (`p.2 ≥ j`) increases the cone. -/
theorem d2_left_ge (p : ℤ × ℤ) (i j : ℤ) (h : p.2 ≥ j) :
    d2 p (i, j - 1) = d2 p (i, j) + 1 := by
  simp only [d2]
  rw [abs_of_nonneg (by omega : (0:ℤ) ≤ p.2 - j),
      abs_of_nonneg (by omega : (0:ℤ) ≤ p.2 - (j - 1))]; ring

/-- **Neighbourhood enumeration.** `w` is at `L¹` distance `1` from `(i,j)` iff it is
one of the four lattice neighbours. -/
theorem nbhd_iff (i j : ℤ) (w : ℤ × ℤ) :
    d2 (i, j) w = 1 ↔ w = (i + 1, j) ∨ w = (i - 1, j) ∨ w = (i, j + 1) ∨ w = (i, j - 1) := by
  constructor
  · intro h
    simp only [d2] at h
    rcases abs_cases (i - w.1) with ⟨e1, h1⟩ | ⟨e1, h1⟩ <;>
    rcases abs_cases (j - w.2) with ⟨e2, h2⟩ | ⟨e2, h2⟩ <;>
    · obtain ⟨w1, w2⟩ := w; simp_all; omega
  · rintro (h | h | h | h) <;> subst h <;> simp only [d2] <;> norm_num

variable {ι : Type*} (p : ι → ℤ × ℤ) (c : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

/-- The lower envelope `E_{A,c}(v) = min_s (c_s + d(p_s, v))`. -/
def Env (v : ℤ × ℤ) : ℤ := S.inf' hS (fun s => c s + d2 (p s) v)

/-- The envelope is a **height function**: adjacent grid cells differ by exactly one.
The paper's standing consequence of `(PAR)` (`lem:parityvalid`), supplied by
`SequelHeight`; taken here as a hypothesis. -/
def IsHeightFn : Prop :=
  ∀ v w, d2 v w = 1 → Env p c S hS w = Env p c S hS v + 1 ∨ Env p c S hS w = Env p c S hS v - 1

/-- `v` is a **strict local maximum** of the envelope: every grid neighbour has
strictly smaller envelope value. -/
def IsSLMax (v : ℤ × ℤ) : Prop := ∀ w, d2 v w = 1 → Env p c S hS w < Env p c S hS v

/-- **Down step** (`lem:maxima`, down clause). `E` decreases on the downward step iff
some active cone has its apex strictly below row `i`. -/
theorem maxima_down (i j : ℤ) :
    Env p c S hS (i + 1, j) = Env p c S hS (i, j) - 1 ↔
      ∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).1 > i := by
  constructor
  · intro hE
    obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) (i + 1, j))
    have hEv' : Env p c S hS (i + 1, j) = c s0 + d2 (p s0) (i + 1, j) := hs0e
    have hge : Env p c S hS (i, j) ≤ c s0 + d2 (p s0) (i, j) := Finset.inf'_le _ hs0
    refine ⟨s0, hs0, ?_, ?_⟩
    · by_cases hp : (p s0).1 > i
      · rw [d2_down_lt _ _ _ hp] at hEv'; omega
      · rw [d2_down_ge _ _ _ (by omega)] at hEv'; omega
    · by_contra hcon; push_neg at hcon
      rw [d2_down_ge _ _ _ hcon] at hEv'; omega
  · rintro ⟨s, hs, hsE, hpi⟩
    apply le_antisymm
    · have hle : Env p c S hS (i + 1, j) ≤ c s + d2 (p s) (i + 1, j) := Finset.inf'_le _ hs
      rw [d2_down_lt _ _ _ hpi] at hle; omega
    · apply Finset.le_inf'
      intro t htS
      have hge : Env p c S hS (i, j) ≤ c t + d2 (p t) (i, j) := Finset.inf'_le _ htS
      by_cases hp : (p t).1 > i
      · rw [d2_down_lt _ _ _ hp]; omega
      · rw [d2_down_ge _ _ _ (by omega)]; omega

/-- **Up step** (`lem:maxima`, up clause). -/
theorem maxima_up (i j : ℤ) :
    Env p c S hS (i - 1, j) = Env p c S hS (i, j) - 1 ↔
      ∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).1 < i := by
  constructor
  · intro hE
    obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) (i - 1, j))
    have hEv' : Env p c S hS (i - 1, j) = c s0 + d2 (p s0) (i - 1, j) := hs0e
    have hge : Env p c S hS (i, j) ≤ c s0 + d2 (p s0) (i, j) := Finset.inf'_le _ hs0
    refine ⟨s0, hs0, ?_, ?_⟩
    · by_cases hp : (p s0).1 < i
      · rw [d2_up_lt _ _ _ hp] at hEv'; omega
      · rw [d2_up_ge _ _ _ (by omega)] at hEv'; omega
    · by_contra hcon; push_neg at hcon
      rw [d2_up_ge _ _ _ hcon] at hEv'; omega
  · rintro ⟨s, hs, hsE, hpi⟩
    apply le_antisymm
    · have hle : Env p c S hS (i - 1, j) ≤ c s + d2 (p s) (i - 1, j) := Finset.inf'_le _ hs
      rw [d2_up_lt _ _ _ hpi] at hle; omega
    · apply Finset.le_inf'
      intro t htS
      have hge : Env p c S hS (i, j) ≤ c t + d2 (p t) (i, j) := Finset.inf'_le _ htS
      by_cases hp : (p t).1 < i
      · rw [d2_up_lt _ _ _ hp]; omega
      · rw [d2_up_ge _ _ _ (by omega)]; omega

/-- **Right step** (`lem:maxima`, right clause). -/
theorem maxima_right (i j : ℤ) :
    Env p c S hS (i, j + 1) = Env p c S hS (i, j) - 1 ↔
      ∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).2 > j := by
  constructor
  · intro hE
    obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) (i, j + 1))
    have hEv' : Env p c S hS (i, j + 1) = c s0 + d2 (p s0) (i, j + 1) := hs0e
    have hge : Env p c S hS (i, j) ≤ c s0 + d2 (p s0) (i, j) := Finset.inf'_le _ hs0
    refine ⟨s0, hs0, ?_, ?_⟩
    · by_cases hp : (p s0).2 > j
      · rw [d2_right_lt _ _ _ hp] at hEv'; omega
      · rw [d2_right_ge _ _ _ (by omega)] at hEv'; omega
    · by_contra hcon; push_neg at hcon
      rw [d2_right_ge _ _ _ hcon] at hEv'; omega
  · rintro ⟨s, hs, hsE, hpj⟩
    apply le_antisymm
    · have hle : Env p c S hS (i, j + 1) ≤ c s + d2 (p s) (i, j + 1) := Finset.inf'_le _ hs
      rw [d2_right_lt _ _ _ hpj] at hle; omega
    · apply Finset.le_inf'
      intro t htS
      have hge : Env p c S hS (i, j) ≤ c t + d2 (p t) (i, j) := Finset.inf'_le _ htS
      by_cases hp : (p t).2 > j
      · rw [d2_right_lt _ _ _ hp]; omega
      · rw [d2_right_ge _ _ _ (by omega)]; omega

/-- **Left step** (`lem:maxima`, left clause). -/
theorem maxima_left (i j : ℤ) :
    Env p c S hS (i, j - 1) = Env p c S hS (i, j) - 1 ↔
      ∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).2 < j := by
  constructor
  · intro hE
    obtain ⟨s0, hs0, hs0e⟩ := S.exists_mem_eq_inf' hS (fun s => c s + d2 (p s) (i, j - 1))
    have hEv' : Env p c S hS (i, j - 1) = c s0 + d2 (p s0) (i, j - 1) := hs0e
    have hge : Env p c S hS (i, j) ≤ c s0 + d2 (p s0) (i, j) := Finset.inf'_le _ hs0
    refine ⟨s0, hs0, ?_, ?_⟩
    · by_cases hp : (p s0).2 < j
      · rw [d2_left_lt _ _ _ hp] at hEv'; omega
      · rw [d2_left_ge _ _ _ (by omega)] at hEv'; omega
    · by_contra hcon; push_neg at hcon
      rw [d2_left_ge _ _ _ hcon] at hEv'; omega
  · rintro ⟨s, hs, hsE, hpj⟩
    apply le_antisymm
    · have hle : Env p c S hS (i, j - 1) ≤ c s + d2 (p s) (i, j - 1) := Finset.inf'_le _ hs
      rw [d2_left_lt _ _ _ hpj] at hle; omega
    · apply Finset.le_inf'
      intro t htS
      have hge : Env p c S hS (i, j) ≤ c t + d2 (p t) (i, j) := Finset.inf'_le _ htS
      by_cases hp : (p t).2 < j
      · rw [d2_left_lt _ _ _ hp]; omega
      · rw [d2_left_ge _ _ _ (by omega)]; omega

/-- **Maxima Criterion** (`lem:maxima`, complete on the full lattice). For a height
-function envelope, a cell `(i,j)` is a strict local maximum iff in each of the four
directions some active cone points back toward its apex. -/
theorem maxima_criterion (hheight : IsHeightFn p c S hS) (i j : ℤ) :
    IsSLMax p c S hS (i, j) ↔
      (∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).1 > i) ∧
      (∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).1 < i) ∧
      (∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).2 > j) ∧
      (∃ s ∈ S, c s + d2 (p s) (i, j) = Env p c S hS (i, j) ∧ (p s).2 < j) := by
  constructor
  · intro hmax
    have hstep : ∀ w, d2 (i, j) w = 1 → Env p c S hS w = Env p c S hS (i, j) - 1 := by
      intro w hw
      rcases hheight (i, j) w hw with h | h
      · have := hmax w hw; omega
      · exact h
    refine ⟨?_, ?_, ?_, ?_⟩
    · rw [← maxima_down]; exact hstep _ (by rw [nbhd_iff]; tauto)
    · rw [← maxima_up]; exact hstep _ (by rw [nbhd_iff]; tauto)
    · rw [← maxima_right]; exact hstep _ (by rw [nbhd_iff]; tauto)
    · rw [← maxima_left]; exact hstep _ (by rw [nbhd_iff]; tauto)
  · rintro ⟨hd, hu, hr, hl⟩ w hw
    rw [nbhd_iff] at hw
    rcases hw with h | h | h | h <;> subst h
    · rw [(maxima_down p c S hS i j).mpr hd]; omega
    · rw [(maxima_up p c S hS i j).mpr hu]; omega
    · rw [(maxima_right p c S hS i j).mpr hr]; omega
    · rw [(maxima_left p c S hS i j).mpr hl]; omega

end OrigamiCone.Sequel

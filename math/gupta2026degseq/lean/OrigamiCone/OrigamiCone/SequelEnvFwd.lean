import Mathlib
import OrigamiCone.SequelEnvThm

/-!
# Sequel: Envelope Structure Theorem — forward direction

Closes the forward direction of `thm:envelope` from the sequel paper,
conditional on the **Envelope Lemma** (`eq:env`, paper-external, taken
here as a named `axiom`).

The reverse direction (`SequelEnvThm.envelope_structure_reverse`,
AVENUE 13) is unconditional. This module closes the forward direction:
every height function `h` on `ℤ × ℤ` with finite nonempty SLM set `P`
is the lower envelope `E_{P, h|_P}` of its minima, and `(P, h|_P)`
satisfies ACT + PAR. Together with the reverse direction, this closes
the structural bijection of `thm:envelope` in Lean (modulo the named
external axiom).

## External dependency: `eq:env`

The Envelope Lemma is the deep external input that requires the
companion paper's full height-function machinery (induction on distance
to the minimum set, plus Lipschitz + integer-edge-difference structure).
We state it as a Lean `axiom` with explicit citation, and verify that
everything downstream is mechanical bookkeeping over the axiom. The
campaign's contribution is **pinning down** that `eq:env` is the unique
non-formalised step.

## Theorems

* `IsHeightFunction h` : `h` changes by exactly `±1` on every edge.
* `IsSLMinOf h v` : `v` is a strict local minimum of `h`.
* `Envelope_Lemma` (**axiom**, `eq:env`) : statement of the Envelope
  Lemma.
* `height_walk_parity` (private) : `h v - h (0,0) ≡ d2 (0,0) v [ZMOD 2]`.
  Strong induction on `(d2 (0, 0) v).toNat`.
* `height_diff_parity` (private) : `h s - h t ≡ d2 s t [ZMOD 2]` for any
  pair `s, t`. Combines `height_walk_parity` at `s` and `t` with
  `dgrid_parity`.
* `slmin_imp_active_id` (private) : SLM implies Active, specialised to
  `ι := ℤ × ℤ`, `p := id`. Re-proved inline because the general
  `SequelActive.slmin_imp_active` cannot be co-imported with
  `SequelEnvThm` (duplicate-definition clash on `IsSLMin`/`Env`).
* `envelope_structure_forward` (**main turn 4 result**, conditional on
  `Envelope_Lemma`) : the forward direction of `thm:envelope`.

## Scope

* Imports `Mathlib` and `OrigamiCone.SequelEnvThm`. Does **not** import
  `SequelActive` (clash). The one needed lemma from there is re-proved.
* No `sorry`. Axioms: `[propext, Classical.choice, Quot.sound,
  OrigamiCone.Sequel.Envelope_Lemma]` — baseline plus the named external
  axiom.
* NOT added to root aggregator `OrigamiCone.lean`.

Check axioms with
`#print axioms OrigamiCone.Sequel.envelope_structure_forward`.
-/

namespace OrigamiCone.Sequel

open Polynomial Matrix Finset

/-- A function `h : ℤ × ℤ → ℤ` is a **height function** on the lattice if
it changes by exactly `±1` across every edge: for every edge-adjacent pair
(`d2 v w = 1`), either `h v + 1 = h w` or `h v = h w + 1`. -/
def IsHeightFunction (h : ℤ × ℤ → ℤ) : Prop :=
  ∀ v w, d2 v w = 1 → h v + 1 = h w ∨ h v = h w + 1

/-- A cell `v` is a **strict local minimum** of `h` if every edge-neighbour
has strictly greater value. -/
def IsSLMinOf (h : ℤ × ℤ → ℤ) (v : ℤ × ℤ) : Prop :=
  ∀ w, d2 v w = 1 → h v < h w

/-- **Envelope Lemma** (`eq:env` of the companion paper). For every height
function `h` on `ℤ × ℤ` with finite nonempty SLM set,
`h v = min_{p ∈ SLM} (h p + d2 p v)` for every `v`.

This is taken as a Lean **axiom**. Its proof in the companion paper uses
the full height-function machinery (induction on distance from the minimum
set; Lipschitz + integer-edge-difference structure) and is the **unique**
external dependency of this module. -/
axiom Envelope_Lemma (h : ℤ × ℤ → ℤ) (_ : IsHeightFunction h)
    (SLM : Finset (ℤ × ℤ)) (hNE : SLM.Nonempty)
    (_ : ∀ v, v ∈ SLM ↔ IsSLMinOf h v) :
    ∀ v, h v = SLM.inf' hNE (fun p => h p + d2 p v)

/-- For a height function `h`, `h v - h (0, 0) ≡ d2 (0, 0) v [ZMOD 2]`.
Strong induction on `(d2 (0, 0) v).toNat`, stepping toward the origin via
`step_toward` (from `SequelEnvThm`). -/
private lemma height_walk_parity (h : ℤ × ℤ → ℤ) (hh : IsHeightFunction h)
    (v : ℤ × ℤ) : h v - h (0, 0) ≡ d2 (0, 0) v [ZMOD 2] := by
  generalize hk : (d2 (0, 0) v).toNat = k
  induction k using Nat.strong_induction_on generalizing v with
  | _ k IH =>
  by_cases hv : v = (0, 0)
  · subst hv; simp [d2]
  · obtain ⟨w, hw1, hw2⟩ := step_toward (0, 0) v (fun heq => hv heq.symm)
    have hd_pos : 1 ≤ d2 (0, 0) v := by
      have : 0 ≤ d2 (0, 0) w := by unfold d2; positivity
      omega
    have hk_w : (d2 (0, 0) w).toNat < k := by
      rw [hw2, ← hk]; omega
    have IH_w := IH (d2 (0, 0) w).toNat hk_w w rfl
    have hvw_par : h v - h w ≡ 1 [ZMOD 2] := by
      rcases hh v w hw1 with heq | heq
      · have : h v - h w = -1 := by omega
        rw [this]; decide
      · have : h v - h w = 1 := by omega
        rw [this]
    have key : h v - h (0, 0) = (h v - h w) + (h w - h (0, 0)) := by ring
    rw [key]
    have hdsum : d2 (0, 0) v = 1 + d2 (0, 0) w := by rw [hw2]; omega
    rw [hdsum]
    exact (Int.ModEq.add hvw_par IH_w)

/-- For any height function `h` and any pair of points, `h s - h t ≡ d2 s t
[ZMOD 2]`. Subtract `height_walk_parity` at `s` and `t`, then convert via
`dgrid_parity` (using `a - b ≡ a + b [ZMOD 2]`). -/
private lemma height_diff_parity (h : ℤ × ℤ → ℤ) (hh : IsHeightFunction h)
    (s t : ℤ × ℤ) : h s - h t ≡ d2 s t [ZMOD 2] := by
  have hs := height_walk_parity h hh s
  have ht := height_walk_parity h hh t
  have hsubt : h s - h t ≡ d2 (0, 0) s - d2 (0, 0) t [ZMOD 2] := by
    have hd : h s - h t = (h s - h (0, 0)) - (h t - h (0, 0)) := by ring
    rw [hd]
    exact hs.sub ht
  have hdg_s := dgrid_parity (0, 0) s
  have hdg_t := dgrid_parity (0, 0) t
  have hdg_st := dgrid_parity s t
  simp at hdg_s hdg_t
  have hsym : d2 (0, 0) s - d2 (0, 0) t ≡ d2 s t [ZMOD 2] := by
    calc d2 (0, 0) s - d2 (0, 0) t
        ≡ (s.1 + s.2) - (t.1 + t.2) [ZMOD 2] := hdg_s.sub hdg_t
      _ ≡ (s.1 + s.2) + (t.1 + t.2) [ZMOD 2] := by
          show ((s.1 + s.2) - (t.1 + t.2)) % 2 = ((s.1 + s.2) + (t.1 + t.2)) % 2
          omega
      _ ≡ d2 s t [ZMOD 2] := hdg_st.symm
  exact hsubt.trans hsym

/-- **SLM implies Active** (the harder direction of `lem:activemin`),
specialised to `ι := ℤ × ℤ`, `p := id`. Adapted from
`SequelActive.slmin_imp_active` (re-proved inline because the two
modules have a duplicate-definition clash on `IsSLMin`/`Env`).

Proof sketch: contradiction. If `s` is a SLM of `Env id h SLM hNE` but
not Active, some `t ∈ SLM`, `t ≠ s`, has `h t + d2 t s ≤ h s`. The inf
is attained by some `u0 ∈ SLM`. By `u = s` in the inf, `Env s ≤ h s`.
Case-split on `Env s = h s` vs `Env s < h s` to pick a "violating" apex
`u ≠ s` with `h u + d2 u s ≤ Env s`. Step from `s` toward `u` to a
neighbour `w` with `d2 u w = d2 u s - 1`; then `Env w ≤ h u + d2 u w ≤
Env s - 1`, contradicting `Env s < Env w` (from `IsSLMin`).

Uses no property of `h` beyond the SLM hypothesis; in particular,
independent of `Envelope_Lemma`. -/
private lemma slmin_imp_active_id (h : ℤ × ℤ → ℤ)
    (SLM : Finset (ℤ × ℤ)) (hNE : SLM.Nonempty)
    (s : ℤ × ℤ) (hsS : s ∈ SLM)
    (hmin : IsSLMin (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE s) :
    Active (id : ℤ × ℤ → ℤ × ℤ) h SLM s := by
  classical
  refine ⟨hsS, ?_⟩
  by_contra hcon
  push_neg at hcon
  obtain ⟨t, htS, hts, htle⟩ := hcon
  obtain ⟨u0, hu0S, hu0e⟩ :=
    SLM.exists_mem_eq_inf' hNE (fun t => h t + d2 t s)
  have hEnvle : Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE s ≤ h s := by
    have := Finset.inf'_le (fun t => h t + d2 t s) hsS
    have h0 : d2 s s = 0 := by unfold d2; simp
    simpa [Env, h0] using this
  have hEnveq : Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE s = h u0 + d2 u0 s := hu0e
  obtain ⟨u, huS, hus, hule⟩ :
      ∃ u ∈ SLM, u ≠ s ∧
        h u + d2 u s ≤ Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE s := by
    by_cases hEq : Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE s = h s
    · exact ⟨t, htS, hts, by rw [hEq]; exact htle⟩
    · have hlt : Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE s < h s :=
        lt_of_le_of_ne hEnvle hEq
      refine ⟨u0, hu0S, ?_, le_of_eq hEnveq.symm⟩
      intro h_eq
      rw [h_eq] at hEnveq
      have : d2 s s = 0 := by unfold d2; simp
      rw [this] at hEnveq
      omega
  obtain ⟨w, hw1, hw2⟩ := step_toward u s hus
  have hwle : Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE w ≤ h u + d2 u w :=
    Finset.inf'_le _ huS
  rw [hw2] at hwle
  have := hmin w hw1
  omega

/-- **Envelope Structure Theorem, forward direction** (`thm:envelope`,
forward, **conditional on `Envelope_Lemma`**).

Given a height function `h : ℤ × ℤ → ℤ` and its strict-local-minimum set
`SLM` (finite, nonempty), the data `(id, h, SLM)` satisfies the PAR and
ACT conditions, and `h` equals the lower envelope `Env id h SLM hNE`.

Combined with `SequelEnvThm.envelope_structure_reverse` (unconditional),
this closes the structural bijection of `thm:envelope` at the codomain
side, modulo the named external axiom `Envelope_Lemma`. -/
theorem envelope_structure_forward
    (h : ℤ × ℤ → ℤ) (hh : IsHeightFunction h)
    (SLM : Finset (ℤ × ℤ)) (hNE : SLM.Nonempty)
    (hSLM : ∀ v, v ∈ SLM ↔ IsSLMinOf h v) :
    Par (id : ℤ × ℤ → ℤ × ℤ) h SLM ∧
    (∀ s ∈ SLM, Active (id : ℤ × ℤ → ℤ × ℤ) h SLM s) ∧
    (∀ v, h v = Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE v) := by
  -- Par: from height_diff_parity.
  have hPar : Par (id : ℤ × ℤ → ℤ × ℤ) h SLM := by
    intro s _ t _
    show h s - h t ≡ d2 s t [ZMOD 2]
    exact height_diff_parity h hh s t
  -- h = Env: directly from Envelope_Lemma.
  have hEnv : ∀ v, h v = Env (id : ℤ × ℤ → ℤ × ℤ) h SLM hNE v := by
    have := Envelope_Lemma h hh SLM hNE hSLM
    intro v
    unfold Env
    exact this v
  refine ⟨hPar, ?_, hEnv⟩
  -- ACT: each s ∈ SLM is a SLM of Env (via hEnv), hence Active by slmin_imp_active_id.
  intro s hsS
  apply slmin_imp_active_id h SLM hNE s hsS
  -- Goal: IsSLMin id h SLM hNE s
  intro w hw1
  -- IsSLMin: Env s < Env w. Via hEnv: Env s = h s, Env w = h w. Need: h s < h w.
  rw [← hEnv s, ← hEnv w]
  exact (hSLM s).mp hsS w hw1

end OrigamiCone.Sequel

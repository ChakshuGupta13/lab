import Mathlib
import OrigamiCone.SequelEd

/-!
# Sequel: active columns — combinatorial foundation of `lem:uniform` (Task E.δ.a)

Paper `lem:uniform` (§8) has two independent proofs — a combinatorial
"per-axis degree" argument via active-column contraction, and an analytic
"onset" argument via resolvent expansion.  This module opens the combinatorial
side by formalising the first step: the pigeonhole `#active columns ≤ #extrema`.

A **column** `j : Fin n` of a height function `h : Cell m n → ℤ` is **active**
if it carries at least one strict local extremum.  The paper's `lem:uniform`
uses:

* boundary columns are active (`lem:boundary`),
* middle columns are active iff not frozen (`lem:frozen`),
* a function with `d` extrema has ≤ `d` active columns (pigeonhole),
* runs of frozen columns have a single slope and can be contracted to a
  single column, preserving all height differences,
* the count over contracted "types" is polynomial in `n` of degree ≤ `d-2`.

This module supplies the third bullet (pigeonhole) as a self-contained
building block.  The frozen-column classification (bullets 1–2), contraction
(bullet 4), and type-count polynomial bound (bullet 5) are heavier and
remain deferred to Task E.δ's own session(s).

## Theorems

* `activeColumn (h : Cell m n → ℤ) (j : Fin n) : Prop` — the column `j` has at
  least one strict local extremum in `h`.
* `numActiveColumns (h : Cell m n → ℤ) : ℕ` — the count of active columns.
* **`numActiveColumns_le_numExtrema`** — the paper's "at most `d` active
  columns for a height function with `d` extrema": pigeonhole via the
  image of the extremum set under column projection.
* `numActiveColumns_le_of_numExtrema` — specialised at `numExtrema h = d`.
* `frozenRunStart`, `numFrozenRuns` — inactive-column run structure.
* **`numFrozenRuns_le`** — with active boundary columns, `#frozen runs + 1 ≤
  #active columns` (predecessor-map injection missing the last column).
* **`numFrozenRuns_lt_numExtrema`** — chained: `#frozen runs + 1 ≤ #extrema`.

## Substrate

Uses `SequelEd`'s ambient `numExtrema` and the grid `IsStrictLocalExtremum`
predicate (`OrigamiCone.IsStrictLocalExtremum` from `DegreeExtrema`,
`IsStrictLocalMax ∨ IsStrictLocalMin`) on `Cell m n = Fin m × Fin n`.  Standalone;
imports only `OrigamiCone.SequelEd`.

## Discipline note

Task E.δ splits into two independent chains:
1. **Combinatorial degree bound** (this module opens): active columns +
   frozen runs + contraction ⟹ degree ≤ `d - 2`.
2. **Analytic onset bound**: resolvent expansion + `lem:frozenbdy`
   ⟹ onset `n ≥ d - 1`.

Neither chain closes `lem:uniform` alone.  Both must be delivered to close
Task E.δ.  This commit contributes to chain (1)'s foundation.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
Check with `#print axioms OrigamiCone.Sequel.numActiveColumns_le_numExtrema`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m n : ℕ}

/-- A **column** `j : Fin n` of a height function `h : Cell m n → ℤ` is
**active** if it carries at least one strict local extremum. -/
def activeColumn (h : Cell m n → ℤ) (j : Fin n) : Prop :=
  ∃ i : Fin m, IsStrictLocalExtremum h (i, j)

instance {h : Cell m n → ℤ} : DecidablePred (activeColumn h) := by
  intro j; unfold activeColumn; infer_instance

/-- Number of active columns of `h`. -/
def numActiveColumns (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (activeColumn h)).card

/-- **Pigeonhole** (paper `lem:uniform`, per-axis-degree proof, active-column
count).  Every active column contains at least one strict local extremum, so
projecting extrema to their column gives a surjection onto active columns,
hence `numActiveColumns h ≤ numExtrema h`. -/
theorem numActiveColumns_le_numExtrema (h : Cell m n → ℤ) :
    numActiveColumns h ≤ numExtrema h := by
  unfold numActiveColumns numExtrema
  set E : Finset (Cell m n) := Finset.univ.filter (IsStrictLocalExtremum h) with hE
  set A : Finset (Fin n) := Finset.univ.filter (activeColumn h) with hA
  suffices A ⊆ E.image Prod.snd by
    calc A.card
        ≤ (E.image Prod.snd).card := Finset.card_le_card this
      _ ≤ E.card := Finset.card_image_le
  intro j hj
  rw [Finset.mem_filter] at hj
  obtain ⟨_, i, hi⟩ := hj
  rw [Finset.mem_image]
  refine ⟨(i, j), ?_, rfl⟩
  rw [Finset.mem_filter]
  exact ⟨Finset.mem_univ _, hi⟩

/-- Specialisation of `numActiveColumns_le_numExtrema` at `numExtrema h = d`.
For a height function with exactly `d` strict local extrema, at most `d`
columns are active. -/
theorem numActiveColumns_le_of_numExtrema (h : Cell m n → ℤ) (d : ℕ)
    (hd : numExtrema h = d) : numActiveColumns h ≤ d :=
  hd ▸ numActiveColumns_le_numExtrema h

/-! ## Frozen-run count bound

The paper's per-axis-degree argument (`lem:uniform`) needs: the number of
maximal runs of inactive ("frozen") columns is at most `#active − 1`.  Given
that the boundary columns are active (`lem:boundary`), each frozen run has a
unique start column whose predecessor is active; the predecessor map is an
injection into the active columns that misses the last column, giving the
bound.  Boundary-column activity is taken as a hypothesis here (its proof is
`lem:boundary`, formalised separately). -/

/-- A column `j` is a **frozen-run start** if it is inactive and either it is
column 0 or the previous column is active.  Each maximal run of inactive
columns has exactly one such start. -/
def frozenRunStart (h : Cell m n → ℤ) (j : Fin n) : Prop :=
  ¬ activeColumn h j ∧ (j.val = 0 ∨ (0 < j.val ∧ activeColumn h ⟨j.val - 1, by omega⟩))

instance {h : Cell m n → ℤ} : DecidablePred (frozenRunStart h) := by
  intro j; unfold frozenRunStart; infer_instance

/-- Number of frozen runs = number of frozen-run starts. -/
def numFrozenRuns (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (frozenRunStart h)).card

/-- **Runs bound** (paper `lem:uniform`, per-axis-degree step): with the first
and last columns active, the number of frozen runs is at most
`#active columns − 1`.  The predecessor map `j ↦ j − 1` sends each frozen-run
start to an active column, is injective, and misses the last column, so its
image lies in the active columns with the last one removed. -/
theorem numFrozenRuns_le (h : Cell m n → ℤ) (hn : 0 < n)
    (h0 : activeColumn h ⟨0, hn⟩)
    (hlast : activeColumn h ⟨n - 1, by omega⟩) :
    numFrozenRuns h + 1 ≤ numActiveColumns h := by
  unfold numFrozenRuns numActiveColumns
  set S : Finset (Fin n) := Finset.univ.filter (frozenRunStart h) with hS
  set A : Finset (Fin n) := Finset.univ.filter (activeColumn h) with hA
  have hlast_mem : (⟨n - 1, by omega⟩ : Fin n) ∈ A := by
    rw [hA, Finset.mem_filter]; exact ⟨Finset.mem_univ _, hlast⟩
  have hjpos : ∀ j ∈ S, 0 < j.val := by
    intro j hj
    rw [hS, Finset.mem_filter] at hj
    obtain ⟨_, hfrz⟩ := hj
    rcases hfrz.2 with h0' | ⟨hpos, _⟩
    · exfalso
      apply hfrz.1
      have : j = ⟨0, hn⟩ := by ext; omega
      rw [this]; exact h0
    · exact hpos
  have h_inj : ∀ j₁ ∈ S, ∀ j₂ ∈ S, j₁.val - 1 = j₂.val - 1 → j₁ = j₂ := by
    intro j₁ hj₁ j₂ hj₂ heq
    have p₁ := hjpos j₁ hj₁
    have p₂ := hjpos j₂ hj₂
    ext; omega
  have h_sub : S.image (fun j => (⟨j.val - 1, by omega⟩ : Fin n))
      ⊆ A.erase ⟨n - 1, by omega⟩ := by
    intro x hx
    rw [Finset.mem_image] at hx
    obtain ⟨j, hjS, hxeq⟩ := hx
    have hjpos' := hjpos j hjS
    rw [hS, Finset.mem_filter] at hjS
    obtain ⟨_, hfrz⟩ := hjS
    rw [Finset.mem_erase]
    constructor
    · intro hcontra
      rw [← hxeq] at hcontra
      have : j.val - 1 = n - 1 := by
        have := Fin.val_eq_of_eq hcontra
        simpa using this
      have hjlt := j.isLt
      omega
    · rw [hA, Finset.mem_filter]
      refine ⟨Finset.mem_univ _, ?_⟩
      rcases hfrz.2 with h0' | ⟨_, hact⟩
      · omega
      · rw [← hxeq]; exact hact
  have h_card_image : (S.image (fun j => (⟨j.val - 1, by omega⟩ : Fin n))).card = S.card :=
    Finset.card_image_of_injOn (fun a ha b hb hab => h_inj a ha b hb (by
      have := Fin.val_eq_of_eq hab; simpa using this))
  have h_le : S.card ≤ (A.erase ⟨n - 1, by omega⟩).card := by
    rw [← h_card_image]; exact Finset.card_le_card h_sub
  rw [Finset.card_erase_of_mem hlast_mem] at h_le
  have hA_pos : 0 < A.card := Finset.card_pos.mpr ⟨_, hlast_mem⟩
  omega

/-- **Runs ≤ extrema − 1** (combined `lem:uniform` step): with active boundary
columns, a height function with `d` strict local extrema has at most `d − 1`
frozen runs.  Chains `numFrozenRuns_le` with
`numActiveColumns_le_numExtrema`. -/
theorem numFrozenRuns_lt_numExtrema (h : Cell m n → ℤ) (hn : 0 < n)
    (h0 : activeColumn h ⟨0, hn⟩)
    (hlast : activeColumn h ⟨n - 1, by omega⟩) :
    numFrozenRuns h + 1 ≤ numExtrema h :=
  le_trans (numFrozenRuns_le h hn h0 hlast) (numActiveColumns_le_numExtrema h)

end OrigamiCone.Sequel

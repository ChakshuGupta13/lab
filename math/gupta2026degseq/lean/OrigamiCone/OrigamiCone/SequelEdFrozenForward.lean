import Mathlib

/-!
# Sequel: extremum-free ⟹ frozen — the abstract three-column core (Task E.δ.g)

The combinatorial arm of paper `lem:uniform` classifies each middle column of a
height function as **active** (carrying a strict local extremum) or **frozen**
(extremum-free).  Paper `lem:frozen` is the classification; module
`SequelEdFrozenCol` supplies the easy `⟸` direction (frozen ⟹ inactive).  This
module supplies the substrate-independent **core of the hard `⟹` direction**:
an extremum-free column is *rainbow* (its two horizontal neighbour-differences
sum to zero at every row), and a rainbow column is *slope-constant*.

## Abstract setup

Fix a column height `M + 1` rows tall.  Three integer walks over `ℕ`:

* `e i` — the vertical step `v(i+1) − v(i)` of the **middle** column (`i < M`);
* `dL i` — the horizontal difference `(left column) − (middle)` at row `i`;
* `dR i` — the horizontal difference `(right column) − (middle)` at row `i`.

All three are `±1` (grid `IsHeight`).  The two *side columns are themselves
height walks*: `e i + (dL(i+1) − dL i)` is the vertical step of the LEFT column
at row `i` (a telescoping identity on the grid), hence `±1`; likewise for `dR`
(hypotheses `hLwalk`, `hRwalk`).

## The extremum predicate

`alignedExt M e dL dR σ i` (`σ = ±1`) says row `i` of the middle column is a
strict local **extremum** — a minimum for `σ = 1`, a maximum for `σ = −1`:
both horizontal neighbours agree in sign (`dL i = dR i = σ`) and both present
vertical neighbours are on the far side (`e(i−1) = −σ` above, `e i = σ` below).
On the grid this is exactly `IsStrictLocalMin`/`Max` of the cell.

## Theorems

* **`cascade_down` / `cascade_up`** — if a row is aligned (`dL = dR = σ`) yet the
  extremum condition fails, the failure propagates monotonically to a grid
  boundary row, where it *forces* an extremum.  These are the two halves of the
  contradiction that powers the main lemma.
* **`extremumFree_rainbow`** — a column with no extremum in any row satisfies
  `dL i + dR i = 0` for every row: the horizontal neighbours point in opposite
  directions (rainbow).  This is `lem:frozen`'s `⟹` direction, abstractly.
* **`rainbow_dL_const`** — under the side-walk hypotheses, a rainbow column has
  `dL` constant: the middle vertical step forces the two side walks to move
  together, pinning `dL(i+1) = dL i`.  This upgrades rainbow to the paper's
  *slope-k frozen* structure.

## Role in Task E.δ

`extremumFree_rainbow` is the substrate-independent heart of the `¬ active ⟹
frozen` implication on the height substrate (`SequelEdFrozenCol.frozenColumn`).
The remaining grid bridge instantiates `e, dL, dR` at an interior column and
converts `alignedExt` to `IsStrictLocalExtremum`; it is deferred to the bridge
module.  `rainbow_dL_const` gives the slope-constancy the frozen-run contraction
map (`hdecomp` of `degreeBound_assembly`) will consume.

## Substrate

Imports `Mathlib` only.  Fully abstract (`ℕ → ℤ` walks); no grid, no `Cell`.

No `sorry`.  Axioms: `[propext, Quot.sound]` (no `Classical.choice`).
Check with `#print axioms OrigamiCone.Sequel.extremumFree_rainbow`.
-/

namespace OrigamiCone.Sequel

/-- Row `i` of the middle column is a strict local **extremum** of sign `σ`
(minimum for `σ = 1`, maximum for `σ = −1`): both horizontal neighbours lie on
the same side (`dL i = dR i = σ`) and each present vertical neighbour lies on
the far side (`e (i−1) = −σ` above when `0 < i`, `e i = σ` below when `i < M`). -/
def alignedExt (M : ℕ) (e dL dR : ℕ → ℤ) (σ : ℤ) (i : ℕ) : Prop :=
  (0 < i → e (i - 1) = -σ) ∧ (i < M → e i = σ) ∧ dL i = σ ∧ dR i = σ

variable {M : ℕ} {e dL dR : ℕ → ℤ}

/-- **Downward cascade.** An aligned row `i0 < M` whose *lower* vertical
neighbour is on the near side (`e i0 = −σ`, so `i0` is not itself an extremum)
forces every row below it to be aligned with the same near-side step, until row
`M`, where the (now absent) lower neighbour makes row `M` a genuine extremum —
contradicting extremum-freeness. -/
theorem cascade_down (σ : ℤ) (hσ : σ = 1 ∨ σ = -1)
    (he : ∀ i, i < M → e i = 1 ∨ e i = -1)
    (hdL : ∀ i, i ≤ M → dL i = 1 ∨ dL i = -1)
    (hdR : ∀ i, i ≤ M → dR i = 1 ∨ dR i = -1)
    (hLwalk : ∀ i, i < M → e i + (dL (i+1) - dL i) = 1 ∨ e i + (dL (i+1) - dL i) = -1)
    (hRwalk : ∀ i, i < M → e i + (dR (i+1) - dR i) = 1 ∨ e i + (dR (i+1) - dR i) = -1)
    (hef : ∀ i, i ≤ M → ¬ alignedExt M e dL dR σ i)
    (i0 : ℕ) (hi0M : i0 < M)
    (hL0 : dL i0 = σ) (hR0 : dR i0 = σ) (he0 : e i0 = -σ) :
    False := by
  have claim : ∀ k, i0 + k ≤ M →
      dL (i0+k) = σ ∧ dR (i0+k) = σ ∧ (i0+k < M → e (i0+k) = -σ) := by
    intro k
    induction k with
    | zero => intro _; exact ⟨hL0, hR0, fun _ => he0⟩
    | succ k IH =>
      intro hle
      have hk : i0 + k ≤ M := by omega
      obtain ⟨hLj, hRj, hej_imp⟩ := IH hk
      set j := i0 + k with hjdef
      have hjM : j < M := by omega
      have hej : e j = -σ := hej_imp hjM
      have hLj1 : dL (j+1) = σ := by
        have hw := hLwalk j hjM
        have hv := hdL (j+1) (by omega)
        rcases hσ with h | h <;> rcases hw with hh | hh <;> rcases hv with hv1 | hv1 <;> omega
      have hRj1 : dR (j+1) = σ := by
        have hw := hRwalk j hjM
        have hv := hdR (j+1) (by omega)
        rcases hσ with h | h <;> rcases hw with hh | hh <;> rcases hv with hv1 | hv1 <;> omega
      have hidx : i0 + (k+1) = j + 1 := by omega
      rw [hidx]
      refine ⟨hLj1, hRj1, ?_⟩
      intro hlt
      by_contra hcontra
      have hene : e (j+1) = σ := by
        have := he (j+1) hlt
        rcases hσ with h | h <;> omega
      apply hef (j+1) (by omega)
      refine ⟨fun _ => ?_, fun _ => hene, hLj1, hRj1⟩
      rw [Nat.add_sub_cancel]; exact hej
  -- Row M: both horizontal neighbours σ, and the upper vertical neighbour on
  -- the far side (`e (M-1) = -σ`), so `alignedExt σ M` holds — contradiction.
  have hLM : dL M = σ := by
    have := (claim (M - i0) (by omega)).1; rwa [show i0 + (M - i0) = M from by omega] at this
  have hRM : dR M = σ := by
    have := (claim (M - i0) (by omega)).2.1; rwa [show i0 + (M - i0) = M from by omega] at this
  have heM : e (M - 1) = -σ := by
    have := (claim (M - 1 - i0) (by omega)).2.2 (by omega)
    rwa [show i0 + (M - 1 - i0) = M - 1 from by omega] at this
  exact hef M (le_refl M) ⟨fun _ => heM, fun h => absurd h (by omega), hLM, hRM⟩

/-- **Upward cascade.** An aligned row `0 < i0` whose *upper* vertical neighbour
is on the near side (`e (i0−1) = σ`) forces every row above it to be aligned
with the same near-side step, until row `0`, where the (now absent) upper
neighbour makes row `0` a genuine extremum — contradicting extremum-freeness. -/
theorem cascade_up (σ : ℤ) (hσ : σ = 1 ∨ σ = -1)
    (he : ∀ i, i < M → e i = 1 ∨ e i = -1)
    (hdL : ∀ i, i ≤ M → dL i = 1 ∨ dL i = -1)
    (hdR : ∀ i, i ≤ M → dR i = 1 ∨ dR i = -1)
    (hLwalk : ∀ i, i < M → e i + (dL (i+1) - dL i) = 1 ∨ e i + (dL (i+1) - dL i) = -1)
    (hRwalk : ∀ i, i < M → e i + (dR (i+1) - dR i) = 1 ∨ e i + (dR (i+1) - dR i) = -1)
    (hef : ∀ i, i ≤ M → ¬ alignedExt M e dL dR σ i)
    (i0 : ℕ) (hi0 : 0 < i0) (hi0M : i0 ≤ M)
    (hL0 : dL i0 = σ) (hR0 : dR i0 = σ) (he0 : e (i0-1) = σ) :
    False := by
  have claim : ∀ k, k ≤ i0 →
      dL (i0-k) = σ ∧ dR (i0-k) = σ ∧ (0 < i0-k → e (i0-k-1) = σ) := by
    intro k
    induction k with
    | zero => intro _; exact ⟨hL0, hR0, fun _ => he0⟩
    | succ k IH =>
      intro hle
      have hk : k ≤ i0 := by omega
      obtain ⟨hLj, hRj, hej_imp⟩ := IH hk
      set j := i0 - k with hjdef
      have hjpos : 0 < j := by omega
      have hej : e (j-1) = σ := hej_imp hjpos
      have hjM : j - 1 < M := by omega
      have hLj1 : dL (j-1) = σ := by
        have hw := hLwalk (j-1) hjM
        have hv := hdL (j-1) (by omega)
        have hsj : j - 1 + 1 = j := by omega
        rw [hsj] at hw
        rcases hσ with h | h <;> rcases hw with hh | hh <;> rcases hv with hv1 | hv1 <;> omega
      have hRj1 : dR (j-1) = σ := by
        have hw := hRwalk (j-1) hjM
        have hv := hdR (j-1) (by omega)
        have hsj : j - 1 + 1 = j := by omega
        rw [hsj] at hw
        rcases hσ with h | h <;> rcases hw with hh | hh <;> rcases hv with hv1 | hv1 <;> omega
      have hidx : i0 - (k+1) = j - 1 := by omega
      rw [hidx]
      refine ⟨hLj1, hRj1, ?_⟩
      intro hpos
      by_contra hcontra
      have hene : e (j-1-1) = -σ := by
        have := he (j-1-1) (by omega)
        rcases hσ with h | h <;> omega
      exact hef (j-1) (by omega)
        ⟨fun _ => hene, fun _ => hej, hLj1, hRj1⟩
  have hL0' : dL 0 = σ := by
    have := (claim i0 (le_refl i0)).1; rwa [show i0 - i0 = 0 from by omega] at this
  have hR0' : dR 0 = σ := by
    have := (claim i0 (le_refl i0)).2.1; rwa [show i0 - i0 = 0 from by omega] at this
  have he0' : e 0 = σ := by
    have := (claim (i0-1) (by omega)).2.2 (by omega)
    rwa [show i0 - (i0-1) - 1 = 0 from by omega] at this
  exact hef 0 (by omega) ⟨fun h => absurd h (by omega), fun _ => he0', hL0', hR0'⟩

/-- **Extremum-free ⟹ rainbow** (`lem:frozen`, `⟹` direction, abstract core).
If no row of the middle column is a strict local extremum of either sign, then
at every row the two horizontal neighbour-differences sum to zero:
`dL i + dR i = 0`.  Equivalently the neighbours point in opposite directions. -/
theorem extremumFree_rainbow
    (he : ∀ i, i < M → e i = 1 ∨ e i = -1)
    (hdL : ∀ i, i ≤ M → dL i = 1 ∨ dL i = -1)
    (hdR : ∀ i, i ≤ M → dR i = 1 ∨ dR i = -1)
    (hLwalk : ∀ i, i < M → e i + (dL (i+1) - dL i) = 1 ∨ e i + (dL (i+1) - dL i) = -1)
    (hRwalk : ∀ i, i < M → e i + (dR (i+1) - dR i) = 1 ∨ e i + (dR (i+1) - dR i) = -1)
    (hef : ∀ σ, (σ = 1 ∨ σ = -1) → ∀ i, i ≤ M → ¬ alignedExt M e dL dR σ i) :
    ∀ i, i ≤ M → dL i + dR i = 0 := by
  intro i0 hi0M
  by_contra hne
  have hLv := hdL i0 hi0M
  have hRv := hdR i0 hi0M
  have hσ : dL i0 = dR i0 := by rcases hLv with h|h <;> rcases hRv with h'|h' <;> omega
  set σ := dL i0 with hσdef
  have hσv : σ = 1 ∨ σ = -1 := hLv
  have hR0 : dR i0 = σ := hσ.symm
  have hefσ := hef σ hσv
  by_cases hd : i0 < M ∧ e i0 = -σ
  · exact cascade_down σ hσv he hdL hdR hLwalk hRwalk hefσ i0 hd.1 rfl hR0 hd.2
  · by_cases hu : 0 < i0 ∧ e (i0-1) = σ
    · exact cascade_up σ hσv he hdL hdR hLwalk hRwalk hefσ i0 hu.1 hi0M rfl hR0 hu.2
    · push_neg at hd hu
      apply hefσ i0 hi0M
      refine ⟨fun hpos => ?_, fun hlt => ?_, rfl, hR0⟩
      · have hne1 := hu hpos
        have hev := he (i0-1) (by omega)
        rcases hσv with h|h <;> omega
      · have hne2 := hd hlt
        have hev := he i0 hlt
        rcases hσv with h|h <;> omega

/-- **Rainbow ⟹ slope-constant.**  A rainbow column (`dL i + dR i = 0`
everywhere) has constant `dL`.  Write `x := dL (i+1) − dL i`.  The left
side-step is `e i + x`, and — since rainbow gives `dR = −dL` — the right
side-step is `e i − x`; both lie in `{±1}` (the side columns are walks) and sum
to `2 e i = ±2`, which forces each to equal `e i` and hence `x = 0`, that is
`dL (i+1) = dL i`.  This lifts the opposite-sign (rainbow) structure to the
paper's *slope-k frozen* column. -/
theorem rainbow_dL_const
    (he : ∀ i, i < M → e i = 1 ∨ e i = -1)
    (hLwalk : ∀ i, i < M → e i + (dL (i+1) - dL i) = 1 ∨ e i + (dL (i+1) - dL i) = -1)
    (hRwalk : ∀ i, i < M → e i + (dR (i+1) - dR i) = 1 ∨ e i + (dR (i+1) - dR i) = -1)
    (rainbow : ∀ i, i ≤ M → dL i + dR i = 0) :
    ∀ i, i < M → dL (i+1) = dL i := by
  intro i hi
  have hr_i := rainbow i (by omega)
  have hr_i1 := rainbow (i+1) (by omega)
  have hL := hLwalk i hi
  have hR := hRwalk i hi
  have hei := he i hi
  rcases hL with hL|hL <;> rcases hR with hR|hR <;> rcases hei with he'|he' <;> omega

end OrigamiCone.Sequel

import Mathlib
import OrigamiCone.SequelPolesIter

/-!
# Sequel: arbitrary-`d` `lem:poles` (matrix-level recurrence theorem)

Standalone formalisation of the **arbitrary-`d` form of `Lemma lem:poles`** of
the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

The `d`-fold convolutional sum

    `RseqMat T A d n := if d = 0 then T^n else convolveMat T A (RseqMat T A (d-1)) n`

is the matrix-valued sequence that arises as the `[x^d]` coefficient of
`(I - z T - z x A)^{-1}` (the resolvent / Neumann series, with the off-by-one
Leibniz shift documented in `SequelPolesConv`). This module proves the
**`p^{d+1}`-recurrence theorem**

    `RseqMat_p_dPlus1_recurrence : ∀ d n, charActIter T (RseqMat T A d) (d+1) n = 0`

— the iterated `p(E_n)`-charpoly-action of depth `d+1` annihilates `RseqMat T A d`.
This is the matrix-level skeleton of `lem:poles` for all `d`, generalising

* `SequelPoles.poles_at_x_zero` (`d=0`, single application kills the
  extremum-free transfer chain),
* `SequelPolesConv.Rseq_p_squared_recurrence` (`d=1`, two applications kill the
  once-convolutional scalar sum).

## How the induction works

The proof is by induction on `d`. The key inductive primitive is

    `convolveMat_kills_step : (∀ n, charActIter T S k n = 0) →
                              ∀ n, charActIter T (convolveMat T A S) (k+1) n = 0`

— if `S` is annihilated by `k` charpoly actions, then `convolveMat T A S` is
annihilated by `k+1` actions. This is itself proved by induction on `k`:

* **Base `k = 0`**: if `S ≡ 0` (the depth-0 sense of "annihilated"), then
  `convolveMat T A S ≡ 0` (each summand `T^j * A * 0 = 0`), so any number of
  charpoly actions still gives `0` (via `charActIter_zero_seq`, depth `1`
  suffices).
* **Step `k → k+1`**: assume the lemma at depth `k`. Given `S` annihilated by
  `k+1` charpoly actions, we want `convolveMat T A S` annihilated by `k+2`.
  The crucial swap lemma `charActIter_succ_swap` lets us push one application
  inside:
  `charActIter T (convolveMat T A S) (k+2) n
       = charActIter T (charActMat T (convolveMat T A S)) (k+1) n`.
  By `SequelPolesIter.charActMat_convolveMat_decomp`, the inner
  `charActMat T (convolveMat T A S)` splits as
  `convolveMat T A (charActMat T S) + BoundaryMat T A S`. By
  `SequelPolesIter.charActIter_add` (linearity), the `(k+1)`-fold iterate
  distributes across this sum:
  * the boundary term vanishes by `charActIter_BoundaryMat_eq_zero` (`k ≥ 0`
    plus one more application kills `BoundaryMat`);
  * the convolution term `convolveMat T A (charActMat T S)` has the IH applied
    to `S' := charActMat T S`, which by `charActIter_succ_swap` is annihilated
    by `k` charpoly actions (using the hypothesis on `S` at depth `k+1`).

The outer induction on `d` is then immediate:

* **Base `d = 0`**: `RseqMat T A 0 n = T^n`, and `charActMat T (T^·) n = 0` by
  `SequelPolesIter.ch_shift_mat` (Cayley-Hamilton at shift `n`).
* **Step `d → d+1`**: `RseqMat T A (d+1) = convolveMat T A (RseqMat T A d)`,
  and apply `convolveMat_kills_step` to the IH.

The proof is the ~30-40 line follow-up forecast by the Adversary review of
`SequelPolesIter`; the two-component invariant (one live `convolveMat`,
one live `BoundaryMat` per step) is captured by the binary split inside
`convolveMat_kills_step`.

## Theorems

* `RseqMat T A d n` : the `d`-fold matrix-valued convolutional sum.
* `charActIter_succ_swap` : `charActIter T S (d+1) n = charActIter T (charActMat T S) d n`
  (the swap lemma; one charpoly action can be moved between the outer iterate
  count and the innermost sequence). Induction on `d`.
* `charActIter_zero_seq` : `charActIter T (fun _ => 0) d ≡ 0`. Induction on `d`.
* `charActIter_BoundaryMat_eq_zero` : `charActIter T (BoundaryMat T A S) (k+1) ≡ 0`.
  Apply swap + `charActMat_BoundaryMat_eq_zero` + `charActIter_zero_seq`.
* `convolveMat_kills_step` : the inductive primitive described above.
  Induction on `k`.
* `RseqMat_p_dPlus1_recurrence` (**main**): for all `d` and `n`,
  `charActIter T (RseqMat T A d) (d+1) n = 0`. Induction on `d` using
  `convolveMat_kills_step` at each step.

## Scope

* The matrix-level `p^{d+1}`-recurrence on the `d`-fold convolutional sum is
  proved end-to-end for arbitrary `d` (no `sorry`).
* The connection to the paper's `[x^d]` Neumann-series coefficient (i.e. that
  `[x^d](u^⊤ T_m(x)^N v)` is a polynomial in `RseqMat T A d (·)` after the
  off-by-one Leibniz shift) is the PowerSeries identification still
  disclaimed in `SequelPolesConv` and not formalised here.
* The scalar-sandwich version of this theorem (i.e. `∀ d n, u ⬝ᵥ
  (charActIter T (RseqMat T A d) (d+1) n) *ᵥ v = 0`) is an immediate
  consequence by sandwiching with `u`/`v` and pulling through the sum; not
  spelled out here.
* The pole-localisation conclusion (the GF of `u ⬝ᵥ (RseqMat T A d ·) *ᵥ v`
  has poles only at reciprocal eigenvalues of `T`, with multiplicity at most
  `d+1`) is the standard `SequelRatGF`-style rational-GF bridge applied to
  the recurrence; not formalised here.
* **Discipline note**: this is the first Sequel module that imports another
  Sequel module (`OrigamiCone.SequelPolesIter`). The Sequel discipline of
  "import Mathlib only" exists to avoid parallel-session edit conflicts on
  shared dependencies; `SequelPolesIter` is a strict downstream foundation
  built and committed in the same session as this module, with no parallel
  edits in progress. The deviation is documented and justified by the cost
  of duplicating ~250 lines of primitives that would otherwise need to be
  inlined.

No `sorry`; check with
`#print axioms OrigamiCone.Sequel.RseqMat_p_dPlus1_recurrence`.
-/

namespace OrigamiCone.Sequel

open Matrix

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- **Swap lemma**: an extra `charActMat T` application at the outer end of
`charActIter T S (d+1)` is equivalent to applying `charActMat T` once to the
input sequence `S` and then iterating `d` times. Both equal the `(d+1)`-fold
iterate of `charActMat T`. -/
lemma charActIter_succ_swap (T : Matrix ι ι R) (S : ℕ → Matrix ι ι R) (d n : ℕ) :
    charActIter T S (d + 1) n = charActIter T (charActMat T S) d n := by
  induction d generalizing n with
  | zero => rfl
  | succ d IH =>
    show charActMat T (charActIter T S (d + 1)) n
      = charActMat T (charActIter T (charActMat T S) d) n
    congr 1
    funext m
    exact IH m

/-- The iterated charpoly action of the identically-zero sequence is
identically zero. Used as a base inside `convolveMat_kills_step`'s `k = 0`
branch and `charActIter_BoundaryMat_eq_zero`. -/
lemma charActIter_zero_seq (T : Matrix ι ι R) (d n : ℕ) :
    charActIter T (fun _ => (0 : Matrix ι ι R)) d n = 0 := by
  induction d generalizing n with
  | zero => rfl
  | succ d IH =>
    show charActMat T (charActIter T (fun _ => 0) d) n = 0
    have hfun : charActIter T (fun _ : ℕ => (0 : Matrix ι ι R)) d
        = fun _ => 0 := by funext m; exact IH m
    rw [hfun]
    show ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
        T.charpoly.coeff k • (0 : Matrix ι ι R) = 0
    simp

/-- The iterated charpoly action of `BoundaryMat T A S` at depth `k + 1`
vanishes. Reduces to one-shot kill of `BoundaryMat` by swapping the outer
iterate into the input sequence (`charActIter_succ_swap`), then propagating
zeros via `charActIter_zero_seq`. -/
lemma charActIter_BoundaryMat_eq_zero (T A : Matrix ι ι R) (S : ℕ → Matrix ι ι R)
    (k n : ℕ) :
    charActIter T (BoundaryMat T A S) (k + 1) n = 0 := by
  rw [charActIter_succ_swap]
  have hfun : charActMat T (BoundaryMat T A S)
      = fun _ => (0 : Matrix ι ι R) := by
    funext m; exact charActMat_BoundaryMat_eq_zero T A S m
  rw [hfun]
  exact charActIter_zero_seq T k n

/-- **Inductive primitive**: if `S` is annihilated by `k` charpoly actions
(`∀ n, charActIter T S k n = 0`), then `convolveMat T A S` is annihilated by
`k + 1` charpoly actions. Proved by induction on `k`.

* **Base `k = 0`**: `S ≡ 0`, so `convolveMat T A S ≡ 0`, and one charpoly
  action kills it.
* **Step `k → k + 1`**: swap one charpoly action inside via
  `charActIter_succ_swap`, then decompose via
  `charActMat_convolveMat_decomp` into a convolution-of-shifted-S part and a
  boundary part; the boundary part vanishes by
  `charActIter_BoundaryMat_eq_zero`; the convolution-of-shifted-S part falls
  to the IH applied to `S' := charActMat T S` (which is annihilated by `k`
  charpoly actions by the swap lemma applied to the hypothesis on `S` at
  depth `k + 1`). -/
lemma convolveMat_kills_step (T A : Matrix ι ι R) :
    ∀ (k : ℕ) (S : ℕ → Matrix ι ι R),
      (∀ n, charActIter T S k n = 0) →
      ∀ n, charActIter T (convolveMat T A S) (k + 1) n = 0 := by
  intro k
  induction k with
  | zero =>
    intro S hS n
    have hS0 : S = fun _ => 0 := by funext m; exact hS m
    have hconv : convolveMat T A S = fun _ => 0 := by
      funext m
      show ∑ j ∈ Finset.range (m + 1), T ^ j * A * S (m - j) = 0
      rw [hS0]
      show ∑ j ∈ Finset.range (m + 1), T ^ j * A * (0 : Matrix ι ι R) = 0
      simp
    rw [hconv]
    exact charActIter_zero_seq T 1 n
  | succ k IH =>
    intro S hS n
    rw [charActIter_succ_swap]
    -- Goal: charActIter T (charActMat T (convolveMat T A S)) (k + 1) n = 0.
    have hdecomp : charActMat T (convolveMat T A S)
        = fun m => convolveMat T A (charActMat T S) m + BoundaryMat T A S m := by
      funext m; exact charActMat_convolveMat_decomp T A S m
    rw [hdecomp]
    rw [charActIter_add]
    rw [charActIter_BoundaryMat_eq_zero, add_zero]
    -- Goal: charActIter T (convolveMat T A (charActMat T S)) (k + 1) n = 0.
    -- Apply IH with `S' := charActMat T S`; need `charActIter T (charActMat T S) k ≡ 0`.
    -- By `charActIter_succ_swap`, this equals `charActIter T S (k + 1) ≡ 0`, which is `hS`.
    apply IH (charActMat T S)
    intro m
    rw [← charActIter_succ_swap]
    exact hS m

/-- **`d`-fold matrix-valued convolutional sum** `RseqMat T A d n`. Defined
recursively on `d`:

* `RseqMat T A 0 n = T ^ n`,
* `RseqMat T A (d + 1) n = convolveMat T A (RseqMat T A d) n`.

For the canonical sandwich `u^⊤ (·) v`, this recovers the d-fold scalar
convolutional sum that arises as `[x^d](u^⊤ T_m(x)^{n+d} v)` (with the
Leibniz shift of `+d` per the d-fold product rule — a generalisation of the
off-by-one `+1` shift documented in `SequelPolesConv` for the `d = 1` case;
at `d = 0` it reduces to `T_m(x)^n` since `[x^0]` of `T_m(x)^n` is `T^n`,
matching the base case definition). -/
noncomputable def RseqMat (T A : Matrix ι ι R) : ℕ → ℕ → Matrix ι ι R
  | 0, n => T ^ n
  | d + 1, n => convolveMat T A (RseqMat T A d) n

/-- **Base case** of the main theorem: the `T^·` chain is annihilated by one
charpoly action. Direct from `SequelPolesIter.ch_shift_mat`. -/
lemma RseqMat_zero_killed (T : Matrix ι ι R) (n : ℕ) :
    charActIter T (fun m => T ^ m) 1 n = 0 := by
  show charActMat T (fun m => T ^ m) n = 0
  exact ch_shift_mat T n

/-- **`p^{d+1}`-recurrence on the `d`-fold convolutional sum** (`lem:poles` at
arbitrary `d`, matrix-level recurrence-complete). For all `d` and `n`,

    `charActIter T (RseqMat T A d) (d + 1) n = 0`,

i.e. the `(d+1)`-fold charpoly action of `T` annihilates the `d`-fold
convolutional sum. This is the arbitrary-`d` form of `lem:poles` at the
matrix level: the sandwich `u^⊤ (·) v` and the `SequelRatGF`-style bridge
upgrade this to "the generating function of `u ⬝ᵥ (RseqMat T A d ·) *ᵥ v`
has poles only at reciprocal eigenvalues of `T` with multiplicity at most
`d + 1`".

Proved by induction on `d`:

* **Base `d = 0`**: `RseqMat T A 0 = fun n => T^n`, so the conclusion is
  `charActMat T (T^·) n = 0`, immediate from `RseqMat_zero_killed`.
* **Step `d → d + 1`**: `RseqMat T A (d + 1) = convolveMat T A (RseqMat T A d)`,
  and the IH says `charActIter T (RseqMat T A d) (d + 1) ≡ 0`, so
  `convolveMat_kills_step` upgrades this to `charActIter T (convolveMat T A
  (RseqMat T A d)) (d + 2) ≡ 0`. -/
theorem RseqMat_p_dPlus1_recurrence (T A : Matrix ι ι R) :
    ∀ (d n : ℕ), charActIter T (RseqMat T A d) (d + 1) n = 0 := by
  intro d
  induction d with
  | zero =>
    intro n
    show charActIter T (RseqMat T A 0) 1 n = 0
    have hfun : RseqMat T A 0 = fun m => T ^ m := by funext m; rfl
    rw [hfun]
    exact RseqMat_zero_killed T n
  | succ d IH =>
    intro n
    show charActIter T (RseqMat T A (d + 1)) (d + 2) n = 0
    have hfun : RseqMat T A (d + 1) = convolveMat T A (RseqMat T A d) := by
      funext m; rfl
    rw [hfun]
    exact convolveMat_kills_step T A (d + 1) (RseqMat T A d) IH n

end OrigamiCone.Sequel

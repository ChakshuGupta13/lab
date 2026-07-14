import Mathlib

/-!
# Sequel meta-theorem: the ±1-walk structure (`lem:binom`) and the
leading-coefficient compensation (`thm:leading`)

Standalone formalisation of two proven, self-contained cores of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

## The ±1 walk (`Lemma lem:binom`)

A degree-$d$ vertex restricted to one boundary row/column is a one-dimensional
`±1` walk; the leading coefficient and the support of each split are read off the
extrema of such walks. We model a walk by its **step list** `s : List Bool`, with
`true` = up (`+1`) and `false` = down (`−1`); a list of `k` steps describes a walk
on the `k+1` cells `0,…,k`. The strict local extrema, counting the two endpoints:

* the **left endpoint** (cell `0`) is a minimum iff the first step is up, a
  maximum iff it is down (it always is an extremum);
* an **interior** cell is a minimum iff its two incident steps read `(down,up)`
  and a maximum iff they read `(up,down)` (a turn);
* the **right endpoint** (cell `k`) is a minimum iff the last step is down, a
  maximum iff it is up.

Write `numMin s`, `numMax s` for these counts. The paper's `lem:binom` states the
support condition `f_{(a,b)} = 0` unless `|a − b| ≤ 1`, the count
`(1 + [a=b])\binom{n-2}{d-2}`, and the degree `d − 2`. This module proves the
**support / alternation** half — the consequence used in `SequelSepDim`:

`abs_numMin_sub_numMax_le_one`: `|numMin s − numMax s| ≤ 1` for every walk.

The mechanism is a clean telescoping invariant (`numMin_sub_numMax`):
`numMin s − numMax s = w(head) − w(last)`, where the interior `(down,up)` and
`(up,down)` turn counts, taken with opposite signs, telescope to
`w(last) − w(head)` along the step list. Both `w(head), w(last) ∈ {0,1}`, so the
difference lies in `{−1,0,1}`. This is exactly the `|ρ − ρ'| ≤ 1`,
`|γ − γ'| ≤ 1` input that `SequelSepDim.sep_dim_bound` takes as a hypothesis.

The full count `(1+[a=b])\binom{n-2}{d-2}` (the run-length / composition
bijection) is not reproduced here.

## The leading-coefficient compensation (`Theorem thm:leading`)

`E_d = \sum_{a\le b}(2-[a=b])N_{(a,b)}` weights each split by the colour-inversion
multiplicity `2 - [a=b]`, while the single-edge walk count carries the endpoint
multiplicity `1 + [a=b]`. The proof of `thm:leading` turns on the arithmetic fact
that these compensate: `(2-[a=b])(1+[a=b]) = 2` whether the balanced split is
even (`a=b`) or odd (`a≠b`), so the pure-power coefficient is
`(2-[a=b])\cdot 2\cdot(1+[a=b]) = 4`, giving `C(d) = 4/(d-2)!`.

No `sorry`; check with `#print axioms OrigamiCone.Sequel.abs_numMin_sub_numMax_le_one`.
-/

namespace OrigamiCone.Sequel

/-- Up/down weight of a step: `true` (up) ↦ `1`, `false` (down) ↦ `0`. -/
def w (b : Bool) : ℤ := if b then 1 else 0

@[simp] lemma w_true : w true = 1 := rfl
@[simp] lemma w_false : w false = 0 := rfl

lemma w_mem (b : Bool) : w b = 0 ∨ w b = 1 := by cases b <;> simp [w]

/-- On an adjacent step pair, the valley indicator `(down,up)` minus the peak
indicator `(up,down)` equals `w y − w x`. This is what makes the signed turn
count telescope. -/
lemma turn_eq (x y : Bool) :
    ((if (!x && y) then (1 : ℤ) else 0) - (if (x && !y) then 1 else 0))
      = w y - w x := by
  cases x <;> cases y <;> simp [w]

/-- Signed count over adjacent step pairs with weight `f`. -/
def adjCount (f : Bool → Bool → ℤ) : List Bool → ℤ
  | [] => 0
  | [_] => 0
  | x :: y :: t => f x y + adjCount f (y :: t)

/-- `adjCount` is additive in its weight: the difference of two adjacent counts is
the adjacent count of the pointwise difference. -/
lemma adjCount_sub (f g : Bool → Bool → ℤ) :
    ∀ s : List Bool,
      adjCount f s - adjCount g s = adjCount (fun a b => f a b - g a b) s
  | [] => by simp [adjCount]
  | [_] => by simp [adjCount]
  | x :: y :: t => by
      simp only [adjCount]
      have := adjCount_sub f g (y :: t)
      ring_nf
      ring_nf at this
      linarith [this]

/-- **Telescoping.** The signed turn count `∑ (w y − w x)` over adjacent pairs of a
nonempty step list collapses to `w(last) − w(head)`. -/
lemma adjCount_telescope :
    ∀ (x : Bool) (t : List Bool),
      adjCount (fun a b => w b - w a) (x :: t)
        = w ((x :: t).getLast (List.cons_ne_nil x t)) - w x := by
  intro x t
  induction t generalizing x with
  | nil => simp [adjCount]
  | cons y t ih =>
      simp only [adjCount]
      rw [ih y, List.getLast_cons (List.cons_ne_nil y t)]
      ring

/-- Number of strict local **minima** of the `±1` walk with step list `x :: t`
(nonempty), counting both endpoints. -/
def numMin (x : Bool) (t : List Bool) : ℤ :=
  w x
  + adjCount (fun a b => if (!a && b) then 1 else 0) (x :: t)
  + (1 - w ((x :: t).getLast (List.cons_ne_nil x t)))

/-- Number of strict local **maxima** of the `±1` walk with step list `x :: t`
(nonempty), counting both endpoints. -/
def numMax (x : Bool) (t : List Bool) : ℤ :=
  (1 - w x)
  + adjCount (fun a b => if (a && !b) then 1 else 0) (x :: t)
  + w ((x :: t).getLast (List.cons_ne_nil x t))

/-- **Telescoping invariant** (`Lemma lem:binom`, support half). For every `±1`
walk, `numMin − numMax = w(head) − w(last)`. -/
lemma numMin_sub_numMax (x : Bool) (t : List Bool) :
    numMin x t - numMax x t
      = w x - w ((x :: t).getLast (List.cons_ne_nil x t)) := by
  have hturn :
      adjCount (fun a b => if (!a && b) then (1 : ℤ) else 0) (x :: t)
        - adjCount (fun a b => if (a && !b) then (1 : ℤ) else 0) (x :: t)
        = w ((x :: t).getLast (List.cons_ne_nil x t)) - w x := by
    rw [adjCount_sub]
    rw [show (fun a b => (if (!a && b) then (1 : ℤ) else 0)
                          - (if (a && !b) then 1 else 0))
            = (fun a b => w b - w a) from by funext a b; exact turn_eq a b]
    exact adjCount_telescope x t
  unfold numMin numMax
  -- difference of the interior turn counts is `w last - w head`; endpoints add up.
  have hx := w_mem x
  linarith [hturn]

/-- **`|numMin − numMax| ≤ 1`** (`Lemma lem:binom`, support condition). A `±1`
walk has its minima and maxima counts differing by at most one — the input that
`SequelSepDim.sep_dim_bound` takes as the walk hypotheses `hrow`, `hcol`. -/
theorem abs_numMin_sub_numMax_le_one (x : Bool) (t : List Bool) :
    |numMin x t - numMax x t| ≤ 1 := by
  rw [numMin_sub_numMax, abs_le]
  rcases w_mem x with hx | hx <;>
    rcases w_mem ((x :: t).getLast (List.cons_ne_nil x t)) with hl | hl <;>
      rw [hx, hl] <;> constructor <;> norm_num

/-- **Leading-coefficient compensation** (proof of `Theorem thm:leading`). With
`e := [a=b] ∈ {0,1}`, the colour-inversion multiplicity `2 − e` times the
endpoint multiplicity `1 + e` is the constant `2`, independent of whether the
balanced split is even or odd. -/
lemma leading_compensation (e : ℕ) (he : e ≤ 1) : (2 - e) * (1 + e) = 2 := by
  interval_cases e <;> rfl

/-- The pure-power numerator: `(2 − e)·2·(1 + e) = 4`, giving `C(d) = 4/(d−2)!` for
the coefficient of `m^{d-2} + n^{d-2}` in `p_d`. -/
lemma leading_coeff_numerator (e : ℕ) (he : e ≤ 1) : (2 - e) * 2 * (1 + e) = 4 := by
  interval_cases e <;> rfl

end OrigamiCone.Sequel

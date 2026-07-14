import Mathlib

/-!
# Sequel: convolutional `p²`-recurrence for `lem:poles` at `d = 1`

Standalone formalisation of the **convolutional recurrence** underlying the
`[x^1]` slice of `Lemma lem:poles` of the sequel paper

> *Degree-$d$ vertex counts of the $m \times n$ origami flip graph:
> a polynomial meta-theorem.*

`Lemma lem:poles` proves that the poles of $\sum_n E_d(m, \cdot)\,z^n =
[x^d](\mathbf u^\top (I - z T_m(x))^{-1} \mathbf v)$ lie among the reciprocal
eigenvalues of $T_0 := T_m(0)$. The paper's proof expands

$$(I - z T_0 - z x B(x))^{-1} = \sum_{k \ge 0} \big[(I - z T_0)^{-1} z x B(x)\big]^k (I - z T_0)^{-1}$$

and observes that, for each fixed `d`, `[x^d]` of the sum is a finite
combination of powers of `(I - z T_0)^{-1}` times constant matrices — so its
denominator divides a power of `det(I - z T_0)`, i.e. the poles are reciprocal
eigenvalues of `T_0`.

`SequelPoles.poles_at_x_zero` formalised the `d = 0` slice: the
extremum-free-only transfer satisfies the Cayley-Hamilton recurrence whose
characteristic polynomial is `T_0.charpoly`. The next substantive step,
`d = 1`, requires showing that the **convolutional sum**

$$R(n) := \sum_{j=0}^{n} \mathbf u^\top T^{j} A T^{n-j} \mathbf v$$

— which arises as the `[x^1]` coefficient of the Neumann-series expansion when
`T_m(x) = T_0 + x A + O(x^2)` (with `T = T_0` and `A` the linear-in-`x` part of
`T_m(x)`) — satisfies the **squared** recurrence whose characteristic polynomial
is `T.charpoly²`. This is the convolutional analogue of the d=0 result: in
the d=0 case a single application of `p(E_n) := \sum_k p_k E_n^k` (where
`p := T.charpoly` and `E_n` is the shift operator) kills the sequence; in the
d=1 case `p(E_n)` reduces the convolutional sum to a finite "boundary" sum, and
a second application of `p(E_n)` kills the boundary. Iterating this argument
`d` times gives the `[x^d]` slice of `lem:poles` with characteristic polynomial
`p^{d+1}` — a multi-day formalisation across all `d`. This module establishes
the **`d = 1` base case** (`p^2`-recurrence on the once-convolutional sum).

## Argument outline

Set `f(j, e) := u^\top T^j A T^e v`, so `R(n) = \sum_{j=0}^{n} f(j, n-j)`.

**First application of `p(E_n)`.** Splitting `R(n + k)` at index `n + 1`:

$$R(n + k) = \sum_{i=0}^{n} f(i, n+k-i) + \sum_{t=0}^{k-1} f(n + 1 + t, k - 1 - t).$$

Multiplying by `p_k` and summing over `k`:

* the first piece collapses by **Cayley-Hamilton on the right-tail power**
  (`scalar_ch_right`): for fixed `i`, the `T^{n-i+k}` factor satisfies
  `\sum_k p_k T^{n-i+k} = T^{n-i} \cdot \sum_k p_k T^k = 0`;
* the second piece is the **boundary**
  `Boundary(n) := \sum_k p_k \sum_{t<k} f(n + 1 + t, k - 1 - t)`, which has *no*
  `n`-dependence on the second `T`-power (it is `T^{k-1-t}` with `t < k` summed
  out, so the second `T`-power ranges over a *finite* set `{0, ..., d-1}`
  independent of `n`).

**Second application of `p(E_n)`.** Each boundary term, as a function of `n`,
is `u^\top T^{n + 1 + t} A T^{k-1-t} v` (with `(k, t)` fixed in the outer sums):
applying `p(E_n)` to this gives `\sum_l p_l T^{l + n + 1 + t} \cdot A T^{k-1-t}
= 0` by **Cayley-Hamilton on the left-tail power** (`scalar_ch_left`).

Composing: `p(E_n)^2 R(n) = p(E_n) \cdot \text{Boundary}(n) = 0`.

## Theorems

* `ch_shift_mat` : the matrix-level Cayley-Hamilton at shift `m`,
  `\sum_k p_k \cdot T^{m+k} = 0`. The d=0 building block, re-derived in-place to
  keep the module self-contained (matches `SequelPoles.poles_at_x_zero`'s
  in-place derivation).
* `scalar_ch_right` : `\sum_k p_k \cdot u^\top (X T^{m+k}) v = 0` for arbitrary
  `X`. The CH-on-the-right-tail building block.
* `scalar_ch_left` : `\sum_k p_k \cdot u^\top (T^{m+k} Y) v = 0` for arbitrary
  `Y`. The CH-on-the-left-tail building block (used for boundary terms).
* `Rseq_split` : the index split `R(n+k) = (\text{old part}) + (\text{boundary
  part})`. Direct from `Finset.sum_range_add`.
* `Rseq_charAct_eq_boundary` : the first application of `p(E_n)` reduces
  `R(n+\cdot)` to `Boundary`. The "old part" vanishes by `scalar_ch_right`; the
  boundary part is `Boundary(n)` by definition.
* `boundary_charAct_eq_zero` : the second application of `p(E_n)` kills
  `Boundary(n)`. Each `(k, t)` term factors as `u^\top T^{n+1+t} \cdot Y v`
  with `Y = A T^{k-1-t}` independent of `n`, so `scalar_ch_left` applies.
* `Rseq_p_squared_recurrence` (**main**): the convolutional sum `R(n)` satisfies
  the squared recurrence `p^2(E_n) R = 0`, where `p = T.charpoly`.

## Scope

* The `p²`-recurrence on the once-convolutional sum is proved end-to-end
  (no `sorry`). This is the `d = 1` base case of `lem:poles` beyond `d = 0`.
* The connection to the paper's actual `[x^1]` coefficient — i.e. that
  `[x^1](T_m(x)^n) = \sum_{j} T_0^j B_0 T_0^{n-1-j}` where `B_0 :=
  [x^1](T_m(x))` — is a separate `PowerSeries` derivation (the
  product-of-power-series Leibniz rule for the linear term). It is *not*
  formalised here. The module proves the **structural recurrence** on the
  convolutional sum; the **specialisation** to the paper's `T_m(x) = T_0 + x A +
  O(x^2)` setting is downstream.
* The arbitrary-`d` generalisation (iterated convolutions on `d`-simplices,
  giving `p^{d+1}`-recurrence) is the natural induction on this base case;
  formalising it is a separate session.
* The conversion of the recurrence to the rational-GF conclusion (poles of the
  generating function lie among reciprocal eigenvalues of `T`) is the bridge
  established for `d = 0` by composing with `SequelRatGF.transfer_GF_rational`;
  the same composition lifts the `p^2`-recurrence to "denominator divides
  `(T.charpoly^*)^2`" for the d=1 GF.
* Per the discipline, this module only imports `Mathlib`. The Cayley-Hamilton
  primitive is re-derived in-place rather than aliased from `SequelPoles`
  (matching `SequelPoles`'s own self-contained derivation; the alias form
  triggers a kernel `whnf` timeout, documented in `SequelPoles`'s docstring).

No `sorry`; check with `#print axioms OrigamiCone.Sequel.Rseq_p_squared_recurrence`.
-/

namespace OrigamiCone.Sequel

open Matrix Polynomial

variable {R : Type*} [CommRing R] {ι : Type*} [Fintype ι] [DecidableEq ι]

/-- The scalar matrix-sandwich kernel `f(j, e) := u^\top T^j A T^e v`. Kept
private; only the convolutional sum `Rseq` and the boundary `Boundary` are
exposed. -/
private def fseq (T A : Matrix ι ι R) (u v : ι → R) (j e : ℕ) : R :=
  u ⬝ᵥ ((T ^ j * A * T ^ e) *ᵥ v)

/-- The **once-convolutional sum** `R(n) := \sum_{j=0}^{n} u^\top T^j A T^{n-j} v`.
By the Leibniz rule, this equals `[x^1](u^\top T_m(x)^{n+1} v)` when
`T_m(x) = T + x A + O(x^2)` (note the off-by-one: the n-th coefficient of `T_m(x)^N`
as a polynomial in `x` has `N` summands of the form `T^j A T^{N-1-j}` with
exponents summing to `N - 1`, so to obtain a sum with `n + 1` summands and
exponents summing to `n` we evaluate at `N = n + 1`). Equivalently, `R(n)` is
the `z^{n+1}` coefficient of `[x^1](u^\top (I - z T_m(x))^{-1} v)`. The
structural claim of `lem:poles` at `d = 1` is that `R(n)` satisfies a linear
recurrence whose characteristic polynomial is `T.charpoly^2`. -/
noncomputable def Rseq (T A : Matrix ι ι R) (u v : ι → R) (n : ℕ) : R :=
  ∑ j ∈ Finset.range (n + 1), fseq T A u v j (n - j)

/-- The **boundary sum** produced by one application of `p(E_n)` to
`Rseq(n+\cdot)`. By `Rseq_charAct_eq_boundary` we have `\sum_k p_k R(n+k) =
Boundary(n)`. Crucially, both `T`-powers in each boundary term are *bounded* by
`d := p.natDegree`, independent of `n`, so a second application of `p(E_n)`
kills it by Cayley-Hamilton on the left-tail. -/
noncomputable def Boundary (T A : Matrix ι ι R) (u v : ι → R) (n : ℕ) : R :=
  ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
    T.charpoly.coeff k *
      ∑ t ∈ Finset.range k, fseq T A u v (n + 1 + t) (k - 1 - t)

/-- **Cayley-Hamilton at shift `m`** (matrix level): `\sum_k p_k \cdot T^{m+k} = 0`
where `p = T.charpoly`. Direct consequence of `Matrix.aeval_self_charpoly`
multiplied through by `T^m` on the left. Re-derived in place to keep the module
self-contained. -/
lemma ch_shift_mat (T : Matrix ι ι R) (m : ℕ) :
    ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • T ^ (m + k) = 0 := by
  have hCH : (Polynomial.aeval T) T.charpoly = 0 := Matrix.aeval_self_charpoly _
  have hExpand : (Polynomial.aeval T) T.charpoly
      = ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff l • T ^ l := Polynomial.aeval_eq_sum_range _
  calc ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff k • T ^ (m + k)
      = ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff k • (T ^ m * T ^ k) := by
        refine Finset.sum_congr rfl ?_
        intro k _; rw [pow_add]
    _ = T ^ m * ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
            T.charpoly.coeff k • T ^ k := by
        rw [Finset.mul_sum]
        refine Finset.sum_congr rfl ?_
        intro k _; rw [Matrix.mul_smul]
    _ = T ^ m * 0 := by rw [← hExpand, hCH]
    _ = 0 := Matrix.mul_zero _

/-- **Cayley-Hamilton on the right-tail power**: `\sum_k p_k \cdot u^\top (X
T^{m+k}) v = 0` for any matrix `X`. Factor `X T^m` out of `X T^{m+k} = X T^m
T^k`, apply CH to the residual `\sum_k p_k T^k = 0`. Used in
`Rseq_charAct_eq_boundary` with `X = T^j A` to kill the "old part" after
applying `p(E_n)` to `Rseq`. -/
lemma scalar_ch_right (T : Matrix ι ι R) (X : Matrix ι ι R) (u v : ι → R) (m : ℕ) :
    ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k * (u ⬝ᵥ ((X * T ^ (m + k)) *ᵥ v)) = 0 := by
  have key : ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • (X * T ^ (m + k)) = 0 := by
    have h := ch_shift_mat T m
    calc ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff k • (X * T ^ (m + k))
        = X * ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff k • T ^ (m + k) := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl ?_
          intro k _; rw [Matrix.mul_smul]
      _ = X * 0 := by rw [h]
      _ = 0 := Matrix.mul_zero _
  have hsum0 : u ⬝ᵥ ((∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • (X * T ^ (m + k))) *ᵥ v) = 0 := by
    rw [key, Matrix.zero_mulVec, dotProduct_zero]
  rw [Matrix.sum_mulVec, dotProduct_sum] at hsum0
  convert hsum0 using 1
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- **Cayley-Hamilton on the left-tail power**: `\sum_k p_k \cdot u^\top (T^{m+k}
Y) v = 0` for any matrix `Y`. Factor `T^m` out on the left (via
`pow_add` + `Matrix.smul_mul` + `Finset.sum_mul`), apply CH to the residual.
Used in `boundary_charAct_eq_zero` with `Y = A T^{k-1-t}` to kill the boundary
after applying `p(E_n)` once more. -/
lemma scalar_ch_left (T : Matrix ι ι R) (Y : Matrix ι ι R) (u v : ι → R) (m : ℕ) :
    ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k * (u ⬝ᵥ ((T ^ (m + k) * Y) *ᵥ v)) = 0 := by
  have key : ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • (T ^ (m + k) * Y) = 0 := by
    have h := ch_shift_mat T m
    calc ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff k • (T ^ (m + k) * Y)
        = (∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
              T.charpoly.coeff k • T ^ (m + k)) * Y := by
          rw [Finset.sum_mul]
          refine Finset.sum_congr rfl ?_
          intro k _; exact (Matrix.smul_mul _ _ _).symm
      _ = 0 * Y := by rw [h]
      _ = 0 := Matrix.zero_mul _
  have hsum0 : u ⬝ᵥ ((∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k • (T ^ (m + k) * Y)) *ᵥ v) = 0 := by
    rw [key, Matrix.zero_mulVec, dotProduct_zero]
  rw [Matrix.sum_mulVec, dotProduct_sum] at hsum0
  convert hsum0 using 1
  refine Finset.sum_congr rfl ?_
  intro k _
  rw [Matrix.smul_mulVec, dotProduct_smul, smul_eq_mul]

/-- **Index split** of the once-convolutional sum: `R(n + k) = (\text{j running
over } [0, n]) + (\text{j running over } [n + 1, n + k])`. The first piece
matches `R(n)` with shifted second `T`-power; the second piece is the boundary
contribution. Direct from `Finset.sum_range_add`. -/
lemma Rseq_split (T A : Matrix ι ι R) (u v : ι → R) (n k : ℕ) :
    Rseq T A u v (n + k)
      = (∑ j ∈ Finset.range (n + 1), fseq T A u v j (n + k - j))
        + (∑ t ∈ Finset.range k, fseq T A u v (n + 1 + t) (k - 1 - t)) := by
  unfold Rseq
  rw [show n + k + 1 = (n + 1) + k from by ring, Finset.sum_range_add]
  congr 1
  refine Finset.sum_congr rfl ?_
  intro t _; congr 1; omega

/-- **First application of `p(E_n)`**: `\sum_k p_k R(n + k) = \text{Boundary}(n)`.
The "old part" `\sum_k p_k \sum_j f(j, n+k-j)` vanishes by `scalar_ch_right`
(swap sums, then apply CH on the right-tail power for each fixed `j ≤ n`); the
remaining boundary part is `Boundary(n)` by definition. -/
lemma Rseq_charAct_eq_boundary (T A : Matrix ι ι R) (u v : ι → R) (n : ℕ) :
    ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k * Rseq T A u v (n + k)
    = Boundary T A u v n := by
  simp_rw [Rseq_split T A u v n, mul_add]
  rw [Finset.sum_add_distrib]
  have hfirst : ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k *
        ∑ j ∈ Finset.range (n + 1), fseq T A u v j (n + k - j) = 0 := by
    simp_rw [Finset.mul_sum]
    rw [Finset.sum_comm]
    refine Finset.sum_eq_zero ?_
    intro j hj
    rw [Finset.mem_range] at hj
    have hreidx : ∀ k, n + k - j = (n - j) + k := by intro k; omega
    simp_rw [hreidx]
    show ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
        T.charpoly.coeff k * fseq T A u v j ((n - j) + k) = 0
    unfold fseq
    exact scalar_ch_right T (T ^ j * A) u v (n - j)
  rw [hfirst, zero_add]
  rfl

/-- **Second application of `p(E_n)`**: `\sum_l p_l \cdot \text{Boundary}(n + l) =
0`. For each fixed `(k, t)` term in the boundary, the `n`-dependence is
`u^\top T^{n+l+1+t} (A T^{k-1-t}) v` — applying `\sum_l p_l` and pulling
`T^{n+l+1+t} = T^{(n+1+t)+l}` reduces to `scalar_ch_left` with `Y := A T^{k-1-t}`
(independent of `l` and `n`). -/
lemma boundary_charAct_eq_zero (T A : Matrix ι ι R) (u v : ι → R) (n : ℕ) :
    ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff l * Boundary T A u v (n + l) = 0 := by
  unfold Boundary
  simp_rw [Finset.mul_sum]
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero ?_
  intro k _
  rw [Finset.sum_comm]
  refine Finset.sum_eq_zero ?_
  intro t _
  have heq1 : ∀ x : ℕ,
      T.charpoly.coeff x * (T.charpoly.coeff k * fseq T A u v (n + x + 1 + t) (k - 1 - t))
      = T.charpoly.coeff k * (T.charpoly.coeff x * fseq T A u v (n + x + 1 + t) (k - 1 - t)) := by
    intro x; ring
  simp_rw [heq1]
  rw [← Finset.mul_sum]
  convert mul_zero (T.charpoly.coeff k)
  unfold fseq
  have hreidx : ∀ x, n + x + 1 + t = (n + 1 + t) + x := by intro x; ring
  have heq : ∀ x,
      u ⬝ᵥ ((T ^ (n + x + 1 + t) * A * T ^ (k - 1 - t)) *ᵥ v)
      = u ⬝ᵥ ((T ^ ((n + 1 + t) + x) * (A * T ^ (k - 1 - t))) *ᵥ v) := by
    intro x; rw [hreidx]; congr 1; rw [Matrix.mul_assoc]
  simp_rw [heq]
  exact scalar_ch_left T (A * T ^ (k - 1 - t)) u v (n + 1 + t)

/-- **`p²`-recurrence** on the once-convolutional sum (`lem:poles` at `d = 1`,
recurrence-complete). The convolutional sum `R(n) := \sum_{j=0}^{n} u^\top T^j A
T^{n-j} v` satisfies the squared Cayley-Hamilton recurrence

`\sum_{l, k} p_l \cdot p_k \cdot R(n + l + k) = 0,    where p := T.charpoly`.

This is the `d = 1` base case of `lem:poles` beyond the `d = 0` slice already
formalised in `SequelPoles.poles_at_x_zero`: the `[x^1]` coefficient of the
Neumann-series expansion of `(I - z T - z x A)^{-1}` has `z^{n+1}` coefficient
equal to `R(n)` (the off-by-one is the Leibniz-rule shift between `T_m(x)^N`
and its `[x^1]`-slice, see the docstring on `Rseq`), and its generating
function has poles dividing `(T.charpoly^*)^2` (where `(·)^*` is the
reciprocal polynomial). Composing with `SequelRatGF.transfer_GF_rational`
upgrades this to the rational-GF conclusion that the `d = 1` GF's poles are
reciprocal eigenvalues of `T`, with multiplicity at most `2`.

Proved by composing `Rseq_charAct_eq_boundary` (the first `p(E_n)` reduces
`R(n+\cdot)` to `Boundary(n)`) with `boundary_charAct_eq_zero` (the second
`p(E_n)` kills `Boundary(n)`).

The arbitrary-`d` generalisation iterates this argument: each application of
`p(E_n)` reduces an order-`d` convolution to an order-`(d-1)` convolution times
finitely many boundary terms, giving the `p^{d+1}`-recurrence after `d+1`
applications. Formalising the full induction is a separate session. -/
theorem Rseq_p_squared_recurrence (T A : Matrix ι ι R) (u v : ι → R) (n : ℕ) :
    ∑ l ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff l *
        ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
          T.charpoly.coeff k * Rseq T A u v (n + l + k) = 0 := by
  have h1 : ∀ l, ∑ k ∈ Finset.range (T.charpoly.natDegree + 1),
      T.charpoly.coeff k * Rseq T A u v (n + l + k) = Boundary T A u v (n + l) := by
    intro l
    exact Rseq_charAct_eq_boundary T A u v (n + l)
  simp_rw [h1]
  exact boundary_charAct_eq_zero T A u v n

end OrigamiCone.Sequel

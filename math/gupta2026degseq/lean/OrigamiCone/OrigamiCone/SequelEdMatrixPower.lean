import OrigamiCone.SequelEdTransfer
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Data.Matrix.Basic

/-!
# Matrix-power identity for `cnPoly` — scaffold (Item 6d, Task B.0)

Foundation module for the transfer-matrix identity of the sequel paper's
`lem:ratGF` (§8.1):
```
c_{m,n}(x) = u(x)ᵀ (I − z T_m(x))⁻¹ v(x)
```
formalised at the coefficient level as
```
cnPoly m (n + 2) = leftBdyVecNorm ⬝ᵥ ((transferMatrix m)^n *ᵥ rightBdyVec)
```

## What this module builds

The **right-extension weight** `phi m n s₀ := ((T_m)^n *ᵥ rightBdyVec) s₀`
packages the sum "extend `s₀ = (c₀, c₁)` to a length-`(n + 2)` admissible
column sequence and weigh by the product of interior + right-boundary
extremum polynomials" as a single function of `s₀`.

* `phi_zero` : `phi m 0 s₀ = rightBdyVec m s₀`.
* `phi_succ` : `phi m (n + 1) s₀ = ∑ s₁, T[s₀, s₁] * phi m n s₁`.

Together these give a structural recurrence on `phi` in `n` that shifts the
work of proving the transfer identity onto downstream modules (`B.1`, `B.2`,
`B.3`) without re-exposing raw `T^n *ᵥ v` arithmetic:

* **B.1**: `phi m n s₀ = ∑_{admissible extensions} (product of column
  weights)`.  Prove by induction on `n`, unfolding `phi_succ` at each step.
* **B.2**: `cnPoly m (n + 2) = ∑_{s₀ normalized} leftBdyVec m s₀ * phi m n s₀`
  via the `hCol ↔ pathToHeight` bijection (`SequelEdTransfer` commits
  5c-iv-c + 5c-v-a).
* **B.3**: `cnPoly_eq_matrix_power` assembly.

## Provenance

* `transferMatrix m` — `SequelEdTransfer` commit 3 (`dc84dc77`).
* `rightBdyVec m` — `SequelEdTransfer` commit 17 (`e1aa1f63`).
* `leftBdyVecNorm m` — `SequelEdTransfer` commit 18 (`1f3472dc`).

## Axioms

`phi_zero` and `phi_succ` depend on axioms `[propext, Classical.choice,
Quot.sound]` — standard baseline.

No `sorry`, no `native_decide`, no `Float` axioms.
-/

namespace OrigamiCone.Sequel

open Matrix

/-- **Right-extension weight** — the value of `T^n *ᵥ rightBdyVec` at a
starting transfer state `s₀ = (c₀, c₁)`.

Interpreted paperwise: this is the sum over admissible column-sequence
extensions `(c₂, c₃, …, c_{n+1})` of length `n` (so the total sequence has
`n + 2` columns, matching an `m × (n + 2)` grid), weighted by the product
of the extremum-count polynomials contributed by columns `1, 2, …, n + 1`:
interior extremum weights for columns `1..n` (via `columnExtremaCount` on
the triple `(c_{k-1}, c_k, c_{k+1})`) plus the right-boundary extremum
weight for column `n + 1` (via `IsBoundaryExtremum`).

The interpretation as a sequence sum is not proved in THIS module; it is
deferred to Task B.1.  The definition as `T^n *ᵥ rightBdyVec` at `s₀` is
what makes the recurrence below immediate. -/
noncomputable def phi (m : ℕ) (n : ℕ) (s₀ : TransferState m) : Polynomial ℤ :=
  ((transferMatrix m) ^ n *ᵥ (rightBdyVec m)) s₀

/-- **Base case**: `phi m 0 s₀ = rightBdyVec m s₀`.

For a 2-column grid, the "extension" is empty; the only weight is that of
column 1 (which is the right boundary), so `phi m 0 s₀ = rightBdyVec m s₀`. -/
lemma phi_zero (m : ℕ) (s₀ : TransferState m) :
    phi m 0 s₀ = rightBdyVec m s₀ := by
  unfold phi
  simp [pow_zero, Matrix.one_mulVec]

/-- **Recurrence**: `phi m (n + 1) s₀ = ∑ s₁, T[s₀, s₁] * phi m n s₁`.

Extending an `(n + 2)`-column sequence to `(n + 3)` columns amounts to
picking the next transfer state `s₁ = (c₁, c₂)` (which forces `c₁` to
match `s₀`'s right column via the T-gate) and weighing by the interior
extremum count at column `1` (captured by `T[s₀, s₁]`), then continuing
with the `phi m n s₁` weight of the remaining `(n + 2)`-column extension.

The proof: `T^(n+1) = T * T^n` (via `pow_succ'`), so
`T^(n+1) *ᵥ v = T *ᵥ (T^n *ᵥ v)`; then `(T *ᵥ w) s₀ = ∑ s₁, T[s₀, s₁] *
w s₁` by definition of `mulVec`. -/
lemma phi_succ (m n : ℕ) (s₀ : TransferState m) :
    phi m (n + 1) s₀ = ∑ s₁ : TransferState m, transferMatrix m s₀ s₁ * phi m n s₁ := by
  unfold phi
  rw [pow_succ' (transferMatrix m) n, ← Matrix.mulVec_mulVec]
  rfl

/-- **Full-transfer identity restated in terms of `phi`**.  The dot-product
form `leftBdyVecNorm ⬝ᵥ (T^n *ᵥ rightBdyVec)` — the RHS of the paper's
`lem:ratGF` — factors as a sum of `leftBdyVec` weights times `phi`.

This is the "left side" companion to `phi`'s "right side" weight, and
combining them recovers the full extremum weight of a length-`(n + 2)`
admissible column sequence normalized at the origin. -/
lemma leftBdyVecNorm_dotProduct_pow_mulVec_rightBdyVec
    (m n : ℕ) :
    leftBdyVecNorm m ⬝ᵥ ((transferMatrix m) ^ n *ᵥ (rightBdyVec m))
      = ∑ s₀ : TransferState m, leftBdyVecNorm m s₀ * phi m n s₀ := by
  unfold phi
  rfl

/-! ## Task B.1: `phi` as a sum over transfer-state paths

The recurrence `phi_zero` + `phi_succ` unfolds `phi m n s₀` into an explicit
sum over length-`(n + 1)` "paths" `p : Fin (n + 1) → TransferState m` starting
at `s₀`, weighted by the product of the transfer-matrix entries along the
path times the right-boundary vector at the terminal state.

This is the specialisation of the standard "matrix power = sum over path
products" identity to `phi`.  Proved by induction on `n` using the
recurrence, without invoking a general path-sum lemma. -/

/-- **Helper**: double-sum collapse.  For a nested sum
`∑ s₁, T[s₀, s₁] · ∑ path ∈ filter (path 0 = s₁), f path` the inner filter
picks the unique `s₁ = path 0`, so the double sum collapses to a single sum
over `path`, with the outer factor evaluated at `path 0`. -/
private lemma sum_over_sInner_collapse (m n : ℕ) (s₀ : TransferState m)
    (f : (Fin (n + 1) → TransferState m) → Polynomial ℤ) :
    (∑ s₁ : TransferState m,
      transferMatrix m s₀ s₁ *
        ∑ path ∈ (Finset.univ : Finset (Fin (n + 1) → TransferState m)).filter
                    (fun path => path 0 = s₁),
          f path)
    = ∑ path : Fin (n + 1) → TransferState m,
        transferMatrix m s₀ (path 0) * f path := by
  simp_rw [Finset.mul_sum, Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun path _ => ?_
  rw [Finset.sum_eq_single (path 0)]
  · simp
  · intros s₁ _ hs
    simp only [ite_eq_right_iff]
    intro hp; exfalso; exact hs hp.symm
  · intro h; exact absurd (Finset.mem_univ _) h

/-- **Helper**: reindex ∑ over `Fin (n + 1) → TransferState m` paths (weighted
by `T[s₀, path 0] · f path`) as a sum over `Fin (n + 2) → TransferState m`
paths in the target filter `path' 0 = s₀`, using the bijection
`path ↦ Fin.cons s₀ path`. -/
private lemma sum_over_paths_reindex (m n : ℕ) (s₀ : TransferState m)
    (f : (Fin (n + 1) → TransferState m) → Polynomial ℤ) :
    (∑ path : Fin (n + 1) → TransferState m, transferMatrix m s₀ (path 0) * f path)
    = ∑ path' ∈ (Finset.univ : Finset (Fin (n + 2) → TransferState m)).filter
                  (fun path' => path' 0 = s₀),
        transferMatrix m (path' 0) (path' (0 : Fin (n + 1)).succ) *
          f (Fin.tail path') := by
  refine Finset.sum_bij
      (fun (path : Fin (n + 1) → TransferState m) (_ : path ∈ (Finset.univ : Finset _))
        => (Fin.cons s₀ path : Fin (n + 2) → TransferState m))
      ?_ ?_ ?_ ?_
  · intros path _; simp
  · intros p₁ _ p₂ _ h
    show p₁ = p₂
    have h' : (Fin.cons s₀ p₁ : Fin (n + 2) → TransferState m) = Fin.cons s₀ p₂ := h
    have : Fin.tail (Fin.cons s₀ p₁ : Fin (n + 2) → TransferState m)
          = Fin.tail (Fin.cons s₀ p₂ : Fin (n + 2) → TransferState m) := by rw [h']
    simpa using this
  · intros path' hpath'
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hpath'
    refine ⟨Fin.tail path', Finset.mem_univ _, ?_⟩
    ext k
    rcases Fin.eq_zero_or_eq_succ k with hk | ⟨j, hk⟩
    · subst hk; simp [hpath'.symm]
    · subst hk; simp [Fin.tail]
  · intros path _; simp

/-- **B.1**: `phi m n s₀` unfolds as a sum over length-`(n + 1)` transfer-state
paths starting at `s₀`, weighted by the product of transfer-matrix entries
along the path times the right-boundary vector at the terminal state.

Interpretation: `phi m n s₀` is the total extremum-weight of all admissible
length-`(n + 2)` column-sequence extensions of `s₀ = (c₀, c₁)`, where each
transfer-state pair `(path k, path (k+1))` = `((c_k, c_{k+1}), (c_{k+1}, c_{k+2}))`
carries the interior-extremum weight of column `k+1` (via `transferMatrix`),
and the final state `path (Fin.last n) = (c_{n}, c_{n+1})` carries the right-
boundary extremum weight (via `rightBdyVec`).

The proof is a clean induction on `n`, using `phi_zero`, `phi_succ`, and the
two `Finset.sum_bij`-based helpers above.  No general "matrix power = path
sum" lemma is needed. -/
theorem phi_eq_path_sum (m n : ℕ) (s₀ : TransferState m) :
    phi m n s₀
      = ∑ path ∈ (Finset.univ : Finset (Fin (n + 1) → TransferState m)).filter
            (fun path => path 0 = s₀),
          (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
            rightBdyVec m (path (Fin.last n)) := by
  induction n generalizing s₀ with
  | zero =>
    rw [phi_zero]
    have hfilter : ((Finset.univ : Finset (Fin 1 → TransferState m)).filter
                      (fun path => path 0 = s₀))
                  = {fun _ => s₀} := by
      ext p
      simp only [Finset.mem_filter, Finset.mem_singleton, Finset.mem_univ, true_and]
      refine ⟨fun hp => ?_, fun hp => ?_⟩
      · ext k
        have : k = 0 := Subsingleton.elim _ _
        rw [this]; exact hp
      · rw [hp]
    rw [hfilter, Finset.sum_singleton]
    simp
  | succ n ih =>
    rw [phi_succ]
    simp_rw [ih]
    rw [sum_over_sInner_collapse m n s₀
        (fun path => (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
                       rightBdyVec m (path (Fin.last n)))]
    rw [sum_over_paths_reindex m n s₀
        (fun path => (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
                       rightBdyVec m (path (Fin.last n)))]
    refine Finset.sum_congr rfl fun path' hpath' => ?_
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hpath'
    rw [Fin.prod_univ_succ (fun k : Fin (n + 1) =>
          transferMatrix m (path' k.castSucc) (path' k.succ))]
    have h_last : Fin.tail path' (Fin.last n) = path' (Fin.last (n + 1)) := by
      show path' (Fin.last n).succ = path' (Fin.last (n + 1))
      rfl
    rw [h_last]
    -- Align the two products indexed by `k : Fin n`: `Fin.tail path' k = path' k.succ`
    -- (definitional), and `k.castSucc.succ = k.succ.castSucc` (proved via `ext; simp`).
    conv_lhs =>
      rw [show
        (∏ k : Fin n, transferMatrix m (Fin.tail path' k.castSucc)
                                        (Fin.tail path' k.succ))
        = ∏ k : Fin n, transferMatrix m (path' k.succ.castSucc)
                                          (path' k.succ.succ) from by
          refine Finset.prod_congr rfl fun k _ => ?_
          show transferMatrix m (path' k.castSucc.succ) (path' k.succ.succ)
                = transferMatrix m (path' k.succ.castSucc) (path' k.succ.succ)
          have : k.castSucc.succ = k.succ.castSucc (n := n + 1) := by ext; simp
          rw [this]]
    -- `(Fin.castSucc 0 : Fin (n + 2)) = 0` is definitional.
    have h_cs0 : (Fin.castSucc (0 : Fin (n + 1)) : Fin (n + 2)) = 0 := rfl
    rw [h_cs0]
    ring

/-! ## Task B.2 (partial): weight decomposition of `heightToPath h`

For a canonical height `h` on `Cell m (n + 2)`, the value

  `leftBdyVec m ((heightToPath h) 0) *
     (∏ k : Fin n, transferMatrix m ((heightToPath h) k.castSucc) ((heightToPath h) k.succ)) *
     rightBdyVec m ((heightToPath h) (Fin.last n))`

equals `monomial (numExtrema h) 1`.

The three factors decompose via the three column-count lemmas in
`SequelEdTransfer` (commit 16, `cf92fd34`):

* `col_count_left_boundary_eq` turns `leftBdyVec` into the strict-local-
  extremum count at column 0.
* `col_count_interior_eq` turns each transferMatrix factor into the SLE count
  at the corresponding interior column.
* `col_count_right_boundary_eq` turns `rightBdyVec` into the SLE count at
  column `n + 1`.

Product of monomials becomes a monomial with summed exponents, and
`numExtrema_eq_sum_over_cols` matches that sum with `numExtrema h`.

This is the "weight preserved" clause of the bijection between canonical
heights and consistent normalized transfer-state paths that Task B.2 requires;
the injection/surjection halves of the bijection remain future work. -/

open Polynomial in
private lemma leftBdyVec_heightToPath {m n : ℕ} {h : Cell m (n + 2) → ℤ}
    (hh : IsHeight h) :
    leftBdyVec m ((heightToPath h hh (by omega : 2 ≤ n + 2))
                    (⟨0, by omega⟩ : Fin (n + 2 - 1)))
      = monomial
          ((Finset.univ.filter fun i : Fin m =>
              IsStrictLocalExtremum h (i, ⟨0, by omega⟩)).card)
          (1 : ℤ) := by
  unfold leftBdyVec
  rw [show ((heightToPath h hh (by omega : 2 ≤ n + 2))
              (⟨0, by omega⟩ : Fin (n + 2 - 1))).val.1
         = hCol h ⟨0, by omega⟩ from rfl,
       show ((heightToPath h hh (by omega : 2 ≤ n + 2))
              (⟨0, by omega⟩ : Fin (n + 2 - 1))).val.2
         = hCol h ⟨1, by omega⟩ from rfl,
       ← col_count_left_boundary_eq hh (by omega : 0 < n + 2) (by omega : 1 < n + 2)]

open Polynomial in
private lemma rightBdyVec_heightToPath {m n : ℕ} {h : Cell m (n + 2) → ℤ}
    (hh : IsHeight h) :
    rightBdyVec m ((heightToPath h hh (by omega : 2 ≤ n + 2))
                     (Fin.last n : Fin (n + 2 - 1)))
      = monomial
          ((Finset.univ.filter fun i : Fin m =>
              IsStrictLocalExtremum h (i, ⟨n + 1, by omega⟩)).card)
          (1 : ℤ) := by
  unfold rightBdyVec
  rw [show ((heightToPath h hh (by omega : 2 ≤ n + 2))
              (Fin.last n : Fin (n + 2 - 1))).val.1
         = hCol h ⟨n, by omega⟩ from rfl,
       show ((heightToPath h hh (by omega : 2 ≤ n + 2))
              (Fin.last n : Fin (n + 2 - 1))).val.2
         = hCol h ⟨n + 1, by omega⟩ from rfl]
  have h_n1 : (⟨(n + 2 : ℕ) - 1, by omega⟩ : Fin (n + 2)) = ⟨n + 1, by omega⟩ := by
    apply Fin.ext; simp
  have h_n2 : (⟨(n + 2 : ℕ) - 2, by omega⟩ : Fin (n + 2)) = ⟨n, by omega⟩ := by
    apply Fin.ext; simp
  have H := col_count_right_boundary_eq hh (by omega : 1 < n + 2)
  rw [h_n1, h_n2] at H
  rw [← H]

open Polynomial in
private lemma transferMatrix_heightToPath {m n : ℕ} {h : Cell m (n + 2) → ℤ}
    (hh : IsHeight h) (k : Fin n) :
    transferMatrix m
      ((heightToPath h hh (by omega : 2 ≤ n + 2)) k.castSucc)
      ((heightToPath h hh (by omega : 2 ≤ n + 2)) k.succ)
    = monomial
        ((Finset.univ.filter fun i : Fin m =>
            IsStrictLocalExtremum h (i, ⟨k.val + 1, by have := k.isLt; omega⟩)).card)
        (1 : ℤ) := by
  unfold transferMatrix
  have h_match : ((heightToPath h hh (by omega : 2 ≤ n + 2)) k.castSucc).val.2
              = ((heightToPath h hh (by omega : 2 ≤ n + 2)) k.succ).val.1 := rfl
  rw [if_pos h_match, X_pow_eq_monomial]
  show monomial (columnExtremaCount
        (hCol h ⟨k.castSucc.val, by have := k.castSucc.isLt; omega⟩)
        (hCol h ⟨k.castSucc.val + 1, by have := k.castSucc.isLt; omega⟩)
        (hCol h ⟨k.succ.val + 1, by have := k.succ.isLt; omega⟩)) 1
    = monomial ((Finset.univ.filter fun i : Fin m =>
          IsStrictLocalExtremum h (i, ⟨k.val + 1, by have := k.isLt; omega⟩)).card) 1
  have H := col_count_interior_eq hh (k.val + 1) (by omega : 0 < k.val + 1)
              (by have := k.isLt; omega : (k.val + 1) + 1 < n + 2)
  congr 1
  rw [H]
  congr 1

open Polynomial in
private lemma prod_monomial_fin_one (n : ℕ) (f : Fin n → ℕ) :
    (∏ x, monomial (f x) (1 : ℤ)) = monomial (∑ x, f x) 1 := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Fin.prod_univ_succ, Fin.sum_univ_succ, ih (fun i => f i.succ),
        monomial_mul_monomial, one_mul]

private lemma sum_over_fin_split_first_last (n : ℕ) (f : Fin (n + 2) → ℕ) :
    ∑ j : Fin (n + 2), f j
    = f ⟨0, by omega⟩ + (∑ x : Fin n, f ⟨x.val + 1, by have := x.isLt; omega⟩)
        + f ⟨n + 1, by omega⟩ := by
  rw [Fin.sum_univ_succ, Fin.sum_univ_castSucc]
  rw [show (f 0 : ℕ) = f ⟨0, by omega⟩ from rfl]
  rw [show f (Fin.last n).succ = f ⟨n + 1, by omega⟩ from rfl]
  rw [show (∑ i : Fin n, f i.castSucc.succ)
        = ∑ x : Fin n, f ⟨x.val + 1, by have := x.isLt; omega⟩ from
      Finset.sum_congr rfl fun _ _ => rfl]
  ring

/-- **Weight decomposition of `heightToPath h`** (B.2 core weight identity).
For a canonical height `h`, the product `leftBdyVec (path 0) * ∏ T-transitions
* rightBdyVec (path last)` where `path = heightToPath h` equals
`monomial (numExtrema h) 1`.

Combining the three per-factor decompositions with `prod_monomial_fin_one`
turns the product-of-monomials into a single monomial with sum-of-counts
exponent; `numExtrema_eq_sum_over_cols` + `sum_over_fin_split_first_last`
match that sum with `numExtrema h`. -/
theorem pathWeight_heightToPath {m n : ℕ} {h : Cell m (n + 2) → ℤ}
    (hh : IsHeight h) :
    leftBdyVec m ((heightToPath h hh (by omega : 2 ≤ n + 2))
                    (⟨0, by omega⟩ : Fin (n + 2 - 1))) *
    (∏ k : Fin n, transferMatrix m
      ((heightToPath h hh (by omega : 2 ≤ n + 2)) k.castSucc)
      ((heightToPath h hh (by omega : 2 ≤ n + 2)) k.succ)) *
    rightBdyVec m ((heightToPath h hh (by omega : 2 ≤ n + 2))
                     (Fin.last n : Fin (n + 2 - 1)))
    = Polynomial.monomial (numExtrema h) (1 : ℤ) := by
  rw [leftBdyVec_heightToPath hh, rightBdyVec_heightToPath hh]
  simp_rw [transferMatrix_heightToPath hh]
  rw [prod_monomial_fin_one]
  rw [Polynomial.monomial_mul_monomial, Polynomial.monomial_mul_monomial, one_mul]
  apply congrArg (fun k => (Polynomial.monomial k) (1 : ℤ))
  rw [numExtrema_eq_sum_over_cols h,
      sum_over_fin_split_first_last n (fun j => (Finset.univ.filter fun i : Fin m =>
                        IsStrictLocalExtremum h (i, j)).card)]

/-! ## Task B.2 (part 2): consistency + path-to-columns bijection infrastructure

To bijection canonical heights with transfer-state paths, we need:

* A predicate `IsConsistent` isolating paths whose consecutive states share
  their middle column (so the transfer product is nonzero).
* The extraction map `pathToCols : path ↦ Fin (n+2) → PathColouring m` that
  reads off the column sequence.
* Round-trip identities linking `heightToPath / pathToCols / pathToHeight /
  hCol`.
* Two "zero-weight" lemmas showing that non-consistent and non-normalized
  paths contribute zero to the full sum, so the sum-over-all-paths on the
  RHS of `lem:ratGF` reduces to the sum over consistent normalized paths.
-/

/-- A path in the transfer-state graph is **consistent** if consecutive states
share their middle column: `(path k).val.2 = (path (k+1)).val.1`. -/
def IsConsistent {m n : ℕ} (path : Fin (n + 1) → TransferState m) : Prop :=
  ∀ k : Fin n, (path k.castSucc).val.2 = (path k.succ).val.1

instance {m n : ℕ} (path : Fin (n + 1) → TransferState m) :
    Decidable (IsConsistent path) := by unfold IsConsistent; infer_instance

/-- Extract a column sequence from a transfer path: the first `n+1` columns
come from the `.val.1` of each state; the last column comes from the
`.val.2` of the final state. -/
noncomputable def pathToCols {m n : ℕ} (path : Fin (n + 1) → TransferState m) :
    Fin (n + 2) → PathColouring m :=
  Fin.snoc (fun k : Fin (n + 1) => (path k).val.1) ((path (Fin.last n)).val.2)

lemma pathToCols_castSucc {m n : ℕ} (path : Fin (n + 1) → TransferState m)
    (k : Fin (n + 1)) :
    pathToCols path k.castSucc = (path k).val.1 := by
  unfold pathToCols; rw [Fin.snoc_castSucc]

lemma pathToCols_last {m n : ℕ} (path : Fin (n + 1) → TransferState m) :
    pathToCols path (Fin.last (n + 1)) = (path (Fin.last n)).val.2 := by
  unfold pathToCols; rw [Fin.snoc_last]

/-- Round-trip on canonical heights: `pathToCols (heightToPath h) = hCol h`. -/
lemma pathToCols_heightToPath {m n : ℕ} {h : Cell m (n + 2) → ℤ}
    (hh : IsHeight h) :
    pathToCols (heightToPath h hh (by omega : 2 ≤ n + 2))
      = fun j : Fin (n + 2) => hCol h j := by
  unfold pathToCols
  ext j
  rcases (Fin.eq_castSucc_or_eq_last j) with ⟨j', hj'⟩ | hj'
  · subst hj'; rw [Fin.snoc_castSucc]; rfl
  · subst hj'; rw [Fin.snoc_last]; rfl

lemma pathToCols_zero {m n : ℕ} (path : Fin (n + 1) → TransferState m) :
    pathToCols path (⟨0, by omega⟩ : Fin (n + 2)) = (path 0).val.1 := by
  have : (⟨0, by omega⟩ : Fin (n + 2)) = (⟨0, by omega⟩ : Fin (n + 1)).castSucc := by
    apply Fin.ext; rfl
  rw [this, pathToCols_castSucc]
  rfl

/-- For consistent path, `pathToCols path ⟨j.val + 1⟩ = (path j).val.2`. -/
lemma pathToCols_succ_eq_val2 {m n : ℕ} (path : Fin (n + 1) → TransferState m)
    (hcons : IsConsistent path) (j : Fin (n + 1)) :
    pathToCols path ⟨j.val + 1, by have := j.isLt; omega⟩ = (path j).val.2 := by
  by_cases h : j.val + 1 < n + 1
  · have hfin : (⟨j.val + 1, by have := j.isLt; omega⟩ : Fin (n + 2))
              = (⟨j.val + 1, h⟩ : Fin (n + 1)).castSucc := by
      apply Fin.ext; simp
    rw [hfin, pathToCols_castSucc]
    have hj_n : j.val < n := by omega
    have := hcons ⟨j.val, hj_n⟩
    have heq1 : (⟨j.val, hj_n⟩ : Fin n).castSucc = j := by apply Fin.ext; rfl
    have heq2 : (⟨j.val, hj_n⟩ : Fin n).succ = ⟨j.val + 1, h⟩ := by apply Fin.ext; rfl
    rw [heq1, heq2] at this
    rw [this]
  · push_neg at h
    have hj_last : j = Fin.last n := by
      apply Fin.ext; have := j.isLt; show j.val = n; omega
    subst hj_last
    have hfin : (⟨(Fin.last n : Fin (n + 1)).val + 1,
                  by have := (Fin.last n : Fin (n + 1)).isLt; omega⟩ : Fin (n + 2))
              = Fin.last (n + 1) := by apply Fin.ext; simp
    rw [hfin, pathToCols_last]

/-- Every column of `pathToCols path` is a proper 3-colouring. -/
lemma pathToCols_col_proper {m n : ℕ} (path : Fin (n + 1) → TransferState m)
    (j : Fin (n + 2)) :
    IsPathProperColouring (pathToCols path j) := by
  rcases (Fin.eq_castSucc_or_eq_last j) with ⟨j', hj'⟩ | hj'
  · subst hj'; rw [pathToCols_castSucc]; exact (path j').property.1
  · subst hj'; rw [pathToCols_last]; exact (path (Fin.last n)).property.2.1

/-- Consecutive columns of `pathToCols path` are horizontally distinct
(assuming path is consistent). -/
lemma pathToCols_adjacent {m n : ℕ} (path : Fin (n + 1) → TransferState m)
    (hcons : IsConsistent path)
    (k : ℕ) (hk : k + 1 < n + 2) (r : Fin m) :
    pathToCols path ⟨k, by omega⟩ r ≠ pathToCols path ⟨k + 1, hk⟩ r := by
  have hk_lt : k < n + 1 := by omega
  have h1 : pathToCols path ⟨k, by omega⟩ = (path ⟨k, hk_lt⟩).val.1 := by
    have : (⟨k, by omega⟩ : Fin (n + 2)) = (⟨k, hk_lt⟩ : Fin (n + 1)).castSucc := by
      apply Fin.ext; rfl
    rw [this, pathToCols_castSucc]
  have h2 : pathToCols path ⟨k + 1, hk⟩ = (path ⟨k, hk_lt⟩).val.2 := by
    have := pathToCols_succ_eq_val2 path hcons ⟨k, hk_lt⟩
    convert this using 2
  rw [h1, h2]
  exact (path ⟨k, hk_lt⟩).property.2.2 r

/-- For consistent normalized `path`, the reconstructed height
`pathToHeight (pathToCols path)` is canonical (vanishes at every origin cell). -/
lemma pathToHeight_pathToCols_isCanonical
    {m n : ℕ} (path : Fin (n + 1) → TransferState m)
    (_hcons : IsConsistent path) (hnorm : IsNormalizedLeft (path 0))
    (p : Cell m (n + 2)) (hp1 : p.1.val = 0) (hp2 : p.2.val = 0) :
    pathToHeight (pathToCols path) p = 0 := by
  obtain ⟨hm, _⟩ := hnorm
  have hp_eq : p = ((⟨0, hm⟩, ⟨0, by omega⟩) : Cell m (n + 2)) := by
    apply Prod.ext
    · apply Fin.ext; exact hp1
    · apply Fin.ext; exact hp2
  rw [hp_eq, pathToHeight_origin]

/-- **Round-trip on the path side**: for consistent normalized path `p`,
`heightToPath (pathToHeight (pathToCols p)) = p`. -/
lemma heightToPath_pathToHeight_pathToCols
    {m n : ℕ} (path : Fin (n + 1) → TransferState m)
    (hcons : IsConsistent path) (hnorm : IsNormalizedLeft (path 0)) :
    heightToPath (pathToHeight (pathToCols path))
      (pathToHeight_isHeight (pathToCols path)
        (pathToCols_col_proper path)
        (fun k hk r => pathToCols_adjacent path hcons k hk r))
      (by omega : 2 ≤ n + 2) = path := by
  obtain ⟨hm, h_00⟩ := hnorm
  have h_pathToCols_00 : pathToCols path (⟨0, by omega⟩ : Fin (n + 2)) ⟨0, hm⟩ = 0 := by
    rw [pathToCols_zero]; exact h_00
  set h := pathToHeight (pathToCols path)
  have hCol_h : ∀ (i : Fin m) (jval : ℕ) (hjv : jval < n + 2),
      hCol h ⟨jval, hjv⟩ i = pathToCols path ⟨jval, hjv⟩ i := by
    intros i jval hjv
    unfold hCol
    have := hCol_pathToHeight (pathToCols path)
      (pathToCols_col_proper path)
      (fun k hk r => pathToCols_adjacent path hcons k hk r)
      i.val i.isLt jval hjv
    have h_fi : (⟨i.val, i.isLt⟩ : Fin m) = i := Fin.ext rfl
    rw [h_fi] at this
    rw [this, h_pathToCols_00, sub_zero]
  ext j
  apply Subtype.ext
  apply Prod.ext
  · show hCol h ⟨j.val, by have := j.isLt; omega⟩ = (path j).val.1
    funext i
    rw [hCol_h i j.val (by have := j.isLt; omega)]
    have hfin : (⟨j.val, by have := j.isLt; omega⟩ : Fin (n + 2))
              = j.castSucc := by apply Fin.ext; rfl
    rw [hfin, pathToCols_castSucc]
  · show hCol h ⟨j.val + 1, by have := j.isLt; omega⟩ = (path j).val.2
    funext i
    rw [hCol_h i (j.val + 1) (by have := j.isLt; omega)]
    rw [pathToCols_succ_eq_val2 path hcons j]

/-- **heightToPath injectivity on canonical heights**. -/
lemma heightToPath_injOn_canonical {m n : ℕ} (hm : 0 < m)
    {h1 h2 : Cell m (n + 2) → ℤ}
    (hh1 : IsHeight h1) (hh2 : IsHeight h2)
    (h1_00 : h1 ((⟨0, hm⟩, ⟨0, by omega⟩) : Cell m (n + 2)) = 0)
    (h2_00 : h2 ((⟨0, hm⟩, ⟨0, by omega⟩) : Cell m (n + 2)) = 0)
    (heq : heightToPath h1 hh1 (by omega : 2 ≤ n + 2)
         = heightToPath h2 hh2 (by omega : 2 ≤ n + 2)) :
    h1 = h2 := by
  funext p
  obtain ⟨i, j⟩ := p
  have H1 := pathToHeight_hCol hh1 i.val i.isLt j.val j.isLt
  have H2 := pathToHeight_hCol hh2 i.val i.isLt j.val j.isLt
  have h1_p : h1 (i, j) = pathToHeight (hCol h1) (i, j) + h1 ((⟨0, hm⟩, ⟨0, by omega⟩)) := by
    have h_fi : (⟨i.val, i.isLt⟩ : Fin m) = i := Fin.ext rfl
    have h_fj : (⟨j.val, j.isLt⟩ : Fin (n + 2)) = j := Fin.ext rfl
    rw [h_fi, h_fj] at H1
    have h_fin : ((⟨0, by omega⟩, ⟨0, by omega⟩) : Cell m (n + 2))
              = ((⟨0, hm⟩, ⟨0, by omega⟩) : Cell m (n + 2)) := rfl
    rw [h_fin] at H1
    linarith
  have h2_p : h2 (i, j) = pathToHeight (hCol h2) (i, j) + h2 ((⟨0, hm⟩, ⟨0, by omega⟩)) := by
    have h_fi : (⟨i.val, i.isLt⟩ : Fin m) = i := Fin.ext rfl
    have h_fj : (⟨j.val, j.isLt⟩ : Fin (n + 2)) = j := Fin.ext rfl
    rw [h_fi, h_fj] at H2
    have h_fin : ((⟨0, by omega⟩, ⟨0, by omega⟩) : Cell m (n + 2))
              = ((⟨0, hm⟩, ⟨0, by omega⟩) : Cell m (n + 2)) := rfl
    rw [h_fin] at H2
    linarith
  have hCol_eq : hCol h1 = hCol h2 := by
    funext j'
    have E1 := pathToCols_heightToPath hh1 (n := n)
    have E2 := pathToCols_heightToPath hh2 (n := n)
    have : pathToCols (heightToPath h1 hh1 (by omega : 2 ≤ n + 2))
         = pathToCols (heightToPath h2 hh2 (by omega : 2 ≤ n + 2)) := by rw [heq]
    rw [E1, E2] at this
    exact congrFun this j'
  rw [h1_p, h2_p, hCol_eq, h1_00, h2_00]

/-- **Zero weight for non-consistent path**: some transferMatrix factor is 0. -/
lemma pathWeight_zero_of_not_consistent {m n : ℕ}
    (path : Fin (n + 1) → TransferState m)
    (hnot : ¬ IsConsistent path) :
    leftBdyVecNorm m (path 0) *
      (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
      rightBdyVec m (path (Fin.last n)) = 0 := by
  unfold IsConsistent at hnot
  push_neg at hnot
  obtain ⟨k, hk⟩ := hnot
  have h_zero : transferMatrix m (path k.castSucc) (path k.succ) = 0 := by
    unfold transferMatrix; rw [if_neg hk]
  have h_prod_zero : (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) = 0 :=
    Finset.prod_eq_zero (Finset.mem_univ k) h_zero
  rw [h_prod_zero]; ring

/-- **Zero weight for non-normalized path**: `leftBdyVecNorm = 0`. -/
lemma pathWeight_zero_of_not_normalized {m n : ℕ}
    (path : Fin (n + 1) → TransferState m)
    (hnot : ¬ IsNormalizedLeft (path 0)) :
    leftBdyVecNorm m (path 0) *
      (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
      rightBdyVec m (path (Fin.last n)) = 0 := by
  rw [leftBdyVecNorm_of_not_normalized hnot]
  ring

/-! ## Task B.2 headline theorem + B.3 assembly

Combining B.1 (`phi_eq_path_sum`) with the bijection infrastructure from
part 2 yields the paper's `lem:ratGF` at the coefficient level:

  `cnPoly m (n + 2) = leftBdyVecNorm m ⬝ᵥ (T_m^n *ᵥ rightBdyVec m)`.

The proof pipeline:

* B.2 headline `cnPoly_eq_phi_sum`:
  1. Unfold RHS via `phi_eq_path_sum` (B.1) and flatten the double sum.
  2. Restrict the sum to consistent + normalized paths (rest is zero by
     the two zero-weight lemmas).
  3. Unfold LHS via `cnPoly_eq_sum_over_heights` (SequelEdTransfer 5d,
     `8e1a8c81`).
  4. Bijection via `Finset.sum_bij`: forward = `heightToPath h`; the four
     obligations discharge to (well-def: consistency is `rfl` +
     normalization is `hCol h 0 ⟨0⟩ = h(0,0) mod 3 = 0`), (injectivity:
     `heightToPath_injOn_canonical`), (surjectivity: `pathToHeight
     (pathToCols path)` is canonical + round-trip via
     `heightToPath_pathToHeight_pathToCols`), (weight equality:
     `pathWeight_heightToPath` + `leftBdyVecNorm_of_normalized`).

* B.3 headline `cnPoly_eq_matrix_power` is a two-line rewrite: apply
  `cnPoly_eq_phi_sum` and `leftBdyVecNorm_dotProduct_pow_mulVec_rightBdyVec`
  (already in B.0). -/

private lemma sum_leftBdyNorm_phi_flatten (m n : ℕ) :
    (∑ s₀ : TransferState m, leftBdyVecNorm m s₀ *
      ∑ path ∈ (Finset.univ : Finset (Fin (n + 1) → TransferState m)).filter
                  (fun path => path 0 = s₀),
        (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
          rightBdyVec m (path (Fin.last n)))
    = ∑ path : Fin (n + 1) → TransferState m,
        leftBdyVecNorm m (path 0) *
          (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
          rightBdyVec m (path (Fin.last n)) := by
  simp_rw [Finset.mul_sum, Finset.sum_filter]
  rw [Finset.sum_comm]
  refine Finset.sum_congr rfl fun path _ => ?_
  rw [Finset.sum_eq_single (path 0)]
  · simp; ring
  · intros s₀ _ hs
    simp only [ite_eq_right_iff]
    intro hp; exfalso; exact hs hp.symm
  · intro h; exact absurd (Finset.mem_univ _) h

private lemma sum_over_paths_restrict (m n : ℕ) :
    (∑ path : Fin (n + 1) → TransferState m,
        leftBdyVecNorm m (path 0) *
          (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
          rightBdyVec m (path (Fin.last n)))
    = ∑ path ∈ (Finset.univ : Finset (Fin (n + 1) → TransferState m)).filter
                    (fun p => IsConsistent p ∧ IsNormalizedLeft (p 0)),
        leftBdyVecNorm m (path 0) *
          (∏ k : Fin n, transferMatrix m (path k.castSucc) (path k.succ)) *
          rightBdyVec m (path (Fin.last n)) := by
  symm
  apply Finset.sum_filter_of_ne
  intros path _ hne
  by_contra hnotP
  apply hne
  rw [not_and_or] at hnotP
  rcases hnotP with h1 | h2
  · exact pathWeight_zero_of_not_consistent path h1
  · exact pathWeight_zero_of_not_normalized path h2

/-- **B.2 headline theorem**: `cnPoly m (n + 2) = Σ s₀, leftBdyVecNorm m s₀ *
phi m n s₀`.  The paper's `lem:ratGF` split by the initial transfer state. -/
theorem cnPoly_eq_phi_sum (m n : ℕ) (hm : 0 < m) :
    cnPoly m (n + 2) = ∑ s₀ : TransferState m, leftBdyVecNorm m s₀ * phi m n s₀ := by
  simp_rw [phi_eq_path_sum]
  rw [sum_leftBdyNorm_phi_flatten]
  rw [sum_over_paths_restrict]
  rw [cnPoly_eq_sum_over_heights m (n + 2) hm (by omega : 1 ≤ n + 2)]
  refine Finset.sum_bij
      (fun h (hh : h ∈ (CanonicalHeights_finite hm (by omega : 1 ≤ n + 2)).toFinset)
        => heightToPath h
              ((Set.Finite.mem_toFinset _).mp hh).1
              (by omega : 2 ≤ n + 2))
      ?_ ?_ ?_ ?_
  · intros h hh
    simp only [Finset.mem_filter, Finset.mem_univ, true_and]
    obtain ⟨hh_isH, hh_00⟩ := (Set.Finite.mem_toFinset _).mp hh
    refine ⟨?_, ?_⟩
    · intro k; rfl
    · refine ⟨hm, ?_⟩
      show hCol h ⟨0, by omega⟩ ⟨0, hm⟩ = 0
      unfold hCol
      have h00 := hh_00 (⟨⟨0, hm⟩, ⟨0, by omega⟩⟩ : Cell m (n + 2)) rfl rfl
      rw [h00]; simp
  · intros h1 hh1 h2 hh2 heq
    obtain ⟨hh1_isH, hh1_00⟩ := (Set.Finite.mem_toFinset _).mp hh1
    obtain ⟨hh2_isH, hh2_00⟩ := (Set.Finite.mem_toFinset _).mp hh2
    have h1_00 := hh1_00 (⟨⟨0, hm⟩, ⟨0, by omega⟩⟩ : Cell m (n + 2)) rfl rfl
    have h2_00 := hh2_00 (⟨⟨0, hm⟩, ⟨0, by omega⟩⟩ : Cell m (n + 2)) rfl rfl
    exact heightToPath_injOn_canonical hm hh1_isH hh2_isH h1_00 h2_00 heq
  · intros path hp
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp
    obtain ⟨hcons, hnorm⟩ := hp
    refine ⟨pathToHeight (pathToCols path),
      (Set.Finite.mem_toFinset _).mpr ⟨?_, ?_⟩, ?_⟩
    · exact pathToHeight_isHeight (pathToCols path)
        (pathToCols_col_proper path)
        (fun k hk r => pathToCols_adjacent path hcons k hk r)
    · intro p hp1 hp2
      exact pathToHeight_pathToCols_isCanonical path hcons hnorm p hp1 hp2
    · exact heightToPath_pathToHeight_pathToCols path hcons hnorm
  · intros h hh
    obtain ⟨hh_isH, hh_00⟩ := (Set.Finite.mem_toFinset _).mp hh
    have hw := pathWeight_heightToPath hh_isH (n := n)
    have h_norm : IsNormalizedLeft ((heightToPath h hh_isH (by omega : 2 ≤ n + 2))
                    (⟨0, by omega⟩ : Fin (n + 2 - 1))) := by
      refine ⟨hm, ?_⟩
      show hCol h ⟨0, by omega⟩ ⟨0, hm⟩ = 0
      unfold hCol
      have h00 := hh_00 (⟨⟨0, hm⟩, ⟨0, by omega⟩⟩ : Cell m (n + 2)) rfl rfl
      rw [h00]; simp
    rw [← leftBdyVecNorm_of_normalized h_norm] at hw
    exact hw.symm

/-- **B.3 headline theorem** (the paper's `lem:ratGF` at the coefficient level):
`cnPoly m (n + 2) = leftBdyVecNorm m ⬝ᵥ ((transferMatrix m)^n *ᵥ rightBdyVec m)`.

Trivial `trans` of `cnPoly_eq_phi_sum` (B.2) and
`leftBdyVecNorm_dotProduct_pow_mulVec_rightBdyVec` (B.0). -/
theorem cnPoly_eq_matrix_power (m n : ℕ) (hm : 0 < m) :
    cnPoly m (n + 2)
      = leftBdyVecNorm m ⬝ᵥ ((transferMatrix m) ^ n *ᵥ rightBdyVec m) := by
  rw [cnPoly_eq_phi_sum m n hm,
      leftBdyVecNorm_dotProduct_pow_mulVec_rightBdyVec]

end OrigamiCone.Sequel

import OrigamiCone.SequelEd
import Mathlib.Algebra.Polynomial.Basic
import Mathlib.Data.Matrix.Basic

/-!
# Transfer-matrix state space for `Ed`'s per-axis polynomiality

Item 6d main substrate: instantiation of `SequelRatGF` at the paper's
column transfer matrix `T_m(x)`.  This module builds the FIRST piece —
the state space of the transfer.

## The paper's transfer picture

A height function `h : Cell m n → ℤ` on `M_{m,n}` is equivalent (via
the Ginepro–Hull bijection) to a proper 3-colouring `c : Cell m n →
ZMod 3` (`ColoringQuotient.Coloring`, `IsProperColoring`).  Reading `c`
column-by-column, each column is a proper 3-colouring of the path
`P_m` (adjacent cells within the column differ), and two adjacent
columns must have every corresponding pair of cells coloured
differently (adjacent horizontally in the grid).

For the transfer-matrix construction, the state is an *adjacent pair
of column colourings* `(u, v)` and the transfer advances to `(v, w)`
where `w` is compatible with `v`.  The weight on the edge
`(u, v) → (v, w)` records the number of strict local extrema of the
middle column `v`, which by `SequelFrozen`'s dichotomy is `0` iff the
triple `(u, v, w)` is *frozen* (constant slope `k ∈ {1, 2}`).

This module builds:

* `PathColouring m` — column colouring type `Fin m → ZMod 3`.
* `IsPathProperColouring` — adjacent-in-column cells differ.
* `PathAdjacent u v` — the two-column horizontal-adjacency condition
  (`u i ≠ v i` for all `i`, plus each column is proper).
* `Fintype` instances for the state space.

Later modules (deferred) will define the transfer matrix `T_m(x)`, the
weight generating function, and connect it to `Ed`.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
-/

namespace OrigamiCone.Sequel

/-- A colouring of the length-`m` path `P_m`: an assignment of one of
three colours (`ZMod 3`) to each of the `m` cells. -/
def PathColouring (m : ℕ) : Type := Fin m → ZMod 3

instance {m : ℕ} : Fintype (PathColouring m) := Pi.instFintype

instance {m : ℕ} : DecidableEq (PathColouring m) := by
  unfold PathColouring; infer_instance

/-- A colouring is **proper** if adjacent cells receive distinct
colours: for every `i`, if `i + 1 < m` then `c i ≠ c (i + 1)`. -/
def IsPathProperColouring {m : ℕ} (c : PathColouring m) : Prop :=
  ∀ i : Fin m, ∀ h : i.val + 1 < m, c i ≠ c ⟨i.val + 1, h⟩

instance {m : ℕ} (c : PathColouring m) : Decidable (IsPathProperColouring c) := by
  unfold IsPathProperColouring; infer_instance

/-- Two column colourings `u, v : PathColouring m` are **adjacent**
(represent two horizontally-adjacent columns of a valid grid
colouring) if both are proper and every corresponding pair of cells
differs: `u i ≠ v i` for all `i`. -/
def PathAdjacent {m : ℕ} (u v : PathColouring m) : Prop :=
  IsPathProperColouring u ∧ IsPathProperColouring v ∧ (∀ i, u i ≠ v i)

instance {m : ℕ} (u v : PathColouring m) : Decidable (PathAdjacent u v) := by
  unfold PathAdjacent; infer_instance

/-- The transfer-matrix state space for `M_{m,·}`: adjacent pairs
of proper column colourings. -/
def TransferState (m : ℕ) : Type :=
  { p : PathColouring m × PathColouring m // PathAdjacent p.1 p.2 }

instance {m : ℕ} : Fintype (TransferState m) := by
  unfold TransferState; infer_instance

instance {m : ℕ} : DecidableEq (TransferState m) := by
  unfold TransferState; infer_instance

/-! ## Column extrema count (Step 2)

For three consecutive columns `u, v, w` of a proper 3-colouring, a
cell `i` of the middle column `v` is a **strict local extremum** iff
all its present neighbours share one colour (which must equal
`u i = w i`, giving the horizontal-agreement condition).  This is the
`ZMod 3` reformulation of `IsStrictLocalExtremum` — a height cell is
a strict local extremum iff its four (or fewer, at the boundary)
neighbours all share the same height, which mod 3 is the same colour.

`SequelFrozen.isExtremum` gives the same condition indexed by `ℕ`;
we adapt to `PathColouring m = Fin m → ZMod 3` here to match the
transfer state space. -/

/-- Predicate: cell `i` of the middle column `v` is a strict local
extremum, given left column `u` and right column `w`.  All present
neighbours (`u i`, `w i` always; `v ⟨i-1, _⟩`, `v ⟨i+1, _⟩` at
interior cells) share the colour `u i`. -/
def IsColExtremum {m : ℕ} (u v w : PathColouring m) (i : Fin m) : Prop :=
  u i = w i ∧
    (∀ h : 0 < i.val, v ⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt h Nat.one_pos) i.isLt.le⟩ = u i) ∧
    (∀ h : i.val + 1 < m, v ⟨i.val + 1, h⟩ = u i)

instance {m : ℕ} (u v w : PathColouring m) (i : Fin m) :
    Decidable (IsColExtremum u v w i) := by
  unfold IsColExtremum; infer_instance

/-- Number of strict local extrema of the middle column `v` in the
adjacent triple `(u, v, w)`.  This is the row-analogue of
`Sequel.numExtrema` for a single column of a height function. -/
def columnExtremaCount {m : ℕ} (u v w : PathColouring m) : ℕ :=
  (Finset.univ.filter (fun i : Fin m => IsColExtremum u v w i)).card

/-- The extremum count is bounded by the column length. -/
theorem columnExtremaCount_le {m : ℕ} (u v w : PathColouring m) :
    columnExtremaCount u v w ≤ m := by
  have h_le : columnExtremaCount u v w ≤ (Finset.univ : Finset (Fin m)).card :=
    Finset.card_filter_le _ _
  rwa [Finset.card_univ, Fintype.card_fin] at h_le

/-! ## The transfer matrix (Step 3)

The paper's `T_m(x)` acts on the state space `TransferState m` and
weights each edge `((u, v), (v, w))` by `x^{columnExtremaCount u v w}`.
The edge exists (weight nonzero) exactly when the second component of
the source equals the first component of the target — i.e., when the
middle column matches.  Otherwise the entry is zero. -/

/-- **Column-transfer matrix `T_m(x)`.**  Entry
`T_m(x)_{(u,v), (v', w)}` is `x^{columnExtremaCount u v w}` when
`v = v'` (transfer state advances by dropping the left column and
appending a new right column), and `0` otherwise.

Formally: given source state `s = ⟨(u, v), _⟩` and target
`t = ⟨(v', w), _⟩`, the matrix entry uses `if s.val.2 = t.val.1` to
gate the transfer; when equal, `s.val.2 = v = v' = t.val.1`, so
`columnExtremaCount s.val.1 s.val.2 t.val.2 = columnExtremaCount u v w`. -/
noncomputable def transferMatrix (m : ℕ) :
    Matrix (TransferState m) (TransferState m) (Polynomial ℤ) :=
  fun s t =>
    if s.val.2 = t.val.1 then
      Polynomial.X ^ (columnExtremaCount s.val.1 s.val.2 t.val.2)
    else 0

/-- **Transfer diagonal:** the transfer matrix has nonzero entries
only on the "matching column" pairs where the source's right column
equals the target's left column. -/
theorem transferMatrix_apply_eq {m : ℕ} (s t : TransferState m)
    (h : s.val.2 = t.val.1) :
    transferMatrix m s t = Polynomial.X ^ (columnExtremaCount s.val.1 s.val.2 t.val.2) := by
  unfold transferMatrix
  rw [if_pos h]

/-- **Transfer off-diagonal:** zero unless columns match. -/
theorem transferMatrix_apply_ne {m : ℕ} (s t : TransferState m)
    (h : s.val.2 ≠ t.val.1) :
    transferMatrix m s t = 0 := by
  unfold transferMatrix
  rw [if_neg h]

/-! ## The extremum-count generating polynomial `c_{m,n}(x)` (Step 5a)

The paper's transfer-matrix identity has the shape

  `c_{m,n}(x) = ∑_{h ∈ CanonicalHeights} x^{numExtrema h}
             = u^⊤ · T_m(x)^n · v`

for suitable boundary vectors `u, v`.  Establishing the SECOND equality
(the bijection between canonical heights and length-`n` paths in the
transfer state space) is the load-bearing combinatorial content of the
sequel paper; it is deferred to future work.

This step packages the FIRST equality — the definition of the polynomial
`c_{m,n}(x)` — as a Finset sum indexed by `d ∈ {0, …, m·n}` weighted
by `Ed d m n`.  Since `numExtrema h ≤ m·n` for every height `h`, this
Ed-weighted form is (provably, via fibre regrouping — see below)
equal to the sum-over-heights form.  The Ed-weighted definition is
UNCONDITIONAL (works on every `(m, n)`, empty grid included),
removing the finiteness side-condition that a raw sum-over-heights
would carry.

The key identity `Ed d m n = (cnPoly m n).coeff d` is then immediate
from the Ed-weighted definition: the coefficient at `d` picks out the
`d`-th summand.

This turns downstream Item 6d work into "prove the polynomial
identity `cnPoly m n = u^⊤ · T_m(x)^n · v`" — a single polynomial
equation replacing coefficient-by-coefficient reasoning.

**Not proved here** (paper-faithfulness bridge, tracked): the equality
`cnPoly m n = ∑_{h ∈ CanonicalHeights} x^{numExtrema h}` (when the RHS
is expressed via `Set.Finite.toFinset` on a nonempty grid, or via the
one-element canonical-heights set on an empty grid).  This is
propositional, not definitional: it requires grouping heights by
`numExtrema` and applying `Ed_eq_finset_card` fibre-wise (the pattern
used in `sum_Ed_eq_ncard_canonicalHeights`, SequelEd.lean ~371).
Off the formal critical path (downstream connects `cnPoly` directly
to `u^⊤ T^n v`); needed only for paper-verbatim `c_{m,n}(x)`. -/

/-- **Extremum-count generating polynomial `c_{m,n}(x)`.**  The
polynomial `∑_{d = 0}^{m·n} Ed(d, m, n) · x^d` in `ℤ[x]`.

By `numExtrema_le_mn`, this equals the raw sum-over-heights
`∑_{h ∈ CanonicalHeights} x^{numExtrema h}` (a propositional equality
via fibre-regrouping by `numExtrema` value; not proved here).  The
Ed-weighted form is unconditional (no `hm`, `hn` hypothesis needed). -/
noncomputable def cnPoly (m n : ℕ) : Polynomial ℤ :=
  ∑ d ∈ Finset.range (m * n + 1), Polynomial.monomial d (Ed d m n : ℤ)

/-- **Coefficient identity (Step 5b): `Ed d m n = coeff (c_{m,n})(x)_d`.**

The `d`-th coefficient of `cnPoly m n` is exactly `Ed d m n`.  Together
with `numExtrema_le_mn` (so `Ed d m n = 0` for `d > m·n`), this is the
unconditional identity that turns Item 6d's per-axis polynomiality
target into a polynomial coefficient extraction.

The proof is a Finset-sum unfolding:
`(∑_d monomial d (Ed d)).coeff k = ∑_d [d = k] · Ed d = Ed k` when
`k ≤ m·n`, and `0` otherwise (which matches `Ed_gt_mn_eq_zero`). -/
theorem Ed_eq_cnPoly_coeff (d m n : ℕ) :
    (Ed d m n : ℤ) = (cnPoly m n).coeff d := by
  unfold cnPoly
  rw [Polynomial.finset_sum_coeff]
  by_cases hd : d ≤ m * n
  · -- On-support: pick out the d-th summand.
    have hd_mem : d ∈ Finset.range (m * n + 1) := by
      rw [Finset.mem_range]; omega
    rw [← Finset.sum_erase_add _ _ hd_mem]
    have h_others : ∑ k ∈ (Finset.range (m * n + 1)).erase d,
        (Polynomial.monomial k ((Ed k m n : ℤ))).coeff d = 0 := by
      refine Finset.sum_eq_zero ?_
      intro k hk
      rw [Finset.mem_erase] at hk
      rw [Polynomial.coeff_monomial]
      simp only [if_neg hk.1]
    rw [h_others, zero_add, Polynomial.coeff_monomial, if_pos rfl]
  · -- Off-support: `Ed d m n = 0` (via `Ed_gt_mn_eq_zero`) and the sum is 0 at coefficient d.
    push_neg at hd
    have h_ed_zero : Ed d m n = 0 :=
      Ed_gt_mn_eq_zero hd
    rw [h_ed_zero, Nat.cast_zero]
    refine (Finset.sum_eq_zero ?_).symm
    intro k hk
    rw [Finset.mem_range] at hk
    rw [Polynomial.coeff_monomial]
    have hkd : k ≠ d := by omega
    simp only [if_neg hkd]

/-! ## Paper-faithfulness bridge (Step 5d)

The Ed-weighted definition of `cnPoly` matches the paper's raw
sum-over-heights form `∑_{h ∈ CanonicalHeights} x^{numExtrema h}` on
every nonempty grid.  Off the formal critical path for polynomiality
in `n` (which uses `cnPoly` directly via `Ed_eq_cnPoly_coeff`), but
required for verbatim paper-fidelity.

Strategy: regroup the sum-over-heights by `numExtrema h` value.  Each
fibre has count `Ed d m n` (via `Ed_eq_finset_card`), and every
weight on that fibre equals `monomial d 1` (since the exponent equals
`numExtrema h = d` on the fibre).  Sum over `d ∈ range (m·n + 1)`
matches `cnPoly`.  The `d > m·n` tail vanishes by `numExtrema_le_mn`
(no height achieves it), and the fibre-count matches `Ed = 0` there. -/

/-- **`cnPoly` as a sum over canonical heights** (paper-verbatim
`c_{m,n}(x)`).  For nonempty grid, the Ed-weighted definition equals
the raw generating polynomial `∑_h x^{numExtrema h}`. -/
theorem cnPoly_eq_sum_over_heights (m n : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    cnPoly m n
      = ∑ h ∈ (CanonicalHeights_finite hm hn).toFinset,
          Polynomial.monomial (numExtrema h) (1 : ℤ) := by
  unfold cnPoly
  -- Every height's numExtrema lies in [0, m·n].
  have h_maps : ∀ h ∈ (CanonicalHeights_finite hm hn).toFinset,
      numExtrema h ∈ Finset.range (m * n + 1) := by
    intro h _
    rw [Finset.mem_range]
    exact Nat.lt_succ_of_le (numExtrema_le_mn h)
  -- Fibre-regroup the RHS: ∑ h, f h = ∑ d ∈ range, ∑ h with numExtrema h = d, f h.
  rw [← Finset.sum_fiberwise_of_maps_to h_maps
        (fun h => Polynomial.monomial (numExtrema h) (1 : ℤ))]
  -- Match term-by-term (fixing d).
  refine Finset.sum_congr rfl fun d _ => ?_
  -- On the fibre {h : numExtrema h = d}, the summand is `monomial d 1`.
  have h_inner_eq :
      (∑ h ∈ (CanonicalHeights_finite hm hn).toFinset with numExtrema h = d,
        Polynomial.monomial (numExtrema h) (1 : ℤ))
      = ∑ h ∈ (CanonicalHeights_finite hm hn).toFinset with numExtrema h = d,
          Polynomial.monomial d (1 : ℤ) := by
    refine Finset.sum_congr rfl fun h hh => ?_
    rw [Finset.mem_filter] at hh
    rw [hh.2]
  rw [h_inner_eq, Finset.sum_const]
  -- Now `|fibre| • monomial d 1 = monomial d (|fibre| : ℤ)`.
  rw [Polynomial.smul_monomial]
  congr 1
  -- Match `|fibre|` with `Ed d m n`.
  have h_ed_card :
      Ed d m n
        = ((CanonicalHeights_finite hm hn).toFinset.filter
            (fun h => numExtrema h = d)).card := by
    rw [Ed_eq_finset_card d hm hn]
    congr 1
    ext h
    simp only [Set.Finite.mem_toFinset, Finset.mem_filter, Set.mem_setOf_eq]
    tauto
  rw [← h_ed_card]
  simp

/-! ## Height → column-colouring extraction (Step 5c-i)

The paper's transfer picture reads a height function column-by-column,
reducing each column modulo `3` to a `PathColouring m`.  This gives the
forward direction of the bijection `CanonicalHeights ↔ paths in
TransferState`.  Two facts are needed downstream:

* Each column is a proper 3-colouring of `P_m` (adjacent cells within
  the column have distinct mod-`3` values).
* Two horizontally-adjacent columns are `PathAdjacent` (both proper,
  and their `i`-th entries differ for every `i`).

Both follow from `IsHeight h` (adjacent cells differ by exactly `1` in
`ℤ`, hence by `1` or `2` in `ZMod 3`, in particular nonzero).

This module supplies the extraction and the two adjacency facts.  The
extremum-count decomposition and the backward direction (mod-`3`
colouring lifted to a height) are deferred to later modules. -/

/-- The `j`-th column of a height function, reduced mod `3` to a
`PathColouring m`. -/
def hCol {m n : ℕ} (h : Cell m n → ℤ) (j : Fin n) : PathColouring m :=
  fun i => ((h (i, j) : ℤ) : ZMod 3)

/-- Rows `⟨i, hi⟩` and `⟨i+1, h_succ⟩` of an `m × n` grid are adjacent
in the grid (Manhattan distance `1`). -/
private theorem adj_row {m n : ℕ} (i : ℕ) (hi : i < m)
    (h_succ : i + 1 < m) (j : Fin n) :
    adj ((⟨i, hi⟩, j) : Cell m n) (⟨i + 1, h_succ⟩, j) := by
  show gdist ((⟨i, hi⟩, j) : Cell m n) (⟨i + 1, h_succ⟩, j) = 1
  show (((((⟨i, hi⟩ : Fin m).val : ℤ) - ((⟨i + 1, h_succ⟩ : Fin m).val : ℤ)).natAbs
        + (((j.val : ℤ) - j.val).natAbs : ℕ)) : ℤ) = 1
  simp

/-- Columns `⟨j, hj⟩` and `⟨j+1, h_succ⟩` at row `i` are adjacent. -/
private theorem adj_col {m n : ℕ} (i : Fin m) (j : ℕ) (hj : j < n)
    (h_succ : j + 1 < n) :
    adj ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j + 1, h_succ⟩) := by
  show gdist ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j + 1, h_succ⟩) = 1
  show ((((i.val : ℤ) - i.val).natAbs
          + (((⟨j, hj⟩ : Fin n).val : ℤ) - ((⟨j + 1, h_succ⟩ : Fin n).val : ℤ)).natAbs : ℕ) : ℤ) = 1
  simp

/-- Symmetric of `adj_row`: an upper row is adjacent to its predecessor. -/
private theorem adj_row_up {m n : ℕ} (i : ℕ) (hi : i < m)
    (hi_lo : 0 < i) (j : Fin n) :
    adj ((⟨i, hi⟩, j) : Cell m n) (⟨i - 1, by omega⟩, j) := by
  show gdist ((⟨i, hi⟩, j) : Cell m n) (⟨i - 1, by omega⟩, j) = 1
  show ((((i : ℤ) - ((i - 1 : ℕ) : ℤ)).natAbs
          + (((j.val : ℤ) - j.val).natAbs) : ℕ) : ℤ) = 1
  rw [Nat.cast_sub (by omega : 1 ≤ i)]
  push_cast; simp

/-- Symmetric of `adj_col`: the right column is adjacent to the current. -/
private theorem adj_col_left {m n : ℕ} (i : Fin m) (j : ℕ) (hj : j < n)
    (hj_lo : 0 < j) :
    adj ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) := by
  show gdist ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) = 1
  show ((((i.val : ℤ) - i.val).natAbs
          + (((j : ℤ) - ((j - 1 : ℕ) : ℤ)).natAbs) : ℕ) : ℤ) = 1
  rw [Nat.cast_sub (by omega : 1 ≤ j)]
  push_cast; simp

/-- **Grid adjacency enumerated at four axis-aligned neighbours.**  Given
`adj p u` (Manhattan distance `1`), `u` is either the row-successor, row-
predecessor, column-successor, or column-predecessor of `p`. -/
private theorem adj_neighbour_cases {m n : ℕ} (p u : Cell m n) (hadj : adj p u) :
    (∃ hi : p.1.val + 1 < m, u = (⟨p.1.val + 1, hi⟩, p.2)) ∨
    (∃ hi : 0 < p.1.val, u = (⟨p.1.val - 1, by omega⟩, p.2)) ∨
    (∃ hj : p.2.val + 1 < n, u = (p.1, ⟨p.2.val + 1, hj⟩)) ∨
    (∃ hj : 0 < p.2.val, u = (p.1, ⟨p.2.val - 1, by omega⟩)) := by
  have hdist : (((((p.1.val : ℤ) - u.1.val).natAbs
                    + ((p.2.val : ℤ) - u.2.val).natAbs : ℕ)) : ℤ) = 1 := hadj
  -- Split on which of the two natAbs summands is 1.
  have h_split : (((p.1.val : ℤ) - u.1.val).natAbs = 1 ∧ ((p.2.val : ℤ) - u.2.val).natAbs = 0) ∨
                 (((p.1.val : ℤ) - u.1.val).natAbs = 0 ∧ ((p.2.val : ℤ) - u.2.val).natAbs = 1) := by
    have : ((p.1.val : ℤ) - u.1.val).natAbs + ((p.2.val : ℤ) - u.2.val).natAbs = 1 := by
      exact_mod_cast hdist
    omega
  rcases h_split with ⟨hrow_ne, hcol_eq⟩ | ⟨hrow_eq, hcol_ne⟩
  · -- Row differs by 1, column matches.
    have hcol : p.2 = u.2 := by
      apply Fin.ext
      have : (p.2.val : ℤ) = u.2.val := by
        have : ((p.2.val : ℤ) - u.2.val).natAbs = 0 := hcol_eq
        omega
      exact_mod_cast this
    -- Row differs: p.1.val = u.1.val + 1 or u.1.val = p.1.val + 1.
    have h_row_pm : p.1.val = u.1.val + 1 ∨ u.1.val = p.1.val + 1 := by
      have : ((p.1.val : ℤ) - u.1.val).natAbs = 1 := hrow_ne
      omega
    rcases h_row_pm with hup | hdown
    · -- u.1.val + 1 = p.1.val, so u = (⟨p.1.val - 1, _⟩, p.2). This is `adj_col_left`-side.
      right; left
      have hi_lo : 0 < p.1.val := by omega
      refine ⟨hi_lo, ?_⟩
      apply Prod.ext
      · show u.1 = ⟨p.1.val - 1, _⟩
        apply Fin.ext; simp; omega
      · exact hcol.symm
    · -- p.1.val + 1 = u.1.val, so u = (⟨p.1.val + 1, _⟩, p.2). This is `adj_row`-side.
      left
      have hi_hi : p.1.val + 1 < m := by
        have := u.1.isLt
        omega
      refine ⟨hi_hi, ?_⟩
      apply Prod.ext
      · show u.1 = ⟨p.1.val + 1, _⟩
        apply Fin.ext; simp; omega
      · exact hcol.symm
  · -- Column differs by 1, row matches.
    have hrow : p.1 = u.1 := by
      apply Fin.ext
      have : (p.1.val : ℤ) = u.1.val := by
        have : ((p.1.val : ℤ) - u.1.val).natAbs = 0 := hrow_eq
        omega
      exact_mod_cast this
    have h_col_pm : p.2.val = u.2.val + 1 ∨ u.2.val = p.2.val + 1 := by
      have : ((p.2.val : ℤ) - u.2.val).natAbs = 1 := hcol_ne
      omega
    rcases h_col_pm with hleft | hright
    · right; right; right
      have hj_lo : 0 < p.2.val := by omega
      refine ⟨hj_lo, ?_⟩
      apply Prod.ext
      · exact hrow.symm
      · show u.2 = ⟨p.2.val - 1, _⟩
        apply Fin.ext; simp; omega
    · right; right; left
      have hj_hi : p.2.val + 1 < n := by
        have := u.2.isLt
        omega
      refine ⟨hj_hi, ?_⟩
      apply Prod.ext
      · exact hrow.symm
      · show u.2 = ⟨p.2.val + 1, _⟩
        apply Fin.ext; simp; omega

/-- **Casting a height difference of `±1` gives distinct mod-`3` values.**
Height functions have adjacent cells differing by exactly `1` in `ℤ`;
reducing mod `3` gives values differing by `1` or `2`, hence distinct. -/
private theorem ne_of_abs_diff_one {a b : ℤ} (h : |a - b| = 1) :
    (a : ZMod 3) ≠ (b : ZMod 3) := by
  -- |a - b| = 1 ⟹ a - b = 1 or a - b = -1.
  rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp h with heq | heq
  · -- a - b = 1: (a : ZMod 3) = (b : ZMod 3) + 1, so distinct.
    intro habs
    have h_ab : a = b + 1 := by linarith
    rw [h_ab] at habs
    push_cast at habs
    -- habs : (b : ZMod 3) + 1 = (b : ZMod 3)
    have : (1 : ZMod 3) = 0 := by linear_combination habs
    exact absurd this (by decide)
  · -- a - b = -1: (a : ZMod 3) = (b : ZMod 3) - 1.
    intro habs
    have h_ab : a = b - 1 := by linarith
    rw [h_ab] at habs
    push_cast at habs
    have : (1 : ZMod 3) = 0 := by linear_combination -habs
    exact absurd this (by decide)

/-- **Every column of a height function is proper.**  Adjacent cells
within the column receive distinct mod-`3` values. -/
theorem hCol_isPathProperColouring {m n : ℕ} {h : Cell m n → ℤ}
    (hh : IsHeight h) (j : Fin n) :
    IsPathProperColouring (hCol h j) := by
  intro i h_succ
  unfold hCol
  have h_adj : adj ((i, j) : Cell m n) (⟨i.val + 1, h_succ⟩, j) :=
    adj_row i.val i.isLt h_succ j
  have h_diff : |h (i, j) - h (⟨i.val + 1, h_succ⟩, j)| = 1 :=
    hh _ _ h_adj
  -- Rewrite `(i, j)` as `(⟨i.val, i.isLt⟩, j)` — Fin.eta.
  have hi_eta : (i, j) = ((⟨i.val, i.isLt⟩ : Fin m), j) := by
    apply Prod.ext <;> simp
  rw [hi_eta] at h_diff
  exact ne_of_abs_diff_one h_diff

/-- **Two horizontally-adjacent columns of a height function are
`PathAdjacent`.**  Both are proper, and their cell-wise mod-`3` values
differ (since adjacent cells across columns differ by `1` in `ℤ`). -/
theorem hCol_pathAdjacent {m n : ℕ} {h : Cell m n → ℤ}
    (hh : IsHeight h) {j : ℕ} (hj : j < n) (h_succ : j + 1 < n) :
    PathAdjacent (hCol h ⟨j, hj⟩) (hCol h ⟨j + 1, h_succ⟩) := by
  refine ⟨hCol_isPathProperColouring hh _, hCol_isPathProperColouring hh _, ?_⟩
  intro i
  unfold hCol
  have h_adj : adj ((i, ⟨j, hj⟩) : Cell m n) (i, ⟨j + 1, h_succ⟩) :=
    adj_col i j hj h_succ
  exact ne_of_abs_diff_one (hh _ _ h_adj)

/-! ## Cell-wise extremum ↔ mod-3 test (Step 5c-ii, forward direction, interior)

The paper's central transfer-matrix claim is that a strict local extremum of a
height function on the m × n grid is equivalent to a mod-`3` colour-agreement
test on adjacent columns.  This module supplies the FORWARD DIRECTION of the
INTERIOR CASE: a cell `(i, j)` with `1 ≤ j` and `j + 1 < n` (so both left and
right columns exist) that is a strict local extremum satisfies
`IsColExtremum (hCol h (j-1)) (hCol h j) (hCol h (j+1)) i`.

Boundary columns `j = 0` and `j = n - 1`, the backward direction, and the
assembly into a full extremum-count decomposition are deferred.

Strategy: unfold `IsStrictLocalMax` / `IsStrictLocalMin` at the four
neighbour witnesses (left, right, up if present, down if present); each
gives a `+1` or `-1` height difference; reduce mod `3` via a cast lemma
and unpack the `IsColExtremum` conjunction. -/

/-- **Forward, max case, interior:** if `(i, j)` is a strict local maximum of
a height function with `1 ≤ j`, `j + 1 < n`, then the mod-`3` test fires. -/
theorem IsColExtremum_of_IsStrictLocalMax_interior
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hmax : IsStrictLocalMax h (i, ⟨j, by omega⟩)) :
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i := by
  -- Notation shorthand: `c = h(i, j)`, `c ± 1` for the 4 neighbours.
  set c : ℤ := h (i, ⟨j, by omega⟩) with hc_def
  -- Left neighbour: (i, j - 1).
  have h_adj_L : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) :=
    adj_col_left i j (by omega) hj_lo
  have h_L : h (i, ⟨j - 1, by omega⟩) = c - 1 := hmax _ h_adj_L
  -- Right neighbour: (i, j + 1).
  have h_adj_R : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_hi⟩) :=
    adj_col i j (by omega) hj_hi
  have h_R : h (i, ⟨j + 1, hj_hi⟩) = c - 1 := hmax _ h_adj_R
  refine ⟨?_, ?_, ?_⟩
  · -- Left colour = right colour: both = (c - 1) mod 3.
    show (hCol h ⟨j - 1, by omega⟩) i = (hCol h ⟨j + 1, hj_hi⟩) i
    unfold hCol
    rw [h_L, h_R]
  · -- Up neighbour (if i > 0): (hCol h j) ⟨i-1, _⟩ = (hCol h (j-1)) i.
    intro hi_lo
    show (hCol h ⟨j, by omega⟩)
              ⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos) i.isLt.le⟩
          = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos)
                          i.isLt.le⟩, ⟨j, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
    have h_U : h (⟨i.val - 1, _⟩, ⟨j, by omega⟩) = c - 1 := hmax _ h_adj_U
    unfold hCol
    rw [h_U, h_L]
  · -- Down neighbour (if i + 1 < m): (hCol h j) ⟨i+1, _⟩ = (hCol h (j-1)) i.
    intro hi_hi
    show (hCol h ⟨j, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c - 1 := hmax _ h_adj_D
    unfold hCol
    rw [h_D, h_L]

/-- **Forward, min case, interior:** if `(i, j)` is a strict local minimum of
a height function with `1 ≤ j`, `j + 1 < n`, then the mod-`3` test fires. -/
theorem IsColExtremum_of_IsStrictLocalMin_interior
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hmin : IsStrictLocalMin h (i, ⟨j, by omega⟩)) :
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i := by
  -- Notation shorthand: `c = h(i, j)`.  All 4 neighbours are `c + 1`.
  set c : ℤ := h (i, ⟨j, by omega⟩) with hc_def
  have h_adj_L : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) :=
    adj_col_left i j (by omega) hj_lo
  have h_L : h (i, ⟨j - 1, by omega⟩) = c + 1 := hmin _ h_adj_L
  have h_adj_R : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_hi⟩) :=
    adj_col i j (by omega) hj_hi
  have h_R : h (i, ⟨j + 1, hj_hi⟩) = c + 1 := hmin _ h_adj_R
  refine ⟨?_, ?_, ?_⟩
  · show (hCol h ⟨j - 1, by omega⟩) i = (hCol h ⟨j + 1, hj_hi⟩) i
    unfold hCol
    rw [h_L, h_R]
  · intro hi_lo
    show (hCol h ⟨j, by omega⟩)
              ⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos) i.isLt.le⟩
          = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val - 1, Nat.lt_of_lt_of_le (Nat.sub_lt hi_lo Nat.one_pos)
                          i.isLt.le⟩, ⟨j, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
    have h_U : h (⟨i.val - 1, _⟩, ⟨j, by omega⟩) = c + 1 := hmin _ h_adj_U
    unfold hCol
    rw [h_U, h_L]
  · intro hi_hi
    show (hCol h ⟨j, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨j - 1, by omega⟩) i
    have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c + 1 := hmin _ h_adj_D
    unfold hCol
    rw [h_D, h_L]

/-- **Forward, extremum case, interior:** combines the max and min cases. -/
theorem IsColExtremum_of_IsStrictLocalExtremum_interior
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hext : IsStrictLocalExtremum h (i, ⟨j, by omega⟩)) :
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i := by
  rcases hext with hmax | hmin
  · exact IsColExtremum_of_IsStrictLocalMax_interior hh i hj_lo hj_hi hmax
  · exact IsColExtremum_of_IsStrictLocalMin_interior hh i hj_lo hj_hi hmin

/-! ## Backward direction (Step 5c-ii backward, interior)

The paper's cell-wise iff needs both directions.  The BACKWARD direction is:
if the three-column mod-`3` test `IsColExtremum` fires at cell `(i, j)`
for interior `j`, then that cell is a strict local extremum.

Strategy:
* `IsHeight h` restricts every neighbour's height to `c ± 1`.
* `IsColExtremum`'s first condition forces left and right neighbours to
  share mod-`3` value.  With `± 1` restriction, this forces them to be
  equal in `ℤ` — either both `c - 1` (potential max) or both `c + 1`
  (potential min).
* Case-split on the sign; verify all present neighbours match by
  enumerating adjacent cells via `adj_neighbour_cases` and applying
  the up/down conditions of `IsColExtremum` when applicable. -/

/-- **Cast helper**: two `± 1` neighbours sharing mod-`3` are equal.  If
`a, b ∈ {c - 1, c + 1}` and `(a : ZMod 3) = (b : ZMod 3)`, then `a = b`. -/
private theorem eq_of_pm1_and_mod_eq {c a b : ℤ}
    (ha : a = c - 1 ∨ a = c + 1) (hb : b = c - 1 ∨ b = c + 1)
    (hmod : (a : ZMod 3) = (b : ZMod 3)) : a = b := by
  rcases ha with rfl | rfl <;> rcases hb with rfl | rfl
  · rfl
  · exfalso; push_cast at hmod
    have h2 : (2 : ZMod 3) = 0 := by linear_combination -hmod
    exact absurd h2 (by decide)
  · exfalso; push_cast at hmod
    have h2 : (2 : ZMod 3) = 0 := by linear_combination hmod
    exact absurd h2 (by decide)
  · rfl

/-- **Backward, extremum case, interior**: if the mod-`3` test fires at an
interior cell `(i, j)` with `1 ≤ j`, `j + 1 < n`, then that cell is a
strict local extremum.

Uses `IsHeight` to restrict each neighbour's height to `c ± 1`, then
`IsColExtremum`'s mod-`3` conditions to force the sign to be uniform
across all four (or fewer) neighbours. -/
theorem IsStrictLocalExtremum_of_IsColExtremum_interior
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n)
    (hcol : IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i) :
    IsStrictLocalExtremum h (i, ⟨j, by omega⟩) := by
  set c : ℤ := h (i, ⟨j, by omega⟩) with hc_def
  -- Left / right neighbours (always present at interior j).
  have h_adj_L : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j - 1, by omega⟩) :=
    adj_col_left i j (by omega) hj_lo
  have h_adj_R : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_hi⟩) :=
    adj_col i j (by omega) hj_hi
  -- From IsHeight, h at these neighbours is c ± 1.
  have h_L_pm : h (i, ⟨j - 1, by omega⟩) = c - 1 ∨ h (i, ⟨j - 1, by omega⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_L) with h1 | h1
    · left; linarith
    · right; linarith
  have h_R_pm : h (i, ⟨j + 1, hj_hi⟩) = c - 1 ∨ h (i, ⟨j + 1, hj_hi⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_R) with h1 | h1
    · left; linarith
    · right; linarith
  -- hcol.1 forces L = R via `eq_of_pm1_and_mod_eq`.
  have h_LR_mod : (h (i, ⟨j - 1, by omega⟩) : ZMod 3) = (h (i, ⟨j + 1, hj_hi⟩) : ZMod 3) := by
    have := hcol.1
    unfold hCol at this
    exact this
  have h_LR_eq : h (i, ⟨j - 1, by omega⟩) = h (i, ⟨j + 1, hj_hi⟩) :=
    eq_of_pm1_and_mod_eq h_L_pm h_R_pm h_LR_mod
  -- Save `h_L_pm` for reuse inside the case branches (rcases consumes it).
  have h_L_pm' : h (i, ⟨j - 1, by omega⟩) = c - 1 ∨ h (i, ⟨j - 1, by omega⟩) = c + 1 := h_L_pm
  -- Case-split on left-neighbour sign.
  rcases h_L_pm with hL | hL
  · -- All neighbours are c - 1: try IsStrictLocalMax.
    left
    intro u hu
    -- Enumerate u via adj_neighbour_cases.
    rcases adj_neighbour_cases ((i, ⟨j, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj2_hi, hu_eq⟩ | ⟨hj2_lo, hu_eq⟩
    · -- u is the row-successor (i.val + 1, j) — down neighbour.
      have h_down_cond := hcol.2.2 hi_hi
      unfold hCol at h_down_cond
      -- h_down_cond: (h (⟨i.val+1, hi_hi⟩, ⟨j, _⟩) : ZMod 3) = (h (i, ⟨j - 1, _⟩) : ZMod 3)
      have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_down_cond
      rw [hu_eq, h_D_eq, hL]
    · -- u is the row-predecessor (i.val - 1, j) — up neighbour.
      have h_up_cond := hcol.2.1 hi_lo
      unfold hCol at h_up_cond
      have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_up_cond
      rw [hu_eq, h_U_eq, hL]
    · -- u is the column-successor (i, j + 1) — right neighbour.
      rw [hu_eq, ← h_LR_eq, hL]
    · -- u is the column-predecessor (i, j - 1) — left neighbour.
      rw [hu_eq, hL]
  · -- All neighbours are c + 1: try IsStrictLocalMin.
    right
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨j, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj2_hi, hu_eq⟩ | ⟨hj2_lo, hu_eq⟩
    · have h_down_cond := hcol.2.2 hi_hi
      unfold hCol at h_down_cond
      have h_adj_D : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨j, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_down_cond
      rw [hu_eq, h_D_eq, hL]
    · have h_up_cond := hcol.2.1 hi_lo
      unfold hCol at h_up_cond
      have h_adj_U : adj ((i, ⟨j, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨j, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨j, by omega⟩) = h (i, ⟨j - 1, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_up_cond
      rw [hu_eq, h_U_eq, hL]
    · rw [hu_eq, ← h_LR_eq, hL]
    · rw [hu_eq, hL]

/-- **Interior iff**: for interior `j`, a cell is a strict local extremum
iff the three-column mod-`3` test fires. -/
theorem IsStrictLocalExtremum_iff_IsColExtremum_interior
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) {j : ℕ} (hj_lo : 0 < j) (hj_hi : j + 1 < n) :
    IsStrictLocalExtremum h (i, ⟨j, by omega⟩) ↔
    IsColExtremum (hCol h ⟨j - 1, by omega⟩) (hCol h ⟨j, by omega⟩)
      (hCol h ⟨j + 1, hj_hi⟩) i :=
  ⟨IsColExtremum_of_IsStrictLocalExtremum_interior hh i hj_lo hj_hi,
   IsStrictLocalExtremum_of_IsColExtremum_interior hh i hj_lo hj_hi⟩

/-! ## Boundary column iff (Step 5c-ii boundary)

For a cell in the LEFTMOST column (`j = 0`) or RIGHTMOST column (`j = n - 1`)
the horizontal neighbour set has only ONE element (the sole present adjacent
column), not two.  The mod-`3` test therefore uses a TWO-COLUMN predicate:
the cell's own column `own` and the sole adjacent column `other`.

Cell `(i, j)` is a boundary extremum iff its present neighbours share one
colour: at `j = 0`, the neighbours are `(i, 1)` (colour = `other i`) and
`(i - 1, 0)`, `(i + 1, 0)` in the own column (if present).  Sharing means
each present vertical neighbour of `own` equals `other i` in colour.

`IsBoundaryExtremum own other i` packages this test symmetrically for both
boundaries.  The iffs below instantiate it at `(hCol h 0, hCol h 1)` for
`j = 0` and at `(hCol h (n - 1), hCol h (n - 2))` for `j = n - 1`. -/

/-- **Two-column boundary extremum test.**  A cell `i` in column `own`
whose sole horizontally-adjacent column is `other` is a mod-`3` extremum
iff each present vertical neighbour of `own` at row `i` shares the colour
of the horizontal neighbour, `other i`. -/
def IsBoundaryExtremum {m : ℕ} (own other : PathColouring m) (i : Fin m) : Prop :=
  (∀ _ : 0 < i.val, own ⟨i.val - 1, by omega⟩ = other i) ∧
  (∀ h : i.val + 1 < m, own ⟨i.val + 1, h⟩ = other i)

instance {m : ℕ} (own other : PathColouring m) (i : Fin m) :
    Decidable (IsBoundaryExtremum own other i) := by
  unfold IsBoundaryExtremum; infer_instance

/-! ### Left boundary column (`j = 0`) -/

/-- **Forward, max, left boundary:** if `(i, 0)` is a strict local maximum
of a height function on a grid with `n ≥ 2`, then the two-column mod-`3`
test fires with `own = hCol h 0`, `other = hCol h 1`. -/
theorem IsBoundaryExtremum_of_IsStrictLocalMax_left
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hmax : IsStrictLocalMax h (i, ⟨0, hn⟩)) :
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i := by
  set c : ℤ := h (i, ⟨0, hn⟩) with hc_def
  -- Right neighbour (i, 1).
  have h_adj_R : adj ((i, ⟨0, hn⟩) : Cell m n) (i, ⟨1, hn_succ⟩) :=
    adj_col i 0 hn hn_succ
  have h_R : h (i, ⟨1, hn_succ⟩) = c - 1 := hmax _ h_adj_R
  refine ⟨?_, ?_⟩
  · -- Up neighbour (i - 1, 0) if present.
    intro hi_lo
    have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c - 1 := hmax _ h_adj_U
    show (hCol h ⟨0, hn⟩) ⟨i.val - 1, by omega⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_U, h_R]
  · -- Down neighbour (i + 1, 0) if present.
    intro hi_hi
    have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
      adj_row i.val i.isLt hi_hi ⟨0, hn⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c - 1 := hmax _ h_adj_D
    show (hCol h ⟨0, hn⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_D, h_R]

/-- **Forward, min, left boundary.**  Symmetric to max with `+1`. -/
theorem IsBoundaryExtremum_of_IsStrictLocalMin_left
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hmin : IsStrictLocalMin h (i, ⟨0, hn⟩)) :
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i := by
  set c : ℤ := h (i, ⟨0, hn⟩) with hc_def
  have h_adj_R : adj ((i, ⟨0, hn⟩) : Cell m n) (i, ⟨1, hn_succ⟩) :=
    adj_col i 0 hn hn_succ
  have h_R : h (i, ⟨1, hn_succ⟩) = c + 1 := hmin _ h_adj_R
  refine ⟨?_, ?_⟩
  · intro hi_lo
    have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c + 1 := hmin _ h_adj_U
    show (hCol h ⟨0, hn⟩) ⟨i.val - 1, by omega⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_U, h_R]
  · intro hi_hi
    have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
      adj_row i.val i.isLt hi_hi ⟨0, hn⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c + 1 := hmin _ h_adj_D
    show (hCol h ⟨0, hn⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨1, hn_succ⟩) i
    unfold hCol
    rw [h_D, h_R]

/-- **Forward, extremum, left boundary.** -/
theorem IsBoundaryExtremum_of_IsStrictLocalExtremum_left
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hext : IsStrictLocalExtremum h (i, ⟨0, hn⟩)) :
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i := by
  rcases hext with hmax | hmin
  · exact IsBoundaryExtremum_of_IsStrictLocalMax_left hh i hn hn_succ hmax
  · exact IsBoundaryExtremum_of_IsStrictLocalMin_left hh i hn hn_succ hmin

/-- **Backward, extremum, left boundary.**  From `IsHeight h` and the
two-column boundary test at `j = 0`, deduce that `(i, 0)` is a strict
local extremum.  The right neighbour's `± 1` sign is fixed by `IsHeight`;
each present vertical neighbour is forced to match it via
`IsBoundaryExtremum` + `eq_of_pm1_and_mod_eq`; then enumerate adjacent
cells via `adj_neighbour_cases`. -/
theorem IsStrictLocalExtremum_of_IsBoundaryExtremum_left
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n)
    (hbdy : IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i) :
    IsStrictLocalExtremum h (i, ⟨0, hn⟩) := by
  set c : ℤ := h (i, ⟨0, hn⟩) with hc_def
  -- Right neighbour: h(i, 1) = c ± 1.
  have h_adj_R : adj ((i, ⟨0, hn⟩) : Cell m n) (i, ⟨1, hn_succ⟩) :=
    adj_col i 0 hn hn_succ
  have h_R_pm : h (i, ⟨1, hn_succ⟩) = c - 1 ∨ h (i, ⟨1, hn_succ⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_R) with h1 | h1
    · left; linarith
    · right; linarith
  -- Save `h_R_pm` for reuse inside the case branches (rcases consumes it).
  have h_R_pm' : h (i, ⟨1, hn_succ⟩) = c - 1 ∨ h (i, ⟨1, hn_succ⟩) = c + 1 := h_R_pm
  -- Case-split on the right neighbour's sign.
  rcases h_R_pm with hR | hR
  · -- Right = c - 1: potential max.
    left
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨0, hn⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    · -- Down neighbour: use hbdy.2 + eq_of_pm1_and_mod_eq.
      have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
        adj_row i.val i.isLt hi_hi ⟨0, hn⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_R_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hR]
    · -- Up neighbour.
      have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_R_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hR]
    · -- Right neighbour: column-succ of (i, 0) is (i, 1).
      rw [hu_eq]
      have h_fin : (⟨(0 : ℕ) + 1, hj_hi⟩ : Fin n) = ⟨1, hn_succ⟩ := by
        apply Fin.ext; simp
      rw [h_fin]; exact hR
    · -- Left neighbour: hj_lo : 0 < (i, ⟨0, hn⟩).2.val = 0, contradiction.
      exact absurd hj_lo (by simp)
  · -- Right = c + 1: potential min.  Symmetric.
    right
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨0, hn⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    · have h_adj_D : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) :=
        adj_row i.val i.isLt hi_hi ⟨0, hn⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_R_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hR]
    · have h_adj_U : adj ((i, ⟨0, hn⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨0, hn⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨0, hn⟩) = h (i, ⟨1, hn_succ⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_R_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hR]
    · rw [hu_eq]
      have h_fin : (⟨(0 : ℕ) + 1, hj_hi⟩ : Fin n) = ⟨1, hn_succ⟩ := by
        apply Fin.ext; simp
      rw [h_fin]; exact hR
    · exact absurd hj_lo (by simp)

/-- **Left boundary iff**: for `j = 0` with `n ≥ 2`, a cell is a strict local
extremum iff the two-column mod-`3` test fires. -/
theorem IsStrictLocalExtremum_iff_IsBoundaryExtremum_left
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn : 0 < n) (hn_succ : 1 < n) :
    IsStrictLocalExtremum h (i, ⟨0, hn⟩) ↔
    IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i :=
  ⟨IsBoundaryExtremum_of_IsStrictLocalExtremum_left hh i hn hn_succ,
   IsStrictLocalExtremum_of_IsBoundaryExtremum_left hh i hn hn_succ⟩

/-! ### Right boundary column (`j = n - 1`)

Structurally symmetric to the left boundary: the sole horizontal neighbour
is now on the LEFT (column `n - 2`).  All arguments carry over with
`hCol h ⟨n - 1, _⟩` as `own` and `hCol h ⟨n - 2, _⟩` as `other`. -/

/-- **Forward, max, right boundary:** if `(i, n - 1)` is a strict local
maximum on a grid with `n ≥ 2`, the two-column mod-`3` test fires. -/
theorem IsBoundaryExtremum_of_IsStrictLocalMax_right
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hmax : IsStrictLocalMax h (i, ⟨n - 1, by omega⟩)) :
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i := by
  set c : ℤ := h (i, ⟨n - 1, by omega⟩) with hc_def
  -- Left neighbour (i, n - 2).
  have h_adj_L : adj ((i, ⟨n - 1, by omega⟩) : Cell m n) (i, ⟨n - 2, by omega⟩) := by
    have := @adj_col_left m n i (n - 1) (by omega) (by omega : 0 < n - 1)
    -- adj_col_left produces (i, ⟨(n-1) - 1, _⟩) = (i, ⟨n - 2, _⟩) up to Fin ext.
    have h_lt1 : n - 1 - 1 < n := by omega
    have h_lt2 : n - 2 < n := by omega
    have h_fin : (⟨n - 1 - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
      apply Fin.ext; simp; omega
    rw [h_fin] at this
    exact this
  have h_L : h (i, ⟨n - 2, by omega⟩) = c - 1 := hmax _ h_adj_L
  refine ⟨?_, ?_⟩
  · -- Up neighbour (i - 1, n - 1).
    intro hi_lo
    have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c - 1 := hmax _ h_adj_U
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val - 1, by omega⟩
          = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_U, h_L]
  · -- Down neighbour (i + 1, n - 1).
    intro hi_hi
    have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c - 1 := hmax _ h_adj_D
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_D, h_L]

/-- **Forward, min, right boundary.**  Symmetric with `+1`. -/
theorem IsBoundaryExtremum_of_IsStrictLocalMin_right
    {m n : ℕ} {h : Cell m n → ℤ} (_hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hmin : IsStrictLocalMin h (i, ⟨n - 1, by omega⟩)) :
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i := by
  set c : ℤ := h (i, ⟨n - 1, by omega⟩) with hc_def
  have h_adj_L : adj ((i, ⟨n - 1, by omega⟩) : Cell m n) (i, ⟨n - 2, by omega⟩) := by
    have := @adj_col_left m n i (n - 1) (by omega) (by omega : 0 < n - 1)
    have h_lt1 : n - 1 - 1 < n := by omega
    have h_lt2 : n - 2 < n := by omega
    have h_fin : (⟨n - 1 - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
      apply Fin.ext; simp; omega
    rw [h_fin] at this
    exact this
  have h_L : h (i, ⟨n - 2, by omega⟩) = c + 1 := hmin _ h_adj_L
  refine ⟨?_, ?_⟩
  · intro hi_lo
    have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
      adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
    have h_U : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c + 1 := hmin _ h_adj_U
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val - 1, by omega⟩
          = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_U, h_L]
  · intro hi_hi
    have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                        (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
      adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
    have h_D : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c + 1 := hmin _ h_adj_D
    show (hCol h ⟨n - 1, by omega⟩) ⟨i.val + 1, hi_hi⟩ = (hCol h ⟨n - 2, by omega⟩) i
    unfold hCol
    rw [h_D, h_L]

/-- **Forward, extremum, right boundary.** -/
theorem IsBoundaryExtremum_of_IsStrictLocalExtremum_right
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hext : IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩)) :
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i := by
  rcases hext with hmax | hmin
  · exact IsBoundaryExtremum_of_IsStrictLocalMax_right hh i hn_succ hmax
  · exact IsBoundaryExtremum_of_IsStrictLocalMin_right hh i hn_succ hmin

/-- **Backward, extremum, right boundary.** -/
theorem IsStrictLocalExtremum_of_IsBoundaryExtremum_right
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n)
    (hbdy : IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i) :
    IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩) := by
  set c : ℤ := h (i, ⟨n - 1, by omega⟩) with hc_def
  -- Left neighbour: h(i, n - 2) = c ± 1.
  have h_adj_L : adj ((i, ⟨n - 1, by omega⟩) : Cell m n) (i, ⟨n - 2, by omega⟩) := by
    have := @adj_col_left m n i (n - 1) (by omega) (by omega : 0 < n - 1)
    have h_lt1 : n - 1 - 1 < n := by omega
    have h_lt2 : n - 2 < n := by omega
    have h_fin : (⟨n - 1 - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
      apply Fin.ext; simp; omega
    rw [h_fin] at this
    exact this
  have h_L_pm : h (i, ⟨n - 2, by omega⟩) = c - 1 ∨ h (i, ⟨n - 2, by omega⟩) = c + 1 := by
    rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_L) with h1 | h1
    · left; linarith
    · right; linarith
  have h_L_pm' : h (i, ⟨n - 2, by omega⟩) = c - 1 ∨ h (i, ⟨n - 2, by omega⟩) = c + 1 := h_L_pm
  rcases h_L_pm with hL | hL
  · left
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨n - 1, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    · have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hL]
    · have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hL]
    · -- Column-succ: (n - 1) + 1 = n, but hj_hi : (n - 1) + 1 < n, contradiction.
      exfalso; simp at hj_hi; omega
    · -- Column-pred: (i, n - 2).
      rw [hu_eq]
      have h_lt1 : (n - 1 : ℕ) - 1 < n := by omega
      have h_lt2 : n - 2 < n := by omega
      have h_fin : (⟨(n - 1 : ℕ) - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
        apply Fin.ext; simp; omega
      rw [h_fin]; exact hL
  · right
    intro u hu
    rcases adj_neighbour_cases ((i, ⟨n - 1, by omega⟩) : Cell m n) u hu with
      ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
    · have h_adj_D : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) :=
        adj_row i.val i.isLt hi_hi ⟨n - 1, by omega⟩
      have h_D_pm : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_D) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_D := hbdy.2 hi_hi
      unfold hCol at h_bdy_D
      have h_D_eq : h (⟨i.val + 1, hi_hi⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_D_pm h_L_pm' h_bdy_D
      rw [hu_eq, h_D_eq, hL]
    · have h_adj_U : adj ((i, ⟨n - 1, by omega⟩) : Cell m n)
                          (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) :=
        adj_row_up i.val i.isLt hi_lo ⟨n - 1, by omega⟩
      have h_U_pm : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c - 1 ∨
                    h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = c + 1 := by
        rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp (hh _ _ h_adj_U) with h1 | h1
        · left; linarith
        · right; linarith
      have h_bdy_U := hbdy.1 hi_lo
      unfold hCol at h_bdy_U
      have h_U_eq : h (⟨i.val - 1, by omega⟩, ⟨n - 1, by omega⟩) = h (i, ⟨n - 2, by omega⟩) :=
        eq_of_pm1_and_mod_eq h_U_pm h_L_pm' h_bdy_U
      rw [hu_eq, h_U_eq, hL]
    · exfalso; simp at hj_hi; omega
    · rw [hu_eq]
      have h_lt1 : (n - 1 : ℕ) - 1 < n := by omega
      have h_lt2 : n - 2 < n := by omega
      have h_fin : (⟨(n - 1 : ℕ) - 1, h_lt1⟩ : Fin n) = ⟨n - 2, h_lt2⟩ := by
        apply Fin.ext; simp; omega
      rw [h_fin]; exact hL

/-- **Right boundary iff**: for `j = n - 1` with `n ≥ 2`, a cell is a strict
local extremum iff the two-column mod-`3` test fires. -/
theorem IsStrictLocalExtremum_iff_IsBoundaryExtremum_right
    {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : Fin m) (hn_succ : 1 < n) :
    IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩) ↔
    IsBoundaryExtremum (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i :=
  ⟨IsBoundaryExtremum_of_IsStrictLocalExtremum_right hh i hn_succ,
   IsStrictLocalExtremum_of_IsBoundaryExtremum_right hh i hn_succ⟩

/-! ## Height → transfer-path packaging (Step 5c-iii)

The forward direction of the height ↔ transfer-path bijection: every height
function `h` on `Cell m n` with `n ≥ 2` gives rise to a sequence of adjacent
column pairs — a path `Fin (n - 1) → TransferState m` where the `j`-th state
packages the adjacent pair `(hCol h j, hCol h (j + 1))`.

Consecutive states in this path share their middle column: state at `j` has
second component `hCol h (j + 1)`, and state at `j + 1` has first component
`hCol h (j + 1)`; both are literally the same column colouring.  This is the
transfer-adjacency condition needed to interpret the sequence as a walk in
the transfer state graph. -/

/-- **Height → transfer-path** (state extraction).  Given a height function
`h` on `Cell m n` with `n ≥ 2`, the `j`-th `TransferState` of `h` packages the
adjacent column pair `(hCol h j, hCol h (j + 1))` with the `PathAdjacent`
witness from `hCol_pathAdjacent`. -/
noncomputable def heightToPath {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1)) : TransferState m :=
  ⟨(hCol h ⟨j.val, by have := j.isLt; omega⟩,
    hCol h ⟨j.val + 1, by have := j.isLt; omega⟩),
   hCol_pathAdjacent hh (by have := j.isLt; omega)
                        (by have := j.isLt; omega)⟩

/-- **First component**: state `j`'s left column is `hCol h j`. -/
theorem heightToPath_fst {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1)) :
    (heightToPath h hh hn j).val.1
      = hCol h ⟨j.val, by have := j.isLt; omega⟩ := rfl

/-- **Second component**: state `j`'s right column is `hCol h (j + 1)`. -/
theorem heightToPath_snd {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1)) :
    (heightToPath h hh hn j).val.2
      = hCol h ⟨j.val + 1, by have := j.isLt; omega⟩ := rfl

/-- **Transfer-adjacency (middle-column match)**: consecutive states in the
height-path share their middle column.  Namely, state `j`'s right column
equals state `j + 1`'s left column — both are `hCol h (j + 1)`.

This is the transfer-matrix "matching column" condition (see
`transferMatrix_apply_eq`): a `TransferState` at index `j` transitions
weight-nonzero to a `TransferState` at index `j + 1` iff the middle columns
match, and along a height-path they match by construction. -/
theorem heightToPath_matches {m n : ℕ} (h : Cell m n → ℤ)
    (hh : IsHeight h) (hn : 2 ≤ n) (j : Fin (n - 1))
    (hj_succ : j.val + 1 < n - 1) :
    (heightToPath h hh hn j).val.2
      = (heightToPath h hh hn ⟨j.val + 1, hj_succ⟩).val.1 := rfl

/-! ## Mod-`3` sign primitive (Step 5c-iv-a)

The backward direction of the height ↔ transfer-path bijection (`pathToHeight`,
5c-iv) requires LIFTING a mod-`3` transition back to a `± 1` integer step.
Given adjacent cells with colours `a, b : ZMod 3` (`a ≠ b`), their height
values must differ by exactly `± 1` in `ℤ`, and the sign is fixed by which
of `a + 1 = b` (positive step) or `a - 1 = b` (negative step) holds.

`modSign a b` packages this choice as an `ℤ` value `± 1`.  When `a = b`
(a case that should never arise in the bijection, since adjacent cells have
distinct colours), the value is `-1` by convention — but the load-bearing
lemmas below assume `a ≠ b`, so the fallback is moot.

The primitive supports the pointwise integer lift in the eventual
`pathToHeight` construction and enables the row/column signed sums that
compute canonical height values along a canonical path. -/

/-- **The sign of a mod-`3` transition.**  Given `a, b : ZMod 3`, returns
`+1` if `a + 1 = b` and `-1` otherwise.  For proper transitions (`a ≠ b`),
this is the sign of the integer step `h b - h a ∈ {+1, -1}` that lifts the
mod-`3` difference `b - a ∈ {1, 2}`. -/
def modSign (a b : ZMod 3) : ℤ :=
  if a + 1 = b then 1 else -1

/-- `modSign` is always `+1` or `-1`. -/
theorem modSign_eq_pos_or_neg_one (a b : ZMod 3) :
    modSign a b = 1 ∨ modSign a b = -1 := by
  unfold modSign
  by_cases h : a + 1 = b
  · left; rw [if_pos h]
  · right; rw [if_neg h]

/-- `|modSign a b| = 1` (unconditional). -/
theorem modSign_abs (a b : ZMod 3) : |modSign a b| = 1 := by
  rcases modSign_eq_pos_or_neg_one a b with h | h <;> rw [h] <;> decide

/-- **Mod-`3` correspondence** (positive case): when `a + 1 = b`,
`modSign a b = 1` and its cast to `ZMod 3` gives `b - a = 1`. -/
theorem modSign_of_succ {a b : ZMod 3} (h : a + 1 = b) :
    modSign a b = 1 := by
  unfold modSign; rw [if_pos h]

/-- **Mod-`3` correspondence** (negative case): when `a ≠ b` and `a + 1 ≠ b`
(equivalently `a - 1 = b` in `ZMod 3`), `modSign a b = -1`. -/
theorem modSign_of_pred {a b : ZMod 3} (h : a + 1 ≠ b) :
    modSign a b = -1 := by
  unfold modSign; rw [if_neg h]

/-- **Cast identity**: for `a ≠ b`, `((modSign a b : ℤ) : ZMod 3) = b - a`.
This is the load-bearing correspondence that makes the eventual
`pathToHeight` well-defined — the integer sign lifts the mod-`3` difference. -/
theorem modSign_cast_eq_diff {a b : ZMod 3} (hab : a ≠ b) :
    ((modSign a b : ℤ) : ZMod 3) = b - a := by
  revert hab
  revert a b
  decide

/-- **Antisymmetry**: reversing a mod-`3` transition negates its sign.  For
`a ≠ b`, `modSign b a = - modSign a b`. -/
theorem modSign_symm {a b : ZMod 3} (hab : a ≠ b) :
    modSign b a = - modSign a b := by
  revert hab
  revert a b
  decide

/-- **Unit-cycle sum vanishes**: along a `2 × 2` cycle of mod-`3` transitions
where all four edge endpoints differ pairwise (as required for a proper
colouring of a `2 × 2` block), the signed sum of the four transitions is
zero.  This is the path-independence property that underpins the eventual
`pathToHeight` construction — the signed integral around a unit cycle
vanishes, so integrating from `(0,0)` to any cell `(i, j)` gives the same
result regardless of the specific path chosen.

Concretely: for a `2 × 2` block with colours `a, b, d, e` at corners
`(0, 0), (0, 1), (1, 0), (1, 1)` respectively, where each pair of adjacent
corners has distinct colours, we have

  `modSign a b + modSign b e + modSign e d + modSign d a = 0`. -/
theorem modSign_cycle_sum {a b d e : ZMod 3}
    (hab : a ≠ b) (hbe : b ≠ e) (hed : e ≠ d) (hda : d ≠ a) :
    modSign a b + modSign b e + modSign e d + modSign d a = 0 := by
  revert hab hbe hed hda
  revert a b d e
  decide

/-! ## `pathToHeight` construction (Step 5c-iv-b)

Given a sequence of column colourings `c : Fin n → PathColouring m` and a
cell `p ∈ Cell m n`, integrate `modSign` along the row-first path from the
origin `(0, 0)` to `p`:

* **Row-`0` walk**: sum `modSign (c k 0) (c (k + 1) 0)` for `k ∈ [0, p.2)`,
  contributing the height difference from `(0, 0)` to `(0, p.2)`.
* **Column-`p.2` walk**: sum `modSign (c p.2 r) (c p.2 (r + 1))` for
  `r ∈ [0, p.1)`, contributing the height difference from `(0, p.2)`
  to `(p.1, p.2)`.

The definition is UNCONDITIONAL: the mere existence of `p : Cell m n`
witnesses `m > 0` and `n > 0`, which is all the construction needs.

**Value at origin** is zero: both empty sums vanish, giving `pathToHeight
c ⟨0, 0⟩ = 0`.  This gives canonicity for free.

**Deferred to later commits** (5c-iv-b-2, 5c-iv-b-3):
* `IsHeight (pathToHeight c)`: adjacent cells differ by exactly `1`.
* `hCol (pathToHeight c) j = c j`: mod-`3` reduction matches the input.
Both require the input colourings to be pairwise `PathAdjacent` — a
hypothesis this base definition does not impose. -/

/-- **Backward map (row-first integration).** Given a sequence of column
colourings `c : Fin n → PathColouring m`, produce a height field
`pathToHeight c : Cell m n → ℤ` by integrating `modSign` along the row-`0`
walk to column `p.2`, then down column `p.2` to row `p.1`.

Unconditional in `m, n`: the existence of `p : Cell m n` gives `m > 0`,
`n > 0` via `p.1.isLt`, `p.2.isLt`. -/
noncomputable def pathToHeight {m n : ℕ} (c : Fin n → PathColouring m)
    (p : Cell m n) : ℤ :=
  have hm : 0 < m := Nat.lt_of_le_of_lt (Nat.zero_le _) p.1.isLt
  -- Row-0 walk: k : Fin p.2.val.  Fin structure gives k.val < p.2.val,
  -- combined with p.2.isLt : p.2.val < n gives k.val + 1 < n.
  (∑ k : Fin p.2.val,
      modSign (c ⟨k.val, by have := k.isLt; have := p.2.isLt; omega⟩ ⟨0, hm⟩)
              (c ⟨k.val + 1, by have := k.isLt; have := p.2.isLt; omega⟩ ⟨0, hm⟩))
  +
  -- Column-p.2 walk: r : Fin p.1.val.
  (∑ r : Fin p.1.val,
      modSign (c p.2 ⟨r.val, by have := r.isLt; have := p.1.isLt; omega⟩)
              (c p.2 ⟨r.val + 1, by have := r.isLt; have := p.1.isLt; omega⟩))

/-- **Origin value is zero**: both empty sums vanish.  This gives canonicity
`pathToHeight c (0, 0) = 0` for free, when `(0, 0)` is a valid cell. -/
theorem pathToHeight_origin {m n : ℕ} (hm : 0 < m) (hn : 0 < n)
    (c : Fin n → PathColouring m) :
    pathToHeight c ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) = 0 := by
  unfold pathToHeight
  simp

/-- **Vertical step identity**: moving down one row in a fixed column adds
`modSign (c j i) (c j (i + 1))` to `pathToHeight`.  Direct from the
column-sum: `Fin.sum_univ_castSucc` extracts the last term.  Does not
require any adjacency hypothesis (the identity is unconditional). -/
theorem pathToHeight_vstep {m n : ℕ} (c : Fin n → PathColouring m)
    (i : ℕ) (hi_succ : i + 1 < m) (j : Fin n) :
    pathToHeight c ((⟨i + 1, hi_succ⟩, j) : Cell m n)
      - pathToHeight c ((⟨i, by omega⟩, j) : Cell m n)
      = modSign (c j ⟨i, by omega⟩) (c j ⟨i + 1, hi_succ⟩) := by
  unfold pathToHeight
  simp only [Fin.sum_univ_castSucc, Fin.val_castSucc, Fin.val_last]
  ring

/-- **Horizontal step identity**: moving right one column at fixed row `i`
adds `modSign (c j i) (c (j + 1) i)` to `pathToHeight`.

Proof: induction on `i`.  Base `i = 0`: both column sums empty; difference
= row-sum's last term.  Inductive step: apply `pathToHeight_vstep` at both
columns to relate `H(i+1, ·) - H(i, ·)` to `modSign` vertical steps, then
`modSign_cycle_sum` at the `2 × 2` block `{(i, j), (i+1, j), (i+1, j+1),
(i, j+1)}` rearranges the vertical difference into a horizontal difference:
`M_v(j+1, i) - M_v(j, i) = M_h(i+1, j) - M_h(i, j)`.

Requires two adjacency hypotheses:
* `hcp` — each column is a proper colouring (adjacent rows within a column
  have distinct colours).
* `hca` — consecutive columns are horizontally adjacent (at each row, they
  have distinct colours). -/
theorem pathToHeight_hstep {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (j : ℕ) (hj_succ : j + 1 < n)
    (i : ℕ) (hi : i < m) :
    pathToHeight c ((⟨i, hi⟩, ⟨j + 1, hj_succ⟩) : Cell m n)
      - pathToHeight c ((⟨i, hi⟩, ⟨j, by omega⟩) : Cell m n)
      = modSign (c ⟨j, by omega⟩ ⟨i, hi⟩)
                (c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩) := by
  induction i with
  | zero =>
    -- Base: column sums are empty; row_sum(j+1) - row_sum(j) = last term.
    unfold pathToHeight
    simp only [Fin.sum_univ_castSucc, Fin.val_castSucc, Fin.val_last]
    ring
  | succ i ih =>
    have hi_prev : i < m := by omega
    -- Vertical step in column j + 1.
    have vstep_jp := pathToHeight_vstep c i hi ⟨j + 1, hj_succ⟩
    -- Vertical step in column j.
    have vstep_j := pathToHeight_vstep c i hi ⟨j, by omega⟩
    -- Cycle-sum adjacency conditions.
    have hab : c ⟨j, by omega⟩ ⟨i, hi_prev⟩ ≠ c ⟨j + 1, hj_succ⟩ ⟨i, hi_prev⟩ :=
      hca j hj_succ ⟨i, hi_prev⟩
    have hbe : c ⟨j + 1, hj_succ⟩ ⟨i, hi_prev⟩ ≠ c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩ :=
      hcp ⟨j + 1, hj_succ⟩ ⟨i, hi_prev⟩ hi
    have hed : c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩ ≠ c ⟨j, by omega⟩ ⟨i + 1, hi⟩ :=
      (hca j hj_succ ⟨i + 1, hi⟩).symm
    have hda : c ⟨j, by omega⟩ ⟨i + 1, hi⟩ ≠ c ⟨j, by omega⟩ ⟨i, hi_prev⟩ :=
      (hcp ⟨j, by omega⟩ ⟨i, hi_prev⟩ hi).symm
    -- Cycle sum: M_h(i, j) + M_v(j+1, i) + modSign(e, d) + modSign(d, a) = 0.
    have cycle := modSign_cycle_sum hab hbe hed hda
    -- Turn modSign(e, d) into -modSign(d, e) = -M_h(i+1, j).
    have hed_symm : modSign (c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩)
                            (c ⟨j, by omega⟩ ⟨i + 1, hi⟩)
                = - modSign (c ⟨j, by omega⟩ ⟨i + 1, hi⟩)
                            (c ⟨j + 1, hj_succ⟩ ⟨i + 1, hi⟩) :=
      modSign_symm hed.symm
    -- Turn modSign(d, a) into -M_v(j, i).
    have hda_symm : modSign (c ⟨j, by omega⟩ ⟨i + 1, hi⟩)
                            (c ⟨j, by omega⟩ ⟨i, hi_prev⟩)
                = - modSign (c ⟨j, by omega⟩ ⟨i, hi_prev⟩)
                            (c ⟨j, by omega⟩ ⟨i + 1, hi⟩) :=
      modSign_symm hda.symm
    -- Now `cycle` combined with `hed_symm` and `hda_symm` gives
    --   M_h(i, j) + M_v(j+1, i) - M_h(i+1, j) - M_v(j, i) = 0
    -- Linarith over cycle, hed_symm, hda_symm, vstep_jp, vstep_j, ih closes.
    have ih' := ih hi_prev
    linarith [cycle, hed_symm, hda_symm, vstep_jp, vstep_j, ih']

/-- **`pathToHeight` is a height function.**  Under proper-column and
column-adjacency hypotheses, `pathToHeight c` produces an integer field
where adjacent cells differ by exactly `1`.

Proof: case-split on `adj_neighbour_cases`; each of the four branches
(row-succ, row-pred, col-succ, col-pred) discharges to `pathToHeight_vstep`
or `pathToHeight_hstep` combined with `modSign_abs`. -/
theorem pathToHeight_isHeight {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r) :
    IsHeight (pathToHeight c) := by
  intro p q hadj
  rcases adj_neighbour_cases p q hadj with
    ⟨hi_hi, hu_eq⟩ | ⟨hi_lo, hu_eq⟩ | ⟨hj_hi, hu_eq⟩ | ⟨hj_lo, hu_eq⟩
  · -- q is row-successor of p: q = (⟨p.1.val + 1, hi_hi⟩, p.2).
    rw [hu_eq]
    have vstep := pathToHeight_vstep c p.1.val hi_hi p.2
    -- vstep : pathToHeight (⟨p.1.val + 1, hi_hi⟩, p.2)
    --         - pathToHeight (⟨p.1.val, _⟩, p.2)
    --         = modSign (c p.2 ⟨p.1.val, _⟩) (c p.2 ⟨p.1.val + 1, hi_hi⟩)
    -- (⟨p.1.val, _⟩, p.2) reduces to p (Fin.eta).
    -- |p.1.val + 1 cell - p| = |-modSign| = 1.
    have h_eq : (⟨p.1.val, by have := p.1.isLt; omega⟩ : Fin m) = p.1 := by
      apply Fin.ext; rfl
    rw [h_eq] at vstep
    have : (p.1, p.2) = p := rfl
    rw [this] at vstep
    have : |pathToHeight c p - pathToHeight c (⟨p.1.val + 1, hi_hi⟩, p.2)|
         = |modSign (c p.2 p.1) (c p.2 ⟨p.1.val + 1, hi_hi⟩)| := by
      rw [show pathToHeight c p - pathToHeight c (⟨p.1.val + 1, hi_hi⟩, p.2)
            = -(pathToHeight c (⟨p.1.val + 1, hi_hi⟩, p.2) - pathToHeight c p) from by ring,
          vstep, abs_neg]
    rw [this]; exact modSign_abs _ _
  · -- q is row-predecessor of p: q = (⟨p.1.val - 1, _⟩, p.2).
    rw [hu_eq]
    -- Apply vstep at i := p.1.val - 1.
    have h_iprev : p.1.val - 1 + 1 = p.1.val := by omega
    have h_iprev_lt : p.1.val - 1 + 1 < m := by
      rw [h_iprev]; exact p.1.isLt
    have vstep := pathToHeight_vstep c (p.1.val - 1) h_iprev_lt p.2
    -- vstep : pathToHeight (⟨p.1.val - 1 + 1, h_iprev_lt⟩, p.2)
    --         - pathToHeight (⟨p.1.val - 1, _⟩, p.2)
    --         = modSign (c p.2 ⟨p.1.val - 1, _⟩) (c p.2 ⟨p.1.val - 1 + 1, h_iprev_lt⟩)
    -- Rewrite p.1.val - 1 + 1 = p.1.val.
    have h_fin1 : (⟨p.1.val - 1 + 1, h_iprev_lt⟩ : Fin m) = p.1 := by
      apply Fin.ext; simp; omega
    rw [h_fin1] at vstep
    have : (p.1, p.2) = p := rfl
    rw [this] at vstep
    have : |pathToHeight c p
            - pathToHeight c (⟨p.1.val - 1, by omega⟩, p.2)|
         = |modSign (c p.2 ⟨p.1.val - 1, by omega⟩) (c p.2 p.1)| := by
      rw [vstep]
    rw [this]; exact modSign_abs _ _
  · -- q is column-successor of p: q = (p.1, ⟨p.2.val + 1, hj_hi⟩).
    rw [hu_eq]
    have hstep := pathToHeight_hstep c hcp hca p.2.val hj_hi p.1.val p.1.isLt
    -- hstep : pathToHeight (⟨p.1.val, p.1.isLt⟩, ⟨p.2.val + 1, hj_hi⟩)
    --         - pathToHeight (⟨p.1.val, p.1.isLt⟩, ⟨p.2.val, _⟩)
    --         = modSign (c ⟨p.2.val, _⟩ ⟨p.1.val, p.1.isLt⟩)
    --                   (c ⟨p.2.val + 1, hj_hi⟩ ⟨p.1.val, p.1.isLt⟩)
    have h_fin_col : (⟨p.2.val, by have := p.2.isLt; omega⟩ : Fin n) = p.2 := by
      apply Fin.ext; rfl
    have h_fin_row : (⟨p.1.val, p.1.isLt⟩ : Fin m) = p.1 := by
      apply Fin.ext; rfl
    rw [h_fin_col, h_fin_row] at hstep
    have : (p.1, p.2) = p := rfl
    rw [this] at hstep
    have : |pathToHeight c p - pathToHeight c (p.1, ⟨p.2.val + 1, hj_hi⟩)|
         = |modSign (c p.2 p.1) (c ⟨p.2.val + 1, hj_hi⟩ p.1)| := by
      rw [show pathToHeight c p - pathToHeight c (p.1, ⟨p.2.val + 1, hj_hi⟩)
            = -(pathToHeight c (p.1, ⟨p.2.val + 1, hj_hi⟩) - pathToHeight c p) from by ring,
          hstep, abs_neg]
    rw [this]; exact modSign_abs _ _
  · -- q is column-predecessor of p: q = (p.1, ⟨p.2.val - 1, _⟩).
    rw [hu_eq]
    have h_jprev : p.2.val - 1 + 1 = p.2.val := by omega
    have h_jprev_lt : p.2.val - 1 + 1 < n := by
      rw [h_jprev]; exact p.2.isLt
    have hstep := pathToHeight_hstep c hcp hca (p.2.val - 1) h_jprev_lt p.1.val p.1.isLt
    have h_fin_col : (⟨p.2.val - 1 + 1, h_jprev_lt⟩ : Fin n) = p.2 := by
      apply Fin.ext; simp; omega
    have h_fin_row : (⟨p.1.val, p.1.isLt⟩ : Fin m) = p.1 := by
      apply Fin.ext; rfl
    rw [h_fin_col, h_fin_row] at hstep
    have : (p.1, p.2) = p := rfl
    rw [this] at hstep
    have : |pathToHeight c p
            - pathToHeight c (p.1, ⟨p.2.val - 1, by omega⟩)|
         = |modSign (c ⟨p.2.val - 1, by omega⟩ p.1) (c p.2 p.1)| := by
      rw [hstep]
    rw [this]; exact modSign_abs _ _

/-! ## `hCol` reduction (Step 5c-iv-b-3)

The mod-`3` shadow of `pathToHeight c` equals the input colouring `c` up
to an origin shift.  Precisely: `(pathToHeight c (i, j) : ZMod 3) = c j i
- c 0 0`.  Under the origin normalisation `c 0 0 = 0` (which we can
always achieve by shifting the whole colouring by a constant), the
reduction matches the input exactly: `hCol (pathToHeight c) = c`.

Proof route:
* **Vertical mod-`3` step**: `hCol (pathToHeight c) j (i + 1) - hCol
  (pathToHeight c) j i = c j (i + 1) - c j i`.  Cast of
  `pathToHeight_vstep` via `modSign_cast_eq_diff`.
* **Horizontal mod-`3` step**: analogous cast of `pathToHeight_hstep`.
* **Assembly**: induction on `i` (vstep), base case induction on `j`
  (hstep). Origin case closes by `pathToHeight_origin`. -/

/-- **Vertical mod-`3` step**: casting `pathToHeight_vstep` mod `3` and
applying `modSign_cast_eq_diff` recovers the direct colour difference. -/
theorem hCol_pathToHeight_vstep {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (i : ℕ) (hi_succ : i + 1 < m) (j : Fin n) :
    ((pathToHeight c ((⟨i + 1, hi_succ⟩, j) : Cell m n) : ℤ) : ZMod 3)
      - ((pathToHeight c ((⟨i, by omega⟩, j) : Cell m n) : ℤ) : ZMod 3)
      = c j ⟨i + 1, hi_succ⟩ - c j ⟨i, by omega⟩ := by
  have vstep := pathToHeight_vstep c i hi_succ j
  have hab : c j ⟨i, by omega⟩ ≠ c j ⟨i + 1, hi_succ⟩ :=
    hcp j ⟨i, by omega⟩ hi_succ
  have h_cast_step :
      ((pathToHeight c ((⟨i + 1, hi_succ⟩, j) : Cell m n) : ℤ) : ZMod 3)
        - ((pathToHeight c ((⟨i, by omega⟩, j) : Cell m n) : ℤ) : ZMod 3)
        = ((modSign (c j ⟨i, by omega⟩) (c j ⟨i + 1, hi_succ⟩) : ℤ) : ZMod 3) := by
    rw [← Int.cast_sub, vstep]
  rw [h_cast_step, modSign_cast_eq_diff hab]

/-- **Horizontal mod-`3` step**: analogous cast of `pathToHeight_hstep`. -/
theorem hCol_pathToHeight_hstep {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (j : ℕ) (hj_succ : j + 1 < n)
    (i : ℕ) (hi : i < m) :
    ((pathToHeight c ((⟨i, hi⟩, ⟨j + 1, hj_succ⟩) : Cell m n) : ℤ) : ZMod 3)
      - ((pathToHeight c ((⟨i, hi⟩, ⟨j, by omega⟩) : Cell m n) : ℤ) : ZMod 3)
      = c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩ - c ⟨j, by omega⟩ ⟨i, hi⟩ := by
  have hstep := pathToHeight_hstep c hcp hca j hj_succ i hi
  have hab : c ⟨j, by omega⟩ ⟨i, hi⟩ ≠ c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩ :=
    hca j hj_succ ⟨i, hi⟩
  have h_cast_step :
      ((pathToHeight c ((⟨i, hi⟩, ⟨j + 1, hj_succ⟩) : Cell m n) : ℤ) : ZMod 3)
        - ((pathToHeight c ((⟨i, hi⟩, ⟨j, by omega⟩) : Cell m n) : ℤ) : ZMod 3)
        = ((modSign (c ⟨j, by omega⟩ ⟨i, hi⟩) (c ⟨j + 1, hj_succ⟩ ⟨i, hi⟩) : ℤ) : ZMod 3) := by
    rw [← Int.cast_sub, hstep]
  rw [h_cast_step, modSign_cast_eq_diff hab]

/-- **Row-`0` reduction**: along the row-`0` walk, the mod-`3` shadow of
`pathToHeight` matches the input colouring's row `0` up to origin shift.
Proof: induction on `j` using `hCol_pathToHeight_hstep`.  Base `j = 0`:
both sides are `0`. -/
theorem hCol_pathToHeight_row0 {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (hm : 0 < m) (j : ℕ) (hj : j < n) :
    ((pathToHeight c ((⟨0, hm⟩, ⟨j, hj⟩) : Cell m n) : ℤ) : ZMod 3)
      = c ⟨j, hj⟩ ⟨0, hm⟩ - c ⟨0, by omega⟩ ⟨0, hm⟩ := by
  induction j with
  | zero =>
    -- Origin: pathToHeight (0, 0) = 0.
    have h_hn : (0 : ℕ) < n := by omega
    rw [pathToHeight_origin hm h_hn c]
    push_cast; ring
  | succ j ih =>
    have hj_prev : j < n := by omega
    have hstep := hCol_pathToHeight_hstep c hcp hca j hj (0 : ℕ) hm
    -- hstep : hCol (0, j+1) - hCol (0, j) = c (j+1) 0 - c j 0
    have ih' := ih hj_prev
    -- ih' : hCol (0, j) = c j 0 - c 0 0
    linear_combination hstep + ih'

/-- **Full `hCol` reduction**: for every cell `(i, j)`, the mod-`3` shadow
of `pathToHeight c` equals `c j i - c 0 0`.  Proof: induction on `i`
using `hCol_pathToHeight_vstep`; base case `i = 0` closes via
`hCol_pathToHeight_row0`. -/
theorem hCol_pathToHeight {m n : ℕ} (c : Fin n → PathColouring m)
    (hcp : ∀ k : Fin n, IsPathProperColouring (c k))
    (hca : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
              c ⟨k, by omega⟩ r ≠ c ⟨k + 1, hk⟩ r)
    (i : ℕ) (hi : i < m) (j : ℕ) (hj : j < n) :
    ((pathToHeight c ((⟨i, hi⟩, ⟨j, hj⟩) : Cell m n) : ℤ) : ZMod 3)
      = c ⟨j, hj⟩ ⟨i, hi⟩ - c ⟨0, by omega⟩ ⟨0, by omega⟩ := by
  induction i with
  | zero =>
    -- Base: use hCol_pathToHeight_row0.
    exact hCol_pathToHeight_row0 c hcp hca hi j hj
  | succ i ih =>
    have hi_prev : i < m := by omega
    have vstep := hCol_pathToHeight_vstep c hcp i hi ⟨j, hj⟩
    -- vstep : hCol (i+1, j) - hCol (i, j) = c j (i+1) - c j i
    have ih' := ih hi_prev
    -- ih' : hCol (i, j) = c j i - c 0 0
    linear_combination vstep + ih'

/-! ## Bijection identity (Step 5c-iv-c)

Composition of the forward (`hCol`) and backward (`pathToHeight`) maps
recovers the identity up to origin shift:

  `pathToHeight (hCol h) (i, j) = h (i, j) - h (0, 0)`

for any height function `h`.

The proof uses two "signed-difference matches integer step" helpers:
each modSign of adjacent hCol values equals the corresponding
`h`-difference in `ℤ`. These come directly from `IsHeight`'s ±1
condition, distinguishing the two cases by sign to match `modSign_of_succ`
or `modSign_of_pred`. -/

/-- **modSign of column-adjacent hCol values matches the h-difference.**
For `h : IsHeight`, `modSign(hCol h j i, hCol h j (i+1)) = h(i+1, j) - h(i, j)`
in `ℤ`.  Distinguishes ±1 cases directly from `IsHeight`. -/
private theorem modSign_hCol_v {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : ℕ) (hi_succ : i + 1 < m) (j : Fin n) :
    modSign ((hCol h) j ⟨i, by omega⟩) ((hCol h) j ⟨i + 1, hi_succ⟩)
      = h ((⟨i + 1, hi_succ⟩, j) : Cell m n) - h ((⟨i, by omega⟩, j) : Cell m n) := by
  have h_adj : adj ((⟨i, by omega⟩, j) : Cell m n) (⟨i + 1, hi_succ⟩, j) :=
    adj_row i (by omega) hi_succ j
  have h_diff := hh _ _ h_adj
  -- h_diff : |h (⟨i, _⟩, j) - h (⟨i + 1, hi_succ⟩, j)| = 1
  rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp h_diff with h1 | h1
  · -- h(i, j) - h(i+1, j) = 1, so h(i+1, j) = h(i, j) - 1, so hCol h j (i+1) = hCol h j i - 1.
    -- hCol h j i + 1 ≠ hCol h j (i+1) (since a + 1 ≠ a - 1 in ZMod 3), so modSign = -1.
    unfold hCol
    rw [modSign_of_pred]
    · linarith
    · -- Goal: (h(i, j) : ZMod 3) + 1 ≠ (h(i+1, j) : ZMod 3)
      intro hcast
      have h_eq : h (⟨i + 1, hi_succ⟩, j) = h (⟨i, by omega⟩, j) - 1 := by linarith
      rw [h_eq] at hcast
      push_cast at hcast
      -- hcast : (h(i, j) : ZMod 3) + 1 = (h(i, j) : ZMod 3) - 1
      have h2 : (2 : ZMod 3) = 0 := by linear_combination hcast
      exact absurd h2 (by decide)
  · -- h(i, j) - h(i+1, j) = -1, so h(i+1, j) = h(i, j) + 1, so hCol h j (i+1) = hCol h j i + 1.
    unfold hCol
    rw [modSign_of_succ]
    · linarith
    · -- Goal: (h(i, j) : ZMod 3) + 1 = (h(i+1, j) : ZMod 3)
      have h_eq : h (⟨i + 1, hi_succ⟩, j) = h (⟨i, by omega⟩, j) + 1 := by linarith
      rw [h_eq]; push_cast; ring

/-- **modSign of row-adjacent hCol values matches the h-difference.**
For `h : IsHeight`, `modSign(hCol h j i, hCol h (j+1) i) = h(i, j+1) - h(i, j)`
in `ℤ`.  Analog of `modSign_hCol_v`. -/
private theorem modSign_hCol_h {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (j : ℕ) (hj_succ : j + 1 < n) (i : Fin m) :
    modSign ((hCol h) ⟨j, by omega⟩ i) ((hCol h) ⟨j + 1, hj_succ⟩ i)
      = h ((i, ⟨j + 1, hj_succ⟩) : Cell m n) - h ((i, ⟨j, by omega⟩) : Cell m n) := by
  have h_adj : adj ((i, ⟨j, by omega⟩) : Cell m n) (i, ⟨j + 1, hj_succ⟩) :=
    adj_col i j (by omega) hj_succ
  have h_diff := hh _ _ h_adj
  rcases abs_eq (by norm_num : (0 : ℤ) ≤ 1) |>.mp h_diff with h1 | h1
  · unfold hCol
    rw [modSign_of_pred]
    · linarith
    · intro hcast
      have h_eq : h (i, ⟨j + 1, hj_succ⟩) = h (i, ⟨j, by omega⟩) - 1 := by linarith
      rw [h_eq] at hcast
      push_cast at hcast
      have h2 : (2 : ZMod 3) = 0 := by linear_combination hcast
      exact absurd h2 (by decide)
  · unfold hCol
    rw [modSign_of_succ]
    · linarith
    · have h_eq : h (i, ⟨j + 1, hj_succ⟩) = h (i, ⟨j, by omega⟩) + 1 := by linarith
      rw [h_eq]; push_cast; ring

/-- **Row-0 bijection**: along row 0, `pathToHeight (hCol h) (0, j) =
h(0, j) - h(0, 0)`.  Induction on `j` using `pathToHeight_hstep` for
`pathToHeight (hCol h)` combined with `modSign_hCol_h`. -/
private theorem pathToHeight_hCol_row0 {m n : ℕ} {h : Cell m n → ℤ}
    (hh : IsHeight h) (hm : 0 < m) (j : ℕ) (hj : j < n) :
    pathToHeight (hCol h) ((⟨0, hm⟩, ⟨j, hj⟩) : Cell m n)
      = h ((⟨0, hm⟩, ⟨j, hj⟩) : Cell m n) - h ((⟨0, hm⟩, ⟨0, by omega⟩) : Cell m n) := by
  induction j with
  | zero =>
    have h_hn : (0 : ℕ) < n := by omega
    rw [pathToHeight_origin hm h_hn (hCol h)]; ring
  | succ j ih =>
    have hj_prev : j < n := by omega
    -- hCol adjacencies for pathToHeight_hstep.
    have hcp_h : ∀ k : Fin n, IsPathProperColouring ((hCol h) k) :=
      fun k => hCol_isPathProperColouring hh k
    have hca_h : ∀ (k : ℕ) (hk : k + 1 < n) (r : Fin m),
        (hCol h) ⟨k, by omega⟩ r ≠ (hCol h) ⟨k + 1, hk⟩ r := fun k hk r => by
      have := hCol_pathAdjacent hh (by omega : k < n) hk
      exact this.2.2 r
    have hstep := pathToHeight_hstep (hCol h) hcp_h hca_h j hj (0 : ℕ) hm
    -- hstep : pathToHeight (hCol h) (0, j+1) - pathToHeight (hCol h) (0, j)
    --         = modSign((hCol h) j 0)((hCol h) (j+1) 0)
    have modSign_match := modSign_hCol_h hh j hj ⟨0, hm⟩
    -- modSign_match : modSign((hCol h) j 0)((hCol h) (j+1) 0)
    --               = h(0, j+1) - h(0, j)
    have ih' := ih hj_prev
    linarith [hstep, modSign_match, ih']

/-- **Bijection identity** (5c-iv-c): the composition of forward (`hCol`)
and backward (`pathToHeight`) recovers the identity up to origin shift.
Induction on `i` using `pathToHeight_vstep` for `pathToHeight (hCol h)`
combined with `modSign_hCol_v`. Base case `i = 0` closes via
`pathToHeight_hCol_row0`. -/
theorem pathToHeight_hCol {m n : ℕ} {h : Cell m n → ℤ} (hh : IsHeight h)
    (i : ℕ) (hi : i < m) (j : ℕ) (hj : j < n) :
    pathToHeight (hCol h) ((⟨i, hi⟩, ⟨j, hj⟩) : Cell m n)
      = h ((⟨i, hi⟩, ⟨j, hj⟩) : Cell m n)
        - h ((⟨0, by omega⟩, ⟨0, by omega⟩) : Cell m n) := by
  induction i with
  | zero =>
    exact pathToHeight_hCol_row0 hh hi j hj
  | succ i ih =>
    have hi_prev : i < m := by omega
    have vstep := pathToHeight_vstep (hCol h) i hi ⟨j, hj⟩
    have modSign_match := modSign_hCol_v hh i hi ⟨j, hj⟩
    have ih' := ih hi_prev
    linarith [vstep, modSign_match, ih']

/-! ## Column decomposition of `numExtrema` (Step 5c-v-a)

The total extremum count of a height function factors as a sum over
columns of per-column extremum counts.  This is pure Fubini on the
`Fin m × Fin n` = Cell m n indexing set. -/

/-- **Column decomposition**: `numExtrema h = ∑ j, (column-j extremum count)`. -/
theorem numExtrema_eq_sum_over_cols {m n : ℕ} (h : Cell m n → ℤ) :
    numExtrema h = ∑ j : Fin n,
        (Finset.univ.filter (fun i : Fin m => IsStrictLocalExtremum h (i, j))).card := by
  unfold numExtrema
  simp_rw [Finset.card_filter]
  rw [Fintype.sum_prod_type (f := fun (p : Fin m × Fin n) =>
    (if IsStrictLocalExtremum h p then (1 : ℕ) else 0))]
  exact Finset.sum_comm

/-- **Interior column count = `columnExtremaCount`.**  For interior `j`
(with `1 ≤ j ≤ n - 2`), the number of extrema in column `j` equals
`columnExtremaCount` on the mod-`3` triple `(hCol h (j-1), hCol h j,
hCol h (j+1))`. -/
theorem col_count_interior_eq {m n : ℕ}
    {h : Cell m n → ℤ} (hh : IsHeight h)
    (j : ℕ) (hj_lo : 0 < j) (hj_hi : j + 1 < n) :
    (Finset.univ.filter
        (fun i : Fin m => IsStrictLocalExtremum h (i, ⟨j, by omega⟩))).card
      = columnExtremaCount (hCol h ⟨j - 1, by omega⟩)
                           (hCol h ⟨j, by omega⟩)
                           (hCol h ⟨j + 1, hj_hi⟩) := by
  unfold columnExtremaCount
  congr 1
  apply Finset.filter_congr
  intro i _
  exact (IsStrictLocalExtremum_iff_IsColExtremum_interior hh i hj_lo hj_hi)

/-- **Left boundary column count = `IsBoundaryExtremum` count**.  For
`j = 0` on a grid with `n ≥ 2`, the number of extrema in column `0`
equals the `IsBoundaryExtremum` count on the pair `(hCol h 0, hCol h 1)`. -/
theorem col_count_left_boundary_eq {m n : ℕ}
    {h : Cell m n → ℤ} (hh : IsHeight h)
    (hn : 0 < n) (hn_succ : 1 < n) :
    (Finset.univ.filter
        (fun i : Fin m => IsStrictLocalExtremum h (i, ⟨0, hn⟩))).card
      = (Finset.univ.filter
          (fun i : Fin m => IsBoundaryExtremum (hCol h ⟨0, hn⟩) (hCol h ⟨1, hn_succ⟩) i)).card := by
  congr 1
  apply Finset.filter_congr
  intro i _
  exact (IsStrictLocalExtremum_iff_IsBoundaryExtremum_left hh i hn hn_succ)

/-- **Right boundary column count = `IsBoundaryExtremum` count**.  For
`j = n - 1` on a grid with `n ≥ 2`, the number of extrema in column `n - 1`
equals the `IsBoundaryExtremum` count on the pair `(hCol h (n-1), hCol h (n-2))`. -/
theorem col_count_right_boundary_eq {m n : ℕ}
    {h : Cell m n → ℤ} (hh : IsHeight h)
    (hn_succ : 1 < n) :
    (Finset.univ.filter
        (fun i : Fin m => IsStrictLocalExtremum h (i, ⟨n - 1, by omega⟩))).card
      = (Finset.univ.filter
          (fun i : Fin m => IsBoundaryExtremum
              (hCol h ⟨n - 1, by omega⟩) (hCol h ⟨n - 2, by omega⟩) i)).card := by
  congr 1
  apply Finset.filter_congr
  intro i _
  exact (IsStrictLocalExtremum_iff_IsBoundaryExtremum_right hh i hn_succ)

/-! ## Boundary vectors (Step 5c-vi-prep)

The paper's transfer identity `c_{m,n}(x) = u^⊤ · T_m(x)^{n-2} · v` uses
boundary vectors `u` and `v` that encode the extremum contributions of
the two end columns (which are NOT captured by `T_m(x)` — the transfer
matrix only handles middle columns of triples).

`u` is indexed by the LEFT-boundary pair state `s = (c_0, c_1)`:
`u(s) = x^{|extrema at col 0 given (c_0, c_1)|}`.

`v` is indexed by the RIGHT-boundary pair state `s = (c_{n-2}, c_{n-1})`:
`v(s) = x^{|extrema at col (n-1) given (c_{n-2}, c_{n-1})|}`.

The extremum counts use `IsBoundaryExtremum`; `u` treats `s.val.1` as
`own` (the boundary column) and `s.val.2` as `other`, while `v` treats
`s.val.2` as `own` and `s.val.1` as `other` (opposite orientation).

Definitional-only; the matrix identity `cnPoly = u^⊤ · T^{n-2} · v` is
deferred to future work. -/

/-- **Left boundary vector**.  `leftBdyVec m s = x^{|extrema at column 0
given adjacent pair (c_0, c_1) = s|}`.  Treats `s.val.1` as the boundary
column and `s.val.2` as its right neighbour. -/
noncomputable def leftBdyVec (m : ℕ) (s : TransferState m) : Polynomial ℤ :=
  Polynomial.monomial
    ((Finset.univ.filter
        (fun i : Fin m => IsBoundaryExtremum s.val.1 s.val.2 i)).card)
    (1 : ℤ)

/-- **Right boundary vector**.  `rightBdyVec m s = x^{|extrema at column
(n-1) given adjacent pair (c_{n-2}, c_{n-1}) = s|}`.  Treats `s.val.2`
as the boundary column and `s.val.1` as its left neighbour (opposite of
`leftBdyVec`). -/
noncomputable def rightBdyVec (m : ℕ) (s : TransferState m) : Polynomial ℤ :=
  Polynomial.monomial
    ((Finset.univ.filter
        (fun i : Fin m => IsBoundaryExtremum s.val.2 s.val.1 i)).card)
    (1 : ℤ)

/-- **Both boundary vectors are monic monomials.**  `leftBdyVec m s`
never vanishes: it is `X^k` for some `k` (the extremum count), and
`monomial k 1` is a monic monomial of degree `k`. -/
theorem leftBdyVec_ne_zero (m : ℕ) (s : TransferState m) :
    leftBdyVec m s ≠ 0 := by
  unfold leftBdyVec
  simp [Polynomial.monomial_eq_zero_iff]

/-- Same for `rightBdyVec`. -/
theorem rightBdyVec_ne_zero (m : ℕ) (s : TransferState m) :
    rightBdyVec m s ≠ 0 := by
  unfold rightBdyVec
  simp [Polynomial.monomial_eq_zero_iff]

/-! ### Origin normalization for the left boundary

The paper's identity `c_{m,n}(x) = u^⊤ · T_m(x)^{n-2} · v` uses **canonical**
height functions on the LHS (normalized so `h(0, 0) = 0`).  The transfer
matrix `T_m` acts on ALL admissible column pairs, which by `lem:quotient`
carry a free `ℤ/3` rotation ρ.  Summing over the full `TransferState m`
without an indicator would triple-count each canonical height (once per
ρ-orbit).

The paper's convention (best guess pending paper re-read): the LEFT
boundary vector `u` includes an **origin-normalization indicator** at
the corner cell — `u(s) = 0` unless `s.val.1(⟨0⟩) = 0`.  The RIGHT
boundary vector `v` does NOT need normalization, since the corner cell
is fixed at the LEFT end and the transfer propagates automatically.

This section supplies the normalized variant `leftBdyVecNorm` alongside
the unnormalized `leftBdyVec` (which is the raw extremum-weight,
retained for its role as the "column-0 weight" independent of any
corner-normalization convention).  Downstream use of the matrix
identity should pick `leftBdyVecNorm` — but confirm against the paper
first.  See the session handoff prompt for the full open question. -/

/-- **Normalized state at the left boundary**: a `TransferState m` whose
first column's row-0 entry is `0`.  Corresponds to a canonical height
with `h(0, 0) = 0` after mod-3 reduction. -/
def IsNormalizedLeft {m : ℕ} (s : TransferState m) : Prop :=
  ∃ h : 0 < m, s.val.1 ⟨0, h⟩ = 0

instance {m : ℕ} (s : TransferState m) : Decidable (IsNormalizedLeft s) := by
  unfold IsNormalizedLeft
  by_cases h : 0 < m
  · exact decidable_of_iff _ ⟨fun heq => ⟨h, heq⟩, fun ⟨_, heq⟩ => heq⟩
  · exact isFalse (fun ⟨h', _⟩ => h h')

/-- **Left boundary vector with origin normalization** (candidate for the
paper's `u` in `c_{m,n}(x) = u^⊤ · T_m(x)^{n-2} · v`).  Includes the
indicator `[s.val.1(⟨0⟩) = 0]` to restrict to canonical-height starting
states, avoiding the Z/3-rotation triple-counting. -/
noncomputable def leftBdyVecNorm (m : ℕ) (s : TransferState m) : Polynomial ℤ :=
  if IsNormalizedLeft s then leftBdyVec m s else 0

/-- **`leftBdyVecNorm` is zero when `s` is not normalized**. -/
theorem leftBdyVecNorm_of_not_normalized {m : ℕ} {s : TransferState m}
    (h : ¬ IsNormalizedLeft s) : leftBdyVecNorm m s = 0 := by
  unfold leftBdyVecNorm; rw [if_neg h]

/-- **`leftBdyVecNorm` equals `leftBdyVec` on normalized states**. -/
theorem leftBdyVecNorm_of_normalized {m : ℕ} {s : TransferState m}
    (h : IsNormalizedLeft s) : leftBdyVecNorm m s = leftBdyVec m s := by
  unfold leftBdyVecNorm; rw [if_pos h]

end OrigamiCone.Sequel

import OrigamiCone.QuotientDegree
import OrigamiCone.SequelPolyAssembly
import Mathlib.Data.Fintype.Pi

/-!
# Formal `E_d` skeleton for the sequel paper's `thm:poly`

Defines the paper's degree-`d` vertex count function `E_d(m, n)` as a Lean
formal object, sitting on top of the existing height-function substrate
(`Basic.lean`: `Cell`, `IsHeight`) and the extrema definitions
(`DegreeExtrema.lean`: `IsStrictLocalExtremum`).

The paper's `E_d(m, n)` (sequel §1, §4) is the number of vertices of
`OFG(M_{m,n})` — the origami flip graph — of vertex-degree exactly `d`.
By paper's `Lemma 2.1` (Lean: `ofgDegree_eq_extrema`, kernel-checked for
`mn ≥ 3`), the OFG-degree of the shift class of a *height function*
`h : Cell m n → ℤ` equals the number of its strict local extrema.

## Definition strategy

Every OFG vertex is a shift class of a height function.  We count shift
classes by counting their **canonical representatives**: a height
function `h` with `h ⟨⟨0, _⟩, ⟨0, _⟩⟩ = 0`.  When `m, n ≥ 1` this fixes
a unique representative per shift class; when the grid is empty the
universal quantifier is vacuous and there is a unique (empty)
representative.

Restricting the counted domain to height functions (rather than
`OFGVertex m n = Quotient (shiftSetoid m n)`, whose setoid is on ALL
functions `Cell m n → ℤ`) is essential: `OFGAdj` (in
`DegreeExtrema.lean`) requires only the *target* of an edge to be a
height function, so many non-height classes have positive `OFGDegree`
and would otherwise inflate the counted fibre to an infinite set,
forcing `Set.ncard = 0` on the paper-relevant degrees.

This module records:

* `numExtrema h` — the extremum count of a function
  (`Finset.univ.filter … .card`);
* `IsCanonicalHeight h` — `h` is a height function and takes value
  `0` at every origin cell `⟨⟨0, _⟩, ⟨0, _⟩⟩` (there is at most one);
* `Ed d m n` — `Set.ncard` of canonical heights with `numExtrema = d`,
  the paper's `E_d`.

## Scope

Skeleton + finiteness + counting API + `mkV` cardinality bijection +
stratified per-degree bijection + `Ed` symmetry + `thm:poly` skeleton
(hsym-closed).  Deliberately does NOT prove:

* **`Ed` split decomposition.**  Paper's `eq:Ed`,
  `E_d = ∑_{a ≤ b, a + b = d} (2 - δ_{a,b}) · N_{(a,b)}`.
* **Per-axis polynomiality** (paper's `thm:poly` `hrow` + `hcol`).  The
  specialisation `Ed_thm_poly_of_perAxis` takes these as explicit
  parameters; closing them at `Ed` needs the transfer-matrix
  polynomiality chain (`SequelRatGF` + `SequelPoles*` + `SequelFrozen` +
  `SequelQuotient`).

DOES prove:

* **Finiteness of `CanonicalHeights m n`** for `m, n ≥ 1` (via
  `height_lipschitz` bound `|h(p) - h(origin)| ≤ (m-1)+(n-1) = m+n-2`).
  Was the top downstream obligation in the previous version's scope
  section — without it, `Set.ncard` collapses to 0 and `Ed d m n = 0`
  on every input.
* **Finiteness of every `Ed` fibre.**  Immediate corollary of
  `CanonicalHeights_finite`.

Two trivial companion theorems record the API-level structure the
canonical-heights domain does support (unlike the whole `OFGVertex m n`
type, which is infinite for `mn ≥ 1`):

* `canonical_heights_partition_by_extrema` — the canonical-heights set
  partitions into extremum-count fibres.
* `Ed_fibres_disjoint` — distinct extremum counts give disjoint fibres.

Two finiteness theorems close the previous version's top downstream
obligation:

* `CanonicalHeights_finite` — `CanonicalHeights m n` is a finite set
  for `m, n ≥ 1` (via `height_lipschitz` + canonical `h(0,0) = 0`).
* `Ed_fibre_finite` — every `Ed`-fibre is finite (immediate
  corollary).

A counting-API layer bridges `Ed` from `Set.ncard` into the
`Nat.card` / `Finset.card` counting flavours downstream work needs:

* `Ed_eq_natCard` — `Ed d m n = Nat.card ↥{h // …}` (unconditional; `rfl`).
* `Ed_eq_finset_card` — `Ed d m n = |(Ed_fibre_finite d hm hn).toFinset|`
  (nonempty grid).
* `Ed_le_ncard_canonicalHeights` — every `Ed` fibre is bounded by the
  total canonical-height count (nonempty grid).
* `numExtrema_le_mn` — every extremum count is at most `m · n`
  (unconditional; trivial `Finset.filter` bound).
* `sum_Ed_eq_ncard_canonicalHeights` — **partition identity**:
  `∑_{d=0}^{m·n} Ed d m n = (CanonicalHeights m n).ncard` (nonempty
  grid).  Uses `Finset.card_eq_sum_card_fiberwise` at `numExtrema`.

A canonicalisation + `mkV`-bijection layer bridges canonical heights
to their OFG shift classes (Item 6c-b substrate):

* `canonicalise h hm hn` — subtracts `h(origin)` pointwise; produces
  a canonical rep in `h`'s shift class.
* `canonical_unique` — shift-equivalent canonical heights are equal.
* `canonicalise_shiftEq` / `canonicalise_isHeight` /
  `canonicalise_isCanonicalHeight` — canonicalisation is
  shift-equivalent, preserves the height property, and lands in
  `CanonicalHeights`.
* `numExtrema_canonicalise` — `numExtrema (canonicalise h) = numExtrema h`.
* `HeightOFGVertex m n` — the set of OFG shift classes containing a
  height-function representative.
* `mkV_injOn_canonical` — `mkV` is injective on `CanonicalHeights m n`.
* `mkV_image_canonical` — the image of `mkV` on `CanonicalHeights m n`
  is exactly `HeightOFGVertex m n`.
* `canonicalHeights_ncard_eq_heightOFGVertex` — the two sets have equal
  `ncard`s (total-count cardinality bijection).

A stratified per-degree bijection promotes the total-count bridge to
per-degree fibres, closing Item 6c-b (requires `mn ≥ 3` for
`ofgDegree_eq_extrema`):

* `mkV_image_fibre` — image of `mkV` on the `d`-extrema fibre of
  `CanonicalHeights` equals the OFG-degree-`d` fibre of `HeightOFGVertex`.
* `Ed_eq_ncard_heightOFG_degree` — **paper's `E_d` faithfulness**:
  `Ed d m n = |{C ∈ HeightOFGVertex m n | OFGDegree C = d}|.ncard`
  for `m, n ≥ 1` and `mn ≥ 3`.  `HeightOFGVertex m n` is the paper's
  OFG vertex set (via the Ginepro–Hull colouring bijection).

An `Ed` symmetry layer, supplying one of the three
`SequelPolyAssembly.thm_poly_abstract` hypotheses at `Ed`:

* `swapH` — transpose a function `Cell m n → ℤ` to `Cell n m → ℤ` by
  swapping coordinates.
* `adj_swap` — grid adjacency is invariant under the swap.
* `swapH_isHeight` / `swapH_isCanonical` — swap preserves height /
  canonicality.
* `swapH_strictMax_iff` / `swapH_strictMin_iff` /
  `swapH_strictExtremum_iff` — extremum condition is preserved under
  swap (up to the `Prod.swap` on cells).
* `swapH_numExtrema` — extremum count is preserved.
* `Ed_symm` — `Ed d m n = Ed d n m` for `m, n ≥ 1`.

A boundary-corollary layer records the paper's `E_d = 0` for
degenerate small-`d`:

* `exists_strictLocalMax` — every height function on a nonempty grid
  has a strict local max (at its argmax cell).
* `exists_strictLocalMin` — dual.
* `numExtrema_ge_two` — every height function on `mn ≥ 2` has
  `numExtrema ≥ 2` (argmax and argmin are distinct extrema via
  `max_min_excl`).
* `Ed_lt_two_eq_zero` — `Ed d m n = 0` for `d < 2` on `mn ≥ 2`
  (specialisations: `Ed 0 m n = Ed 1 m n = 0`).

A `thm:poly` specialisation skeleton packages `SequelPolyAssembly.thm_poly_*`
at `Ed`:

* `Ed_thm_poly_of_perAxis (d D lo) (hlo : 1 ≤ lo) (hrow hcol)` —
  specialises `thm_poly_polynomial_symmetry` at `F := fun m n => (Ed d m n : ℚ)`,
  supplying the symmetry hypothesis via `Ed_symm` + `exact_mod_cast`.
  Requires `hrow`, `hcol` (per-axis polynomiality along rows/columns)
  as explicit parameters — the deep substrate obligation deferred to
  the sequel's transfer-matrix machinery (`SequelRatGF` + `SequelPoles*`
  + `SequelFrozen` + `SequelQuotient`).

Fully-closed `thm:poly` instances at the degenerate `d < 2` boundary
(the paper's `E_d = 0` for `d ∈ {0, 1}`):

* `Ed_hrow_of_lt_two` / `Ed_hcol_of_lt_two` — for `d < 2` on
  `lo ≥ 2`, the per-axis witness is the zero polynomial (of degree 0),
  discharged unconditionally via `Ed_lt_two_eq_zero`.
* `Ed_lt_two_thm_poly` — full `thm:poly` at `d < 2` on the region
  `{a, b ≥ 2}` (strictly weaker than the paper's `{a, b ≥ 0}` because
  `Ed 1 1 1 = 1 ≠ 0` — the 1×1 grid has a vacuously-both-max-and-min
  cell — but strictly stronger than the paper's `mn ≥ 3` bridge).
  The paper's `E_d = 0` for `d ∈ {0, 1}` is kernel-certified as
  polynomiality with `p_d = 0` on `{a, b ≥ 2}`.

Upper-bound corollary — dual boundary:

* `Ed_gt_mn_eq_zero` — `Ed d m n = 0` for `d > m * n` (via
  `numExtrema_le_mn`).  Combined with `Ed_lt_two_eq_zero`, this gives
  `Ed d m n = 0` outside the "reachable" band `{d : 2 ≤ d ≤ m * n}`.
  The upper bound does not directly yield a uniform `thm:poly` on a
  high region — see the module note.

No `sorry`.  Axioms: `[propext, Classical.choice, Quot.sound]` baseline.
Check with `#print axioms OrigamiCone.Sequel.Ed`.
-/

namespace OrigamiCone.Sequel

open OrigamiCone

variable {m n : ℕ}

/-- Number of strict local extrema of a function `h : Cell m n → ℤ`. -/
def numExtrema (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (fun v : Cell m n => IsStrictLocalExtremum h v)).card

/-- A **canonical** height-function representative of an OFG shift class:
`h` is a height function and vanishes at every origin cell
`⟨⟨0, _⟩, ⟨0, _⟩⟩`.  When `m, n ≥ 1` there is a unique such origin cell
and `IsCanonicalHeight` picks the unique canonical representative in
each shift class; when the grid is empty, the condition is vacuous. -/
def IsCanonicalHeight (h : Cell m n → ℤ) : Prop :=
  IsHeight h ∧ ∀ p : Cell m n, p.1.val = 0 → p.2.val = 0 → h p = 0

/-- The set of canonical height-function representatives on `Cell m n`. -/
def CanonicalHeights (m n : ℕ) : Set (Cell m n → ℤ) :=
  { h | IsCanonicalHeight h }

/-- Formal `E_d(m, n)`: number of canonical height-function
representatives on `M_{m,n}` with exactly `d` strict local extrema.

By paper's `Lemma 2.1` (Lean: `ofgDegree_eq_extrema`, for `mn ≥ 3`) and
the (deferred) bijection between OFG shift classes and canonical
representatives, this equals the paper's `E_d(m, n)` — the number of
degree-`d` vertices of the origami flip graph on `M_{m,n}`. -/
noncomputable def Ed (d m n : ℕ) : ℕ :=
  { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }.ncard

/-- Unfolding lemma: `Ed d m n` is the `Set.ncard` of the extrema-count
fibre of the canonical-heights set. -/
theorem Ed_def (d m n : ℕ) :
    Ed d m n
      = { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }.ncard :=
  rfl

/-- Partition of `CanonicalHeights m n` by extremum count: every
canonical height lies in the fibre for its own `numExtrema`. -/
theorem canonical_heights_partition_by_extrema :
    CanonicalHeights m n
      = ⋃ d, { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } := by
  ext h
  simp only [CanonicalHeights, Set.mem_iUnion, Set.mem_setOf_eq]
  refine ⟨fun hc => ⟨numExtrema h, hc, rfl⟩, ?_⟩
  rintro ⟨_, hc, _⟩
  exact hc

/-- `Ed` fibres for distinct extremum counts are disjoint. -/
theorem Ed_fibres_disjoint (d₁ d₂ : ℕ) (hd : d₁ ≠ d₂) :
    Disjoint
      ({ h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d₁ })
      ({ h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d₂ }) := by
  rw [Set.disjoint_iff_forall_ne]
  rintro h ⟨_, hd₁⟩ h' ⟨_, hd₂⟩ rfl
  exact hd (hd₁.symm.trans hd₂)

/-- **Finiteness of `CanonicalHeights m n`** for nonempty grid.

Every canonical height `h : Cell m n → ℤ` is bounded pointwise by
`|h p| ≤ m + n - 2`: from `h(0,0) = 0` (canonicity) and the one-Lipschitz
property `|h(0,0) - h p| ≤ gdist (0,0) p` (`height_lipschitz` in
`Basic.lean`), combined with `gdist (0,0) p ≤ (m-1) + (n-1) = m + n - 2`.

Hence `CanonicalHeights m n` embeds into the finite pi-set of functions
`Cell m n → Set.Icc (-B) B` with `B = m + n - 2`, and `Set.Finite.pi'`
closes it.

This is the top downstream obligation flagged by the module docstring —
without finiteness, `Set.ncard` collapses to 0 and `Ed d m n = 0` on
every input. -/
theorem CanonicalHeights_finite (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (CanonicalHeights m n).Finite := by
  set B : ℕ := m + n - 2 with hB_def
  -- Every value of a canonical height sits in [-(B:ℤ), B].
  have h_bounded_finite :
      {h : Cell m n → ℤ | ∀ p, h p ∈ Set.Icc (-(B : ℤ)) (B : ℤ)}.Finite :=
    Set.Finite.pi' (fun _ => Set.finite_Icc _ _)
  refine h_bounded_finite.subset ?_
  unfold CanonicalHeights IsCanonicalHeight
  rintro h ⟨hh_height, hh_zero⟩ p
  -- Origin cell (uses `hm`, `hn` for nonemptiness).
  let origin : Cell m n := (⟨0, hm⟩, ⟨0, hn⟩)
  -- `h origin = 0` from canonicity.
  have h_origin_zero : h origin = 0 := hh_zero origin rfl rfl
  -- `gdist origin p ≤ B` (direct via omega after unfolding).
  have hp1 : p.1.val ≤ m - 1 := Nat.le_sub_one_of_lt p.1.isLt
  have hp2 : p.2.val ≤ n - 1 := Nat.le_sub_one_of_lt p.2.isLt
  have h_gdist_bound : gdist origin p ≤ (B : ℤ) := by
    show (((((⟨0, hm⟩ : Fin m)).val : ℤ) - (p.1.val : ℤ)).natAbs
          + (((⟨0, hn⟩ : Fin n).val : ℤ) - (p.2.val : ℤ)).natAbs : ℕ) ≤ (B : ℤ)
    have hn1 : ((((⟨0, hm⟩ : Fin m)).val : ℤ) - (p.1.val : ℤ)).natAbs = p.1.val := by
      simp
    have hn2 : ((((⟨0, hn⟩ : Fin n)).val : ℤ) - (p.2.val : ℤ)).natAbs = p.2.val := by
      simp
    rw [hn1, hn2]
    push_cast
    omega
  -- Apply `height_lipschitz` at `k = (gdist origin p).toNat`.
  have h_gd_nonneg : (0 : ℤ) ≤ gdist origin p := gdist_nonneg _ _
  have h_toNat : ((gdist origin p).toNat : ℤ) = gdist origin p :=
    Int.toNat_of_nonneg h_gd_nonneg
  have h_lip : |h origin - h p| ≤ gdist origin p :=
    height_lipschitz hh_height (gdist origin p).toNat origin p (by rw [h_toNat])
  rw [h_origin_zero, zero_sub, abs_neg] at h_lip
  -- Conclude `h p ∈ [-(B), B]`.
  rw [Set.mem_Icc]
  have h_abs : |h p| ≤ (B : ℤ) := le_trans h_lip h_gdist_bound
  exact abs_le.mp h_abs

/-- **Finiteness of the `Ed`-fibres for nonempty grid.**  The
degree-`d` fibre inside `CanonicalHeights m n` is finite: as a subset
of `CanonicalHeights m n` (itself finite by `CanonicalHeights_finite`),
it inherits finiteness. -/
theorem Ed_fibre_finite (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }.Finite := by
  refine (CanonicalHeights_finite hm hn).subset ?_
  rintro h ⟨hc, _⟩
  exact hc

/-! ## Counting API bridges

Three trivial bridges convert `Ed` from a `Set.ncard` expression into
the equivalent `Nat.card` / `Finset.card` forms, so downstream count
identities (partition sums, monotonicity in `d`) can be stated in
whichever counting-flavour is most convenient. -/

/-- **`Ed` as `Nat.card` on the subtype.**  `Set.ncard` on a set equals
`Nat.card` on the coerced subtype (definitional identity from
`Nat.card_coe_set_eq`). -/
theorem Ed_eq_natCard (d m n : ℕ) :
    Ed d m n
      = Nat.card
          { h : Cell m n → ℤ // IsCanonicalHeight h ∧ numExtrema h = d } :=
  (Nat.card_coe_set_eq _).symm

/-- **`Ed` as a `Finset.card`** for nonempty grid.  The `.toFinset` of the
finite fibre gives a concrete `Finset` witness whose cardinality equals
`Ed d m n`. -/
theorem Ed_eq_finset_card (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Ed d m n = (Ed_fibre_finite d hm hn).toFinset.card :=
  Set.ncard_eq_toFinset_card _ (Ed_fibre_finite d hm hn)

/-- **`Ed` is bounded by the total canonical-height count** for nonempty
grid.  Every degree-`d` fibre is a subset of `CanonicalHeights m n`,
which is finite by `CanonicalHeights_finite`; `Set.ncard_le_ncard`
closes it. -/
theorem Ed_le_ncard_canonicalHeights (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Ed d m n ≤ (CanonicalHeights m n).ncard :=
  Set.ncard_le_ncard (fun _ ⟨hc, _⟩ => hc) (CanonicalHeights_finite hm hn)

/-- **`numExtrema` upper bound**: the number of strict local extrema is
at most the number of cells `m * n`.  Trivial (a `Finset.filter` is a
subset of the ambient `Finset.univ`). -/
theorem numExtrema_le_mn (h : Cell m n → ℤ) : numExtrema h ≤ m * n := by
  have h_le : numExtrema h ≤ (Finset.univ : Finset (Cell m n)).card :=
    Finset.card_filter_le _ _
  rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin] at h_le
  exact h_le

/-- **`Ed` sum decomposition** for nonempty grid.  Summing `Ed d m n`
over `d ∈ {0, …, m·n}` recovers the total canonical-height count.

This is the partition identity: `CanonicalHeights m n` decomposes as the
disjoint union of its extremum-count fibres, and every fibre has index
`d ≤ m · n` because `numExtrema h ≤ Fintype.card (Cell m n) = m · n`.
Applying `Finset.card_eq_sum_card_fiberwise` to `numExtrema` gives the
sum, and `Ed_eq_finset_card` matches each summand. -/
theorem sum_Ed_eq_ncard_canonicalHeights (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∑ d ∈ Finset.range (m * n + 1), Ed d m n = (CanonicalHeights m n).ncard := by
  rw [Set.ncard_eq_toFinset_card _ (CanonicalHeights_finite hm hn)]
  have h_num_le :
      Set.MapsTo numExtrema
        (↑((CanonicalHeights_finite hm hn).toFinset) : Set (Cell m n → ℤ))
        (↑(Finset.range (m * n + 1)) : Set ℕ) := by
    intro h _
    rw [Finset.mem_coe, Finset.mem_range]
    exact Nat.lt_succ_of_le (numExtrema_le_mn h)
  rw [Finset.card_eq_sum_card_fiberwise h_num_le]
  refine Finset.sum_congr rfl fun d _ => ?_
  rw [Ed_eq_finset_card _ hm hn]
  congr 1
  ext h
  simp only [Set.Finite.mem_toFinset, Finset.mem_filter, Set.mem_setOf_eq]
  tauto

/-! ## Canonicalisation: unique canonical representative per shift class

Every height function has a shift-equivalent canonical representative
(`canonicalise h := fun p => h p - h origin`), and canonical heights
are shift-inequivalent unless equal.  Together these witness a
bijection `{OFG shift class of some height function} ↔ CanonicalHeights m n`,
which is the substrate needed to state `Ed = |OFG height-class fibre of
degree d|` (Item 6c-b main theorem, deferred).

Uses the shift infrastructure from `QuotientDegree.lean`:
`ShiftEq`, `mkV`, `mkV_eq_iff`, `isHeight_add_const`, `strictExtremum_add_const`. -/

open QuotientModel in
/-- **Canonical uniqueness.**  For a nonempty grid, two shift-equivalent
canonical heights are equal (the origin-fixing normalisation makes the
shift constant zero). -/
theorem canonical_unique (hm : 1 ≤ m) (hn : 1 ≤ n)
    {h h' : Cell m n → ℤ} (hc : IsCanonicalHeight h)
    (hc' : IsCanonicalHeight h') (heq : ShiftEq h h') : h = h' := by
  obtain ⟨k, hk⟩ := heq
  -- `h origin = 0` and `h' origin = 0`; but `h' origin = h origin + k = k`.
  have h_orig : h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) = 0 :=
    hc.2 _ rfl rfl
  have h'_orig : h' ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) = 0 :=
    hc'.2 _ rfl rfl
  have hk_zero : k = 0 := by
    have := congrFun hk ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)
    rw [h'_orig, h_orig] at this
    linarith
  subst hk_zero
  funext v
  have := congrFun hk v
  simpa using this.symm

/-- **Canonicalisation function.**  Subtract `h origin` from every value.
This is a shift-equivalent representative in the OFG shift class of `h`,
and it is canonical (vanishes at origin) whenever `h` is a height. -/
def canonicalise (h : Cell m n → ℤ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Cell m n → ℤ :=
  fun p => h p - h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)

/-- **Canonicalisation is shift-equivalent** to the original. -/
theorem canonicalise_shiftEq (h : Cell m n → ℤ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    QuotientModel.ShiftEq h (canonicalise h hm hn) := by
  refine ⟨-h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n), ?_⟩
  funext v
  unfold canonicalise
  ring

/-- **Canonicalisation preserves the height property.**  Adding a
constant does not change edge differences. -/
theorem canonicalise_isHeight (hh : IsHeight h) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    IsHeight (canonicalise h hm hn) := by
  -- `canonicalise h = fun p => h p + (- h origin)`, apply `isHeight_add_const`.
  have : canonicalise h hm hn
      = fun p => h p + (-h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)) := by
    funext p
    unfold canonicalise
    ring
  rw [this]
  exact QuotientModel.isHeight_add_const hh _

/-- **Canonicalisation is canonical.**  If `h` is a height, then
`canonicalise h` is a canonical height (satisfies both height and
origin-vanishing conditions). -/
theorem canonicalise_isCanonicalHeight (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) :
    IsCanonicalHeight (canonicalise h hm hn) := by
  refine ⟨canonicalise_isHeight hh hm hn, ?_⟩
  intro p hp1 hp2
  -- `canonicalise h p = h p - h origin`; for `p = origin` this is `0`.
  have hp_eq : p = ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n) := by
    ext <;> simp [hp1, hp2]
  rw [hp_eq]
  unfold canonicalise
  ring

/-- **Extremum count is invariant under canonicalisation.**  Since
`canonicalise h = fun p => h p + (-h origin)`, `strictExtremum_add_const`
gives shift-invariance, so the extremum sets — and hence their
cardinalities — agree. -/
theorem numExtrema_canonicalise (h : Cell m n → ℤ) (hm : 1 ≤ m) (hn : 1 ≤ n) :
    numExtrema (canonicalise h hm hn) = numExtrema h := by
  unfold numExtrema
  congr 1
  ext v
  simp only [Finset.mem_filter, Finset.mem_univ, true_and]
  have heq : canonicalise h hm hn
      = fun p => h p + (-h ((⟨0, hm⟩, ⟨0, hn⟩) : Cell m n)) := by
    funext p; unfold canonicalise; ring
  rw [heq]
  exact QuotientModel.strictExtremum_add_const

/-! ## `mkV` bijection between canonical heights and OFG height classes

The map `mkV : CanonicalHeights m n → OFGVertex m n` is injective (from
`canonical_unique`) and its image is exactly the OFG shift classes that
contain a height-function representative (surjectivity via
`canonicalise`, which produces a canonical rep in every height class).
Together these witness the bijection needed to state the main Item 6c-b
theorem `Ed d m n = |{height-OFG-vertex of degree d}|` (deferred). -/

/-- The set of OFG shift classes that contain a height-function
representative. -/
def HeightOFGVertex (m n : ℕ) : Set (QuotientModel.OFGVertex m n) :=
  { C | ∃ h : Cell m n → ℤ, IsHeight h ∧ QuotientModel.mkV h = C }

/-- **Injectivity of `mkV` on canonical heights.**  Two canonical heights
mapping to the same OFG shift class must be equal (via
`canonical_unique`). -/
theorem mkV_injOn_canonical (hm : 1 ≤ m) (hn : 1 ≤ n) :
    Set.InjOn QuotientModel.mkV (CanonicalHeights m n) := by
  intro h hh h' hh' heq
  rw [QuotientModel.mkV_eq_iff] at heq
  exact canonical_unique hm hn hh hh' heq

/-- **Image of `mkV` on canonical heights.**  The image is exactly the
OFG shift classes with a height-function representative
(`HeightOFGVertex m n`).

The forward inclusion is direct (a canonical `h` is a height).  The
reverse uses `canonicalise` to produce a canonical representative in
every height OFG class, using `canonicalise_shiftEq` +
`mkV_eq_iff` to conclude `mkV (canonicalise h) = mkV h`. -/
theorem mkV_image_canonical (hm : 1 ≤ m) (hn : 1 ≤ n) :
    QuotientModel.mkV '' (CanonicalHeights m n) = HeightOFGVertex m n := by
  ext C
  constructor
  · rintro ⟨h, hh, rfl⟩
    exact ⟨h, hh.1, rfl⟩
  · rintro ⟨h, hh, rfl⟩
    refine ⟨canonicalise h hm hn, canonicalise_isCanonicalHeight hh hm hn, ?_⟩
    -- `mkV (canonicalise h) = mkV h` from `canonicalise_shiftEq`.
    exact QuotientModel.mkV_eq_iff.mpr (canonicalise_shiftEq h hm hn).symm

/-- **Cardinality bridge.**  `CanonicalHeights m n` and `HeightOFGVertex m n`
have equal `ncard`s: `mkV` is an `InjOn` on the former with image the
latter, so `Set.ncard_image_of_injOn` closes it. -/
theorem canonicalHeights_ncard_eq_heightOFGVertex (hm : 1 ≤ m) (hn : 1 ≤ n) :
    (CanonicalHeights m n).ncard = (HeightOFGVertex m n).ncard := by
  rw [← mkV_image_canonical hm hn]
  exact (Set.ncard_image_of_injOn (mkV_injOn_canonical hm hn)).symm

/-! ## Stratified per-degree bijection: `Ed = |HeightOFGVertex-degree-d|`

The paper's `E_d(m, n)` is the number of degree-`d` vertices in the
origami flip graph `OFG(M_{m,n})`.  The final step in identifying
formal `Ed d m n` with the paper's count is a **stratified**
cardinality bijection: at each fixed `d`, the degree-`d` canonical
heights biject with the degree-`d` height-OFG vertices.  This uses:

* `mkV_injOn_canonical` — injectivity of `mkV` on canonical heights;
* `numExtrema_canonicalise` — extremum count invariant under shift;
* `ofgDegree_eq_extrema` (paper's `Lemma 2.1`, kernel-checked for
  `mn ≥ 3` in `QuotientDegree.lean`) — `OFGDegree (mkV h) = numExtrema h`.

The bijection is `mkV` restricted to the extremum-count fibre; the
image is exactly the OFG-degree fibre inside `HeightOFGVertex`. -/

/-- **Image of `mkV` on the `d`-extrema fibre**.  For `mn ≥ 3`, the
image of `mkV` on `{h : canonical | numExtrema h = d}` is exactly
`{C : HeightOFGVertex | OFGDegree C = d}`.

Requires `mn ≥ 3` so `ofgDegree_eq_extrema` applies.  This is the
per-degree strengthening of `mkV_image_canonical`. -/
theorem mkV_image_fibre (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 3 ≤ m * n) :
    QuotientModel.mkV ''
        { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }
      = { C : QuotientModel.OFGVertex m n |
          C ∈ HeightOFGVertex m n ∧ QuotientModel.OFGDegree C = d } := by
  ext C
  constructor
  · rintro ⟨h, ⟨hc, hd⟩, rfl⟩
    refine ⟨⟨h, hc.1, rfl⟩, ?_⟩
    -- `OFGDegree (mkV h) = numExtrema h` for height `h`, `mn ≥ 3`.
    rw [QuotientModel.ofgDegree_eq_extrema hc.1 hmn]
    exact hd
  · rintro ⟨⟨h_orig, hh_orig, hmkv⟩, hOFGd⟩
    refine ⟨canonicalise h_orig hm hn,
      ⟨canonicalise_isCanonicalHeight hh_orig hm hn, ?_⟩, ?_⟩
    · -- `numExtrema (canonicalise h_orig) = numExtrema h_orig = OFGDegree C = d`.
      rw [numExtrema_canonicalise h_orig hm hn]
      have h1 : QuotientModel.OFGDegree (QuotientModel.mkV h_orig)
                  = (Finset.univ.filter (IsStrictLocalExtremum h_orig)).card :=
        QuotientModel.ofgDegree_eq_extrema hh_orig hmn
      show (Finset.univ.filter (IsStrictLocalExtremum h_orig)).card = d
      rw [← h1, hmkv]
      exact hOFGd
    · -- `mkV (canonicalise h_orig) = mkV h_orig = C`.
      rw [← hmkv]
      exact QuotientModel.mkV_eq_iff.mpr (canonicalise_shiftEq h_orig hm hn).symm

/-- **`Ed` faithfulness to the paper's `E_d`** (Item 6c-b main theorem).

For `min(m, n) ≥ 1` and `mn ≥ 3`, `Ed d m n` equals the number of
degree-`d` height-OFG vertices, i.e., the paper's `E_d(m, n)` — the
number of degree-`d` vertices of the origami flip graph on `M_{m,n}`.

The paper's OFG vertices ARE height classes (by the Ginepro–Hull
colouring bijection), so `HeightOFGVertex m n` is the paper's actual
vertex set inside the Lean `OFGVertex m n = Quotient (shiftSetoid m n)`
model.  Faithfulness is a `Set.InjOn`-image cardinality equality via
`mkV_image_fibre` and `Set.ncard_image_of_injOn`. -/
theorem Ed_eq_ncard_heightOFG_degree
    (d : ℕ) (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 3 ≤ m * n) :
    Ed d m n = { C : QuotientModel.OFGVertex m n |
                 C ∈ HeightOFGVertex m n ∧ QuotientModel.OFGDegree C = d }.ncard := by
  unfold Ed
  have inj : Set.InjOn QuotientModel.mkV
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } :=
    (mkV_injOn_canonical hm hn).mono (fun _ ⟨hc, _⟩ => hc)
  rw [← mkV_image_fibre d hm hn hmn]
  exact (Set.ncard_image_of_injOn inj).symm

/-! ## Symmetry: `Ed d m n = Ed d n m`

The paper's `E_d` is symmetric in `(m, n)` (the origami-flip graph on
`M_{m,n}` and `M_{n,m}` are isomorphic via grid transposition).  This
symmetry is one of the three hypotheses `SequelPolyAssembly.thm_poly_abstract`
takes (alongside per-axis polynomiality and its column dual).

Proof strategy: define `swapH : (Cell m n → ℤ) → (Cell n m → ℤ)` by
composition with `Prod.swap`, show it preserves height, canonicality,
and extremum count, and use `Set.ncard_image_of_injOn`. -/

/-- Transpose a function `Cell m n → ℤ` to `Cell n m → ℤ` by swapping
coordinates: `swapH h q := h q.swap`. -/
def swapH (h : Cell m n → ℤ) : Cell n m → ℤ := fun q => h q.swap

/-- Adjacency in `Cell n m` is preserved by swapping to `Cell m n`. -/
theorem adj_swap {p q : Cell n m} : adj p q ↔ adj p.swap q.swap := by
  unfold adj gdist
  simp only [Prod.fst_swap, Prod.snd_swap]
  constructor
  · intro h; omega
  · intro h; omega

/-- `swapH` preserves the height property. -/
theorem swapH_isHeight {h : Cell m n → ℤ} (hh : IsHeight h) :
    IsHeight (swapH h) := by
  intro p q hpq
  show |h p.swap - h q.swap| = 1
  exact hh p.swap q.swap (adj_swap.mp hpq)

/-- `swapH` preserves the canonical-height property. -/
theorem swapH_isCanonical {h : Cell m n → ℤ} (hc : IsCanonicalHeight h) :
    IsCanonicalHeight (swapH h) := by
  refine ⟨swapH_isHeight hc.1, ?_⟩
  intro p hp1 hp2
  show h p.swap = 0
  exact hc.2 p.swap hp2 hp1

/-- Strict local max is preserved under the swap bijection. -/
theorem swapH_strictMax_iff {h : Cell m n → ℤ} (v : Cell n m) :
    IsStrictLocalMax (swapH h) v ↔ IsStrictLocalMax h v.swap := by
  constructor
  · intro hmax w hw
    have hu : adj v w.swap := by
      rw [adj_swap]; simpa using hw
    simpa [swapH, Prod.swap_swap] using hmax w.swap hu
  · intro hmax u hu
    show h u.swap = h v.swap - 1
    exact hmax u.swap (adj_swap.mp hu)

/-- Strict local min is preserved under the swap bijection. -/
theorem swapH_strictMin_iff {h : Cell m n → ℤ} (v : Cell n m) :
    IsStrictLocalMin (swapH h) v ↔ IsStrictLocalMin h v.swap := by
  constructor
  · intro hmin w hw
    have hu : adj v w.swap := by
      rw [adj_swap]; simpa using hw
    simpa [swapH, Prod.swap_swap] using hmin w.swap hu
  · intro hmin u hu
    show h u.swap = h v.swap + 1
    exact hmin u.swap (adj_swap.mp hu)

/-- Strict local extremum is preserved under the swap bijection. -/
theorem swapH_strictExtremum_iff {h : Cell m n → ℤ} (v : Cell n m) :
    IsStrictLocalExtremum (swapH h) v ↔ IsStrictLocalExtremum h v.swap := by
  unfold IsStrictLocalExtremum
  rw [swapH_strictMax_iff, swapH_strictMin_iff]

/-- `swapH` preserves the extremum count.  The extrema of `swapH h`
(in `Cell n m`) biject with the extrema of `h` (in `Cell m n`) via
`Prod.swap`. -/
theorem swapH_numExtrema (h : Cell m n → ℤ) :
    numExtrema (swapH h) = numExtrema h := by
  unfold numExtrema
  have h_inj : Function.Injective (Prod.swap : Cell n m → Cell m n) :=
    Prod.swap_injective
  rw [← Finset.card_image_of_injective _ h_inj]
  congr 1
  ext v
  simp only [Finset.mem_image, Finset.mem_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨w, hw, rfl⟩
    exact (swapH_strictExtremum_iff w).mp hw
  · intro hv
    refine ⟨v.swap, ?_, by simp⟩
    rw [swapH_strictExtremum_iff]
    simpa using hv

/-- **`Ed` symmetry**: `Ed d m n = Ed d n m` for `m, n ≥ 1`.

The map `swapH` bijects the `d`-extrema fibre of `CanonicalHeights m n`
with that of `CanonicalHeights n m`, via `Set.InjOn` + image equality. -/
theorem Ed_symm (d : ℕ) (_hm : 1 ≤ m) (_hn : 1 ≤ n) :
    Ed d m n = Ed d n m := by
  unfold Ed
  -- Both fibres have the same ncard via the swap bijection.
  have swapH_inj : Function.Injective (@swapH m n) := by
    intro h h' heq
    funext p
    have := congrFun heq p.swap
    simpa [swapH] using this
  have inj : Set.InjOn (@swapH m n)
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } :=
    swapH_inj.injOn
  have himg : swapH ''
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d }
      = { h : Cell n m → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } := by
    ext h'
    constructor
    · rintro ⟨h, ⟨hc, hd⟩, rfl⟩
      exact ⟨swapH_isCanonical hc, (swapH_numExtrema h).trans hd⟩
    · rintro ⟨hc', hd'⟩
      -- Witness `swapH h' : Cell m n → ℤ` (the reverse swap).
      refine ⟨swapH h', ⟨swapH_isCanonical hc', ?_⟩, ?_⟩
      · exact (swapH_numExtrema h').trans hd'
      · funext p
        simp [swapH]
  rw [← himg, Set.ncard_image_of_injOn inj]

/-! ## Boundary corollary: `Ed d m n = 0` for `d < 2` on nonempty grid

Every height function on a grid with `mn ≥ 2` has at least two strict
local extrema — one at its argmax cell (strict local max by the height
property) and one at its argmin cell (dual), which must be distinct by
`max_min_excl` (no cell can be simultaneously a strict local max and a
strict local min when a neighbour exists).  Hence `numExtrema h ≥ 2`
unconditionally on `mn ≥ 2`, so `Ed 0 m n = Ed 1 m n = 0`.

This is a paper-attested boundary claim (the paper's `E_d(m, n) = 0`
for the small-`d` degenerate cases). -/

/-- Every height function on a nonempty grid has a strict local maximum
at its argmax cell.  Uses `Finset.exists_max_image` + the height
property (adjacent cells differ by `±1`). -/
theorem exists_strictLocalMax {h : Cell m n → ℤ} (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∃ v : Cell m n, IsStrictLocalMax h v := by
  have h_ne : (Finset.univ : Finset (Cell m n)).Nonempty :=
    ⟨(⟨0, hm⟩, ⟨0, hn⟩), Finset.mem_univ _⟩
  obtain ⟨v, _, hv⟩ := Finset.exists_max_image Finset.univ h h_ne
  refine ⟨v, ?_⟩
  intro u hu
  have h_le : h u ≤ h v := hv u (Finset.mem_univ u)
  have h_diff : |h v - h u| = 1 := hh v u hu
  have h_nonneg : (0 : ℤ) ≤ h v - h u := by linarith
  have : h v - h u = 1 := by rw [abs_of_nonneg h_nonneg] at h_diff; exact h_diff
  linarith

/-- Dual: every height function on a nonempty grid has a strict local
minimum at its argmin cell. -/
theorem exists_strictLocalMin {h : Cell m n → ℤ} (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) :
    ∃ v : Cell m n, IsStrictLocalMin h v := by
  have h_ne : (Finset.univ : Finset (Cell m n)).Nonempty :=
    ⟨(⟨0, hm⟩, ⟨0, hn⟩), Finset.mem_univ _⟩
  obtain ⟨v, _, hv⟩ := Finset.exists_min_image Finset.univ h h_ne
  refine ⟨v, ?_⟩
  intro u hu
  have h_ge : h v ≤ h u := hv u (Finset.mem_univ u)
  have h_diff : |h v - h u| = 1 := hh v u hu
  have h_nonpos : h v - h u ≤ 0 := by linarith
  have : h v - h u = -1 := by
    rw [abs_of_nonpos h_nonpos] at h_diff; linarith
  linarith

/-- **Every height function on `mn ≥ 2` has at least two strict local
extrema.**  The argmax and argmin cells are both strict local extrema
(via `exists_strictLocalMax` / `_Min`), and they are distinct by
`max_min_excl` (no cell can be simultaneously a strict local max and
min when a neighbour exists). -/
theorem numExtrema_ge_two {h : Cell m n → ℤ} (hh : IsHeight h)
    (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 2 ≤ m * n) :
    2 ≤ numExtrema h := by
  obtain ⟨v_max, hv_max⟩ := exists_strictLocalMax hh hm hn
  obtain ⟨v_min, hv_min⟩ := exists_strictLocalMin hh hm hn
  have h_ne : v_max ≠ v_min :=
    fun heq => max_min_excl hmn hv_max (heq ▸ hv_min)
  unfold numExtrema
  have hsub : ({v_max, v_min} : Finset (Cell m n)) ⊆
      Finset.univ.filter (IsStrictLocalExtremum h) := by
    intro x hx
    rw [Finset.mem_insert, Finset.mem_singleton] at hx
    refine Finset.mem_filter.mpr ⟨Finset.mem_univ _, ?_⟩
    rcases hx with rfl | rfl
    · exact Or.inl hv_max
    · exact Or.inr hv_min
  calc 2 = ({v_max, v_min} : Finset (Cell m n)).card := by
        rw [Finset.card_insert_of_notMem (by simp [h_ne]), Finset.card_singleton]
    _ ≤ _ := Finset.card_le_card hsub

/-- **`Ed d m n = 0` for `d < 2` on `mn ≥ 2`.**  Every canonical height
has `numExtrema ≥ 2`, so the `d`-extrema fibre is empty for `d < 2`. -/
theorem Ed_lt_two_eq_zero {d : ℕ} (hd : d < 2)
    (hm : 1 ≤ m) (hn : 1 ≤ n) (hmn : 2 ≤ m * n) : Ed d m n = 0 := by
  suffices h_empty :
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } = ∅ by
    rw [Ed_def, h_empty, Set.ncard_empty]
  ext h
  simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
  rintro ⟨hc, hd_eq⟩
  have := numExtrema_ge_two hc.1 hm hn hmn
  omega

/-! ## `Ed`-specialisation of `thm_poly_abstract` (Item 6d main skeleton)

The paper's `thm:poly` (Polynomiality on the high region) instantiates
`SequelPolyAssembly.thm_poly_abstract` at `F := fun m n => (Ed d m n : ℚ)`
for fixed `d`.  Three hypotheses are required at each `lo` and `D`:

* `hrow` — per-axis polynomiality along rows (fixed `a`, varying `b`);
* `hcol` — per-axis polynomiality along columns (fixed `b`, varying `a`);
* `hsym` — symmetry `F a b = F b a`.

This module supplies `hsym` unconditionally (via `Ed_symm` + the ℕ→ℚ
cast).  `hrow` and `hcol` remain deferred: they come from the sequel's
transfer-matrix machinery (`SequelRatGF` for the generating function,
`SequelPoles*` for pole location, `SequelFrozen` for the frozen
classification, `SequelQuotient` for peripheral-spectrum period-`1`).
The theorem below **packages the specialisation** — it takes `hrow`
and `hcol` as explicit parameters, so once the deep substrate is
threaded, `thm:poly` at `Ed` follows in one application. -/

/-- **`Ed` polynomiality skeleton**: specialisation of
`SequelPolyAssembly.thm_poly_polynomial_symmetry` at
`F := fun m n => (Ed d m n : ℚ)`.

For `lo ≥ 1` (so `Ed_symm`'s nonempty-grid hypothesis is discharged
along the diagonal `lo ≤ a`), given per-axis polynomiality along rows
and columns, `Ed d` agrees on `{a, b ≥ lo}` with a bivariate polynomial
witness in factored Lagrange form, which is symmetric on all of `ℚ²`.

This is a paper-`thm:poly` corollary MODULO `hrow` and `hcol`, which
package the transfer-matrix polynomiality proof.  The two paper
polynomiality hypotheses are the sole remaining substrate for a full
kernel-checked `thm:poly` at `Ed`.

Note: the paper's exact `thm:poly` uses `D := d − 2` and `lo := d − 1`;
recovering that specialisation requires `hrow`/`hcol` to include the
tight degree bound `natDegree ≤ d − 2` (not just some `D`).  The
degree-pinning obligation is part of `hrow`/`hcol`, not a separate
lemma. -/
theorem Ed_thm_poly_of_perAxis (d D lo : ℕ) (hlo : 1 ≤ lo)
    (hrow : ∀ a, lo ≤ a → ∃ p : Polynomial ℚ, p.natDegree ≤ D ∧
      ∀ b, lo ≤ b → (Ed d a b : ℚ) = p.eval (b : ℚ))
    (hcol : ∀ b, lo ≤ b → ∃ q : Polynomial ℚ, q.natDegree ≤ D ∧
      ∀ a, lo ≤ a → (Ed d a b : ℚ) = q.eval (a : ℚ)) :
    ∃ (g L : Fin (D + 1) → Polynomial ℚ),
      (∀ i, (g i).natDegree ≤ D) ∧ (∀ i, (L i).natDegree ≤ D) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        (Ed d a b : ℚ) = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b : ℚ,
        (∑ i, (g i).eval b * (L i).eval a) =
          (∑ i, (g i).eval a * (L i).eval b)) := by
  apply thm_poly_polynomial_symmetry (fun m n => (Ed d m n : ℚ)) hrow hcol
  intro a b ha hb
  have h_ed_symm : Ed d a b = Ed d b a :=
    Ed_symm d (le_trans hlo ha) (le_trans hlo hb)
  exact_mod_cast h_ed_symm

/-! ## Fully-closed `thm:poly` instances at `d < 2`

For `d < 2` on `mn ≥ 2`, `Ed d m n = 0` (via `Ed_lt_two_eq_zero`),
so the per-axis polynomial witness is the zero polynomial.  This
supplies `hrow`, `hcol` unconditionally and closes the full paper
`thm:poly` at `d = 0` and `d = 1` — the two degenerate cases the
paper attests as `E_d = 0`. -/

/-- `hrow` at `d < 2` on `lo ≥ 2`: `Ed d a b = 0` (zero polynomial). -/
theorem Ed_hrow_of_lt_two {d : ℕ} (hd : d < 2) {lo : ℕ} (hlo : 2 ≤ lo) :
    ∀ a, lo ≤ a → ∃ p : Polynomial ℚ, p.natDegree ≤ 0 ∧
      ∀ b, lo ≤ b → (Ed d a b : ℚ) = p.eval (b : ℚ) := by
  intro a ha
  refine ⟨0, by simp, ?_⟩
  intro b hb
  have h_ab_ge_2 : 2 ≤ a * b := by nlinarith
  have h_zero : Ed d a b = 0 :=
    Ed_lt_two_eq_zero hd (by omega) (by omega) h_ab_ge_2
  simp [h_zero]

/-- `hcol` at `d < 2` on `lo ≥ 2`: dual. -/
theorem Ed_hcol_of_lt_two {d : ℕ} (hd : d < 2) {lo : ℕ} (hlo : 2 ≤ lo) :
    ∀ b, lo ≤ b → ∃ q : Polynomial ℚ, q.natDegree ≤ 0 ∧
      ∀ a, lo ≤ a → (Ed d a b : ℚ) = q.eval (a : ℚ) := by
  intro b hb
  refine ⟨0, by simp, ?_⟩
  intro a ha
  have h_ab_ge_2 : 2 ≤ a * b := by nlinarith
  have h_zero : Ed d a b = 0 :=
    Ed_lt_two_eq_zero hd (by omega) (by omega) h_ab_ge_2
  simp [h_zero]

/-- **Fully-closed `thm:poly` at `d < 2` on `{a, b ≥ 2}`.**  For
`d ∈ {0, 1}` on `a, b ≥ 2`, `Ed d a b = 0` (via `Ed_lt_two_eq_zero`),
so the paper's polynomial witness is `p_d = 0` (degree `0`).  This is
`Ed_thm_poly_of_perAxis` applied at `D = 0` with `hrow`, `hcol`
discharged via `Ed_lt_two_eq_zero`.

**Scope of the instance.**  The `lo ≥ 2` gate is not a stylistic
convenience — it is load-bearing.  At `(m, n) = (1, 1)` the single
cell is *vacuously* both a strict local max and min (no neighbours to
falsify the universally-quantified condition), so `numExtrema = 1`
and `Ed 1 1 1 = 1 ≠ 0`.  The zero polynomial would then fail.  The
paper's `thm:poly` region `{a, b ≥ d - 1}` collapses to the full plane
for `d ∈ {0, 1}`; this Lean instance is therefore strictly weaker than
the paper's full boundary claim, but strictly stronger than the paper's
`mn ≥ 3` bridge (via `numExtrema_ge_two` which holds at `mn ≥ 2`).

An unconditional Item 6d instance in its region — no per-axis
polynomiality substrate needed at the boundary, but the region itself
is `{a, b ≥ 2}`, not the paper's `{a, b ≥ 0}`. -/
theorem Ed_lt_two_thm_poly {d : ℕ} (hd : d < 2) {lo : ℕ} (hlo : 2 ≤ lo) :
    ∃ (g L : Fin 1 → Polynomial ℚ),
      (∀ i, (g i).natDegree ≤ 0) ∧ (∀ i, (L i).natDegree ≤ 0) ∧
      (∀ a b, lo ≤ a → lo ≤ b →
        (Ed d a b : ℚ) = ∑ i, (g i).eval (b : ℚ) * (L i).eval (a : ℚ)) ∧
      (∀ a b : ℚ,
        (∑ i, (g i).eval b * (L i).eval a) =
          (∑ i, (g i).eval a * (L i).eval b)) :=
  Ed_thm_poly_of_perAxis d 0 lo (by omega)
    (Ed_hrow_of_lt_two hd hlo) (Ed_hcol_of_lt_two hd hlo)

/-! ## Upper-bound corollary: `Ed d m n = 0` for `d > mn`

Every height function has `numExtrema h ≤ m * n` (via
`numExtrema_le_mn`), so the `d`-extrema fibre is empty for `d > m * n`.
This is the dual of `Ed_lt_two_eq_zero` at the other boundary, and
completes the trivial-zero coverage of `Ed`:

* `Ed_lt_two_eq_zero`: `Ed d m n = 0` for `d < 2` on `mn ≥ 2`.
* `Ed_gt_mn_eq_zero`: `Ed d m n = 0` for `d > m * n`.

Combined, `Ed d m n = 0` unless `2 ≤ d ≤ m * n` (on `mn ≥ 2`).

Note: unlike `Ed_lt_two_thm_poly`, the upper bound does not directly
yield a fully-closed `thm:poly` instance — a "high region" `{a, b ≥ lo}`
that uniformly satisfies `d > a * b` requires `a * b < d` for all
`a, b ≥ lo`, which is unsatisfiable when `b` can grow.  The upper
bound is a per-cell zero-check, not a per-row polynomial witness. -/

/-- **`Ed d m n = 0` for `d > m * n`.**  No canonical height has
extremum count exceeding the total cell count. -/
theorem Ed_gt_mn_eq_zero {d m n : ℕ} (hd : m * n < d) :
    Ed d m n = 0 := by
  suffices h_empty :
      { h : Cell m n → ℤ | IsCanonicalHeight h ∧ numExtrema h = d } = ∅ by
    rw [Ed_def, h_empty, Set.ncard_empty]
  ext h
  simp only [Set.mem_setOf_eq, Set.mem_empty_iff_false, iff_false]
  rintro ⟨_, hd_eq⟩
  have := numExtrema_le_mn h
  omega

end OrigamiCone.Sequel

import MiuraCounting.ConcreteSeparableProducts
import OrigamiCone.SequelSepDim

namespace OrigamiCone.Sequel.SeparableRunBound

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SignedHeightCounts SingleSideFamilies SeparableProducts

theorem signed_count_one_unique {rows width : Nat} (height : Cell rows width -> Int)
    (kind : SignedExtrema.Kind) (count : SignedExtrema.count height kind = 1) :
    ∃ apex : Cell rows width, SignedExtrema.HasKind height kind apex ∧
      ∀ point, SignedExtrema.HasKind height kind point -> point = apex := by
  classical
  let sites := Finset.univ.filter (SignedExtrema.HasKind height kind)
  let equivalence : SignedExtrema.Sites height kind ≃ sites :=
    Equiv.subtypeEquivRight (by intro point; simp [sites])
  have cardinality : Nat.card (SignedExtrema.Sites height kind) = sites.card :=
    (Nat.card_congr equivalence).trans (by rw [Nat.card_eq_fintype_card, Fintype.card_coe])
  have one : sites.card = 1 := cardinality.symm.trans count
  obtain ⟨apex, singleton⟩ := Finset.card_eq_one.mp one
  have member : apex ∈ sites := by rw [singleton]; exact Finset.mem_singleton_self _
  refine ⟨apex, (Finset.mem_filter.mp member).2, ?_⟩
  intro point property
  have belongs : point ∈ sites := Finset.mem_filter.mpr ⟨Finset.mem_univ _, property⟩
  rw [singleton] at belongs
  exact Finset.mem_singleton.mp belongs

theorem unique_minimum_maximum_boundary {length : Nat} (walk : CanonicalWalk length)
    (lengthValid : 2 ≤ length) (apex : Cell 1 length)
    (minimum : IsStrictLocalMin walk.val apex)
    (unique : ∀ point, IsStrictLocalMin walk.val point -> point = apex)
    (point : Cell 1 length) (maximum : IsStrictLocalMax walk.val point) :
    point.2.val = 0 ∨ point.2.val + 1 = length := by
  have apexRow : apex.1.val = 0 := by have := apex.1.isLt; omega
  have pointRow : point.1.val = 0 := by have := point.1.isLt; omega
  by_cases before : point.2.val ≤ apex.2.val
  · let endpoint : Cell 1 length := ((0 : Fin 1), ⟨0, by omega⟩)
    have profile := cone_min walk.property.1 unique endpoint
    have saturated : walk.val endpoint - walk.val apex = 1 * gdist apex endpoint := by omega
    have between : gdist apex point + gdist point endpoint = gdist apex endpoint := by
      unfold gdist
      dsimp only [endpoint]
      omega
    rcases OppositeAxisTrees.saturated_geodesic_extremum_endpoints walk.val walk.property.1
        apex endpoint point 1 (Or.inl rfl) saturated between (Or.inl maximum) with sameApex | sameEndpoint
    · have apexMaximum : IsStrictLocalMax walk.val apex := by simpa only [sameApex] using maximum
      exact (max_min_excl (by simpa using lengthValid) apexMaximum minimum).elim
    · left
      exact congrArg (fun cell : Cell 1 length => cell.2.val) sameEndpoint
  · let endpoint : Cell 1 length := ((0 : Fin 1), ⟨length - 1, by omega⟩)
    have profile := cone_min walk.property.1 unique endpoint
    have saturated : walk.val endpoint - walk.val apex = 1 * gdist apex endpoint := by omega
    have between : gdist apex point + gdist point endpoint = gdist apex endpoint := by
      have pointBound := point.2.isLt
      have apexBound := apex.2.isLt
      unfold gdist
      dsimp only [endpoint]
      omega
    rcases OppositeAxisTrees.saturated_geodesic_extremum_endpoints walk.val walk.property.1
        apex endpoint point 1 (Or.inl rfl) saturated between (Or.inl maximum) with sameApex | sameEndpoint
    · have apexMaximum : IsStrictLocalMax walk.val apex := by simpa only [sameApex] using maximum
      exact (max_min_excl (by simpa using lengthValid) apexMaximum minimum).elim
    · right
      have index := congrArg (fun cell : Cell 1 length => cell.2.val) sameEndpoint
      change point.2.val = length - 1 at index
      omega

theorem endpoint_sites_card_le_two {length : Nat} (walk : CanonicalWalk length) (kind : SignedExtrema.Kind)
    (endpoints : ∀ point : SignedExtrema.Sites walk.val kind,
      point.val.2.val = 0 ∨ point.val.2.val + 1 = length) :
    SignedExtrema.count walk.val kind ≤ 2 := by
  classical
  let encode : SignedExtrema.Sites walk.val kind -> Bool := fun point => decide (point.val.2.val = 0)
  have injective : Function.Injective encode := by
    intro first second equality
    apply Subtype.ext
    apply Prod.ext
    · exact Subsingleton.elim _ _
    · apply Fin.ext
      by_cases firstZero : first.val.2.val = 0
      · have secondZero : second.val.2.val = 0 := by
          by_contra secondNonzero
          simp [encode, firstZero, secondNonzero] at equality
        omega
      · have secondNonzero : second.val.2.val ≠ 0 := by
          intro secondZero
          simp [encode, firstZero, secondZero] at equality
        have firstLast := (endpoints first).resolve_left firstZero
        have secondLast := (endpoints second).resolve_left secondNonzero
        omega
  have bound := Fintype.card_le_of_injective encode injective
  simpa only [SignedExtrema.count, Nat.card_eq_fintype_card, Fintype.card_bool] using bound

theorem unique_minimum_maximum_count_le_two {length : Nat} (walk : CanonicalWalk length)
    (lengthValid : 2 ≤ length) (oneMinimum : SignedExtrema.count walk.val SignedExtrema.Kind.minimum = 1) :
    SignedExtrema.count walk.val SignedExtrema.Kind.maximum ≤ 2 := by
  obtain ⟨apex, minimum, unique⟩ := signed_count_one_unique walk.val SignedExtrema.Kind.minimum oneMinimum
  apply endpoint_sites_card_le_two walk SignedExtrema.Kind.maximum
  intro point
  exact unique_minimum_maximum_boundary walk lengthValid apex minimum unique point.val point.property

def negateWalk {length : Nat} (walk : CanonicalWalk length) : CanonicalWalk length :=
  ⟨GridNegation.negH walk.val, GridNegation.negH_isHeight walk.val walk.property.1, by
    intro point rowZero columnZero
    have root := walk.property.2 point rowZero columnZero
    change -walk.val point = 0
    rw [root, neg_zero]⟩

theorem negateWalk_signed_counts {length : Nat} (walk : CanonicalWalk length) :
    SignedExtrema.count (negateWalk walk).val SignedExtrema.Kind.minimum =
        SignedExtrema.count walk.val SignedExtrema.Kind.maximum ∧
      SignedExtrema.count (negateWalk walk).val SignedExtrema.Kind.maximum =
        SignedExtrema.count walk.val SignedExtrema.Kind.minimum := by
  let minima : SignedExtrema.Sites (negateWalk walk).val SignedExtrema.Kind.minimum ≃
      SignedExtrema.Sites walk.val SignedExtrema.Kind.maximum :=
    Equiv.subtypeEquivRight (by intro point; exact strictMin_neg_iff)
  let maxima : SignedExtrema.Sites (negateWalk walk).val SignedExtrema.Kind.maximum ≃
      SignedExtrema.Sites walk.val SignedExtrema.Kind.minimum :=
    Equiv.subtypeEquivRight (by intro point; exact strictMax_neg_iff)
  exact ⟨Nat.card_congr minima, Nat.card_congr maxima⟩

theorem unique_maximum_minimum_count_le_two {length : Nat} (walk : CanonicalWalk length)
    (lengthValid : 2 ≤ length) (oneMaximum : SignedExtrema.count walk.val SignedExtrema.Kind.maximum = 1) :
    SignedExtrema.count walk.val SignedExtrema.Kind.minimum ≤ 2 := by
  have counts := negateWalk_signed_counts walk
  have bound := unique_minimum_maximum_count_le_two (negateWalk walk) lengthValid (counts.1.trans oneMaximum)
  rw [counts.2] at bound
  exact bound

theorem one_minimum_one_maximum_boundary {length : Nat} (walk : CanonicalWalk length)
    (lengthValid : 2 ≤ length)
    (oneMinimum : SignedExtrema.count walk.val SignedExtrema.Kind.minimum = 1)
    (oneMaximum : SignedExtrema.count walk.val SignedExtrema.Kind.maximum = 1) :
    ∃ apex : Cell 1 length, IsStrictLocalMin walk.val apex ∧
      (∀ point, IsStrictLocalMin walk.val point -> point = apex) ∧
      (apex.2.val = 0 ∨ apex.2.val + 1 = length) := by
  obtain ⟨apex, minimum, unique⟩ := signed_count_one_unique walk.val SignedExtrema.Kind.minimum oneMinimum
  have counts := negateWalk_signed_counts walk
  obtain ⟨peak, negativeMinimum, negativeUnique⟩ :=
    signed_count_one_unique (negateWalk walk).val SignedExtrema.Kind.minimum (counts.1.trans oneMaximum)
  have negativeMaximum : IsStrictLocalMax (negateWalk walk).val apex := strictMax_neg_iff.mpr minimum
  exact ⟨apex, minimum, unique, unique_minimum_maximum_boundary (negateWalk walk)
    lengthValid peak negativeMinimum negativeUnique apex negativeMaximum⟩

theorem product_singleSide_of_row_counts_one {rows width : Nat}
    (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width) (rowsValid : 2 ≤ rows)
    (oneMinimum : SignedExtrema.count rowWalk.val SignedExtrema.Kind.minimum = 1)
    (oneMaximum : SignedExtrema.count rowWalk.val SignedExtrema.Kind.maximum = 1) :
    IsSingleSide (productHeight rowWalk columnWalk) := by
  obtain ⟨apex, _, unique, boundary⟩ := one_minimum_one_maximum_boundary rowWalk rowsValid oneMinimum oneMaximum
  rcases boundary with top | bottom
  · refine ⟨Side.top, ?_⟩
    intro point minimum
    have rowMinimum := ((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.minimum point).mp minimum).1
    have equality := congrArg (fun cell : Cell 1 rows => cell.2.val) (unique (0, point.1) rowMinimum)
    change point.1.val = 0
    change point.1.val = apex.2.val at equality
    omega
  · refine ⟨Side.bottom, ?_⟩
    intro point minimum
    have rowMinimum := ((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.minimum point).mp minimum).1
    have equality := congrArg (fun cell : Cell 1 rows => cell.2.val) (unique (0, point.1) rowMinimum)
    change point.1.val + 1 = rows
    change point.1.val = apex.2.val at equality
    omega

theorem product_singleSide_of_column_counts_one {rows width : Nat}
    (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width) (widthValid : 2 ≤ width)
    (oneMinimum : SignedExtrema.count columnWalk.val SignedExtrema.Kind.minimum = 1)
    (oneMaximum : SignedExtrema.count columnWalk.val SignedExtrema.Kind.maximum = 1) :
    IsSingleSide (productHeight rowWalk columnWalk) := by
  obtain ⟨apex, _, unique, boundary⟩ := one_minimum_one_maximum_boundary columnWalk widthValid oneMinimum oneMaximum
  rcases boundary with left | right
  · refine ⟨Side.left, ?_⟩
    intro point minimum
    have columnMinimum := ((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.minimum point).mp minimum).2
    have equality := congrArg (fun cell : Cell 1 width => cell.2.val) (unique (0, point.2) columnMinimum)
    change point.2.val = 0
    change point.2.val = apex.2.val at equality
    omega
  · refine ⟨Side.right, ?_⟩
    intro point minimum
    have columnMinimum := ((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.minimum point).mp minimum).2
    have equality := congrArg (fun cell : Cell 1 width => cell.2.val) (unique (0, point.2) columnMinimum)
    change point.2.val + 1 = width
    change point.2.val = apex.2.val at equality
    omega

theorem path_has_signed_site {length : Nat} (walk : CanonicalWalk length)
    (lengthPositive : 0 < length) (kind : SignedExtrema.Kind) :
    Nonempty (SignedExtrema.Sites walk.val kind) := by
  haveI : Nonempty (Cell 1 length) := ⟨((0 : Fin 1), ⟨0, lengthPositive⟩)⟩
  cases kind with
  | minimum =>
    obtain ⟨point, least⟩ := Finite.exists_min walk.val
    refine ⟨⟨point, ?_⟩⟩
    intro neighbor near
    have order := least neighbor
    have unit := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp (walk.property.1 point neighbor near)
    omega
  | maximum =>
    obtain ⟨point, greatest⟩ := Finite.exists_max walk.val
    refine ⟨⟨point, ?_⟩⟩
    intro neighbor near
    have order := greatest neighbor
    have unit := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp (walk.property.1 point neighbor near)
    omega

theorem path_signed_count_pos {length : Nat} (walk : CanonicalWalk length)
    (lengthPositive : 0 < length) (kind : SignedExtrema.Kind) :
    0 < SignedExtrema.count walk.val kind := by
  letI : Nonempty (SignedExtrema.Sites walk.val kind) := path_has_signed_site walk lengthPositive kind
  exact Nat.card_pos

theorem product_active_column_iff {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (rowsPositive : 0 < rows) (column : Fin width) :
    activeColumn (productHeight rowWalk columnWalk) column ↔ activeColumn columnWalk.val column := by
  constructor
  · rintro ⟨row, maximum | minimum⟩
    · exact ⟨0, Or.inl (((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.maximum (row, column)).mp maximum).2)⟩
    · exact ⟨0, Or.inr (((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.minimum (row, column)).mp minimum).2)⟩
  · rintro ⟨index, extremum⟩
    have indexZero : index = (0 : Fin 1) := Subsingleton.elim _ _
    subst index
    rcases extremum with maximum | minimum
    · obtain ⟨rowPoint⟩ := path_has_signed_site rowWalk rowsPositive SignedExtrema.Kind.maximum
      have rowMaximum : SignedExtrema.HasKind rowWalk.val SignedExtrema.Kind.maximum (0, rowPoint.val.2) := by
        simpa only [pathPoint_eta] using rowPoint.property
      exact ⟨rowPoint.val.2, Or.inl ((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.maximum _).mpr
        ⟨rowMaximum, maximum⟩)⟩
    · obtain ⟨rowPoint⟩ := path_has_signed_site rowWalk rowsPositive SignedExtrema.Kind.minimum
      have rowMinimum : SignedExtrema.HasKind rowWalk.val SignedExtrema.Kind.minimum (0, rowPoint.val.2) := by
        simpa only [pathPoint_eta] using rowPoint.property
      exact ⟨rowPoint.val.2, Or.inr ((product_kind_iff rowWalk columnWalk SignedExtrema.Kind.minimum _).mpr
        ⟨rowMinimum, minimum⟩)⟩

theorem product_column_runs {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (rowsPositive : 0 < rows) : numFrozenRuns (productHeight rowWalk columnWalk) = numFrozenRuns columnWalk.val := by
  classical
  unfold numFrozenRuns
  congr 1
  ext column
  simp only [Finset.mem_filter, frozenRunStart, product_active_column_iff rowWalk columnWalk rowsPositive]

theorem product_row_runs {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (widthPositive : 0 < width) : numFrozenRuns (swapH (productHeight rowWalk columnWalk)) = numFrozenRuns rowWalk.val := by
  have sameHeight : swapH (productHeight rowWalk columnWalk) = productHeight columnWalk rowWalk := by
    funext point
    change rowWalk.val (0, point.2) + columnWalk.val (0, point.1) =
      columnWalk.val (0, point.1) + rowWalk.val (0, point.2)
    exact add_comm _ _
  rw [sameHeight]
  exact product_column_runs columnWalk rowWalk widthPositive

theorem product_family_dimension_bound {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 5 ≤ numExtrema (productHeight rowWalk columnWalk))
    (notSingleSide : ¬ IsSingleSide (productHeight rowWalk columnWalk)) :
    (SignedExtrema.count rowWalk.val SignedExtrema.Kind.minimum +
      SignedExtrema.count rowWalk.val SignedExtrema.Kind.maximum - 2) +
      (SignedExtrema.count columnWalk.val SignedExtrema.Kind.minimum +
        SignedExtrema.count columnWalk.val SignedExtrema.Kind.maximum - 2) ≤
      numExtrema (productHeight rowWalk columnWalk) - 3 := by
  let rowMin := SignedExtrema.count rowWalk.val SignedExtrema.Kind.minimum
  let rowMax := SignedExtrema.count rowWalk.val SignedExtrema.Kind.maximum
  let columnMin := SignedExtrema.count columnWalk.val SignedExtrema.Kind.minimum
  let columnMax := SignedExtrema.count columnWalk.val SignedExtrema.Kind.maximum
  change (rowMin + rowMax - 2) + (columnMin + columnMax - 2) ≤
    numExtrema (productHeight rowWalk columnWalk) - 3
  have rowMinPositive : 1 ≤ rowMin := Nat.succ_le_of_lt (path_signed_count_pos rowWalk (by omega) SignedExtrema.Kind.minimum)
  have rowMaxPositive : 1 ≤ rowMax := Nat.succ_le_of_lt (path_signed_count_pos rowWalk (by omega) SignedExtrema.Kind.maximum)
  have columnMinPositive : 1 ≤ columnMin := Nat.succ_le_of_lt (path_signed_count_pos columnWalk (by omega) SignedExtrema.Kind.minimum)
  have columnMaxPositive : 1 ≤ columnMax := Nat.succ_le_of_lt (path_signed_count_pos columnWalk (by omega) SignedExtrema.Kind.maximum)
  have counts : numExtrema (productHeight rowWalk columnWalk) = rowMin * columnMin + rowMax * columnMax := by
    rw [numExtrema_eq_signed_counts _ (by nlinarith), product_signed_count, product_signed_count]
  have identity := dim_identity rowMin columnMin rowMax columnMax
    rowMinPositive columnMinPositive rowMaxPositive columnMaxPositive
  rw [← counts] at identity
  by_contra tooLarge
  have minimumSlack : (rowMin - 1) * (columnMin - 1) = 0 := by omega
  have maximumSlack : (rowMax - 1) * (columnMax - 1) = 0 := by omega
  rcases Nat.mul_eq_zero.mp minimumSlack with rowMinZero | columnMinZero
  · have rowMinOne : rowMin = 1 := by omega
    rcases Nat.mul_eq_zero.mp maximumSlack with rowMaxZero | columnMaxZero
    · have rowMaxOne : rowMax = 1 := by omega
      exact notSingleSide (product_singleSide_of_row_counts_one rowWalk columnWalk rowsValid rowMinOne rowMaxOne)
    · have columnMaxOne : columnMax = 1 := by omega
      have rowCap : rowMax ≤ 2 := unique_minimum_maximum_count_le_two rowWalk rowsValid rowMinOne
      have columnCap : columnMin ≤ 2 := unique_maximum_minimum_count_le_two columnWalk widthValid columnMaxOne
      rw [rowMinOne, columnMaxOne] at counts
      simp only [one_mul, mul_one] at counts
      omega
  · have columnMinOne : columnMin = 1 := by omega
    rcases Nat.mul_eq_zero.mp maximumSlack with rowMaxZero | columnMaxZero
    · have rowMaxOne : rowMax = 1 := by omega
      have rowCap : rowMin ≤ 2 := unique_maximum_minimum_count_le_two rowWalk rowsValid rowMaxOne
      have columnCap : columnMax ≤ 2 := unique_minimum_maximum_count_le_two columnWalk widthValid columnMinOne
      rw [columnMinOne, rowMaxOne] at counts
      simp only [one_mul, mul_one] at counts
      omega
    · have columnMaxOne : columnMax = 1 := by omega
      exact notSingleSide (product_singleSide_of_column_counts_one rowWalk columnWalk widthValid columnMinOne columnMaxOne)

theorem product_nonsingleSide_run_bound {rows width : Nat} (rowWalk : CanonicalWalk rows) (columnWalk : CanonicalWalk width)
    (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 5 ≤ numExtrema (productHeight rowWalk columnWalk))
    (notSingleSide : ¬ IsSingleSide (productHeight rowWalk columnWalk)) :
    numFrozenRuns (swapH (productHeight rowWalk columnWalk)) +
      numFrozenRuns (productHeight rowWalk columnWalk) + 1 ≤ numExtrema (productHeight rowWalk columnWalk) := by
  have rowRuns := numFrozenRuns_lt_numExtrema rowWalk.val (by omega)
    (firstColumn_active rowWalk.val rowWalk.property.1 (by omega : 0 < 1) rowsValid)
    (lastColumn_active rowWalk.val rowWalk.property.1 (by omega : 0 < 1) rowsValid)
  have columnRuns := numFrozenRuns_lt_numExtrema columnWalk.val (by omega)
    (firstColumn_active columnWalk.val columnWalk.property.1 (by omega : 0 < 1) widthValid)
    (lastColumn_active columnWalk.val columnWalk.property.1 (by omega : 0 < 1) widthValid)
  have rowTotal := numExtrema_eq_signed_counts rowWalk.val (by simpa using rowsValid)
  have columnTotal := numExtrema_eq_signed_counts columnWalk.val (by simpa using widthValid)
  rw [rowTotal] at rowRuns
  rw [columnTotal] at columnRuns
  have rowMinimumPositive := path_signed_count_pos rowWalk (by omega) SignedExtrema.Kind.minimum
  have rowMaximumPositive := path_signed_count_pos rowWalk (by omega) SignedExtrema.Kind.maximum
  have columnMinimumPositive := path_signed_count_pos columnWalk (by omega) SignedExtrema.Kind.minimum
  have columnMaximumPositive := path_signed_count_pos columnWalk (by omega) SignedExtrema.Kind.maximum
  have dimension := product_family_dimension_bound rowWalk columnWalk rowsValid widthValid degreeLarge notSingleSide
  rw [product_row_runs rowWalk columnWalk (by omega), product_column_runs rowWalk columnWalk (by omega)]
  omega

theorem separable_nonsingleSide_run_bound {rows width : Nat} (height : Cell rows width -> Int)
    (canonical : IsCanonicalHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (separable : AdditivelySeparable height) (degreeLarge : 5 ≤ numExtrema height)
    (notSingleSide : ¬ IsSingleSide height) :
    numFrozenRuns (swapH height) + numFrozenRuns height + 1 ≤ numExtrema height := by
  let equivalence := canonicalSeparableEquiv rows width (by omega) (by omega)
  let source : CanonicalSeparableHeights rows width := ⟨height, canonical, separable⟩
  let walks := equivalence source
  have reconstruction := congrArg (fun value : CanonicalSeparableHeights rows width => value.val)
    (equivalence.symm_apply_apply source)
  change productHeight walks.1 walks.2 = height at reconstruction
  have large : 5 ≤ numExtrema (productHeight walks.1 walks.2) := by
    rw [reconstruction]
    exact degreeLarge
  have outside : ¬ IsSingleSide (productHeight walks.1 walks.2) := by
    rw [reconstruction]
    exact notSingleSide
  have bound := product_nonsingleSide_run_bound walks.1 walks.2 rowsValid widthValid large outside
  simpa only [reconstruction] using bound

theorem nonsingleSide_run_bound {rows width : Nat} (height : Cell rows width -> Int)
    (canonical : IsCanonicalHeight height) (rowsValid : 2 ≤ rows) (widthValid : 2 ≤ width)
    (degreeLarge : 5 ≤ numExtrema height) (notSingleSide : ¬ IsSingleSide height) :
    numFrozenRuns (swapH height) + numFrozenRuns height + 1 ≤ numExtrema height := by
  by_cases separable : AdditivelySeparable height
  · exact separable_nonsingleSide_run_bound height canonical rowsValid widthValid separable degreeLarge notSingleSide
  · exact UniformRunBound.nonseparable_run_bound height canonical.1 rowsValid widthValid degreeLarge separable

#print axioms signed_count_one_unique
#print axioms unique_minimum_maximum_boundary
#print axioms endpoint_sites_card_le_two
#print axioms unique_minimum_maximum_count_le_two
#print axioms negateWalk_signed_counts
#print axioms unique_maximum_minimum_count_le_two
#print axioms one_minimum_one_maximum_boundary
#print axioms product_singleSide_of_row_counts_one
#print axioms product_singleSide_of_column_counts_one
#print axioms path_has_signed_site
#print axioms path_signed_count_pos
#print axioms product_active_column_iff
#print axioms product_column_runs
#print axioms product_row_runs
#print axioms product_family_dimension_bound
#print axioms product_nonsingleSide_run_bound
#print axioms separable_nonsingleSide_run_bound
#print axioms nonsingleSide_run_bound

end OrigamiCone.Sequel.SeparableRunBound

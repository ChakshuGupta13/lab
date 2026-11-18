import MiuraCounting.ConcreteSeparableProducts
import MiuraCounting.ConcretePathWalks

namespace OrigamiCone.Sequel.SignedPathWalks

open OrigamiCone FrozenExpansion JointReduction JointFibers JointTypePartition SplitPolynomials SignedHeightCounts SingleSideFamilies SeparableProducts

def pathStep {junctions : Nat} (walk : CanonicalWalk (junctions + 2)) (index : Fin (junctions + 1)) : Int :=
  walk.val (0, ⟨index.val + 1, by have := index.isLt; omega⟩) -
    walk.val (0, ⟨index.val, by have := index.isLt; omega⟩)

theorem pathStep_unit {junctions : Nat} (walk : CanonicalWalk (junctions + 2)) (index : Fin (junctions + 1)) :
    pathStep walk index = 1 ∨ pathStep walk index = -1 := by
  have near : adj (((0 : Fin 1), ⟨index.val, by have := index.isLt; omega⟩) : Cell 1 (junctions + 2))
      (0, ⟨index.val + 1, by have := index.isLt; omega⟩) := by
    simp only [adj, gdist]
    omega
  have unit := (abs_eq (by norm_num : (0 : Int) ≤ 1)).mp (walk.property.1 _ _ near)
  unfold pathStep
  omega

def pathTurns {junctions : Nat} (walk : CanonicalWalk (junctions + 2)) (junction : Fin junctions) : Bool :=
  decide (pathStep walk ⟨junction.val, by have := junction.isLt; omega⟩ ≠
    pathStep walk ⟨junction.val + 1, by have := junction.isLt; omega⟩)

theorem pathTurns_step {junctions : Nat} (walk : CanonicalWalk (junctions + 2)) (junction : Fin junctions) :
    pathStep walk ⟨junction.val + 1, by have := junction.isLt; omega⟩ =
      if pathTurns walk junction then -pathStep walk ⟨junction.val, by have := junction.isLt; omega⟩
      else pathStep walk ⟨junction.val, by have := junction.isLt; omega⟩ := by
  have firstUnit := pathStep_unit walk (⟨junction.val, by have := junction.isLt; omega⟩ : Fin (junctions + 1))
  have secondUnit := pathStep_unit walk (⟨junction.val + 1, by have := junction.isLt; omega⟩ : Fin (junctions + 1))
  by_cases changed : pathStep walk ⟨junction.val, by have := junction.isLt; omega⟩ ≠
      pathStep walk ⟨junction.val + 1, by have := junction.isLt; omega⟩
  · simp [pathTurns, changed]
    omega
  · simp [pathTurns, changed]
    omega

theorem pathStep_reconstruction {junctions : Nat} (walk : CanonicalWalk (junctions + 2)) :
    ∀ index : Nat, ∀ within : index < junctions + 1,
      pathStep walk ⟨index, within⟩ = pathStep walk 0 * ConcreteWalk.turnSign (pathTurns walk) index := by
  intro index
  induction index with
  | zero =>
    intro within
    simp only [ConcreteWalk.turnSign, mul_one]
    exact congrArg (pathStep walk) (Fin.ext (by simp))
  | succ index inductionHypothesis =>
    intro within
    let junction : Fin junctions := ⟨index, by omega⟩
    have previous := inductionHypothesis (by omega)
    have actual := pathTurns_step walk junction
    have generated := ConcreteWalk.turnSign_step (pathTurns walk) junction
    change pathStep walk ⟨index + 1, within⟩ =
      if pathTurns walk junction then -pathStep walk ⟨index, by omega⟩
      else pathStep walk ⟨index, by omega⟩ at actual
    change ConcreteWalk.turnSign (pathTurns walk) (index + 1) =
      if pathTurns walk junction then -ConcreteWalk.turnSign (pathTurns walk) index
      else ConcreteWalk.turnSign (pathTurns walk) index at generated
    rw [actual, generated]
    split_ifs <;> nlinarith [previous]

theorem walk_reconstruction {junctions : Nat} (walk : CanonicalWalk (junctions + 2))
    (column : Fin (junctions + 2)) :
    walk.val (0, column) = pathStep walk 0 * ConcreteWalk.walkAt (pathTurns walk) column.val := by
  have recover : ∀ index : Nat, ∀ within : index < junctions + 2,
      walk.val (0, ⟨index, within⟩) = pathStep walk 0 * ConcreteWalk.walkAt (pathTurns walk) index := by
    intro index
    induction index with
    | zero =>
      intro within
      have root := walk.property.2 ((0 : Fin 1), ⟨0, within⟩) rfl rfl
      simpa only [ConcreteWalk.walkAt, mul_zero] using root
    | succ index inductionHypothesis =>
      intro within
      have previous := inductionHypothesis (by omega)
      have step := pathStep_reconstruction walk index (by omega)
      change walk.val (0, ⟨index + 1, within⟩) - walk.val (0, ⟨index, by omega⟩) =
        pathStep walk 0 * ConcreteWalk.turnSign (pathTurns walk) index at step
      rw [ConcreteWalk.walkAt_succ]
      nlinarith [previous, step]
  exact recover column.val column.isLt

def signedTurnWalk {junctions : Nat} : Bool -> (Fin junctions -> Bool) -> CanonicalWalk (junctions + 2)
  | true, turns => ⟨ConcreteWalk.turnHeight 1 turns, ConcreteWalk.turnHeight_canonical 1 turns⟩
  | false, turns => ⟨fun point => -ConcreteWalk.turnHeight 1 turns point,
      isHeight_neg (ConcreteWalk.turnHeight_isHeight 1 turns), by
        intro point rowZero columnZero
        have root := (ConcreteWalk.turnHeight_canonical 1 turns).2 point rowZero columnZero
        change -ConcreteWalk.turnHeight 1 turns point = 0
        rw [root, neg_zero]⟩

theorem signedTurnWalk_step {junctions : Nat} (initial : Bool) (turns : Fin junctions -> Bool)
    (index : Fin (junctions + 1)) :
    pathStep (signedTurnWalk initial turns) index =
      (if initial then 1 else -1) * ConcreteWalk.turnSign turns index.val := by
  have step := ConcreteWalk.walkAt_succ turns index.val
  cases initial <;> dsimp only [signedTurnWalk, pathStep, ConcreteWalk.turnHeight] <;>
    norm_num <;> nlinarith [step]

theorem signedTurnWalk_initial {junctions : Nat} (initial : Bool) (turns : Fin junctions -> Bool) :
    pathStep (signedTurnWalk initial turns) 0 = if initial then 1 else -1 := by
  rw [signedTurnWalk_step]
  simp [ConcreteWalk.turnSign]

theorem signedTurnWalk_surjective (junctions : Nat) :
    Function.Surjective (fun choice : Bool × (Fin junctions -> Bool) => signedTurnWalk choice.1 choice.2) := by
  intro walk
  rcases pathStep_unit walk 0 with upward | downward
  · refine ⟨(true, pathTurns walk), ?_⟩
    apply Subtype.ext
    funext point
    have value := walk_reconstruction walk point.2
    rw [upward, one_mul] at value
    rw [← pathPoint_eta point]
    simpa [signedTurnWalk, ConcreteWalk.turnHeight] using value.symm
  · refine ⟨(false, pathTurns walk), ?_⟩
    apply Subtype.ext
    funext point
    have value := walk_reconstruction walk point.2
    rw [downward, neg_one_mul] at value
    rw [← pathPoint_eta point]
    simpa [signedTurnWalk, ConcreteWalk.turnHeight] using value.symm

theorem signedTurnWalk_injective (junctions : Nat) :
    Function.Injective (fun choice : Bool × (Fin junctions -> Bool) => signedTurnWalk choice.1 choice.2) := by
  rintro ⟨firstInitial, firstTurns⟩ ⟨secondInitial, secondTurns⟩ equality
  have signEquality := congrArg (fun walk : CanonicalWalk (junctions + 2) => pathStep walk 0) equality
  change pathStep (signedTurnWalk firstInitial firstTurns) 0 =
    pathStep (signedTurnWalk secondInitial secondTurns) 0 at signEquality
  rw [signedTurnWalk_initial, signedTurnWalk_initial] at signEquality
  have sameInitial : firstInitial = secondInitial := by
    cases firstInitial <;> cases secondInitial
    · rfl
    · norm_num at signEquality
    · norm_num at signEquality
    · rfl
  subst secondInitial
  refine Prod.ext rfl ?_
  apply ConcreteWalk.turnHeight_injective (rows := 1) (by omega)
  cases firstInitial with
  | false =>
    funext point
    change ConcreteWalk.turnHeight 1 firstTurns point = ConcreteWalk.turnHeight 1 secondTurns point
    have value := congrFun (congrArg (fun walk : CanonicalWalk (junctions + 2) => walk.val) equality) point
    change -ConcreteWalk.turnHeight 1 firstTurns point = -ConcreteWalk.turnHeight 1 secondTurns point at value
    omega
  | true => exact congrArg (fun walk : CanonicalWalk (junctions + 2) => walk.val) equality

noncomputable def signedTurnWalkEquiv (junctions : Nat) :
    (Bool × (Fin junctions -> Bool)) ≃ CanonicalWalk (junctions + 2) :=
  Equiv.ofBijective (fun choice => signedTurnWalk choice.1 choice.2)
    ⟨signedTurnWalk_injective junctions, signedTurnWalk_surjective junctions⟩

theorem signedTurnWalk_numExtrema {junctions : Nat} (initial : Bool) (turns : Fin junctions -> Bool) :
    numExtrema (signedTurnWalk initial turns).val =
      (Finset.univ.filter (fun junction => turns junction = true)).card + 2 := by
  cases initial with
  | true => exact ConcreteWalk.turnHeight_numExtrema (by omega : 0 < 1) turns
  | false =>
    change numExtrema (GridNegation.negH (ConcreteWalk.turnHeight 1 turns)) = _
    have same : numExtrema (GridNegation.negH (ConcreteWalk.turnHeight 1 turns)) =
        numExtrema (ConcreteWalk.turnHeight 1 turns) := by
      unfold numExtrema
      congr 1
      ext point
      simp only [Finset.mem_filter, GridNegation.negH_extremum_iff]
    rw [same]
    exact ConcreteWalk.turnHeight_numExtrema (by omega : 0 < 1) turns

abbrev WalksWithExtrema (junctions turnCount : Nat) :=
  {walk : CanonicalWalk (junctions + 2) // numExtrema walk.val = turnCount + 2}

noncomputable def countedTurnEquiv (junctions turnCount : Nat) :
    (Bool × (turnPatterns junctions turnCount)) ≃ WalksWithExtrema junctions turnCount := by
  classical
  let realize : (Bool × (turnPatterns junctions turnCount)) -> WalksWithExtrema junctions turnCount :=
    fun choice => ⟨signedTurnWalk choice.1 choice.2.val, by
      have count : (Finset.univ.filter (fun junction => choice.2.val junction = true)).card = turnCount :=
        (Finset.mem_filter.mp choice.2.property).2
      rw [signedTurnWalk_numExtrema, count]⟩
  apply Equiv.ofBijective realize
  constructor
  · intro first second equality
    have values := congrArg (fun walk : WalksWithExtrema junctions turnCount => walk.val) equality
    change signedTurnWalk first.1 first.2.val = signedTurnWalk second.1 second.2.val at values
    have choices : ((first.1, first.2.val) : Bool × (Fin junctions -> Bool)) = (second.1, second.2.val) :=
      signedTurnWalk_injective junctions values
    exact Prod.ext (congrArg (fun choice : Bool × (Fin junctions -> Bool) => choice.1) choices)
      (Subtype.ext (congrArg (fun choice : Bool × (Fin junctions -> Bool) => choice.2) choices))
  · intro walk
    obtain ⟨⟨initial, turns⟩, same⟩ := signedTurnWalk_surjective junctions walk.val
    have count := signedTurnWalk_numExtrema initial turns
    change signedTurnWalk initial turns = walk.val at same
    rw [same] at count
    have turnCard : (Finset.univ.filter (fun junction => turns junction = true)).card = turnCount := by
      have actual := walk.property
      omega
    have member : turns ∈ turnPatterns junctions turnCount :=
      Finset.mem_filter.mpr ⟨Finset.mem_univ _, turnCard⟩
    exact ⟨(initial, ⟨turns, member⟩), Subtype.ext same⟩

theorem canonicalWalk_extrema_card (junctions turnCount : Nat) :
    Nat.card (WalksWithExtrema junctions turnCount) = 2 * junctions.choose turnCount := by
  have counts := Nat.card_congr (countedTurnEquiv junctions turnCount)
  rw [Nat.card_prod] at counts
  have directions : Nat.card Bool = 2 := by simp [Nat.card_eq_fintype_card]
  have patterns : Nat.card (turnPatterns junctions turnCount) = junctions.choose turnCount := by
    rw [Nat.card_eq_fintype_card, Fintype.card_coe, card_turnPatterns]
  rw [directions, patterns] at counts
  exact counts.symm

#print axioms pathStep_unit
#print axioms pathTurns_step
#print axioms pathStep_reconstruction
#print axioms walk_reconstruction
#print axioms signedTurnWalk_step
#print axioms signedTurnWalk_initial
#print axioms signedTurnWalk_surjective
#print axioms signedTurnWalk_injective
#print axioms signedTurnWalkEquiv
#print axioms signedTurnWalk_numExtrema
#print axioms countedTurnEquiv
#print axioms canonicalWalk_extrema_card

end OrigamiCone.Sequel.SignedPathWalks

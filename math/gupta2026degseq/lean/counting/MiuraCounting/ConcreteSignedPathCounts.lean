import MiuraCounting.ConcreteSignedPathWalks

namespace OrigamiCone.Sequel.SignedPathCounts

open OrigamiCone SeparableProducts SignedPathWalks

def turnKind {junctions : Nat} (turns : Fin junctions -> Bool) (kind : SignedExtrema.Kind)
    (column : Fin (junctions + 2)) : Prop :=
  match kind with
  | SignedExtrema.Kind.minimum => ConcreteWalk.walkMinimum turns column
  | SignedExtrema.Kind.maximum => ConcreteWalk.walkMaximum turns column

instance turnKindDecidable {junctions : Nat} (turns : Fin junctions -> Bool) (kind : SignedExtrema.Kind)
    (column : Fin (junctions + 2)) : Decidable (turnKind turns kind column) := by
  cases kind <;> dsimp [turnKind, ConcreteWalk.walkMinimum, ConcreteWalk.walkMaximum] <;> infer_instance

def turnKindSitesEquiv {junctions : Nat} (turns : Fin junctions -> Bool) (kind : SignedExtrema.Kind) :
    SignedExtrema.Sites (ConcreteWalk.turnHeight 1 turns) kind ≃
      {column : Fin (junctions + 2) // turnKind turns kind column} where
  toFun point := ⟨point.val.2, by
    cases kind with
    | minimum => exact ((ConcreteWalk.turnHeight_minimum turns point.val.1 point.val.2).mp point.property).2
    | maximum => exact ((ConcreteWalk.turnHeight_maximum turns point.val.1 point.val.2).mp point.property).2⟩
  invFun column := ⟨(0, column.val), by
    cases kind with
    | minimum => exact (ConcreteWalk.turnHeight_minimum turns 0 column.val).mpr ⟨rfl, column.property⟩
    | maximum => exact (ConcreteWalk.turnHeight_maximum turns 0 column.val).mpr ⟨rfl, column.property⟩⟩
  left_inv point := Subtype.ext (Prod.ext (Subsingleton.elim _ _) rfl)
  right_inv _ := Subtype.ext rfl

theorem positive_signed_count {junctions : Nat} (turns : Fin junctions -> Bool) (kind : SignedExtrema.Kind) :
    SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) kind =
      (Finset.univ.filter (fun column => turnKind turns kind column)).card := by
  classical
  rw [SignedExtrema.count, Nat.card_congr (turnKindSitesEquiv turns kind),
    Nat.card_eq_fintype_card, Fintype.card_subtype]

def oppositeKind : SignedExtrema.Kind -> SignedExtrema.Kind
  | SignedExtrema.Kind.minimum => SignedExtrema.Kind.maximum
  | SignedExtrema.Kind.maximum => SignedExtrema.Kind.minimum

def negatedKindSitesEquiv {rows width : Nat} (height : Cell rows width -> Int) (kind : SignedExtrema.Kind) :
    SignedExtrema.Sites (fun point => -height point) kind ≃ SignedExtrema.Sites height (oppositeKind kind) :=
  (Equiv.refl (Cell rows width)).subtypeEquiv (by
    intro point
    cases kind with
    | minimum => exact strictMin_neg_iff
    | maximum => exact strictMax_neg_iff)

theorem signedTurnWalk_count {junctions : Nat} (initial : Bool) (turns : Fin junctions -> Bool)
    (kind : SignedExtrema.Kind) :
    SignedExtrema.count (signedTurnWalk initial turns).val kind =
      if initial then SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) kind
      else SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) (oppositeKind kind) := by
  cases initial with
  | true => rfl
  | false => exact Nat.card_congr (negatedKindSitesEquiv (ConcreteWalk.turnHeight 1 turns) kind)

def upIndicator {junctions : Nat} (turns : Fin junctions -> Bool) (index : Nat) : Int :=
  if ConcreteWalk.turnSign turns index = 1 then 1 else 0

def extremumBalance {junctions : Nat} (turns : Fin junctions -> Bool) (column : Fin (junctions + 2)) : Int :=
  (if turnKind turns SignedExtrema.Kind.minimum column then 1 else 0) -
    (if turnKind turns SignedExtrema.Kind.maximum column then 1 else 0)

theorem extremumBalance_first {junctions : Nat} (turns : Fin junctions -> Bool) :
    extremumBalance turns (0 : Fin (junctions + 2)) = 1 := by
  simp [extremumBalance, turnKind, ConcreteWalk.walkMinimum, ConcreteWalk.walkMaximum, ConcreteWalk.turnSign]

theorem extremumBalance_last {junctions : Nat} (turns : Fin junctions -> Bool) :
    extremumBalance turns (Fin.last (junctions + 1)) = 1 - 2 * upIndicator turns junctions := by
  rcases ConcreteWalk.turnSign_unit turns junctions with direction | direction <;>
    simp [extremumBalance, turnKind, ConcreteWalk.walkMinimum, ConcreteWalk.walkMaximum, upIndicator, direction]

theorem extremumBalance_middle {junctions : Nat} (turns : Fin junctions -> Bool) (junction : Fin junctions) :
    extremumBalance turns ⟨junction.val + 1, by have := junction.isLt; omega⟩ =
      upIndicator turns (junction.val + 1) - upIndicator turns junction.val := by
  have rightPresent : junction.val + 1 + 1 < junctions + 2 := by have := junction.isLt; omega
  rcases ConcreteWalk.turnSign_unit turns junction.val with firstDirection | firstDirection <;>
    rcases ConcreteWalk.turnSign_unit turns (junction.val + 1) with secondDirection | secondDirection <;>
    simp [extremumBalance, turnKind, ConcreteWalk.walkMinimum, ConcreteWalk.walkMaximum,
      upIndicator, rightPresent, firstDirection, secondDirection]

theorem extremumBalance_sum {junctions : Nat} (turns : Fin junctions -> Bool) :
    (∑ column : Fin (junctions + 2), extremumBalance turns column) = 1 - upIndicator turns junctions := by
  rw [Fin.sum_univ_succ, extremumBalance_first,
    Fin.sum_univ_castSucc (fun column : Fin (junctions + 1) => extremumBalance turns column.succ)]
  have middle : (∑ junction : Fin junctions, extremumBalance turns junction.castSucc.succ) =
      upIndicator turns junctions - 1 := by
    change (∑ junction : Fin junctions, extremumBalance turns
      ⟨junction.val + 1, by have := junction.isLt; omega⟩) = _
    simp_rw [extremumBalance_middle]
    rw [Fin.sum_univ_eq_sum_range (fun index : Nat =>
      upIndicator turns (index + 1) - upIndicator turns index) junctions, Finset.sum_range_sub]
    simp [upIndicator, ConcreteWalk.turnSign]
  change (1 : Int) + ((∑ junction : Fin junctions, extremumBalance turns junction.castSucc.succ) +
    extremumBalance turns (Fin.last (junctions + 1))) = _
  rw [middle, extremumBalance_last]
  ring

theorem positive_signed_difference {junctions : Nat} (turns : Fin junctions -> Bool) :
    (SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.minimum : Int) -
      (SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.maximum : Int) =
        1 - upIndicator turns junctions := by
  calc
    _ = ∑ column : Fin (junctions + 2), extremumBalance turns column := by
      rw [positive_signed_count, positive_signed_count]
      simp only [Finset.card_filter, Nat.cast_sum, Nat.cast_ite, Nat.cast_one, Nat.cast_zero,
        ← Finset.sum_sub_distrib, extremumBalance]
    _ = _ := extremumBalance_sum turns

theorem positive_signed_balance {junctions : Nat} (turns : Fin junctions -> Bool) :
    SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.maximum ≤
      SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.minimum ∧
    SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.minimum ≤
      SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.maximum + 1 := by
  have difference := positive_signed_difference turns
  by_cases up : ConcreteWalk.turnSign turns junctions = 1
  · simp only [upIndicator, if_pos up] at difference
    omega
  · simp only [upIndicator, if_neg up] at difference
    omega

theorem positive_signed_total {junctions : Nat} (turns : Fin junctions -> Bool) :
    SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.minimum +
      SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.maximum =
        (Finset.univ.filter (fun junction => turns junction = true)).card + 2 := by
  rw [← JointTypePartition.numExtrema_eq_signed_counts _ (by omega)]
  exact ConcreteWalk.turnHeight_numExtrema (by omega : 0 < 1) turns

theorem positive_signed_split_iff {junctions : Nat} (turns : Fin junctions -> Bool) (minima maxima : Nat) :
    (SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.minimum = minima ∧
      SignedExtrema.count (ConcreteWalk.turnHeight 1 turns) SignedExtrema.Kind.maximum = maxima) ↔
    (Finset.univ.filter (fun junction => turns junction = true)).card + 2 = minima + maxima ∧
      maxima ≤ minima ∧ minima ≤ maxima + 1 := by
  have balance := positive_signed_balance turns
  have total := positive_signed_total turns
  constructor
  · rintro ⟨minimumCount, maximumCount⟩
    omega
  · rintro ⟨sum, ordered, close⟩
    constructor <;> omega

abbrev InitialChoices (minima maxima : Nat) :=
  {initial : Bool // if initial then maxima ≤ minima else minima ≤ maxima}

abbrev SignedWalks (junctions minima maxima : Nat) :=
  {walk : CanonicalWalk (junctions + 2) //
    SignedExtrema.count walk.val SignedExtrema.Kind.minimum = minima ∧
      SignedExtrema.count walk.val SignedExtrema.Kind.maximum = maxima}

theorem signedTurnWalk_split_iff {junctions : Nat} (initial : Bool) (turns : Fin junctions -> Bool)
    (minima maxima : Nat) :
    (SignedExtrema.count (signedTurnWalk initial turns).val SignedExtrema.Kind.minimum = minima ∧
      SignedExtrema.count (signedTurnWalk initial turns).val SignedExtrema.Kind.maximum = maxima) ↔
    (Finset.univ.filter (fun junction => turns junction = true)).card + 2 = minima + maxima ∧
      if initial then maxima ≤ minima ∧ minima ≤ maxima + 1 else minima ≤ maxima ∧ maxima ≤ minima + 1 := by
  have balance := positive_signed_balance turns
  have total := positive_signed_total turns
  cases initial with
  | true =>
    exact positive_signed_split_iff turns minima maxima
  | false =>
    simp only [signedTurnWalk_count, Bool.false_eq_true, if_false, oppositeKind]
    constructor
    · rintro ⟨minimumCount, maximumCount⟩
      omega
    · rintro ⟨sum, ordered, close⟩
      constructor <;> omega

theorem initialChoices_card (minima maxima : Nat) :
    Nat.card (InitialChoices minima maxima) = if minima = maxima then 2 else 1 := by
  classical
  by_cases same : minima = maxima
  · subst maxima
    simp [InitialChoices, Nat.card_eq_fintype_card, Fintype.card_subtype]
  · rcases Nat.lt_or_gt_of_ne same with smaller | larger
    · have forward : minima ≤ maxima := by omega
      have notBackward : ¬ maxima ≤ minima := by omega
      simp [InitialChoices, Nat.card_eq_fintype_card, same, forward, notBackward]
    · have backward : maxima ≤ minima := by omega
      have notForward : ¬ minima ≤ maxima := by omega
      simp [InitialChoices, Nat.card_eq_fintype_card, same, backward, notForward]

theorem selected_turns_realize_split (junctions minima maxima : Nat) (minimaPositive : 0 < minima)
    (maximaPositive : 0 < maxima) (close : minima ≤ maxima + 1 ∧ maxima ≤ minima + 1)
    (initial : InitialChoices minima maxima) (turns : turnPatterns junctions (minima + maxima - 2)) :
    SignedExtrema.count (signedTurnWalk initial.val turns.val).val SignedExtrema.Kind.minimum = minima ∧
      SignedExtrema.count (signedTurnWalk initial.val turns.val).val SignedExtrema.Kind.maximum = maxima := by
  apply (signedTurnWalk_split_iff initial.val turns.val minima maxima).mpr
  have turnCount : (Finset.univ.filter (fun junction => turns.val junction = true)).card = minima + maxima - 2 :=
    (Finset.mem_filter.mp turns.property).2
  refine ⟨by omega, ?_⟩
  have selected := initial.property
  cases initialValue : initial.val <;> simp only [initialValue, Bool.false_eq_true, if_false, if_true] at selected ⊢ <;> omega

noncomputable def signedCountedTurnEquiv (junctions minima maxima : Nat) (minimaPositive : 0 < minima)
    (maximaPositive : 0 < maxima) (close : minima ≤ maxima + 1 ∧ maxima ≤ minima + 1) :
    (InitialChoices minima maxima × (turnPatterns junctions (minima + maxima - 2))) ≃ SignedWalks junctions minima maxima := by
  classical
  let realize : (InitialChoices minima maxima × (turnPatterns junctions (minima + maxima - 2))) ->
      SignedWalks junctions minima maxima := fun choice =>
    ⟨signedTurnWalk choice.1.val choice.2.val,
      selected_turns_realize_split junctions minima maxima minimaPositive maximaPositive close choice.1 choice.2⟩
  apply Equiv.ofBijective realize
  constructor
  · intro first second equality
    have values := congrArg (fun walk : SignedWalks junctions minima maxima => walk.val) equality
    change signedTurnWalk first.1.val first.2.val = signedTurnWalk second.1.val second.2.val at values
    have choices : ((first.1.val, first.2.val) : Bool × (Fin junctions -> Bool)) = (second.1.val, second.2.val) :=
      signedTurnWalk_injective junctions values
    exact Prod.ext
      (Subtype.ext (congrArg (fun choice : Bool × (Fin junctions -> Bool) => choice.1) choices))
      (Subtype.ext (congrArg (fun choice : Bool × (Fin junctions -> Bool) => choice.2) choices))
  · intro walk
    obtain ⟨⟨initial, turns⟩, same⟩ := signedTurnWalk_surjective junctions walk.val
    have split := walk.property
    rw [← same] at split
    have conditions := (signedTurnWalk_split_iff initial turns minima maxima).mp split
    have turnCount : (Finset.univ.filter (fun junction => turns junction = true)).card = minima + maxima - 2 := by omega
    have selected : if initial then maxima ≤ minima else minima ≤ maxima := by
      cases initial <;> simp only [Bool.false_eq_true, if_false, if_true] at conditions ⊢ <;> omega
    refine ⟨(⟨initial, selected⟩, ⟨turns, Finset.mem_filter.mpr ⟨Finset.mem_univ _, turnCount⟩⟩), ?_⟩
    exact Subtype.ext same

theorem signedWalks_empty_of_unbalanced (junctions minima maxima : Nat)
    (unbalanced : ¬ (minima ≤ maxima + 1 ∧ maxima ≤ minima + 1)) :
    IsEmpty (SignedWalks junctions minima maxima) := by
  refine ⟨fun walk => ?_⟩
  obtain ⟨⟨initial, turns⟩, same⟩ := signedTurnWalk_surjective junctions walk.val
  have split := walk.property
  rw [← same] at split
  have conditions := (signedTurnWalk_split_iff initial turns minima maxima).mp split
  apply unbalanced
  cases initial <;> simp only [Bool.false_eq_true, if_false, if_true] at conditions <;> omega

theorem canonicalWalk_signed_card (junctions minima maxima : Nat) (minimaPositive : 0 < minima)
    (maximaPositive : 0 < maxima) :
    Nat.card (SignedWalks junctions minima maxima) =
      if minima ≤ maxima + 1 ∧ maxima ≤ minima + 1 then
        (if minima = maxima then 2 else 1) * junctions.choose (minima + maxima - 2)
      else 0 := by
  classical
  by_cases close : minima ≤ maxima + 1 ∧ maxima ≤ minima + 1
  · rw [if_pos close, ← Nat.card_congr
      (signedCountedTurnEquiv junctions minima maxima minimaPositive maximaPositive close),
      Nat.card_prod, initialChoices_card]
    have patterns : Nat.card (turnPatterns junctions (minima + maxima - 2)) = junctions.choose (minima + maxima - 2) := by
      rw [Nat.card_eq_fintype_card, Fintype.card_coe, card_turnPatterns]
    rw [patterns]
  · letI : IsEmpty (SignedWalks junctions minima maxima) := signedWalks_empty_of_unbalanced junctions minima maxima close
    rw [if_neg close, Nat.card_of_isEmpty]

theorem canonical_path_signed_binomial (length minima maxima : Nat) (lengthValid : 2 ≤ length)
    (minimaPositive : 0 < minima) (maximaPositive : 0 < maxima) :
    Nat.card {walk : CanonicalWalk length //
      SignedExtrema.count walk.val SignedExtrema.Kind.minimum = minima ∧
        SignedExtrema.count walk.val SignedExtrema.Kind.maximum = maxima} =
      if |(minima : Int) - (maxima : Int)| ≤ 1 then
        (1 + if minima = maxima then 1 else 0) * (length - 2).choose (minima + maxima - 2)
      else 0 := by
  obtain ⟨junctions, rfl⟩ : ∃ junctions : Nat, length = junctions + 2 := ⟨length - 2, by omega⟩
  change Nat.card (SignedWalks junctions minima maxima) =
    if |(minima : Int) - (maxima : Int)| ≤ 1 then
      (1 + if minima = maxima then 1 else 0) * junctions.choose (minima + maxima - 2) else 0
  have supported : (|(minima : Int) - (maxima : Int)| ≤ 1) ↔ minima ≤ maxima + 1 ∧ maxima ≤ minima + 1 := by
    rw [abs_le]
    omega
  have multiplicity : (if minima = maxima then 2 else 1) = 1 + if minima = maxima then 1 else 0 := by
    by_cases same : minima = maxima <;> simp [same]
  simpa only [supported, multiplicity] using
    canonicalWalk_signed_card junctions minima maxima minimaPositive maximaPositive

#print axioms turnKind
#print axioms turnKindSitesEquiv
#print axioms positive_signed_count
#print axioms oppositeKind
#print axioms negatedKindSitesEquiv
#print axioms signedTurnWalk_count
#print axioms upIndicator
#print axioms extremumBalance
#print axioms extremumBalance_first
#print axioms extremumBalance_last
#print axioms extremumBalance_middle
#print axioms extremumBalance_sum
#print axioms positive_signed_difference
#print axioms positive_signed_balance
#print axioms positive_signed_total
#print axioms positive_signed_split_iff
#print axioms signedTurnWalk_split_iff
#print axioms initialChoices_card
#print axioms selected_turns_realize_split
#print axioms signedCountedTurnEquiv
#print axioms signedWalks_empty_of_unbalanced
#print axioms canonicalWalk_signed_card
#print axioms canonical_path_signed_binomial

end OrigamiCone.Sequel.SignedPathCounts

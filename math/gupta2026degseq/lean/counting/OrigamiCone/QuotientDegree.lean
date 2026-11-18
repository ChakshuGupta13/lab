import OrigamiCone.DegreeExtrema

namespace OrigamiCone
namespace QuotientModel

variable {m n : ℕ}

def ShiftEq (h h' : Cell m n → ℤ) : Prop := ∃ k : ℤ, h' = fun v => h v + k

lemma ShiftEq.rfl' (h : Cell m n → ℤ) : ShiftEq h h := ⟨0, by funext v; ring⟩

lemma ShiftEq.symm {h h' : Cell m n → ℤ} : ShiftEq h h' → ShiftEq h' h := by
  rintro ⟨k, rfl⟩; exact ⟨-k, by funext v; ring⟩

lemma ShiftEq.trans {h h' h'' : Cell m n → ℤ} :
    ShiftEq h h' → ShiftEq h' h'' → ShiftEq h h'' := by
  rintro ⟨k, rfl⟩ ⟨l, rfl⟩; exact ⟨k + l, by funext v; ring⟩

instance shiftSetoid (m n : ℕ) : Setoid (Cell m n → ℤ) where
  r := ShiftEq
  iseqv := ⟨ShiftEq.rfl', ShiftEq.symm, ShiftEq.trans⟩

abbrev OFGVertex (m n : ℕ) := Quotient (shiftSetoid m n)

def mkV (h : Cell m n → ℤ) : OFGVertex m n := ⟦h⟧

lemma mkV_eq_iff {h h' : Cell m n → ℤ} : mkV h = mkV h' ↔ ShiftEq h h' :=
  Quotient.eq

lemma mkV_add_const (h : Cell m n → ℤ) (k : ℤ) :
    mkV (fun v => h v + k) = mkV h :=
  Quotient.sound ⟨-k, by funext v; ring⟩

lemma isHeight_add_const {h : Cell m n → ℤ} (hh : IsHeight h) (k : ℤ) :
    IsHeight (fun v => h v + k) := by
  intro p q hpq
  have h1 := hh p q hpq
  show |(h p + k) - (h q + k)| = 1
  rw [show (h p + k) - (h q + k) = h p - h q by ring]; exact h1

lemma strictMax_add_const {h : Cell m n → ℤ} {k : ℤ} {v : Cell m n} :
    IsStrictLocalMax (fun w => h w + k) v ↔ IsStrictLocalMax h v := by
  unfold IsStrictLocalMax
  constructor
  · intro hmax u hu; have := hmax u hu; simp only at this; omega
  · intro hmax u hu; have := hmax u hu; simp only; omega

lemma strictMin_add_const {h : Cell m n → ℤ} {k : ℤ} {v : Cell m n} :
    IsStrictLocalMin (fun w => h w + k) v ↔ IsStrictLocalMin h v := by
  unfold IsStrictLocalMin
  constructor
  · intro hmin u hu; have := hmin u hu; simp only at this; omega
  · intro hmin u hu; have := hmin u hu; simp only; omega

lemma strictExtremum_add_const {h : Cell m n → ℤ} {k : ℤ} {v : Cell m n} :
    IsStrictLocalExtremum (fun w => h w + k) v ↔ IsStrictLocalExtremum h v := by
  unfold IsStrictLocalExtremum
  rw [strictMax_add_const, strictMin_add_const]

lemma flipMax_add_const {h : Cell m n → ℤ} (k : ℤ) (v w : Cell m n) :
    flipMax (fun u => h u + k) v w = flipMax h v w + k := by
  rcases eq_or_ne w v with hw | hw
  · subst hw; simp only [flipMax_apply_self]; ring
  · rw [flipMax_apply_ne hw, flipMax_apply_ne hw]

lemma flipMin_add_const {h : Cell m n → ℤ} (k : ℤ) (v w : Cell m n) :
    flipMin (fun u => h u + k) v w = flipMin h v w + k := by
  rcases eq_or_ne w v with hw | hw
  · subst hw; simp only [flipMin_apply_self]; ring
  · rw [flipMin_apply_ne hw, flipMin_apply_ne hw]

lemma flipAt_add_const {h : Cell m n → ℤ} (k : ℤ) (v : Cell m n) :
    flipAt (fun u => h u + k) v = fun w => flipAt h v w + k := by
  unfold flipAt
  by_cases hm : IsStrictLocalMax h v
  · rw [if_pos (strictMax_add_const.mpr hm), if_pos hm]
    funext w; exact flipMax_add_const k v w
  · rw [if_neg (fun hc => hm (strictMax_add_const.mp hc)), if_neg hm]
    funext w; exact flipMin_add_const k v w

def IsOFGNeighbor (C D : OFGVertex m n) : Prop :=
  C ≠ D ∧ ∃ g g', mkV g = C ∧ mkV g' = D ∧ OFGAdj g g'

def OFGNeighbors (C : OFGVertex m n) : Set (OFGVertex m n) :=
  {D | IsOFGNeighbor C D}

noncomputable def OFGDegree (C : OFGVertex m n) : ℕ := (OFGNeighbors C).ncard

lemma flip_mkV_ne {h : Cell m n → ℤ} (hmn : 2 ≤ m * n) {v : Cell m n}
    (hv : IsStrictLocalExtremum h v) : mkV (flipAt h v) ≠ mkV h := by
  rw [Ne, mkV_eq_iff]
  rintro ⟨k, hk⟩

  obtain ⟨u, hu⟩ := exists_neighbor hmn v
  have hune : u ≠ v := by rintro rfl; simp [adj, gdist_self] at hu
  have hku : h u = flipAt h v u + k := congrFun hk u
  rw [flipAt_apply_ne hune] at hku
  have hk0 : k = 0 := by omega
  have hkv : h v = flipAt h v v + k := congrFun hk v
  have heq : flipAt h v v = h v := by omega
  exact flipAt_self_ne hmn hv heq

lemma exists_third (hmn : 3 ≤ m * n) (v v' : Cell m n) :
    ∃ w, w ≠ v ∧ w ≠ v' := by
  by_contra hcon
  push_neg at hcon
  have hsub : (Finset.univ : Finset (Cell m n)) ⊆ {v, v'} := by
    intro w _
    rcases eq_or_ne w v with hw | hw
    · simp [hw]
    · simp [hcon w hw]
  have h1 := Finset.card_le_card hsub
  have h2 : ({v, v'} : Finset (Cell m n)).card ≤ 2 :=
    le_trans (Finset.card_insert_le _ _) (by simp)
  have h3 : (Finset.univ : Finset (Cell m n)).card = m * n := by
    rw [Finset.card_univ, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
  omega

lemma flip_mkV_injOn {h : Cell m n → ℤ} (hmn : 3 ≤ m * n) :
    Set.InjOn (fun v => mkV (flipAt h v))
      ↑(Finset.univ.filter (IsStrictLocalExtremum h)) := by
  intro v hv v' _ heq
  have hext : IsStrictLocalExtremum h v := by
    have := Finset.mem_coe.mp hv; exact (Finset.mem_filter.mp this).2
  by_contra hne

  rw [mkV_eq_iff] at heq
  obtain ⟨k, hk⟩ := heq

  obtain ⟨w, hwv, hwv'⟩ := exists_third hmn v v'
  have e1 : flipAt h v w = h w := flipAt_apply_ne hwv
  have e2 : flipAt h v' w = h w := flipAt_apply_ne hwv'
  have hkw : flipAt h v' w = flipAt h v w + k := congrFun hk w
  rw [e1, e2] at hkw
  have hk0 : k = 0 := by omega
  have eqfun : flipAt h v' = flipAt h v := by
    rw [hk]; funext u; rw [hk0]; ring
  have ev1 : flipAt h v' v = h v := flipAt_apply_ne hne
  rw [eqfun] at ev1
  exact flipAt_self_ne (by omega) hext ev1

lemma ofgNeighbors_eq_image {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 3 ≤ m * n) :
    OFGNeighbors (mkV h)
      = (fun v => mkV (flipAt h v)) ''
          ↑(Finset.univ.filter (IsStrictLocalExtremum h)) := by
  have hmn' : 2 ≤ m * n := by omega
  ext D
  simp only [OFGNeighbors, Set.mem_setOf_eq, IsOFGNeighbor, Set.mem_image,
             Finset.coe_filter, Finset.mem_univ, true_and]
  constructor
  · rintro ⟨_, g, g', hg, hg', hadj⟩

    rw [mkV_eq_iff] at hg
    obtain ⟨k, hk⟩ := hg

    have hg_eq : g = fun v => h v + (-k) := by
      funext v; have := congrFun hk v; simp only at this; omega
    have hgH : IsHeight g := by rw [hg_eq]; exact isHeight_add_const hh (-k)
    obtain ⟨w, hwext, hflip⟩ := neighbor_is_flipAt hmn' hgH hadj
    refine ⟨w, ?_, ?_⟩
    ·
      rw [hg_eq] at hwext; exact strictExtremum_add_const.mp hwext
    ·
      rw [← hg', ← hflip, hg_eq, flipAt_add_const, mkV_add_const]
  · rintro ⟨v, hvext, rfl⟩
    exact ⟨(flip_mkV_ne hmn' hvext).symm, h, flipAt h v, rfl, rfl,
           ofgAdj_flipAt hmn' hh hvext⟩

theorem ofgDegree_eq_extrema {h : Cell m n → ℤ} (hh : IsHeight h) (hmn : 3 ≤ m * n) :
    OFGDegree (mkV h)
      = (Finset.univ.filter (IsStrictLocalExtremum h)).card := by
  unfold OFGDegree
  rw [ofgNeighbors_eq_image hh hmn, Set.ncard_image_of_injOn (flip_mkV_injOn hmn),
      Set.ncard_coe_finset]

end QuotientModel
end OrigamiCone

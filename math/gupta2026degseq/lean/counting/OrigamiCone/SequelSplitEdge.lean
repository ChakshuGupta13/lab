import OrigamiCone.SequelEdge
import OrigamiCone.SequelBinom

namespace OrigamiCone.Sequel

inductive Side where
  | top | bottom | left | right
  deriving DecidableEq, Repr

open Side

variable {ι : Type*} (p : ι → ℤ × ℤ) (S : Finset ι)

def AllOnSide (m n : ℤ) (sd : Side) : Prop :=
  match sd with
  | top    => ∀ s ∈ S, (p s).1 = 1
  | bottom => ∀ s ∈ S, (p s).1 = m
  | left   => ∀ s ∈ S, (p s).2 = 1
  | right  => ∀ s ∈ S, (p s).2 = n

private lemma two_mem_of_two_le_card {α : Type*} {S : Finset α} (h : 2 ≤ S.card) :
    ∃ s ∈ S, ∃ t ∈ S, s ≠ t := by
  have := Finset.one_lt_card.mp h
  exact this

theorem AllOnSide_top_bottom_disjoint (m n : ℤ) (hm : 2 ≤ m) (hS : S.Nonempty)
    (htop : AllOnSide p S m n top) (hbot : AllOnSide p S m n bottom) : False := by
  obtain ⟨s, hsS⟩ := hS
  have h1 := htop s hsS
  have h2 := hbot s hsS
  omega

theorem AllOnSide_left_right_disjoint (m n : ℤ) (hn : 2 ≤ n) (hS : S.Nonempty)
    (hlt : AllOnSide p S m n left) (hrt : AllOnSide p S m n right) : False := by
  obtain ⟨s, hsS⟩ := hS
  have h1 := hlt s hsS
  have h2 := hrt s hsS
  omega

theorem AllOnSide_top_left_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (htop : AllOnSide p S m n top) (hlt : AllOnSide p S m n left) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (1, 1) := by
    ext
    · exact htop s hsS
    · exact hlt s hsS
  have hpt : p t = (1, 1) := by
    ext
    · exact htop t htS
    · exact hlt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

theorem AllOnSide_top_right_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (htop : AllOnSide p S m n top) (hrt : AllOnSide p S m n right) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (1, n) := by
    ext
    · exact htop s hsS
    · exact hrt s hsS
  have hpt : p t = (1, n) := by
    ext
    · exact htop t htS
    · exact hrt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

theorem AllOnSide_bottom_left_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (hbot : AllOnSide p S m n bottom) (hlt : AllOnSide p S m n left) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (m, 1) := by
    ext
    · exact hbot s hsS
    · exact hlt s hsS
  have hpt : p t = (m, 1) := by
    ext
    · exact hbot t htS
    · exact hlt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

theorem AllOnSide_bottom_right_disjoint (m n : ℤ) (hcard : 2 ≤ S.card)
    (hinj : Set.InjOn p S)
    (hbot : AllOnSide p S m n bottom) (hrt : AllOnSide p S m n right) : False := by
  obtain ⟨s, hsS, t, htS, hne⟩ := two_mem_of_two_le_card hcard
  have hps : p s = (m, n) := by
    ext
    · exact hbot s hsS
    · exact hrt s hsS
  have hpt : p t = (m, n) := by
    ext
    · exact hbot t htS
    · exact hrt t htS
  exact hne (hinj hsS htS (hps.trans hpt.symm))

theorem AllOnSide_pairwise_disjoint (m n : ℤ) (hm : 2 ≤ m) (hn : 2 ≤ n)
    (hcard : 2 ≤ S.card) (hinj : Set.InjOn p S) (sd sd' : Side) (hne : sd ≠ sd')
    (hsd : AllOnSide p S m n sd) (hsd' : AllOnSide p S m n sd') : False := by
  have hnonempty : S.Nonempty := Finset.card_pos.mp (by omega)

  cases sd <;> cases sd' <;> first
    | exact hne rfl
    | exact AllOnSide_top_bottom_disjoint p S m n hm hnonempty hsd hsd'
    | exact AllOnSide_top_bottom_disjoint p S m n hm hnonempty hsd' hsd
    | exact AllOnSide_left_right_disjoint p S m n hn hnonempty hsd hsd'
    | exact AllOnSide_left_right_disjoint p S m n hn hnonempty hsd' hsd
    | exact AllOnSide_top_left_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_top_left_disjoint p S m n hcard hinj hsd' hsd
    | exact AllOnSide_top_right_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_top_right_disjoint p S m n hcard hinj hsd' hsd
    | exact AllOnSide_bottom_left_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_bottom_left_disjoint p S m n hcard hinj hsd' hsd
    | exact AllOnSide_bottom_right_disjoint p S m n hcard hinj hsd hsd'
    | exact AllOnSide_bottom_right_disjoint p S m n hcard hinj hsd' hsd

end OrigamiCone.Sequel

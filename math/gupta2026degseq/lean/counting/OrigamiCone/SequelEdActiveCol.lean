import Mathlib
import OrigamiCone.SequelEd

namespace OrigamiCone.Sequel

open OrigamiCone Finset

variable {m n : ℕ}

def activeColumn (h : Cell m n → ℤ) (j : Fin n) : Prop :=
  ∃ i : Fin m, IsStrictLocalExtremum h (i, j)

instance {h : Cell m n → ℤ} : DecidablePred (activeColumn h) := by
  intro j; unfold activeColumn; infer_instance

def numActiveColumns (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (activeColumn h)).card

theorem numActiveColumns_le_numExtrema (h : Cell m n → ℤ) :
    numActiveColumns h ≤ numExtrema h := by
  unfold numActiveColumns numExtrema
  set E : Finset (Cell m n) := Finset.univ.filter (IsStrictLocalExtremum h) with hE
  set A : Finset (Fin n) := Finset.univ.filter (activeColumn h) with hA
  suffices A ⊆ E.image Prod.snd by
    calc A.card
        ≤ (E.image Prod.snd).card := Finset.card_le_card this
      _ ≤ E.card := Finset.card_image_le
  intro j hj
  rw [Finset.mem_filter] at hj
  obtain ⟨_, i, hi⟩ := hj
  rw [Finset.mem_image]
  refine ⟨(i, j), ?_, rfl⟩
  rw [Finset.mem_filter]
  exact ⟨Finset.mem_univ _, hi⟩

theorem numActiveColumns_le_of_numExtrema (h : Cell m n → ℤ) (d : ℕ)
    (hd : numExtrema h = d) : numActiveColumns h ≤ d :=
  hd ▸ numActiveColumns_le_numExtrema h

def frozenRunStart (h : Cell m n → ℤ) (j : Fin n) : Prop :=
  ¬ activeColumn h j ∧ (j.val = 0 ∨ (0 < j.val ∧ activeColumn h ⟨j.val - 1, by omega⟩))

instance {h : Cell m n → ℤ} : DecidablePred (frozenRunStart h) := by
  intro j; unfold frozenRunStart; infer_instance

def numFrozenRuns (h : Cell m n → ℤ) : ℕ :=
  (Finset.univ.filter (frozenRunStart h)).card

theorem numFrozenRuns_le (h : Cell m n → ℤ) (hn : 0 < n)
    (h0 : activeColumn h ⟨0, hn⟩)
    (hlast : activeColumn h ⟨n - 1, by omega⟩) :
    numFrozenRuns h + 1 ≤ numActiveColumns h := by
  unfold numFrozenRuns numActiveColumns
  set S : Finset (Fin n) := Finset.univ.filter (frozenRunStart h) with hS
  set A : Finset (Fin n) := Finset.univ.filter (activeColumn h) with hA
  have hlast_mem : (⟨n - 1, by omega⟩ : Fin n) ∈ A := by
    rw [hA, Finset.mem_filter]; exact ⟨Finset.mem_univ _, hlast⟩
  have hjpos : ∀ j ∈ S, 0 < j.val := by
    intro j hj
    rw [hS, Finset.mem_filter] at hj
    obtain ⟨_, hfrz⟩ := hj
    rcases hfrz.2 with h0' | ⟨hpos, _⟩
    · exfalso
      apply hfrz.1
      have : j = ⟨0, hn⟩ := by ext; omega
      rw [this]; exact h0
    · exact hpos
  have h_inj : ∀ j₁ ∈ S, ∀ j₂ ∈ S, j₁.val - 1 = j₂.val - 1 → j₁ = j₂ := by
    intro j₁ hj₁ j₂ hj₂ heq
    have p₁ := hjpos j₁ hj₁
    have p₂ := hjpos j₂ hj₂
    ext; omega
  have h_sub : S.image (fun j => (⟨j.val - 1, by omega⟩ : Fin n))
      ⊆ A.erase ⟨n - 1, by omega⟩ := by
    intro x hx
    rw [Finset.mem_image] at hx
    obtain ⟨j, hjS, hxeq⟩ := hx
    have hjpos' := hjpos j hjS
    rw [hS, Finset.mem_filter] at hjS
    obtain ⟨_, hfrz⟩ := hjS
    rw [Finset.mem_erase]
    constructor
    · intro hcontra
      rw [← hxeq] at hcontra
      have : j.val - 1 = n - 1 := by
        have := Fin.val_eq_of_eq hcontra
        simpa using this
      have hjlt := j.isLt
      omega
    · rw [hA, Finset.mem_filter]
      refine ⟨Finset.mem_univ _, ?_⟩
      rcases hfrz.2 with h0' | ⟨_, hact⟩
      · omega
      · rw [← hxeq]; exact hact
  have h_card_image : (S.image (fun j => (⟨j.val - 1, by omega⟩ : Fin n))).card = S.card :=
    Finset.card_image_of_injOn (fun a ha b hb hab => h_inj a ha b hb (by
      have := Fin.val_eq_of_eq hab; simpa using this))
  have h_le : S.card ≤ (A.erase ⟨n - 1, by omega⟩).card := by
    rw [← h_card_image]; exact Finset.card_le_card h_sub
  rw [Finset.card_erase_of_mem hlast_mem] at h_le
  have hA_pos : 0 < A.card := Finset.card_pos.mpr ⟨_, hlast_mem⟩
  omega

theorem numFrozenRuns_lt_numExtrema (h : Cell m n → ℤ) (hn : 0 < n)
    (h0 : activeColumn h ⟨0, hn⟩)
    (hlast : activeColumn h ⟨n - 1, by omega⟩) :
    numFrozenRuns h + 1 ≤ numExtrema h :=
  le_trans (numFrozenRuns_le h hn h0 hlast) (numActiveColumns_le_numExtrema h)

end OrigamiCone.Sequel

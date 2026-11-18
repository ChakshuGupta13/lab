import Mathlib

namespace OrigamiCone.Sequel

open Finset

def turnPatterns (N c : ℕ) : Finset (Fin N → Bool) :=
  univ.filter (fun f => (univ.filter (fun i => f i = true)).card = c)

theorem card_turnPatterns (N c : ℕ) : (turnPatterns N c).card = N.choose c := by
  have key : (powersetCard c (univ : Finset (Fin N))).card = N.choose c := by
    rw [card_powersetCard, card_univ, Fintype.card_fin]
  rw [← key]
  refine card_nbij' (fun f => univ.filter (fun i => f i = true))
                    (fun S => fun i => decide (i ∈ S)) ?_ ?_ ?_ ?_
  · intro f hf
    simp only [turnPatterns, coe_filter, mem_univ, true_and, Set.mem_setOf_eq] at hf
    simp only [mem_coe, mem_powersetCard]
    exact ⟨filter_subset _ _, hf⟩
  · intro S hS
    simp only [mem_coe, mem_powersetCard] at hS
    simp only [turnPatterns, coe_filter, mem_univ, true_and, Set.mem_setOf_eq,
               decide_eq_true_eq]
    rw [← hS.2]; congr 1; ext i; simp
  · intro f hf
    funext i; simp
  · intro S hS
    ext i; simp

def runs {N : ℕ} (f : Fin N → Bool) : ℕ :=
  (univ.filter (fun i => f i = true)).card + 1

theorem runs_count (N r : ℕ) (hr : 1 ≤ r) :
    (univ.filter (fun f : Fin N → Bool => runs f = r)).card = N.choose (r - 1) := by
  have : (univ.filter (fun f : Fin N → Bool => runs f = r))
      = turnPatterns N (r - 1) := by
    apply filter_congr
    intro f _
    simp only [runs]
    omega
  rw [this, card_turnPatterns]

theorem sum_card_turnPatterns (N : ℕ) :
    ∑ c ∈ range (N + 1), (turnPatterns N c).card = 2 ^ N := by
  simp only [card_turnPatterns]
  exact Nat.sum_range_choose N

end OrigamiCone.Sequel

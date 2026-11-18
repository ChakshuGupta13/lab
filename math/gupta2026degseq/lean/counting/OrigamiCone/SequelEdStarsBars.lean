import Mathlib

namespace OrigamiCone.Sequel

open Finset

theorem antidiagonalTuple_succ_card (r m : ℕ) :
    (Finset.Nat.antidiagonalTuple (r+1) m).card
      = ∑ a ∈ range (m+1), (Finset.Nat.antidiagonalTuple r (m - a)).card := by
  rw [← Finset.card_sigma]
  apply Finset.card_nbij'
    (fun f => (⟨f 0, Fin.tail f⟩ : Σ _ : ℕ, (Fin r → ℕ)))
    (fun p => Fin.cons p.1 p.2)
  · intro f hf
    simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple] at hf
    simp only [Finset.mem_coe, Finset.mem_sigma, Finset.mem_range,
      Finset.Nat.mem_antidiagonalTuple]
    have hsum : f 0 + ∑ i, Fin.tail f i = m := by
      have h := hf; rw [Fin.sum_univ_succ] at h; exact h
    refine ⟨by omega, by omega⟩
  · intro p hp
    simp only [Finset.mem_coe, Finset.mem_sigma, Finset.mem_range,
      Finset.Nat.mem_antidiagonalTuple] at hp
    simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple, Fin.sum_univ_succ,
      Fin.cons_zero, Fin.cons_succ]
    omega
  · intro f _; exact Fin.cons_self_tail f
  · intro p _; simp only [Fin.cons_zero, Fin.tail_cons]

theorem antidiagonalTuple_card (r m : ℕ) :
    (Finset.Nat.antidiagonalTuple (r+1) m).card = (m + r).choose r := by
  induction r generalizing m with
  | zero => simp
  | succ r ih =>
    rw [antidiagonalTuple_succ_card, Finset.sum_congr rfl (fun a _ => ih (m - a))]
    have hrefl : ∑ a ∈ range (m+1), ((m - a) + r).choose r
        = ∑ i ∈ range (m+1), (i + r).choose r := by
      rw [← Finset.sum_range_reflect (fun i => (i + r).choose r) (m+1)]
      apply Finset.sum_congr rfl
      intro a ha
      simp only [Finset.mem_range] at ha
      congr 1
    rw [hrefl, Nat.sum_range_add_choose m r]
    congr 1 <;> omega

theorem positiveComp_card (r S : ℕ) (hr : 1 ≤ r) (hS : r ≤ S) :
    ((Finset.Nat.antidiagonalTuple r S).filter (fun f => ∀ i, 1 ≤ f i)).card
      = (S - 1).choose (r - 1) := by
  have hbij : ((Finset.Nat.antidiagonalTuple r S).filter (fun f => ∀ i, 1 ≤ f i)).card
      = (Finset.Nat.antidiagonalTuple r (S - r)).card := by
    apply Finset.card_nbij' (fun f => (fun i => f i - 1)) (fun g => (fun i => g i + 1))
    · intro f hf
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple] at hf
      simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple]
      obtain ⟨hsum, hpos⟩ := hf
      have hh : (∑ i, (f i - 1)) + ∑ _i : Fin r, 1 = ∑ i, f i := by
        rw [← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro i _; have := hpos i; omega
      simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
        mul_one, hsum] at hh
      omega
    · intro g hg
      simp only [Finset.mem_coe, Finset.Nat.mem_antidiagonalTuple] at hg
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple]
      refine ⟨?_, fun i => by omega⟩
      rw [Finset.sum_add_distrib, hg, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
        smul_eq_mul, mul_one]
      omega
    · intro f hf
      simp only [Finset.mem_coe, Finset.mem_filter, Finset.Nat.mem_antidiagonalTuple] at hf
      funext i; have := hf.2 i; dsimp only; omega
    · intro g _; funext i; dsimp only; omega
  rw [hbij]
  obtain ⟨r', rfl⟩ : ∃ r', r = r' + 1 := ⟨r - 1, by omega⟩
  rw [antidiagonalTuple_card r' (S - (r' + 1))]
  congr 1 <;> omega

theorem runExtension_card (r W n : ℕ) (hr : 1 ≤ r) (hW : r ≤ W) (hn : W ≤ n) :
    ((Finset.Nat.antidiagonalTuple r (n - W + r)).filter (fun f => ∀ i, 1 ≤ f i)).card
      = (n - (W - r + 1)).choose (r - 1) := by
  rw [positiveComp_card r (n - W + r) hr (by omega)]
  congr 1 <;> omega

end OrigamiCone.Sequel

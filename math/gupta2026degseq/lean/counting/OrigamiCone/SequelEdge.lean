import Mathlib

namespace OrigamiCone.Sequel

open Finset

variable {ι : Type*} (o col : ι → ℤ) (S : Finset ι) (hS : S.Nonempty)

def tau (j : ℤ) : ℤ := S.inf' hS (fun s => o s + |col s - j|)

def E2D (i j : ℤ) : ℤ := S.inf' hS (fun s => o s + (|(1 : ℤ) - i| + |col s - j|))

theorem dgrid_toprow (c i j : ℤ) (hi : 1 ≤ i) :
    |(1 : ℤ) - i| + |c - j| = (i - 1) + |c - j| := by
  rw [abs_of_nonpos (by omega)]; ring

theorem edge_reduction (i j : ℤ) (hi : 1 ≤ i) :
    E2D o col S hS i j = (i - 1) + tau o col S hS j := by
  unfold E2D tau
  have hrw : (fun s => o s + (|(1 : ℤ) - i| + |col s - j|))
           = (fun s => (i - 1) + (o s + |col s - j|)) := by
    funext s; rw [abs_of_nonpos (by omega : (1 : ℤ) - i ≤ 0)]; ring
  rw [hrw]
  apply le_antisymm
  · obtain ⟨s, hsS, hs⟩ := S.exists_mem_eq_inf' hS (fun s => o s + |col s - j|)
    calc S.inf' hS (fun s => (i - 1) + (o s + |col s - j|))
          ≤ (i - 1) + (o s + |col s - j|) := Finset.inf'_le _ hsS
      _ = (i - 1) + S.inf' hS (fun s => o s + |col s - j|) := by rw [hs]
  · apply Finset.le_inf'
    intro s hsS
    have : S.inf' hS (fun s => o s + |col s - j|) ≤ o s + |col s - j| :=
      Finset.inf'_le _ hsS
    omega

theorem edge_strict_down (i j : ℤ) (hi : 1 ≤ i) :
    E2D o col S hS (i + 1) j = E2D o col S hS i j + 1 := by
  rw [edge_reduction o col S hS (i + 1) j (by omega),
      edge_reduction o col S hS i j hi]
  ring

theorem edge_col_strictMono (j a b : ℤ) (ha : 1 ≤ a) (hab : a < b) :
    E2D o col S hS a j < E2D o col S hS b j := by
  rw [edge_reduction o col S hS a j ha, edge_reduction o col S hS b j (by omega)]
  omega

end OrigamiCone.Sequel

import Mathlib
import TxGraffitiC1FC.Invariants
import TxGraffitiC1FC.Vehicle
import TxGraffitiC1FC.CaroWei
import TxGraffitiC1FC.Delta2
import TxGraffitiC1FC.Favaron

/-!
# TxGraffiti Conjecture 1 conditional on residue monotonicity

Source: Davila–Brimkov–Pepper, *In Reverie Together* (arXiv:2507.17780), Conjecture 1;
proved in Gupta, arXiv:2606.29553.

  Every connected graph with `Δ ≥ 2` satisfies `α ≥ (a + R) / Δ`, i.e. `a + R ≤ Δ · α`,
  where α = independence number, Δ = max degree, a = annihilation number, R = residue.

The Caro-Wei and annihilation bounds and the connected maximum-degree-two
branch are proved over the graph invariants. The residue bound used in the
assembly depends on `residue_le_residue_induce_compl_of_maxDegree` in
`Favaron.lean`. Consequently `#print axioms txgraffiti_conjecture_1` lists
that custom assumption alongside the standard Lean/Mathlib axioms. The finite
probes do not discharge it.
-/

namespace SimpleGraph

variable {V : Type*} [Fintype V] [DecidableEq V] (G : SimpleGraph V) [DecidableRel G.Adj]

/-- Caro-Wei weight `W(G) = ∑_v 1/(deg v + 1)` over ℚ. -/
def caroWeiWeight : ℚ := ∑ v : V, 1 / ((G.degree v : ℚ) + 1)

set_option linter.unusedSectionVars false in
lemma caroWeiWeight_nonneg : 0 ≤ G.caroWeiWeight := by
  unfold caroWeiWeight; positivity

/-- Caro-Wei bound `W ≤ α`, proved by Finset-relative deletion induction. -/
theorem caroWei_le_indepNum : G.caroWeiWeight ≤ (G.indepNum : ℚ) := by
  unfold caroWeiWeight
  exact G.caroWei_bound

/-- Annihilation-number bound `a ≤ (Δ+1)/2 · W` for positive maximum degree,
obtained from `vehicle_bound` by unfolding the Caro-Wei weight. -/
theorem annih_vehicle (hΔ : 1 ≤ G.maxDegree) :
    (G.annihilationNumber : ℚ) ≤ ((G.maxDegree : ℚ) + 1) / 2 * G.caroWeiWeight := by
  unfold caroWeiWeight
  exact G.vehicle_bound hΔ

/-- **Head bound**: for a connected graph with `Δ ≥ 2`, `a ≤ (Δ−1)·α`.
Δ≥3 via the vehicle + Caro–Wei + `(Δ+1)/2 ≤ Δ−1`; Δ=2 via the path/cycle branch. -/
theorem annih_le_predMaxDegree_indepNum
    (hconn : G.Connected) (hΔ : 2 ≤ G.maxDegree) :
    (G.annihilationNumber : ℚ) ≤ ((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) := by
  by_cases h3 : 3 ≤ G.maxDegree
  · -- Δ ≥ 3: a ≤ (Δ+1)/2·W ≤ (Δ+1)/2·α ≤ (Δ−1)·α
    have hv := G.annih_vehicle (by omega)
    have hcw := G.caroWei_le_indepNum
    have hΔQ : (3 : ℚ) ≤ (G.maxDegree : ℚ) := by exact_mod_cast h3
    have hα0 : (0 : ℚ) ≤ (G.indepNum : ℚ) := by positivity
    have hhalf0 : (0 : ℚ) ≤ ((G.maxDegree : ℚ) + 1) / 2 := by linarith
    have hratio : ((G.maxDegree : ℚ) + 1) / 2 ≤ (G.maxDegree : ℚ) - 1 := by linarith
    calc (G.annihilationNumber : ℚ)
        ≤ ((G.maxDegree : ℚ) + 1) / 2 * G.caroWeiWeight := hv
      _ ≤ ((G.maxDegree : ℚ) + 1) / 2 * (G.indepNum : ℚ) :=
            mul_le_mul_of_nonneg_left hcw hhalf0
      _ ≤ ((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) :=
            mul_le_mul_of_nonneg_right hratio hα0
  · -- Δ = 2: (Δ−1)·α = α ≥ a
    have hΔ2 : G.maxDegree = 2 := by omega
    have ha := G.annih_le_indepNum_of_maxDegree_le_two hconn (by omega)
    have hrw : ((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) = (G.indepNum : ℚ) := by
      rw [hΔ2]; push_cast; ring
    rw [hrw]; exact ha

/-- **TxGraffiti Conjecture 1.** Every connected graph with `Δ ≥ 2` satisfies
`a + R ≤ Δ · α`. -/
theorem txgraffiti_conjecture_1
    (hconn : G.Connected) (hΔ : 2 ≤ G.maxDegree) :
    (G.annihilationNumber : ℚ) + (G.residue : ℚ)
      ≤ (G.maxDegree : ℚ) * (G.indepNum : ℚ) := by
  have hB := G.annih_le_predMaxDegree_indepNum hconn hΔ
  have hA := G.residue_le_indepNum_rat
  have hsum : (G.maxDegree : ℚ) * (G.indepNum : ℚ)
            = ((G.maxDegree : ℚ) - 1) * (G.indepNum : ℚ) + (G.indepNum : ℚ) := by ring
  linarith

#print axioms txgraffiti_conjecture_1

end SimpleGraph

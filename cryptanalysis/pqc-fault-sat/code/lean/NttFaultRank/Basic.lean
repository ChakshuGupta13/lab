/-
Theorem 1 (single-fault rank) of the Kyber NTT twiddle-zeroing fault model
— structural building blocks.

  For a single Cooley-Tukey butterfly with twiddle `z`, zeroing the twiddle
  produces a rank-1 perturbation of the butterfly map (provided `z ≠ 0`).
  This is the per-butterfly content of Theorem 1: a fault at layer `ℓ`
  controls `L = n / 2^(ℓ+1)` butterflies on disjoint coordinate pairs,
  giving combined rank `L`.

  We state the lemmas over an abstract field `K`, then specialise to
  `F = ZMod 3329` at the bottom. The abstraction avoids `Fact (Nat.Prime
  3329)`-triggered typeclass-search slowdowns inside each lemma.

  Reference: domains/cryptanalysis/docs/pqc-fault-sat/main.tex (Thm 1).
  Verified empirically in scratch/ntt_fault_rank.py for n ∈ {16, 32, 256}.
-/

import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.LinearAlgebra.Dimension.Finrank
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.IntervalCases

namespace NttFaultRank

/-! ### Generic single-butterfly rank-1 lemma -/

section Field
variable {K : Type*} [Field K]

/-- A single Cooley-Tukey butterfly with twiddle `z`. -/
def butterfly (z : K) : (Fin 2 → K) →ₗ[K] (Fin 2 → K) where
  toFun v := fun i => if i = 0 then v 0 + z * v 1 else v 0 - z * v 1
  map_add' x y := by
    funext i; by_cases h : i = 0 <;> simp [h] <;> ring
  map_smul' c x := by
    funext i; by_cases h : i = 0 <;> simp [h] <;> ring

@[simp] lemma butterfly_apply_zero (z : K) (v : Fin 2 → K) :
    butterfly z v 0 = v 0 + z * v 1 := by simp [butterfly]

@[simp] lemma butterfly_apply_one (z : K) (v : Fin 2 → K) :
    butterfly z v 1 = v 0 - z * v 1 := by simp [butterfly]

/-- The faulted butterfly: twiddle zeroed. -/
def butterflyFault : (Fin 2 → K) →ₗ[K] (Fin 2 → K) := butterfly (0 : K)

@[simp] lemma butterflyFault_apply_zero (v : Fin 2 → K) :
    (butterflyFault (K := K)) v 0 = v 0 := by simp [butterflyFault]

@[simp] lemma butterflyFault_apply_one (v : Fin 2 → K) :
    (butterflyFault (K := K)) v 1 = v 0 := by simp [butterflyFault]

/-- The difference `butterfly z - butterflyFault`. -/
def butterflyDiff (z : K) : (Fin 2 → K) →ₗ[K] (Fin 2 → K) :=
  butterfly z - butterflyFault

@[simp] lemma butterflyDiff_apply_zero (z : K) (v : Fin 2 → K) :
    butterflyDiff z v 0 = z * v 1 := by
  simp [butterflyDiff, LinearMap.sub_apply]

@[simp] lemma butterflyDiff_apply_one (z : K) (v : Fin 2 → K) :
    butterflyDiff z v 1 = -(z * v 1) := by
  simp [butterflyDiff, LinearMap.sub_apply]

/-- The spanning direction `(1, -1)`. -/
def diffDir : Fin 2 → K := fun i => if i = 0 then 1 else -1

@[simp] lemma diffDir_zero : (diffDir : Fin 2 → K) 0 = 1 := rfl
@[simp] lemma diffDir_one : (diffDir : Fin 2 → K) 1 = -1 := rfl

/-- For any `i : Fin 2`, `i = 0 ∨ i = 1`. -/
lemma fin2_cases (i : Fin 2) : i = 0 ∨ i = 1 := by
  rcases i with ⟨n, hn⟩
  interval_cases n
  · left; rfl
  · right; rfl

/-- `diffDir` is nonzero. -/
lemma diffDir_ne_zero : (diffDir : Fin 2 → K) ≠ 0 := by
  intro h
  have h0 : (diffDir : Fin 2 → K) 0 = (0 : Fin 2 → K) 0 := by rw [h]
  simp at h0

/-- Difference image ⊆ span of `(1, -1)`. -/
lemma butterflyDiff_range_le_span (z : K) :
    LinearMap.range (butterflyDiff z) ≤ K ∙ (diffDir : Fin 2 → K) := by
  rintro w ⟨v, rfl⟩
  refine Submodule.mem_span_singleton.mpr ⟨z * v 1, ?_⟩
  funext i
  rcases fin2_cases i with rfl | rfl
  · simp
  · simp

/-- The input `(0, 1)` is mapped to `z • diffDir`. -/
lemma butterflyDiff_apply_basis (z : K) :
    butterflyDiff z (fun i => if i = 0 then 0 else 1) = z • (diffDir : Fin 2 → K) := by
  funext i
  rcases fin2_cases i with rfl | rfl
  · simp
  · simp

/-- When `z ≠ 0`, `z • diffDir ∈ range`. -/
lemma smul_diffDir_mem_range (z : K) :
    z • (diffDir : Fin 2 → K) ∈ LinearMap.range (butterflyDiff z) :=
  ⟨_, butterflyDiff_apply_basis z⟩

/-- When `z ≠ 0`, the range *equals* `K ∙ diffDir`. -/
lemma butterflyDiff_range_eq_span {z : K} (hz : z ≠ 0) :
    LinearMap.range (butterflyDiff z) = K ∙ (diffDir : Fin 2 → K) := by
  apply le_antisymm (butterflyDiff_range_le_span z)
  -- K ∙ diffDir ≤ range, via K ∙ diffDir = K ∙ (z • diffDir) ≤ range.
  have hu : IsUnit z := isUnit_iff_ne_zero.mpr hz
  have hspan_eq : (K ∙ (z • (diffDir : Fin 2 → K))) = K ∙ (diffDir : Fin 2 → K) :=
    Submodule.span_singleton_smul_eq hu (diffDir : Fin 2 → K)
  rw [← hspan_eq]
  rw [Submodule.span_singleton_le_iff_mem]
  exact smul_diffDir_mem_range z

/-- **Per-butterfly rank-1.** When `z ≠ 0`, the difference between the
    unfaulted and faulted butterfly is a rank-1 linear map. -/
theorem butterflyDiff_rank_eq_one_of_ne_zero {z : K} (hz : z ≠ 0) :
    Module.finrank K (LinearMap.range (butterflyDiff z)) = 1 := by
  rw [butterflyDiff_range_eq_span hz]
  exact finrank_span_singleton diffDir_ne_zero

end Field

/-! ### Kyber instantiation (`F = ZMod 3329`)

  The abstract theorem `butterflyDiff_rank_eq_one_of_ne_zero` above already
  applies verbatim to any field `K`, in particular to `ZMod 3329`. We do
  not state a separately-named Kyber instantiation here: doing so triggers
  a Mathlib 4.27 typeclass-search slowdown on `Fact (Nat.Prime 3329)` →
  `Field (ZMod 3329)` that adds many seconds per use. Callers in the
  forthcoming `NTT.lean` will simply instantiate at use sites with the
  `Field (ZMod q)` instance carried in their context (where `q := 3329`
  is defined once and the `Fact` instance is provided locally).
-/

/-- Kyber prime, exposed for downstream files. -/
abbrev q : ℕ := 3329

end NttFaultRank


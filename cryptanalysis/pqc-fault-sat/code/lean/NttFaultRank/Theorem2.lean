/-
Phase D — Theorem 2 statement primitives.

This file defines the structures used in the kernel-equals-span theorem
on the indexed Cooley-Tukey fault model:

  ker (ntt Z - nttFault Z F) = span {e_0, e_1}

when `F` selects exactly one fault per layer and all twiddles in `Z`
are nonzero (`hZ`).

The full kernel equality is proved as `theorem2_kyber_final` in
`Theorem2Final.lean` (which composes everything: ⊇ inclusion from
`Theorem2Subset`, upper bound from `RankCeiling`, lower bound from
the Assembly pass in `Assembly`, plus rank–nullity). This file
defines only the LayerSpec primitives (`basis`, `OneFaultPerLayer`,
`faultDiff`) used throughout the development.
-/

import NttFaultRank.Ntt

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-- Standard basis vector `e p : Fin spec.n → K`, 1 at `p` and 0 elsewhere. -/
def basis (p : Fin spec.n) : Fin spec.n → K :=
  fun i => if i = p then 1 else 0

/-- "Exactly one fault per layer": every per-layer fault set has cardinality 1. -/
def OneFaultPerLayer (F : spec.FaultSet) : Prop :=
  ∀ ℓ : Fin spec.N, (F ℓ).card = 1

/-- The difference operator induced by the fault. -/
def faultDiff (Z : spec.Twiddles K) (F : spec.FaultSet) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.ntt Z - spec.nttFault Z F

end LayerSpec

end NttFaultRank

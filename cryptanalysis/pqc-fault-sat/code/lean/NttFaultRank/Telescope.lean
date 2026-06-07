/-
Phase F1 — Telescope identity for the fault difference.

The difference `ntt Z - nttFault Z F` is expressed as a sum of `N`
per-layer "perturbation" terms via the standard hybrid argument.

Avoiding the dependent-type friction of varying-bound `Fin.foldl`, we
package the hybrids as a single `Fin.foldl spec.N`-shaped fold whose
step function chooses the twiddle set (`Z'` for layers `< k`, `Z` for
layers `≥ k`) by a threshold `k : ℕ`. Boundary cases collapse to
`ntt Z` (`k = 0`) and `ntt Z'` (`k ≥ N`); the telescope identity then
reduces to `Finset.sum_range_sub'`.

This file is the foundation of the (⊆)-direction proof of Theorem 2:
all subsequent rank-bound work (F2–F9) studies `teleTerm` instead of
the full difference.
-/

import NttFaultRank.Theorem2Subset

namespace NttFaultRank

variable {K : Type*} [Field K]

namespace LayerSpec
variable (spec : LayerSpec)

/-- Hybrid linear map at threshold `k`: compose all `N` layers, using
    twiddles `Z'` on layers with index `< k` and twiddles `Z` on the
    rest. The threshold is a plain `ℕ`; no bound is required because
    the upper bound is fixed at `spec.N`. -/
def hybrid (Z Z' : spec.Twiddles K) (k : ℕ) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  Fin.foldl spec.N
    (fun acc ℓ => (spec.layer (if ℓ.val < k then Z' else Z) ℓ).comp acc)
    LinearMap.id

/-- At threshold `0` the hybrid uses `Z` on every layer: it is the
    unfaulted NTT. -/
lemma hybrid_zero (Z Z' : spec.Twiddles K) :
    spec.hybrid Z Z' 0 = spec.ntt Z := by
  unfold hybrid ntt
  congr 1 with acc ℓ

/-- At threshold `≥ N` the hybrid uses `Z'` on every layer: it is
    the NTT with twiddles `Z'`. -/
lemma hybrid_ge (Z Z' : spec.Twiddles K) {k : ℕ} (hk : spec.N ≤ k) :
    spec.hybrid Z Z' k = spec.ntt Z' := by
  unfold hybrid ntt
  congr 1 with acc ℓ
  simp [show ℓ.val < k from lt_of_lt_of_le ℓ.isLt hk]

/-- One telescope term: the difference between consecutive hybrids,
    `hybrid k - hybrid (k+1)`. Concretely this is the "layer-`k`
    perturbation": layers `< k` agree (both use `Z'`), layers `> k`
    agree (both use `Z`), and only layer `k` differs. -/
def teleTerm (Z Z' : spec.Twiddles K) (k : ℕ) :
    (Fin spec.n → K) →ₗ[K] (Fin spec.n → K) :=
  spec.hybrid Z Z' k - spec.hybrid Z Z' (k + 1)

/-- **Telescope identity for the fault difference.**

    `ntt Z - nttFault Z F = ∑_{k < N} teleTerm Z (faultedTwiddles Z F) k`.

    Proof: `nttFault Z F = ntt (faultedTwiddles Z F)` by definition;
    set `Z' := faultedTwiddles Z F`. The hybrids interpolate from
    `ntt Z` (at `k = 0`, by `hybrid_zero`) to `ntt Z'` (at `k = N`,
    by `hybrid_ge`). The telescope is `Finset.sum_range_sub'`. -/
theorem diff_ntt_telescope (Z : spec.Twiddles K) (F : spec.FaultSet) :
    spec.ntt Z - spec.nttFault Z F =
      ∑ k ∈ Finset.range spec.N,
        spec.teleTerm Z (spec.faultedTwiddles Z F) k := by
  set Z' := spec.faultedTwiddles Z F with hZ'
  -- ∑ k ∈ range N, (hybrid k - hybrid (k+1)) = hybrid 0 - hybrid N
  have hsum :
      ∑ k ∈ Finset.range spec.N,
          (spec.hybrid Z Z' k - spec.hybrid Z Z' (k + 1))
        = spec.hybrid Z Z' 0 - spec.hybrid Z Z' spec.N :=
    Finset.sum_range_sub' (fun k => spec.hybrid Z Z' k) spec.N
  -- Rewrite each side.
  change spec.ntt Z - spec.nttFault Z F =
       ∑ k ∈ Finset.range spec.N, spec.teleTerm Z Z' k
  simp only [teleTerm]
  rw [hsum, spec.hybrid_zero Z Z', spec.hybrid_ge Z Z' (le_refl spec.N)]
  rfl

/-- **Generalized telescope identity.** -/
theorem diff_ntt_telescope_gen (Z Z' : spec.Twiddles K) (F : spec.FaultSet) :
    spec.ntt Z - spec.nttFaultGen Z Z' F =
      ∑ k ∈ Finset.range spec.N,
        spec.teleTerm Z (spec.faultedTwiddlesGen Z Z' F) k := by
  show spec.ntt Z - spec.ntt (spec.faultedTwiddlesGen Z Z' F) = _
  set W := spec.faultedTwiddlesGen Z Z' F
  have hsum :
      ∑ k ∈ Finset.range spec.N,
          (spec.hybrid Z W k - spec.hybrid Z W (k + 1))
        = spec.hybrid Z W 0 - spec.hybrid Z W spec.N :=
    Finset.sum_range_sub' (fun k => spec.hybrid Z W k) spec.N
  simp only [teleTerm]
  rw [hsum, spec.hybrid_zero Z W, spec.hybrid_ge Z W (le_refl spec.N)]

end LayerSpec
end NttFaultRank

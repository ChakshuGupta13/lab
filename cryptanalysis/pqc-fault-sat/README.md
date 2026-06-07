# Rank Ceiling for Twiddle-Perturbation Faults on the Forward NTT

Reference code and Lean 4 formalisation for:

> Gupta, C. *Rank Ceiling for Twiddle-Perturbation Faults on the
> Forward NTT.* Cryptology ePrint Archive (2026).

The paper proves three properties of the information ceiling that
twiddle-perturbation faults ($\zeta_k \mapsto \zeta'_k$,
$\zeta'_k \ne \zeta_k$) can reach against the forward NTT of
ML-KEM key generation and ML-DSA signing:

1. **Theorem 1 (single-fault rank).** Each single-twiddle fault leaks
   exactly $\mathrm{len}_\ell = n/2^{\ell+1}$ secret coefficients,
   where $\ell$ is the transform layer at which the twiddle sits.
2. **Theorem 2 (one-per-layer rank and kernel).**
   - **ML-KEM** (incomplete NTT, $m = 7$ layers):
     $\mathrm{rank}\,D_K = n-2$ and
     $\ker D_K = \mathrm{span}(\mathbf{e}_0, \mathbf{e}_1)$.
   - **ML-DSA** (complete NTT, $m = 8$ layers):
     $\mathrm{rank}\,D_K = n-1$ and
     $\ker D_K = \mathrm{span}(\mathbf{e}_0)$.
3. **Corollary (universal kernel inclusion).** For every fault set $K$,
   the kernel of Theorem 2 is contained in the kernel of any
   super-set of faults, so no combination of twiddle perturbation
   reduces the residual ambiguity below the stated dimension.

## Artifact map

| Paper artifact | Repository location |
|---|---|
| Reference NTT + single-fault rank (Table 1) | [code/python/ntt_fault_rank.py](code/python/ntt_fault_rank.py) |
| Arbitrary-perturbation sweep, ML-KEM (§6) | [code/python/generalization_check.py](code/python/generalization_check.py) |
| ML-DSA rank verification (§6) | [code/python/dsa_rank_check.py](code/python/dsa_rank_check.py) |
| ML-DSA single-fault exhaustive sweep (§6) | [code/python/dsa_single_fault_sweep.py](code/python/dsa_single_fault_sweep.py) |
| Universal kernel inclusion check (§5) | [code/python/single_fault_universal.py](code/python/single_fault_universal.py) |
| One-per-layer exhaustive sweeps (§6) | [code/python/telescope_independence.py](code/python/telescope_independence.py) |
| Telescope decomposition sanity check | [code/python/telescope_decomp.py](code/python/telescope_decomp.py) |
| Lean 4: ML-KEM Theorem 2 (`theorem2_kyber_final_gen`) | [code/lean/NttFaultRank/Theorem2Final.lean](code/lean/NttFaultRank/Theorem2Final.lean) |
| Lean 4: ML-DSA Theorem 2 (`theorem2_dsa_final_gen`) | [code/lean/NttFaultRank/Theorem2DsaFinal.lean](code/lean/NttFaultRank/Theorem2DsaFinal.lean) |
| Lean 4: kernel inclusion | [code/lean/NttFaultRank/Theorem2Subset.lean](code/lean/NttFaultRank/Theorem2Subset.lean) |
| Lean 4: rank upper bound | [code/lean/NttFaultRank/RankCeiling.lean](code/lean/NttFaultRank/RankCeiling.lean) |
| Lean 4: rank lower bound (block-triangular LI) | [code/lean/NttFaultRank/Assembly.lean](code/lean/NttFaultRank/Assembly.lean) |

## Lean 4 formalisation

Both headline theorems are parametric over a single `CooleyTukeyLike`
type-class (in `Basic.lean`) instantiated by `kyberSpec` (incomplete
NTT, $m=7$) and `dsaSpec` (complete NTT, $m=8$).

- `theorem2_kyber_final_gen` in
  [Theorem2Final.lean](code/lean/NttFaultRank/Theorem2Final.lean):
  rank $n-2$, kernel $\mathrm{span}(\mathbf{e}_0, \mathbf{e}_1)$
  for ML-KEM under arbitrary twiddle perturbation.
- `theorem2_dsa_final_gen` in
  [Theorem2DsaFinal.lean](code/lean/NttFaultRank/Theorem2DsaFinal.lean):
  rank $n-1$, kernel $\mathrm{span}(\mathbf{e}_0)$ for ML-DSA
  under arbitrary twiddle perturbation.

### Reproducing the build

```bash
cd code/lean
lake exe cache get      # fetches Mathlib v4.27.0 olean cache
lake build
```

### Verifying the axioms

**ML-KEM:**

```bash
cd code/lean
echo 'import NttFaultRank
#print axioms NttFaultRank.theorem2_kyber_final_gen' > /tmp/check.lean
lake env lean /tmp/check.lean
```

**ML-DSA:**

```bash
cd code/lean
echo 'import NttFaultRank
#print axioms NttFaultRank.theorem2_dsa_final_gen' > /tmp/check.lean
lake env lean /tmp/check.lean
```

Expected output (both):

```
'NttFaultRank.theorem2_..._final_gen' depends on axioms:
  [propext, Classical.choice, Quot.sound]
```

These are the three standard Mathlib axioms; no `sorry`, no custom
axioms.

## Python reference implementation

```bash
cd code/python
python3 ntt_fault_rank.py          # Table 1 (ML-KEM + ML-DSA)
python3 generalization_check.py    # arbitrary-delta sweep (ML-KEM)
python3 dsa_rank_check.py          # ML-DSA rank verification
python3 dsa_single_fault_sweep.py  # ML-DSA single-fault exhaustive
python3 single_fault_universal.py  # universal kernel inclusion
python3 telescope_independence.py  # §6 one-per-layer sweeps
```

Requirements: Python 3.10+, NumPy. No other dependencies.

## License

MIT. See [LICENSE](../../LICENSE).

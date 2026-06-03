# Twiddle-Zeroing Faults on the ML-KEM Forward NTT

Reference code and Lean 4 formalisation for:

> Gupta, C. *Rank Ceiling for Twiddle-Zeroing Faults on the ML-KEM Forward
> NTT.* Cryptology ePrint Archive (2026).

The paper proves three properties of the information ceiling that
twiddle-zeroing faults can reach against ML-KEM key generation:

1. **Theorem 1 (single-fault rank).** Each single-twiddle fault leaks
   exactly $\mathrm{len}_\ell = n/2^{\ell+1}$ secret coefficients,
   where $\ell$ is the transform layer at which the twiddle sits.
2. **Theorem 2 (one-per-layer rank and kernel).** Combining one fault
   per layer gives $\mathrm{rank}\,D_K = n-2$ and
   $\ker D_K = \mathrm{span}(\mathbf{e}_0, \mathbf{e}_1)$, regardless
   of which twiddle within each layer is faulted.
3. **Corollary 3 (universal kernel inclusion).** For every $K$,
   $\mathrm{span}(\mathbf{e}_0, \mathbf{e}_1) \subseteq \ker D_K$, so
   no combination of twiddle-zeroing faults reduces the residual
   ambiguity below two dimensions.

## Artifact map

| Paper artifact | Repository location |
|---|---|
| Reference NTT + single-fault rank (Table 1) | [code/python/ntt_fault_rank.py](code/python/ntt_fault_rank.py) |
| One-per-layer exhaustive sweeps (§6) | [code/python/telescope_independence.py](code/python/telescope_independence.py) |
| Telescope decomposition sanity check | [code/python/telescope_decomp.py](code/python/telescope_decomp.py) |
| Lean 4 formalisation, headline Theorem 2 | [code/lean/NttFaultRank/Theorem2Final.lean](code/lean/NttFaultRank/Theorem2Final.lean) |
| Lean 4: Theorem 2 kernel direction (⊇) | [code/lean/NttFaultRank/Theorem2Subset.lean](code/lean/NttFaultRank/Theorem2Subset.lean) |
| Lean 4: rank upper bound | [code/lean/NttFaultRank/RankCeiling.lean](code/lean/NttFaultRank/RankCeiling.lean) |
| Lean 4: rank lower bound (block-triangular LI) | [code/lean/NttFaultRank/Assembly.lean](code/lean/NttFaultRank/Assembly.lean) |

## Lean 4 formalisation

The headline theorem `theorem2_kyber_final` in
[code/lean/NttFaultRank/Theorem2Final.lean](code/lean/NttFaultRank/Theorem2Final.lean)
proves both the rank equality $\mathrm{rank}\,D_K = n-2$ and the kernel
equality $\ker D_K = \mathrm{span}(\mathbf{e}_0, \mathbf{e}_1)$ for the
Kyber parameters (one fault per layer, all twiddles nonzero, $2 \ne 0$
in the field).

### Reproducing the build

```bash
cd code/lean
lake exe cache get      # fetches Mathlib v4.27.0 olean cache
lake build
```

### Verifying the axioms

```bash
cd code/lean
echo 'import NttFaultRank
#print axioms NttFaultRank.theorem2_kyber_final' > /tmp/check.lean
lake env lean /tmp/check.lean
```

Expected output:

```
'NttFaultRank.theorem2_kyber_final' depends on axioms:
  [propext, Classical.choice, Quot.sound]
```

These are the three standard Mathlib axioms; no `sorry`, no custom
axioms.

## Python reference implementation

```bash
cd code/python
python3 ntt_fault_rank.py        # Table 1
python3 telescope_independence.py # §6 sweeps
```

Requirements: Python 3.10+, NumPy. No other dependencies.

## License

MIT. See [LICENSE](../../LICENSE).

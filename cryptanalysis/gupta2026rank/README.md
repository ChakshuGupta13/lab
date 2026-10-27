# Rank Ceiling for Twiddle-Perturbation Faults on the Forward NTT

Reference code and Lean 4 formalisation for:

> Gupta, C. *Rank Ceiling for Twiddle-Perturbation Faults on the
> Forward NTT.* Cryptology ePrint Archive, Report 2026/1188 (2026).
> <https://eprint.iacr.org/2026/1188>

The programs compute ranks of the correct-minus-faulted linear maps
under static twiddle replacements. Rank counts independent linear
combinations, not individually recovered secret coefficients.
The forward results cover these two butterfly shapes:

1. **Theorem 1 (single-fault rank).** Each single-twiddle difference
  map has rank $\mathrm{len}_\ell = n/2^{\ell+1}$, where $\ell$
  is the transform layer at which the twiddle sits.
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
| Complete-shape C zeroing cross-check (§6) | [code/c/ntt_fault_rank.c](code/c/ntt_fault_rank.c) |
| Verification configurations and implementation records (§6) | [verification/](verification/) |
| Universal kernel inclusion check (§5) | [code/python/single_fault_universal.py](code/python/single_fault_universal.py) |
| One-per-layer exhaustive sweeps (§6) | [code/python/telescope_independence.py](code/python/telescope_independence.py) |
| Telescope decomposition sanity check | [code/python/telescope_decomp.py](code/python/telescope_decomp.py) |
| Lean 4: ML-KEM Theorem 2 (`theorem2_kyber_final_gen`) | [code/lean/NttFaultRank/Theorem2Final.lean](code/lean/NttFaultRank/Theorem2Final.lean) |
| Lean 4: ML-DSA Theorem 2 (`theorem2_dsa_final_gen`) | [code/lean/NttFaultRank/Theorem2DsaFinal.lean](code/lean/NttFaultRank/Theorem2DsaFinal.lean) |
| Lean 4: kernel inclusion | [code/lean/NttFaultRank/Theorem2Subset.lean](code/lean/NttFaultRank/Theorem2Subset.lean) |
| Lean 4: rank upper bound | [code/lean/NttFaultRank/RankCeiling.lean](code/lean/NttFaultRank/RankCeiling.lean) |
| Lean 4: rank lower bound (block-triangular LI) | [code/lean/NttFaultRank/Assembly.lean](code/lean/NttFaultRank/Assembly.lean) |

## Lean 4 formalisation

The two general forward kernel declarations use concrete $n=256$
specifications. The incomplete shape `kyberSpec` has seven layers;
the complete shape `dsaSpec` has eight. Their common hypotheses
require a field with $2\ne0$, nonzero original twiddles, and one
changed twiddle per layer. Replacements may be zero or nonzero.

- `theorem2_kyber_final_gen` in
  [Theorem2Final.lean](code/lean/NttFaultRank/Theorem2Final.lean):
  kernel $\mathrm{span}(\mathbf{e}_0, \mathbf{e}_1)$ for the
  incomplete shape; rank $254$ follows by rank-nullity.
- `theorem2_dsa_final_gen` in
  [Theorem2DsaFinal.lean](code/lean/NttFaultRank/Theorem2DsaFinal.lean):
  kernel $\mathrm{span}(\mathbf{e}_0)$ for the complete shape;
  rank $255$ follows by rank-nullity.

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

## C cross-check

From this artifact directory, compile and run with a C11 compiler.

```sh
cc -std=c11 -O2 -Wall -Wextra -Wpedantic code/c/ntt_fault_rank.c -o /tmp/ntt_fault_rank
/tmp/ntt_fault_rank
```

The program uses plain modular arithmetic at $n=256$, $q=8380417$
and root $1753$, with bit-reversed twiddle exponents as in
[FIPS 204, Algorithm 41](https://doi.org/10.6028/NIST.FIPS.204).
It checks 12 selected single-twiddle zeroings, one first-twiddle-per-layer
selection and the pair $\{2,3\}$ in layer 1, plus a no-fault control.
The final output is `ALL CHECKS PASSED` when all 14 fault-case ranks
match their expected values. The no-fault difference has rank zero;
that control alone does not establish transform invertibility.

These are finite zeroing rank checks, not a kernel-basis computation,
optimized-code equivalence test, firmware reproduction or physical
fault experiment. Python small-field sweeps and the concrete Lean
certificate have different coverage and should not be conflated.

## License

CC BY 4.0. See [LICENSE](../../LICENSE).

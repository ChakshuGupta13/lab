# Python reference

Plain Python + NumPy. Tested with Python 3.10+.

## Files

- `ntt_fault_rank.py` — Cooley–Tukey forward NTT for ML-KEM
  ($n=256$, $q=3329$) and ML-DSA ($n=256$, $q=8380417$),
  single-fault rank computation. Reproduces Table 1.
- `generalization_check.py` — Arbitrary twiddle perturbation
  ($\zeta_k \mapsto \zeta'_k$) sweep for ML-KEM. Confirms
  rank $n-2$ under non-zeroing replacements (§6).
- `dsa_rank_check.py` — ML-DSA rank verification at small
  $n \in \{8, 16, 32\}$: confirms rank $n-1$ and kernel
  $\mathrm{span}(\mathbf{e}_0)$ for both zeroing and
  general perturbation (§6).
- `dsa_single_fault_sweep.py` — ML-DSA exhaustive single-fault
  sweep; reproduces the ML-DSA columns of Table 1.
- `single_fault_universal.py` — Verifies universal kernel
  inclusion (Corollary 2, §5).
- `telescope_independence.py` — Exhaustive one-per-layer sweeps
  at $n \in \{32, 64\}$ and representative selections at $n = 256$
  for both ML-KEM and ML-DSA. Reproduces §6.
- `telescope_decomp.py` — Telescope decomposition sanity check;
  verifies $D_K = \sum_\ell T_\ell$ and per-term ranks.

## Running

```bash
python3 ntt_fault_rank.py
python3 generalization_check.py
python3 dsa_rank_check.py
python3 dsa_single_fault_sweep.py
python3 single_fault_universal.py
python3 telescope_independence.py
python3 telescope_decomp.py
```

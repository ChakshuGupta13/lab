# Python reference

Plain Python + NumPy. Tested with Python 3.10+.

## Files

- `ntt_fault_rank.py` — Cooley–Tukey forward NTT for ML-KEM
  ($n=256$, $q=3329$), single-fault rank computation. Reproduces
  Table 1 of the paper.
- `telescope_independence.py` — Exhaustive one-per-layer sweeps
  at $n \in \{32, 64\}$ and representative selections at $n = 256$
  for both incomplete (ML-KEM) and full Cooley–Tukey NTTs.
  Reproduces §6.
- `telescope_decomp.py` — Telescope decomposition sanity check;
  verifies $D_K = \sum_\ell T_\ell$ and per-term ranks.

## Running

```bash
python3 ntt_fault_rank.py
python3 telescope_independence.py
python3 telescope_decomp.py
```

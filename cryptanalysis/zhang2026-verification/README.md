# Zhang–Li–Gao–Wang 2026 — SFS Collision Verification

Independent verification artifacts for:

> Zhang, Z., Li, M., Gao, L. & Wang, M. (2026).
> *Collision Attacks on SHA-256 up to 37 Steps with Improved Trail Search.*
> EUROCRYPT 2026.
> Authors' code: [github.com/Zhang-SDU/AutoSHA2Collision](https://github.com/Zhang-SDU/AutoSHA2Collision)

## Contents

| File | Description |
|------|-------------|
| `verify_sfs.cpp` | Verifies the 37-step (Table 9) and 36-step (Table 11) semi-free-start collision pairs via step-reduced SHA-256 compression with full output check |
| `table8_char.hpp` | 37-step differential characteristic (Table 8), computationally verified against the Table 9 SFS pair. Corrects extraction errors in the published PDF (38/246 value-condition bits were garbled by pdftotext) |
| `table10_char.hpp` | 36-step differential characteristic (Table 10), verified against the Table 11 SFS pair |
| `lc_sweep.cpp` | Word-level local collision parameter sweep: for a given round count R, scans all (V, K) pairs and reports the best shapes by OBJ. Fills the gap at R = 33–35 between Mendel 2013 and Zhang 2026 |
| `carry_chain_model.cpp` | Carry-chain Markov model for β-condition cost prediction. Validates the model against direct sampling (<0.006 bits error) and distinguishes the unconstrained (2.00 bits/diff-cond) from constrained (~1.71 bits/diff-cond) regimes |
| `local_collision.hpp` | Word-level local collision search engine (Zhang et al. 2026 Algorithm 1) |
| `carry_chain.hpp` | Carry-chain Markov model primitives for modular addition differentials |
| `sha256.hpp` | Self-contained SHA-256 step function (shared primitive) |

## Build and run

```bash
# SFS collision verification
g++ -std=c++17 -O2 -Wall -o verify_sfs verify_sfs.cpp
./verify_sfs    # exits 0 on VERIFIED for both pairs

# Local collision parameter sweep (e.g. R=35)
g++ -std=c++17 -O2 -Wall -o lc_sweep lc_sweep.cpp
./lc_sweep 35   # prints best (V, K, OBJ) shapes for 35-step

# Carry-chain Markov model validation
g++ -std=c++17 -O2 -Wall -o carry_chain_model carry_chain_model.cpp
./carry_chain_model   # validates model against sampling, prints per-regime costs
```

## Note on Table 8

The published PDF's Table 8 contains dense value-condition rows where
`pdftotext` garbles `0`↔`1` in approximately 15% of positions (38/246
conditions affected, concentrated in DE steps 4–7). The `table8_char.hpp`
in this directory was reconstructed by computational verification against
the published Table 9 SFS collision pair: each condition was checked against
the actual message pair, and disagreements were corrected. The `.txt`
extraction retains the original garbled text for reference; the `.hpp` is
the ground truth.

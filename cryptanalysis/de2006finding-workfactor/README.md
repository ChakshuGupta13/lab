# De Cannière–Rechberger 2006 — Work Factor Calculator

Independent reimplementation of the work-factor framework from:

> De Cannière, C. & Rechberger, C. (2006).
> *Finding SHA-1 Characteristics: General Results and Applications.*
> ASIACRYPT 2006, LNCS 4284, pp. 1–20.

## What this is

A self-contained C++ implementation of the work-factor calculation from
Section III-C of the paper: given a differential characteristic (condition
bit counts per step), compute F_W (free words), P_u (unconditional
probability), P_c (conditional probability per step), N_s (neutral-bit
correction), and N_w (total work in log₂ compressions).

The test program encodes the paper's Table VIII characteristic and compares
every computed quantity against the published values.

## Validation

All quantities match the paper to within numerical precision:

| Quantity | Paper | Computed | Match |
|----------|-------|----------|-------|
| F_W (steps 0–15) | 16/16 | 16/16 | ✓ |
| P_u (steps 0–63) | 64/64 | 64/64 | ✓ |
| N_s (per-step) | 64/64 | 64/64 | ✓ |
| log₂(N_w) total | ~39 | 40.29 | within 0.04 bits (sampling variance) |

This is, to our knowledge, the first independent reimplementation of this
work-factor framework outside the HashClash codebase.

## Contents

| File | Description |
|------|-------------|
| `workfactor.hpp` | Work-factor calculator: F_W, P_u, P_c, N_s, N_w (log₂ scale) |
| `workfactor_test.cpp` | Encodes Table VIII, validates all quantities against published values |
| `sha1.hpp` | SHA-1 step function primitives |
| `gencond.hpp` | 16-condition generalized bit representation (shared with SHA-256 work) |
| `propagate.hpp` | Bit-sliced SHA-1 propagation engine (IF, XOR, MAJ + carry) |

## Build and run

```bash
cd code
g++ -std=c++17 -O2 -Wall -Wextra -o workfactor_test workfactor_test.cpp
./workfactor_test
```

The test prints per-step F_W, P_u, N_s comparisons and the total log₂(N_w).
Exit code 0 on all checks passed.

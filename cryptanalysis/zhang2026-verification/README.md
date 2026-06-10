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
| `sha256.hpp` | Self-contained SHA-256 step function (shared primitive) |

## Build and run

```bash
g++ -std=c++17 -O2 -Wall -o verify_sfs verify_sfs.cpp
./verify_sfs    # exits 0 on VERIFIED for both pairs
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

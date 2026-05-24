# 38-Step SHA-512 SFS Collision via SAT (Eichlseder 2014)

Reproduction of:

> Eichlseder, M., Mendel, F. & Schläffer, M. (2014).
> *Branching Heuristics in Differential Collision Search with Applications to SHA-512.*
> FSE 2014, IACR final version 2014-04-30.
> [eprint 2014/302](https://eprint.iacr.org/2014/302)

This directory packages a 38-step **semi-free-start (SFS) collision**
on reduced SHA-512, encoded as a SAT instance from the paper's
published full differential characteristic (Table 3) and solved with
CaDiCaL. A self-contained C++ verifier checks the SAT-extracted message
pair against the SHA-512 step function.

This is a **reproduction artifact**, not new cryptanalysis. The paper's
own characteristic data is what makes the SAT instance tractable in
under a minute; no novel attack or speedup is claimed. In particular,
the paper's central contribution — the look-ahead branching heuristic
(Algorithm 2) — is **not exercised** by this reproduction; the dense
SAT encoding bypasses the search problem the heuristic was designed
for. See [Cross-paper context](#cross-paper-context) below.

## Attack reproduced

### 38-step SHA-512 SFS collision (Table 3)

Semi-free-start (IV is free; both copies use the same value).
Local collision spans 18 steps; 6 active expanded message words
($W_7, W_8, W_{10}, W_{15}, W_{23}, W_{24}$).

- **Reproducer**: [`code/reproduce_table3_38step_sfs_sha512.cpp`](code/reproduce_table3_38step_sfs_sha512.cpp)
- **Source**: paper Table 3 (full characteristic), encoded as
  `eichlseder2014::starting_point_38_sfs()` in [`code/starting_points_512.hpp`](code/starting_points_512.hpp).
- **SAT result**: SAT in ~38 s / ~640K conflicts on a single CPU core
  (M-class laptop). Paper estimate for the same instance via the
  authors' guess-and-determine engine + look-ahead heuristic was
  $2^{40.5}$ compressions ≈ 1.5 h on a 40-CPU cluster.
- **Verification**: [`code/verify_collision.cpp`](code/verify_collision.cpp)
  embeds a minimal SHA-512 step function and rechecks the paper's
  Table 2 example collision (and any candidate produced by the
  reproducer) for full 512-bit output equality.

## Cross-paper context

This reproduction reuses the encoding methodology from the
[`mendel2011`](../mendel2011/) and
[`mendel2013improving`](../mendel2013improving/) reproduction packages
(SAT over Tseitin-encoded dual-copy step function), ported from SHA-256
to SHA-512 word width. The port is mechanical — rotation constants
updated for SHA-512 Σ/σ, word size 32→64.

## Build

C++17. External dependency: [CaDiCaL](https://github.com/arminbiere/cadical) SAT solver.

```
# Get and build CaDiCaL once
git clone https://github.com/arminbiere/cadical /tmp/cadical
( cd /tmp/cadical && ./configure && make )

# Build the reproducer
cd code
g++ -std=c++17 -O2 -Wall -Wextra -Wno-trigraphs -pthread -I. \
    -I/tmp/cadical/src \
    -o reproduce_table3_38step_sfs_sha512 reproduce_table3_38step_sfs_sha512.cpp \
    /tmp/cadical/build/libcadical.a

# Build the standalone verifier (no CaDiCaL needed)
g++ -std=c++17 -O2 -Wall -Wextra -o verify_collision verify_collision.cpp
```

## Run

```
cd code
./reproduce_table3_38step_sfs_sha512 <conflict_limit>
# e.g. ./reproduce_table3_38step_sfs_sha512 1000000   (SAT in ~38 s)

./verify_collision
# Re-verifies the paper's published Table 2 example collision.
```

## Files

| File | Purpose |
|---|---|
| [`code/reproduce_table3_38step_sfs_sha512.cpp`](code/reproduce_table3_38step_sfs_sha512.cpp) | SAT reproduction driver for 38-step SHA-512 SFS collision |
| [`code/verify_collision.cpp`](code/verify_collision.cpp) | Standalone SHA-512 step-function verifier (paper Table 2 example) |
| [`code/starting_points_512.hpp`](code/starting_points_512.hpp) | `eichlseder2014::starting_point_38_sfs()` — Table 3 characteristic |
| [`code/cnf_encoder_512.hpp`](code/cnf_encoder_512.hpp) | Tseitin-style CNF encoder for the dual-copy SHA-512 step function |
| [`code/sha512.hpp`](code/sha512.hpp) | Reference SHA-512 step function and constants |
| [`code/sha256.hpp`](code/sha256.hpp) | Reference SHA-256 step function (pulled in transitively by shared headers; not used at runtime by the SHA-512 reproducer) |
| [`code/gencond.hpp`](code/gencond.hpp) | 16-valued generalized bit conditions (De Cannière–Rechberger) |
| [`code/search_512.hpp`](code/search_512.hpp) | Guess-and-determine engine for SHA-512. Defines `CharState` type used by the reproducer; the search routine itself is not invoked. |
| `code/propagate.hpp`, `code/propagate_512.hpp`, `code/twobit.hpp`, `code/alt_step_512.hpp` | Headers transitively required by `search_512.hpp` |

## License

See repository root [`LICENSE`](../../LICENSE).

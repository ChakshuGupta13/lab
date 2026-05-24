# Reduced-Step SHA-256 Collisions via SAT (Mendel 2011)

Reproductions of:

> Mendel, F., Nad, T. & Schläffer, M. (2011).
> *Finding SHA-2 Characteristics: Searching Through a Minefield of Contradictions.*
> ASIACRYPT 2011, LNCS 7073, pp. 288–307.
> [DOI: 10.1007/978-3-642-25385-0_16](https://doi.org/10.1007/978-3-642-25385-0_16)

This directory packages two collision reproductions on reduced-step
SHA-256: the **27-step real collision** (paper Table 7/8) and the
**32-step semi-free-start collision** (paper Table 3/4). Both are
produced by encoding the paper's full differential characteristic as a
SAT instance and solving with CaDiCaL.

This is a **reproduction artifact**, not new cryptanalysis. The paper's
own characteristic data is what makes the SAT instances tractable in
seconds; no novel attack or speedup is claimed.

## Attacks reproduced

### 27-step real collision (Table 7)

Standard SHA-256 IV. Local collision spans steps 7–17. Message words
$W_0, \dots, W_6$ are identical between the two copies; differences in
$W_7, W_8, W_{12}, W_{15}, W_{17}$.

- **Reproducer**: [`code/reproduce_table7_27step.cpp`](code/reproduce_table7_27step.cpp)
- **Source**: paper Table 7 (full characteristic). The reproducer
  embeds the characteristic data locally (function
  `starting_point_27_table7_local()`); an identical copy lives in
  [`code/starting_points.hpp`](code/starting_points.hpp) as `starting_point_27_table7()`
  for reuse by other consumers.
- **SAT result** (single core, n_fixed=0): SAT in ~40 s / 764K
  conflicts. With $W_0, \dots, W_{n-1}$ pinned to Table 8 values
  (n_fixed=9): SAT in ~0.1 s / ~750 conflicts.
- **Verification**: SAT-extracted message pair is fed through the
  reference SHA-256 step function ([`code/sha256.hpp`](code/sha256.hpp)) with
  full 256-bit output equality check.

### 32-step semi-free-start collision (Table 3)

IV is free (both copies use the same IV value but it is not the
standard SHA-256 IV). Local collision spans steps 2–17.

- **Reproducer**: [`code/reproduce_table3_32step_sfs.cpp`](code/reproduce_table3_32step_sfs.cpp)
- **Source**: paper Table 3 (full characteristic), with a documented
  hybrid relaxation: Table 3 marks 19 bits in $W_{3..7}$ low-half as
  `-` (equal) but Table 4's example collision has those bits
  differing. The hybrid SP keeps Table 3's upper-half conditions (the
  cryptanalytically meaningful active region) and relaxes the lower
  ~16 bits of $W_{3..7}$ to free. The reproducer embeds the
  characteristic locally (function `starting_point_32_sfs_table3_local()`);
  a near-identical copy lives in
  [`code/starting_points.hpp`](code/starting_points.hpp) as `starting_point_32_sfs_table3()`.
- **SAT result**: SAT in ~5.6 s / 170K conflicts.
- **Verification**: as above.

### A documented paper typo

Mendel 2011 Table 4 prints $h_1[1]$ as `e1f519a2`; running SHA-256 on
the paper's message pair and IV produces `f5abb78c`. The verifier
[`code/verify_collisions.cpp`](code/verify_collisions.cpp) checks against the
corrected value.

## Build

C++17. External dependency: [CaDiCaL](https://github.com/arminbiere/cadical) SAT solver.

```
# Get and build CaDiCaL once
git clone https://github.com/arminbiere/cadical /tmp/cadical
( cd /tmp/cadical && ./configure && make )

# Build the two reproducers
g++ -std=c++17 -O2 -Wall -Wextra -Wno-trigraphs -pthread -I. \
    -I/tmp/cadical/src \
    -o reproduce_table7_27step reproduce_table7_27step.cpp \
    /tmp/cadical/build/libcadical.a

g++ -std=c++17 -O2 -Wall -Wextra -Wno-trigraphs -pthread -I. \
    -I/tmp/cadical/src \
    -o reproduce_table3_32step_sfs reproduce_table3_32step_sfs.cpp \
    /tmp/cadical/build/libcadical.a
```

## Run

```
# 27-step (Table 6 baseline vs Table 7 dense head-to-head)
./reproduce_table7_27step <conflict_limit> <n_fixed>
# e.g. ./reproduce_table7_27step 1000000 0   (full search, ~40 s)
#      ./reproduce_table7_27step 50000   9   (fast smoke, < 1 s)

# 32-step SFS
./reproduce_table3_32step_sfs <conflict_limit>
# e.g. ./reproduce_table3_32step_sfs 5000000   (SAT in ~5.6 s)
```

Each reproducer prints the SAT statistics, then the verified
message-pair words ($M_1$ and $M_2$).

## Files

| File | Purpose |
|---|---|
| `reproduce_table7_27step.cpp` | SAT reproduction driver for 27-step real collision (Table 7 vs Table 6) |
| `reproduce_table3_32step_sfs.cpp` | SAT reproduction driver for 32-step SFS collision (Table 3) |
| `verify_collisions.cpp` | Independent paper-table verifier (catches the Table 4 `h₁[1]` typo) |
| `cnf_encoder.hpp` | Tseitin-style CNF encoder for the dual-copy SHA-256 step function |
| `starting_points.hpp` | The four starting points: `starting_point_27()` (Table 6 sparse), `starting_point_27_table7()` (Table 7 dense), `starting_point_32_sfs()` (Table 2 sparse), `starting_point_32_sfs_table3()` (Table 3 dense hybrid) |
| `sha256.hpp` | Reference SHA-256 step function and constants |
| `gencond.hpp` | 16-valued generalized bit conditions (De Cannière–Rechberger) |
| `search.hpp` | Guess-and-determine engine. Defines the `CharState` data type that the reproducers use to organize starting-point data; the search routine itself is not invoked by the SAT reproducers above. |
| `propagate.hpp`, `twobit.hpp`, `alt_step.hpp`, `wordwise.hpp` | Headers transitively required by `search.hpp` |
| `embedded_bcp.hpp` | Experimental 2-watched-literal BCP engine. Included transitively; not instantiated or called by the SAT reproducers. |

## License

See repository root [`LICENSE`](../../LICENSE).

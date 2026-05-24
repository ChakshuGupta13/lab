# Reduced-Step SHA-256 Collisions via SAT (Mendel 2013)

Reproductions of:

> Mendel, F., Nad, T. & Schläffer, M. (2013).
> *Improving Local Collisions: New Attacks on Reduced SHA-256.*
> EUROCRYPT 2013, LNCS 7881, pp. 262–278.
> [DOI: 10.1007/978-3-642-38348-9_16](https://doi.org/10.1007/978-3-642-38348-9_16)

Three collision reproductions on reduced-step SHA-256, encoded as SAT
instances from the paper's published full differential characteristics:
**28-step real collision** (Table 3/6), **31-step semi-free-start
collision** (Table 4/7), and **38-step semi-free-start collision**
(Table 5/8). Reuses the CNF encoder from the
[`mendel2011`](../mendel2011/) reproduction package.

This is a **reproduction artifact**, not new cryptanalysis. The paper's
own characteristic data is what makes the SAT instances tractable in
seconds; no novel attack or speedup is claimed.

## Attacks reproduced

### 28-step real collision (Table 3)

Standard SHA-256 IV. Local collision spans 11 steps.

- **Reproducer**: [`reproduce_table3_28step.cpp`](reproduce_table3_28step.cpp)
- **Source**: paper Table 3 (full characteristic), encoded as
  `mendel2013::starting_point_28_full()` in [`starting_points.hpp`](starting_points.hpp).
- **SAT result**: SAT in ~0.1 s / ~350 conflicts.
- **Verification**: SAT-extracted message pair fed through reference
  SHA-256 step function with full 256-bit output equality check.

### 31-step SFS collision (Table 4)

Semi-free-start. Local collision spans 14 steps; 7 active message words.

- **Reproducer**: [`reproduce_table4_31step_sfs.cpp`](reproduce_table4_31step_sfs.cpp)
- **Source**: paper Table 4 (full characteristic), encoded as
  `mendel2013::starting_point_31_full()` in [`starting_points.hpp`](starting_points.hpp).
- **SAT result**: SAT in ~11.7 s / ~272K conflicts.
- **Verification**: as above.

### 38-step SFS collision (Table 5)

Semi-free-start. Local collision spans 18 steps; 6 active message words.
Paper estimates this attack at ~8 hours of single-CPU time using their
bespoke guess-and-determine engine.

- **Reproducer**: [`reproduce_table5_38step_sfs.cpp`](reproduce_table5_38step_sfs.cpp)
- **Source**: paper Table 5 (full characteristic), encoded as
  `mendel2013::starting_point_38_sfs_full()` in [`starting_points.hpp`](starting_points.hpp).
- **SAT result**: SAT in ~9.6 s / ~212K conflicts.
- **Verification**: as above.

## Build

C++17. External dependency: [CaDiCaL](https://github.com/arminbiere/cadical) SAT solver.

```
# Get and build CaDiCaL once
git clone https://github.com/arminbiere/cadical /tmp/cadical
( cd /tmp/cadical && ./configure && make )

# Build the three reproducers
for r in reproduce_table3_28step reproduce_table4_31step_sfs reproduce_table5_38step_sfs; do
  g++ -std=c++17 -O2 -Wall -Wextra -Wno-trigraphs -pthread -I. \
      -I/tmp/cadical/src \
      -o $r $r.cpp \
      /tmp/cadical/build/libcadical.a
done
```

## Run

```
./reproduce_table3_28step  <conflict_limit>    # SAT in ~0.1 s
./reproduce_table4_31step_sfs <conflict_limit> # SAT in ~11.7 s
./reproduce_table5_38step_sfs <conflict_limit> # SAT in ~9.6 s
# example: ./reproduce_table5_38step_sfs 500000
```

Each reproducer runs a head-to-head: first the sparse Table 1 starting
point (expected TIMEOUT), then the dense Table 3/4/5 characteristic
(expected SAT with verified message pair).

## Files

| File | Purpose |
|---|---|
| `reproduce_table3_28step.cpp` | SAT reproduction for 28-step real collision |
| `reproduce_table4_31step_sfs.cpp` | SAT reproduction for 31-step SFS collision |
| `reproduce_table5_38step_sfs.cpp` | SAT reproduction for 38-step SFS collision |
| `starting_points.hpp` | Mendel 2013 starting points: `mendel2013::starting_point_28_full()`, `_31_full()`, `_38_sfs_full()` |
| `starting_points_2011.hpp` | Mendel 2011 starting points (the 2013 SPs extend the 2011 base SPs) |
| `sha256_starting_points.hpp` | Sparse Table 1 starting points for head-to-head comparison |
| `cnf_encoder.hpp` | Tseitin-style CNF encoder for the dual-copy SHA-256 step function (reused from mendel2011) |
| `sha256.hpp` | Reference SHA-256 step function and constants |
| `gencond.hpp` | 16-valued generalized bit conditions (De Cannière–Rechberger) |
| `search.hpp` | Guess-and-determine engine. Defines the `CharState` data type used by the reproducers; the search routine itself is not invoked by the SAT reproducers. |
| `propagate.hpp`, `twobit.hpp`, `alt_step.hpp`, `wordwise.hpp` | Headers transitively required by `search.hpp` |
| `embedded_bcp.hpp` | Experimental 2-watched-literal BCP engine. Included transitively; not instantiated or called by the SAT reproducers. |

## License

See repository root [`LICENSE`](../../LICENSE).

# Verification Configurations

## Matrix Construction

The algebraic implementations use plain, non-Montgomery forward
Cooley--Tukey networks. Correct-minus-faulted evaluations on every coordinate
basis vector construct each difference matrix. A selected replacement is fixed
at every use of its indexed twiddle. Exact finite-field elimination computes
the rank.

## Singleton Checks

- The incomplete checker uses the bit-reversed ML-KEM table at $n=256$ and
  $q=3329$. It zeroes all 127 indexed positions separately.
- Complete small-field checks use seeded uniform nonzero twiddles at
  $(n,q)=(8,17)$, $(16,97)$ and $(32,193)$. They cover 7, 15 and 31 positions.
- Every singleton rank equals the butterfly length. The incomplete layer ranks
  are 128, 64, 32, 16, 8, 4 and 2. At $(32,193)$, the complete layer ranks are
  16, 8, 4, 2 and 1.

## One Replacement Per Layer

The incomplete exhaustive sweeps at $q=3329$ use an unreversed power ladder.
All 64 selections at $n=32$ and all 1,024 selections at $n=64$ have zeroing
rank $n-2$ and nullity 2. These sweeps do not compute kernel bases.

At $(n,q)=(256,3329)$, six bit-reversed-table selections choose the first,
last or middle group in each layer, followed by three seeded selections.
Zeroing and the changed value $\zeta+1\bmod q$ give twelve cases. Every case
has rank 254 and zero columns 0 and 1.

The complete helper selects each layer's first group in the three small-field
configurations. Zeroing and a changed nonzero value give six rank-$(n-1)$
matrices with zero column 0.

## Standard ML-DSA Parameters

The C check uses $n=256$, $q=8380417$, ordinary modular arithmetic and
bit-reversed powers of the 512th root 1753.

- Twelve singleton zeroing matrices use each layer's first twiddle and indices
  3, 7, 200 and 255.
- First-group-per-layer zeroing has rank 255.
- Zeroing group twiddles 2 and 3 in layer 1 has rank 128.

The replacement-value checker uses the standardized moduli. Its ML-KEM table
contains bit-reversed powers of 17 with seven-bit indices. Its ML-DSA table
contains bit-reversed powers of 1753 with eight-bit indices.

## Roundtrips

Each unfaulted roundtrip recovers one seeded input per configuration. The
incomplete routine covers $q=3329$ and $n$ in 16, 32, 64 and 256. The complete
routine covers the three small-field configurations. These checks do not test
rank or kernel predictions.
# Counting Proof Interface

## Model

`OrigamiCone.Cell m n` is `Fin m x Fin n`. Grid adjacency uses Manhattan
distance one. Heights are integer-valued functions whose adjacent values
differ in absolute value by one. A canonical height is zero at the corner
with both coordinates zero.

`OrigamiCone.Sequel.Ed d m n` counts actual canonical heights having exactly
`d` strict local extrema. The signed counts distinguish strict local minima
and maxima. The one-dimensional path model uses `Cell 1 n`, includes both
endpoints as extrema, and requires `n >= 2` in the signed counting formula.

## Entry Points

- `MiuraCounting.ConcreteSmallDegreePolynomials` provides the full total-degree
  theorem, literal degree-three/four polynomial definitions, exact node
  computations and polynomial uniqueness on the stable region.
- `MiuraCounting.ConcreteSignedPathCounts` provides actual signed-site maps,
  sign/turn-set counting, support and the complete signed binomial formula.
- `MiuraCounting.ConcreteSignedSideCounts` provides actual signed side-to-path
  maps, finite side decomposition, its true remainder complement and the
  leading homogeneous component.
- `MiuraCounting.ConcreteDegreeTwoCounts` provides the four-gradient
  classification, distinctness and exact degree-two count.

The aggregator `MiuraCounting` imports the three selected release entry
points; their dependency closure includes the signed path module.

## Mathematical Domains

The total-degree theorem requires `d >= 3` and represents the count on
`m,n >= d-1`. Its polynomial has total degree `d-2`, specializes to
`4(m+n)-16` at degree three and to
`2m^2+2n^2+6mn-10(m+n)-4` at degree four. For `d >= 5` its homogeneous part
of degree `d-2` is `4/(d-2)!` times `m^(d-2)+n^(d-2)`.

The signed path count requires positive `a,b` and `n >= 2`. It is zero
unless `abs(a-b) <= 1`; otherwise it is
`(1+[a=b]) * choose(n-2,a+b-2)`.

The signed side identity requires `m,n >= 2` and `a+b >= 3`. Degree-two
corner gradients occur on more than one side, so that threshold must not be
removed. The four-side count is `2*f(n)+2*f(m)`. The remainder is the actual
complement of the single-side family in the canonical signed-height family.

## Trust and Validation

The toolchain and dependency graph are pinned. The validator checks source
hashes and import closure before compilation, rejects incomplete proofs and
custom-axiom declarations, compiles the local source graph, and inspects the
six declarations listed in `sources.json` using `Check.lean`.

Only `propext`, `Classical.choice` and `Quot.sound` are accepted in those
theorems' transitive axiom closures. Existing foundation style warnings are
recorded; warnings from new counting modules fail the validator. The normal
incremental build may reuse locally produced objects. `--require-clean`
rejects existing local proof objects at startup for a fresh replay.

This is not a formalization of the entire articles or a proof of the physical
crease-pattern correspondence, quotient graph distance, boundary-correction
conjectures or all numerical tables.
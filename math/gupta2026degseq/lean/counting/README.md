# Miura-ori Counting Proofs

This Lean project proves selected counting results for normalized integer
height functions on rectangular grids. It is a partial formalization, not a
verification of every result in the accompanying origami papers.

The project contains 94 source modules and no declared custom axioms. The
public theorem checks allow only `propext`, `Classical.choice` and `Quot.sound`.
The historical sibling project remains separate and retains its own disclosed
assumptions.

## Build

Requirements are Git, Node.js 18 or newer, and elan. The committed toolchain
selects Lean 4.27.0. Mathlib and its transitive dependencies are locked by
commit; Mathlib is pinned to
`a3a10db0e9d66acbebf76c5e6a135066525ac900`.

From a fresh clone of this repository:

```sh
cd math/gupta2026degseq/lean/counting
lake exe cache get
node verify.mjs --require-clean
```

The setup command downloads the pinned third-party dependencies and their
published cache. The validator does not download dependencies or rebuild
Mathlib. It fails if the dependency cache is missing or out of date. Our own
proof modules are compiled from the source files in this directory.

The validator builds local modules in dependency order, one target at a time.
Each compiler uses two threads and a memory limit of 8 GiB. Compilation can
take tens of minutes. The final normal library build and theorem checks must
also pass. A local `validation.json` records results and is not committed.

Existing style warnings in the preserved `OrigamiCone` foundation are reported
in that local receipt. Warnings in the new `MiuraCounting` modules, compilation
errors, incomplete proofs and nonstandard theorem axioms fail validation.

For source integrity checks without compilation:

```sh
node verify.mjs --sources-only
```

For an ordinary incremental build:

```sh
lake build MiuraCounting
lake env lean Check.lean
```

The expected final validator output includes `"build":"pass"`,
`"declarations":6` and `"standardAxiomsOnly":true`.

[BUILD.json](BUILD.json) records the release build, theorem axiom closures and
the retained foundation diagnostics. It distinguishes the fresh exported
source tree from third-party cache reuse and records the build continuation.

## Coverage

All statements below use actual normalized unit-step height functions and
their actual strict local extrema, including endpoints of paths.

| Result | Public declaration |
| --- | --- |
| Total degree `d-2` for every `d >= 3`, explicit degree-three/four polynomials, and the leading homogeneous part for `d >= 5` | `OrigamiCone.Sequel.SmallDegreePolynomials.Ed_full_totalDegree_and_leading` |
| Exactly four degree-two heights | `OrigamiCone.Sequel.DegreeTwoCounts.Ed_two` |
| Complete four-gradient classification at degree two | `OrigamiCone.Sequel.DegreeTwoCounts.degree_two_iff_gradient` |
| Full prescribed minimum/maximum path count and unsupported-split zero | `OrigamiCone.Sequel.SignedPathCounts.canonical_path_signed_binomial` |
| Prescribed signed side/remainder decomposition | `OrigamiCone.Sequel.SignedSideCounts.canonicalSplit_signed_side_decomposition` |
| Leading homogeneous part of the signed side contribution | `OrigamiCone.Sequel.SignedSideCounts.signedSidePolynomial_leading` |

The stable counting region for degree `d >= 3` is `m,n >= d-1`. The degree-two
and degree-three counts hold for `m,n >= 2`. The signed side decomposition
requires `m,n >= 2` and a total of at least three extrema. Its count identity
holds throughout that region; the particular side polynomial's evaluation
theorem is currently stated on the smaller tail `m,n >= d`.

The explicit degree-four formula is obtained from exact kernel computations
at the nine points of `{3,4,5} x {3,4,5}` and a proved separate-degree bound.
Repeated quadratic interpolation proves the result on its full stable region.
It is not an extrapolation from numerical samples.

See [SPEC.md](SPEC.md) for precise interfaces and exclusions.

## Source Integrity

[sources.json](sources.json) lists each exported module, its SHA-256 hash and
the corresponding original module hash. Source transformations remove Lean
comments and move standalone module imports under `MiuraCounting`. Declaration
and proof text is otherwise retained. No compiled local proof objects, private
paths or external proof source are shipped.

## Related Formalization

The Omega Institute provides a separate Lean formalization of the
rectangular-grid dispersion inequality in
[PR #21588](https://github.com/the-omega-institute/trureturing/pull/21588),
which closes [issue #19839](https://github.com/the-omega-institute/trureturing/issues/19839).
The exact merged source is
[GuptaMiuraGridDispersion.lean](https://github.com/the-omega-institute/trureturing/blob/92b6f950ddee9e1601f4808a66bdccba2b315605/D5/S3/Combinatorics/Geometry/GuptaMiuraGridDispersion.lean).

Its theorem bounds median-dispersion for integer grid functions with unit
edge bounds. The upstream module explicitly excludes the origami distance
and diameter-lower-bound reductions. It is not imported or rebuilt by this
project and does not verify our counting or farthest-pair results. Its Lean
toolchain differs from this project's pin. Upstream code is Apache-2.0; this
release links to it and does not redistribute it.

The exact rectangular diameter and the communicated covariance comparison
are credited to Simon Watts in the accompanying work. See
[The Slowest-Chain Conjecture and the Exact Diameter of Rectangular Miura-Ori Flip Graphs](https://doi.org/10.5281/zenodo.22019186).
The upstream formalization link does not replace that attribution.

## Limits

This release does not establish the full physical origami/height graph
correspondence, every admissible-configuration transport, every separable-family
statement, boundary-correction claims, higher explicit coefficient tables,
quotient graph distance, diameter, farthest-pair classification or every
appendix result. Unreviewed configuration and colouring-vertex extensions are
not included.

The finite envelope and transfer calculations already present in the selected
source dependencies should not be read as a claim that every similarly named
paper statement has a complete formalization.

## Attribution and License

These proofs accompany Chakshu Gupta's
[degree-sequence paper](https://arxiv.org/abs/2607.05567) and its
[height-function companion](https://arxiv.org/abs/2606.22614). The mathematical
background and external inputs are attributed in those papers. Mathlib is a
third-party dependency under its own license.

The exported project uses the repository's
[CC BY 4.0 license](../../../../LICENSE).
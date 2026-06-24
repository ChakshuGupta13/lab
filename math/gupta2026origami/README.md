# Height functions on the $m \times n$ Miura-ori flip graph: degree sequence and diameter

Verification code for the paper *"Height functions on the $m \times n$ Miura-ori
flip graph: degree sequence and diameter"* (Chakshu Gupta, Georgia Institute of
Technology).

**Paper**: [arXiv:2606.22614](https://arxiv.org/abs/2606.22614) (math.CO, 2026).

## The problem

The **origami flip graph** $\mathrm{OFG}(C)$ of a flat-foldable crease pattern $C$
has the flat-foldable mountain–valley assignments of $C$ as its vertices, with an
edge between two assignments that differ by a single **face flip** (switching every
crease bordering one face). For the $m \times n$ Miura-ori $M_{m,n}$,
Christensen–Hull et al. (2025,
[arXiv:2506.19700](https://arxiv.org/abs/2506.19700)) determine the degree
sequence and diameter of $\mathrm{OFG}(M_{2,n})$ and leave the case $m \ge 3$
open. This paper resolves the degree sequence up to degree five and a closed-form
diameter lower bound, for all $m, n$.

## The method — height functions

By the Ginepro–Hull bijection, the flat-foldable assignments of $M_{m,n}$
correspond to the proper 3-colourings of the grid graph $G_{m,n}$ (one corner
fixed), with a face flip corresponding to recolouring a single grid vertex, so

$$\mathrm{OFG}(M_{m,n}) \;\cong\; R_3(G_{m,n}) \,/\, (\mathbb{Z}/3\mathbb{Z}),$$

the 3-colouring reconfiguration graph of the grid modulo the global colour
rotation. Each proper 3-colouring lifts to an integer **height function** $h$ with
$|h(u) - h(v)| = 1$ across every edge and colour $= h \bmod 3$. The key lemma is:

> **the degree of a vertex equals the number of strict local extrema of its height function.**

Every degree/diameter question thus becomes combinatorics of $\pm 1$ height
functions on the grid.

## Results

| Result | Statement | Status |
|---|---|---|
| Degree–extrema lemma | $\deg(h) = $ number of strict local extrema of $h$ | proved + verified |
| Cone Lemma | a unique local max forces $h$ to be a distance cone from it | proved + verified |
| Cone Classification | the $mn$ cones have degree $2/3/5$ for a corner / edge / interior apex | proved + verified |
| Envelope Lemma | every $h$ is the lower envelope of the cones at its strict local minima | proved + verified |
| $\#\{\deg 2\}$ | $= 4$ for all $m,n \ge 2$ (the four opposite-corner gradients) | proved + verified |
| $\#\{\deg 3\}$ | $= 4(m+n-4)$ for all $m,n \ge 2$ | proved + verified |
| Degree-4 class | $= $ exactly the vertices with two minima and two maxima | proved + verified |
| $\#\{\deg 4\}$ | $= 2m^2 + 2n^2 + 6mn - 10(m+n) - 4$ for $\min(m,n) \ge 3$ | proved + verified |
| $\#\{\deg 5\}$ | $= \tfrac13(2m^3 + 2n^3 + 6m^2 + 6n^2 + 150mn - 392m - 392n + 792)$ for $\min(m,n) \ge 4$ | proved + verified |
| Diameter lower bound | $\mathrm{diam} \ge D(m,n) := \min_K \sum_{i,j} \lvert i+j-K \rvert$ for all $m,n \ge 1$ | proved + verified |
| Diameter | $\mathrm{diam} = D(m,n)$; proved for $m=2$, reduced for $m \ge 3$ to an extremal inequality for $1$-Lipschitz functions (the slowest-chain conjecture) | $m=2$ proved; $m\ge3$ open, verified by enumeration |

The single closed form $D(m,n)$ unifies the separately known diameters
$\lceil n^2/2 \rceil$ ($m=2$), $\lfloor 3n^2/4 \rfloor + 2$ ($m=3$),
$n^2 + 4 + [n\text{ odd}]$ ($m=4$), and the diagonal $(m^3-m)/3$.

## Code

Pure Python with [NetworkX](https://networkx.org); every script runs in seconds on
a laptop.

```sh
pip install networkx
cd code
python3 verify_claims.py        # master: reproduces every numerical claim (PASS/FAIL)
```

| File | Role |
|---|---|
| `ofg.py` | builds $\mathrm{OFG}(M_{m,n})$ by direct enumeration of 3-colourings |
| `heights.py` | height-function lift; degree $=$ number of strict local extrema |
| `verify_claims.py` | **master** — reproduces every numerical claim (M$_{2,n}$, degree counts, diameter forms, symmetry) |
| `verify_deg2_theorem.py` | the four degree-2 corner gradients |
| `verify_cone_classification.py` | cone degrees $2/3/5$ by apex position |
| `verify_envelope_lemma.py` | every height function is a lower envelope of cones |
| `verify_deg4_formula.py` | degree-4 count, including held-out grids |
| `verify_deg4_families.py` | degree-4 cone-pair families |
| `verify_deg4_conjecture.py` | degree-4 structural characterisation |
| `verify_deg5_count.py` | degree-5 count via the cone-pair bijection |
| `verify_diameter_upper.py` | the extremal inequality $\max_\phi \mathrm{disp}(\phi) = D(m,n)$ (diameter upper bound) on small grids |
| `exact_diameter.py` | exact diameter via iFUB for the larger $m \ge 3$ grids |

## Validation

The enumerator reproduces all four of Christensen–Hull et al.'s $M_{2,n}$ results
exactly for $n = 1..7$ — vertex count $2 \cdot 3^{n-1}$, edge count
$8(n+1)3^{n-3}$, diameter $\lceil n^2/2 \rceil$, and the full degree distribution
cell-for-cell — which is what justifies trusting it on the open $m \ge 3$ cases.

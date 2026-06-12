# Architecture-Induced Recoverability Bias in Differentiable Symbolic Regression

Code and paper for [arXiv:2604.23256](https://arxiv.org/abs/2604.23256)
(MLSP 2026, Atlanta).

**Authors**: Chakshu Gupta, Theodore J. LaGrow (Georgia Institute of Technology)

## Visualization

![Loss landscape of EML input routing](code/landscape.png)

The image above is generated from this paper's own operator and training data
by [`code/viz_landscape.py`](code/viz_landscape.py). It is a **decorative
visualization, not a figure from the paper.** It shows the loss landscape of a
deliberately simplified toy problem: a single EML node,
`eml(x, y) = exp(x) - ln(y)`, whose two inputs are blended between the
variables `x` and `y` by a pair of logits `(a, b)`:

```
inL = sigmoid(a) * x + (1 - sigmoid(a)) * y      # left input  -> exp
inR = sigmoid(b) * x + (1 - sigmoid(b)) * y      # right input -> ln
loss(a, b) = mean over x, y in [1, 3] of (eml(inL, inR) - eml(x, y))^2
```

Brightness is closeness to the optimum; the `(a, b)` plane is the routing
space.

**What this is and is not.** This toy *shares the paper's operator and
training data*, but its routing is **not** the paper's routing: the paper uses
a 3-way softmax selection over `{1, x, child}` inside multi-level trees, while
this uses a 2-way continuous blend between `x` and `y` with no constant
channel and no tree. So it is "inspired by", not "a reduction of", the paper's
mechanism. The visible large-scale asymmetry of the landscape is driven mostly
by a simple fact — `exp` of a wrong input explodes while `ln` stays bounded on
`[1, 3]` — i.e. the operator's *value* asymmetry; the *gradient* asymmetry
that the paper studies (its factor F4) governs the fine basin curvature near
the optimum, which is not resolvable at this scale. The image is meant to be
pretty and made from real ingredients, nothing more — it is not evidence for
any claim in the paper.

It doubles as the paper's tile image on the author's site
([haveli](https://haveli.dev-e79.workers.dev/people/chakshu/research/)); the
motivation was to generate that tile from the paper's own operator rather than
use stock imagery. (EML is the exp-minus-log operator of Odrzywolek 2026,
arXiv:2603.21852, which this paper studies; the name "EML" is used there.)

Regenerate:

```bash
cd code && python viz_landscape.py --out landscape.png
```

## Key finding

In gradient-based symbolic regression using parameterized binary trees,
architectures with **provably identical expressiveness** produce recovery
rates from 0% to 100% depending on the interaction between the architecture's
variable routing, the target's tree topology, and the operator's gradient
asymmetry.

Three architectures are compared — all sharing the same operator, target
language, and tree depth — across three operators (EML, SML, RML) and
multiple target topologies. The ranking reverses across targets and
operators: one structure recovers a target at 100% while another scores 0%,
and which structure wins depends on the specific combination.

## Repository structure

```
paper/
  main.tex / main.pdf       LaTeX source + compiled PDF (arXiv version)
  content/                  Section .tex files (input by main.tex)
  references.bib            Bibliography
  mlspconf.sty, IEEEbib.bst LaTeX style files
  fig_*.pdf                 Figures

code/
  Core
    eml.py                  EML operator (NumPy, complex arithmetic)
    master_formula.py       Eq.6 / V16 / Hybrid architectures (PyTorch)
    formalization.py        Architecture / target / operator definitions
    reproduce.py            Self-contained quick-start runner
    results.json            Pre-computed results for Tables 1 and 5

  Experiment runners
    phase1_runner.py        Phase 1 sweep (Table 1: heatmap matrix)
    phase2_runner.py        Phase 2 sweep (Table 3: rho_LR / branch-ratio)
    d4_matrix_runner.py     Depth-4 robustness sweep (Section 5)

  Pre-computed results (used by plots and tables)
    phase1_results/
    phase2_results/
    d4_matrix_results/
    taxonomy_results/
    panel_a_trajectory_cache.json

  Figure generators
    plot_heatmap.py         Fig. 1 (recoverability heatmap)
    plot_gradient_2panel.py Fig. 2 (gradient trajectories)
    plot_branchratio.py     Fig. 3 (branch-ratio vs rho_LR)
    plot_selector.py        Selector / supplementary
    viz_landscape.py        Routing loss-landscape tile (README / website)
```

## Requirements

- Python 3.9+
- PyTorch >= 2.0 (CPU is sufficient)
- NumPy
- matplotlib with TeX Live (optional, for figure regeneration only)

## Quick start

```bash
cd code

# Single cell (V16 on the paper's EML target, depth 3, 64 seeds):
python reproduce.py --arch v16 --target paper --depth 3 --seeds 64

# View pre-computed results matching the paper's tables:
python reproduce.py --mode verify
```

Each cell runs 64 seeds x 4 init strategies = 256 independent training runs
(Eq.6 uses 1 strategy = 64 runs). A single cell takes 1-5 minutes on CPU.
The full heatmap (48 cells) takes several hours.

## Architectures

| Name   | Variable routing                                     | Params (d=3) |
|--------|------------------------------------------------------|--------------|
| Eq.6   | x at all internal nodes + leaves; y at leaves only   | 42           |
| V16    | x, y at leaves only; internal nodes blend {1, child} | 38           |
| Hybrid | V16 body + 4-way softmax {1, x, y, child} at root    | 44           |

All three can represent the same set of depth-3 EML trees (identical
expressiveness). They differ only in how variables enter the tree, which
changes the optimization landscape.

## Erratum (vs. submitted MLSP version)

The paper text (Sections 3, 4, 6) states the training domain as `[-3,3]^2`.
The actual experiments use `[1,3]^2` (step 0.1, 21x21 = 441 grid points),
as reflected in the code and `results.json`. The positive domain is required
because EML targets involve `log(x)`. To be corrected in arXiv v2 /
camera-ready.

## Citation

```bibtex
@inproceedings{gupta2026architecture,
  title     = {Architecture-Induced Recoverability Bias in Differentiable
               Symbolic Regression},
  author    = {Gupta, Chakshu and LaGrow, Theodore J.},
  booktitle = {2026 IEEE International Workshop on Machine Learning for
               Signal Processing (MLSP)},
  year      = {2026},
  address   = {Atlanta, GA, USA},
  eprint    = {2604.23256},
  archivePrefix = {arXiv}
}
```

## License

CC BY 4.0. See [LICENSE](../../LICENSE).

## Acknowledgment

This work builds on the EML (Exp-Minus-Log) Sheffer operator introduced by
Odrzywolek (2026), [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).

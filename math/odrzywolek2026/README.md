# Why Architecture Choice Matters in Symbolic Regression

Code and paper for [arXiv:2604.23256](https://arxiv.org/abs/2604.23256).

## Key finding

In gradient-based symbolic regression using parameterized binary trees,
architectures with **provably identical expressiveness** produce recovery rates
from 0% to 100% depending on the interaction between the architecture's
variable routing, the target's tree topology, and the operator's gradient
asymmetry.

Three architectures are compared — all sharing the same operator, target
language, and tree depth — across three operators (EML, SML, RML) and
multiple target topologies. The ranking reverses across targets and operators:
one structure recovers a target at 100% while another scores 0%, and which
structure wins depends on the specific combination.

## Repository structure

```
code/
  eml.py             # EML operator (NumPy, complex arithmetic)
  master_formula.py  # Three architectures: Eq.6, V16, Hybrid (PyTorch)
  reproduce.py       # Self-contained experiment runner
  results.json       # Pre-computed results for Tables 1 and 5
paper/
  main.tex           # LaTeX source (arXiv version)
  main.pdf           # Compiled PDF
  fig_odrzywolek2026_heatmap.pdf
  fig_odrzywolek2026_gradient.pdf
```

## Requirements

- Python 3.9+
- PyTorch >= 2.0 (CPU is sufficient)
- NumPy
- matplotlib with TeX Live (optional, for figure generation only)

## Erratum

The paper text (Sections 3, 4, 6) states the training domain as $[-3,3]^2$.
The actual experiments use $[1,3]^2$ (step 0.1, 21×21 = 441 grid points),
as reflected in the code and `results.json`. The positive domain is required
because EML targets involve $\log(x)$. This will be corrected in arXiv v2.

## Quick start

```bash
cd code

# Run a single cell (V16 on the paper's EML target, depth 3, 64 seeds):
python reproduce.py --arch v16 --target paper --depth 3 --seeds 64

# Run the full heatmap matrix (Table 1 / Figure 1):
python reproduce.py --mode heatmap --seeds 64

# View pre-computed results:
python reproduce.py --mode verify

# Reproduce gradient trajectory (Figure 2):
python reproduce.py --mode gradient --seeds 10
```

Each cell runs 64 seeds × 4 init strategies = 256 independent training runs
(Eq.6 uses 1 strategy = 64 runs). A single cell takes 1–5 minutes on CPU.
The full heatmap (48 cells) takes several hours.

## Architectures

| Name | Variable routing | Parameters (d=3) |
|------|-----------------|-------------------|
| Eq.6 | x at all internal nodes + leaves; y at leaves only | 42 |
| V16 | x, y at leaves only; internal nodes blend {1, child} | 38 |
| Hybrid | V16 body + 4-way softmax {1, x, y, child} at root | 44 |

All three can represent the same set of depth-3 EML trees (identical
expressiveness). They differ only in how variables enter the tree, which
changes the optimization landscape.

## Citation

```bibtex
@article{gupta2026architecture,
  title   = {Why Architecture Choice Matters in Symbolic Regression},
  author  = {Gupta, Chakshu},
  journal = {arXiv preprint arXiv:2604.23256},
  year    = {2026}
}
```

## License

Code: MIT. Paper: CC BY 4.0.

## Acknowledgment

This work builds on the EML (Exp-Minus-Log) Sheffer operator introduced by
Odrzywolek (2026), [arXiv:2603.21852](https://arxiv.org/abs/2603.21852).

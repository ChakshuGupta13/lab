# Collision-based logic in Lenia and its composition boundary

Accompanying code for the paper *"Collision-based logic in Lenia and its
composition boundary"* (Chakshu Gupta, Georgia Institute of Technology).

**Preprint**: [arXiv:2609.01348](https://arxiv.org/abs/2609.01348).

## Overview

[Lenia](https://chakazul.github.io/lenia.html) is a continuous cellular
automaton whose smooth update rule spontaneously produces hundreds of lifelike,
self-organizing patterns, among them the self-propelling **Orbium** glider. This
code builds a logic gate from collisions of the Orbium glider and then measures
how far that computation composes.

The gate is **emergent**: it arises from collisions between the automaton's own
self-organized patterns under a fixed rule, rather than from logic trained into
the update rule. The code demonstrates a working gate and a two-gate chain, and
then maps the boundary at which composition into a general circuit breaks down.

## Results

| Artifact | What it shows | Scripts |
|---|---|---|
| **INHIBIT gate** | `out = S AND NOT C`: a control glider `C` deflects the signal `S` off its output track. Blocks at all 24 breathing phases across nine integer offsets (`b = 0..8`). | `gate.py`, `gate_robustness.py` |
| **AND-NOT cascade** | Two gates in series, one signal line and two controls, correct on all eight input combinations. | `cascade.py` |
| **Generality** | Of the patterns searched across four continuous-CA rule types (standard Lenia, asymptotic Lenia, SmoothLife, Glaberish), Orbium is the only one whose collision leaves both copies intact. | `orbium_collide.py`, `orbium_phase_map.py` |
| **Composition boundary** | Beyond a single straight chain, a deflected signal is not restored to a fixed landing position (8–16 px spread vs a 9 px gate window), and no reusable absorber for surviving gliders was found. Universality is left undetermined. | `depth2.py`, `routing.py`, `clocking.py`, `eater_search.py`, `static_eater.py` |

## Requirements

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r code/requirements.txt      # jax, numpy, scipy, matplotlib
```

The simulator (`lenia.py`) uses JAX with double precision (`float64`); colliding
gliders are sensitive to rounding.

## Reproduce

Run from the `code/` directory (scripts read data from `code/assets/`):

```bash
cd code

python orbium.py              # the canonical Orbium glider travels in a straight line
python orbium_collide.py headon   # Orbium collision sweep (scatter vs annihilate)
python orbium_phase_map.py headon # operating map: outcome over phase x impact parameter
python gate.py                # INHIBIT gate: truth table + 9 px phase-robust window
python gate_robustness.py     # delta-phi = 1 sweep over 24 phases: b = 0..8 zero-leak
python cascade.py             # two-gate AND-NOT cascade: 8/8 truth table
python depth2.py              # turned-signal landing spread vs the gate window (restoration)
python routing.py             # routing / turn reproducibility (depth 1)
python clocking.py            # clocked routing phase-tolerance vs hop distance
python crossing.py            # signal crossing (needs temporal scheduling)
python fanout.py              # fanout 1 -> 2 (depth 1, needs a helper stream)
python eater_search.py        # eater search across geometries: no phase-robust absorber
python static_eater.py        # static still-life "eater" / wall search (none found)
python plot_paper_figures.py  # regenerate the paper figures
```

The Orbium seed is parsed verbatim (`parse_orbium.py`) from Bert Chan's
[Chakazul/Lenia](https://github.com/Chakazul/Lenia) reference implementation
(`R=13, mu=0.15, sigma=0.014, dt=0.1`) and is tracked at
`code/assets/orbium_seeds.npz`. Evidence figures produced by the scripts are in
`code/assets/`.

## Layout

```
code/
  lenia.py               faithful Lenia simulator (JAX, float64, FFT convolution)
  orbium.py              the canonical Orbium glider
  parse_orbium.py        parse the Orbium seed from Chan's reference file
  collide.py             substrate-general two-glider collision harness (exact D4 symmetries)
  orbium_collide.py      Orbium impact-parameter sweep
  orbium_phase_map.py    phase x impact-parameter operating map
  gate.py                the INHIBIT gate and its readout
  gate_robustness.py     phase-robustness sweep of the gate
  cascade.py             two-gate AND-NOT cascade
  depth2.py, routing.py, clocking.py, crossing.py, fanout.py   composition primitives and boundary
  eater_search.py, static_eater.py, static_eat.py, absorb.py   survivor-absorption search
  plot_paper_figures.py  regenerate the figures in the paper
  assets/                Orbium seed data and evidence figures
```

## Citation

```bibtex
@misc{gupta2026lenia,
  author        = {Gupta, Chakshu},
  title         = {Collision-based logic in Lenia and its composition boundary},
  year          = {2026},
  eprint        = {2609.01348},
  archivePrefix = {arXiv},
  primaryClass  = {cs.ET},
  doi           = {10.48550/arXiv.2609.01348}
}
```

## License

Released under the terms of the repository's [LICENSE](../../LICENSE).

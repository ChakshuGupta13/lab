# Lab

Public code artifacts from research work.

## Papers

Implementations and supporting code for specific research papers.

### math

| Anchor | Description |
|--------|-------------|
| [gupta2026architecture](math/gupta2026architecture/) | Architecture-induced recoverability bias in differentiable symbolic regression ([arXiv:2604.23256](https://arxiv.org/abs/2604.23256), MLSP 2026) |
| [gupta2026origami](math/gupta2026origami/) | Degree sequence and diameter of the $m\times n$ Miura-ori origami flip graph, via height functions ([arXiv:2606.22614](https://arxiv.org/abs/2606.22614)) |
| [gupta2026saturation](https://github.com/ChakshuGupta13/lab/blob/main/math/gupta2026saturation) | The saturation number is not bounded by the harmonic index ([arXiv:2606.15761](https://arxiv.org/abs/2606.15761)) — refutation of TxGraffiti Conjecture 4 ([arXiv:2507.17780](https://arxiv.org/abs/2507.17780)) |

### cryptanalysis

| Anchor | Description |
|--------|-------------|
| [sanadhya2008](cryptanalysis/sanadhya2008/) | 22- and 24-step SHA-256 collision attacks |
| [mendel2011](cryptanalysis/mendel2011/) | 27/32-step SHA-256 collisions via SAT + [novel findings](cryptanalysis/mendel2011/README.md#additional-findings) (32K collision family, 10× phase-seed speedup, W[7] characterization) |
| [mendel2013improving](cryptanalysis/mendel2013improving/) | 28/31/38-step SHA-256 collisions via SAT on dense characteristics |
| [eichlseder2014](cryptanalysis/eichlseder2014/) | 38-step SHA-512 SFS collision via SAT + [GnD plateau analysis](cryptanalysis/eichlseder2014/PLATEAU-ANALYSIS.md) |
| [lamberger2011](cryptanalysis/lamberger2011/) | 46-step SHA-256 second-order differential: [message-modification analysis](cryptanalysis/lamberger2011/MSG-MOD-ANALYSIS.md) (two-layer decomposition, revised 2⁴⁷·⁶ complexity) |
| [de2006finding-workfactor](cryptanalysis/de2006finding-workfactor/) | SHA-1 work-factor calculator (De Cannière–Rechberger 2006 §III-C), first independent reimplementation |
| [gupta2026rank](cryptanalysis/gupta2026rank/) | Rank ceiling for twiddle-perturbation faults on the forward NTT of ML-KEM and ML-DSA ([ePrint 2026/1188](https://eprint.iacr.org/2026/1188)) |
| [zhang2026-verification](cryptanalysis/zhang2026-verification/) | SFS collision verification + corrected Table 8 characteristic for Zhang–Li–Gao–Wang, EUROCRYPT 2026 |

## Tools

Stand-alone utilities not tied to a single paper. Organized by domain.

### quantum

| Tool | Description |
|------|-------------|
| [qbudget](tools/quantum/qbudget/) | Report T-count and AND-uncomputation accounting for compute-uncompute Grover oracles in Qiskit or Q# |

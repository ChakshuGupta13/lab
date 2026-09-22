# Verification Contract

`code/validate.py` has no command-line arguments. It exits zero only when all checks below pass.

## Fixed model

- Ambient temperature: $T_0=298.15\,\mathrm{K}$.
- Dimensionless integration interval: $[10^{-12},80]$.
- Finite-band grid: $T_h\in\{1000,1480,2673\}\,\mathrm{K}$, $x_g\in\{10^{-6},0.5,2,5.21,12\}$, and $W_d\in\{0.5,3,40\}$, retaining the 45 members with $x_g+W_d\le75$.
- Sub-gap absorptances: $a\in\{0,0.07,0.25,0.5,0.75,1\}$ for the 270 covariance checks and $a\in\{0,0.05,\ldots,1\}$ for the 945-label sweep.
- Table 1 rows: four ideal above-gap steps, specified by source temperature and bandgap in `code/validate.py`.

## Success conditions

The script verifies:

1. The affine finite-band covariance agrees with an independently integrated covariance to less than $10^{-11}$ on all 270 grid points.
2. The full-spectrum benchmark agrees with the Petela factor at each validation temperature.
3. Every resolvable grid sign agrees with the affine threshold, and all 15 resolvable interior thresholds agree with bisection to less than $10^{-8}$.
4. The 21-point sweep has 870 resolved and 75 below-floor labels, using a $10^{-13}$ sign floor.
5. The four Table 1 benchmarks round to the values reported in the manuscript.

The integration uses SciPy adaptive quadrature. The exact least-significant digits can depend on the numerical library, while the asserted tolerances define the contract.
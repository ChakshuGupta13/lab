# Exergy-Correction Bounds for Selective Absorbers

Numerical verification for *Sharp exergy-correction bounds for selective absorbers under linear allocation*.

The paper studies the absorbed-spectrum benchmark

$$
\psi_\alpha = \frac{\int \alpha(x)w(x)f(x)\,\mathrm{d}x}
{\int \alpha(x)w(x)\,\mathrm{d}x}
$$

for a finite-band absorptance. The verifier checks the affine covariance formula, its threshold sign classification, and the four ideal above-gap rows in Table 1.

## Run

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python3 code/validate.py
```

A successful run prints:

- 45 finite-band family members and 270 independent covariance checks;
- the worst covariance discrepancy, $1.85\times 10^{-14}$;
- 15 interior thresholds checked by bisection, with worst discrepancy $7.83\times 10^{-12}$;
- 870 resolved and 75 unresolved labels in the 21-point sign sweep; and
- the full-spectrum and absorbed-spectrum factors for the four Table 1 rows.

## Scope

The implementation evaluates the paper's linear-allocation convention. It does not model an absorber as a physical attenuation law for the photon occupation number, and it does not infer device spectra from published efficiency measurements.

The four table rows use the temperature-bandgap pairs reported in [Thermophotovoltaic efficiency of 40%](https://doi.org/10.1038/s41586-022-04473-y) and [Present Efficiencies and Future Opportunities in Thermophotovoltaics](https://doi.org/10.1016/j.joule.2020.06.021).

## Attribution

The whole-spectrum exergy factor follows [Exergy of Heat Radiation](https://doi.org/10.1115/1.3687092). The per-mode exergy expression follows [The Exergy of Incoherent Electromagnetic Radiation](https://doi.org/10.1088/0031-8949/26/4/009). This package verifies the finite-band specialization and its numerical checks; it does not replace either foundational result.

## Files

| File | Purpose |
|---|---|
| `code/validate.py` | Reproduces the numerical-validation grid and Table 1. |
| `SPEC.md` | Defines the verifier's fixed inputs and success contract. |
| `requirements.txt` | Python dependencies. |

## License

See the repository [CC BY 4.0 license](https://github.com/ChakshuGupta13/lab/blob/main/LICENSE).
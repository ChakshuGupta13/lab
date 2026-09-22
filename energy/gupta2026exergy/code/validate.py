#!/usr/bin/env python3
"""Reproduce the numerical checks for selective-absorber exergy bounds."""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad


T_AMBIENT = 298.15
X_MAX = 80.0
X_MIN = 1e-12
SIGN_FLOOR = 1e-13
K_B_EV_PER_K = 8.617333262145e-5

TEMPERATURES = (1000.0, 1480.0, 2673.0)
GAPS = (1e-6, 0.5, 2.0, 5.21, 12.0)
WIDTHS = (0.5, 3.0, 40.0)
ABSORPTANCES = (0.0, 0.07, 0.25, 0.5, 0.75, 1.0)

TABLE_ROWS = (
    ("1.4/1.2 eV tandem", 2673.0, 1.20, 0.8513, 0.8715, 2.37),
    ("1.2/1.0 eV tandem", 2400.0, 1.00, 0.8344, 0.8558, 2.56),
    ("LM-InGaAs", 1480.0, 0.75, 0.7319, 0.7706, 5.28),
    ("Si", 2300.0, 1.12, 0.8273, 0.8518, 2.97),
)


def integrate(function, lower, upper):
    if upper <= lower:
        return 0.0
    return quad(function, lower, upper, limit=800)[0]


def entropy(occupation):
    if occupation <= 1e-300:
        return 0.0
    return (1.0 + occupation) * np.log1p(occupation) - occupation * np.log(occupation)


def hot_occupation(x):
    return 1.0 / np.expm1(x)


def ambient_occupation(x, ratio):
    exponent = x / ratio
    if exponent >= 700.0:
        return np.exp(-exponent)
    return 1.0 / np.expm1(exponent)


def decomposition_kernels(temperature):
    ratio = T_AMBIENT / temperature

    def weight(x):
        return x**3 * hot_occupation(x)

    def mode_exergy(x):
        hot = hot_occupation(x)
        ambient = ambient_occupation(x, ratio)
        hot_availability = x**3 * hot - ratio * x**2 * entropy(hot)
        ambient_availability = x**3 * ambient - ratio * x**2 * entropy(ambient)
        return hot_availability - ambient_availability

    return weight, mode_exergy


def finite_band_moments(temperature, gap, width):
    if temperature < 1000.0:
        raise ValueError("temperature must be at least 1000 K for the validation grid")
    if not X_MIN < gap < X_MAX:
        raise ValueError("gap must lie inside the integration interval")
    if not np.spacing(gap) < width and gap + width <= gap:
        raise ValueError("bandwidth must be representable at the requested gap")
    if gap + width >= X_MAX:
        raise ValueError("band must end below the finite integration cutoff")

    weight, mode_exergy = decomposition_kernels(temperature)
    below_weight = integrate(weight, X_MIN, gap)
    below_exergy = integrate(mode_exergy, X_MIN, gap)
    above_weight = integrate(weight, gap, X_MAX)
    above_exergy = integrate(mode_exergy, gap, X_MAX)
    band_weight = integrate(weight, gap, gap + width)
    band_exergy = integrate(mode_exergy, gap, gap + width)
    total_weight = below_weight + above_weight
    full_benchmark = (below_exergy + above_exergy) / total_weight
    positive_intercept = band_exergy - band_weight * full_benchmark
    slope = (
        below_weight
        * above_weight
        * (above_exergy / above_weight - below_exergy / below_weight)
        / total_weight
    )
    return positive_intercept, slope, total_weight, full_benchmark


def direct_covariance(temperature, gap, width, absorptance):
    ratio = T_AMBIENT / temperature

    def direct_entropy(occupation):
        if occupation <= 1e-300:
            return 0.0
        return (1.0 + occupation) * np.log1p(occupation) - occupation * np.log(occupation)

    def direct_hot_occupation(x):
        return 1.0 / np.expm1(x)

    def direct_ambient_occupation(x):
        exponent = x / ratio
        if exponent >= 700.0:
            return np.exp(-exponent)
        return 1.0 / np.expm1(exponent)

    def availability(x, occupation):
        return x**3 * occupation - ratio * x**2 * direct_entropy(occupation)

    def direct_weight(x):
        return x**3 * direct_hot_occupation(x)

    def direct_mode_exergy(x):
        return availability(x, direct_hot_occupation(x)) - availability(x, direct_ambient_occupation(x))

    def alpha(x):
        if x < gap:
            return absorptance
        if x < gap + width:
            return 1.0
        return 0.0

    cuts = tuple(sorted({X_MIN, gap, min(gap + width, X_MAX), X_MAX}))

    def integral(function):
        return sum(integrate(function, lower, upper) for lower, upper in zip(cuts, cuts[1:]))

    total_weight = integral(direct_weight)
    mean = lambda function: integral(lambda x: direct_weight(x) * function(x)) / total_weight
    exergy_ratio = lambda x: direct_mode_exergy(x) / direct_weight(x)
    mean_alpha = mean(alpha)
    mean_exergy_ratio = mean(exergy_ratio)
    mean_product = mean(lambda x: alpha(x) * exergy_ratio(x))
    return mean_product - mean_alpha * mean_exergy_ratio


def selected_benchmark(temperature, gap):
    weight, mode_exergy = decomposition_kernels(temperature)
    full_weight = integrate(weight, X_MIN, X_MAX)
    full_exergy = integrate(mode_exergy, X_MIN, X_MAX)
    selected_weight = integrate(weight, gap, X_MAX)
    selected_exergy = integrate(mode_exergy, gap, X_MAX)
    return full_exergy / full_weight, selected_exergy / selected_weight


def petela(temperature):
    ratio = T_AMBIENT / temperature
    return 1.0 - 4.0 * ratio / 3.0 + ratio**4 / 3.0


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def validation_grid():
    return tuple(
        (temperature, gap, width)
        for temperature in TEMPERATURES
        for gap in GAPS
        for width in WIDTHS
        if gap + width <= X_MAX - 5.0
    )


def validate_covariance_grid():
    grid = validation_grid()
    require(len(grid) == 45, f"expected 45 grid members, got {len(grid)}")

    worst_covariance_error = 0.0
    for temperature, gap, width in grid:
        for absorptance in ABSORPTANCES:
            positive_intercept, slope, total_weight, _ = finite_band_moments(temperature, gap, width)
            decomposed = (positive_intercept - absorptance * slope) / total_weight
            independent = direct_covariance(temperature, gap, width, absorptance)
            worst_covariance_error = max(worst_covariance_error, abs(decomposed - independent))
    require(worst_covariance_error < 1e-11, f"covariance mismatch {worst_covariance_error:.3e}")

    for temperature in TEMPERATURES:
        _, _, _, full_benchmark = finite_band_moments(temperature, 2.0, 3.0)
        require(abs(full_benchmark - petela(temperature)) < 1e-12, "full-spectrum anchor failed")

    interior_thresholds = 0
    worst_bisection_error = 0.0
    resolved_signs = 0
    unresolved_signs = 0
    for temperature, gap, width in grid:
        positive_intercept, slope, total_weight, _ = finite_band_moments(temperature, gap, width)
        require(slope > 0.0, "finite-band slope must be positive")

        for absorptance in ABSORPTANCES:
            predicted = (positive_intercept - absorptance * slope) / total_weight
            observed = direct_covariance(temperature, gap, width, absorptance)
            if abs(observed) <= SIGN_FLOOR:
                require(abs(predicted) <= 1e-12, "near-zero direct covariance lacks a near-zero prediction")
                unresolved_signs += 1
            else:
                require((predicted > 0.0) == (observed > 0.0), "covariance sign mismatch")
                resolved_signs += 1

        threshold = positive_intercept / slope
        if 0.0 < threshold < 1.0 and abs((positive_intercept - slope) / total_weight) > SIGN_FLOOR:
            lower, upper = 0.0, 1.0
            for _ in range(60):
                midpoint = (lower + upper) / 2.0
                if direct_covariance(temperature, gap, width, midpoint) > 0.0:
                    lower = midpoint
                else:
                    upper = midpoint
            worst_bisection_error = max(worst_bisection_error, abs((lower + upper) / 2.0 - threshold))
            interior_thresholds += 1

    require((resolved_signs, unresolved_signs) == (240, 30), "six-point sign counts changed")
    require(interior_thresholds == 15, f"expected 15 interior thresholds, got {interior_thresholds}")
    require(worst_bisection_error < 1e-8, f"bisection mismatch {worst_bisection_error:.3e}")

    classified = 0
    unresolved = 0
    for temperature, gap, width in grid:
        positive_intercept, slope, total_weight, _ = finite_band_moments(temperature, gap, width)
        for index in range(21):
            absorptance = index / 20.0
            covariance = abs((positive_intercept - absorptance * slope) / total_weight)
            if covariance <= SIGN_FLOOR:
                unresolved += 1
            else:
                classified += 1

    require((classified, unresolved) == (870, 75), "21-point sign counts changed")
    return worst_covariance_error, worst_bisection_error, classified, unresolved


def validate_table():
    computed_rows = []
    for name, temperature, bandgap_ev, expected_full, expected_selected, expected_overstatement in TABLE_ROWS:
        gap = bandgap_ev / (K_B_EV_PER_K * temperature)
        full_benchmark, selected = selected_benchmark(temperature, gap)
        overstatement = 100.0 * (selected - full_benchmark) / full_benchmark
        require(abs(full_benchmark - expected_full) < 5e-5, f"{name}: full-spectrum benchmark changed")
        require(abs(selected - expected_selected) < 5e-5, f"{name}: selected benchmark changed")
        require(abs(overstatement - expected_overstatement) < 0.005, f"{name}: overstatement changed")
        computed_rows.append((name, full_benchmark, selected, overstatement))
    return computed_rows


def main():
    worst_covariance_error, worst_bisection_error, classified, unresolved = validate_covariance_grid()
    table_rows = validate_table()

    print("Numerical validation passed")
    print(f"45 finite-band family members; 270 covariance checks; worst absolute error {worst_covariance_error:.2e}")
    print(f"15 interior thresholds; worst bisection error {worst_bisection_error:.2e}")
    print(f"21-point sign sweep: {classified} classified, {unresolved} unresolved below {SIGN_FLOOR:.0e}")
    print()
    print(f"{'row':<20} {'psi':>8} {'psi_alpha':>10} {'overstatement':>15}")
    for name, full_benchmark, selected, overstatement in table_rows:
        print(f"{name:<20} {full_benchmark:8.4f} {selected:10.4f} {overstatement:14.2f}%")


if __name__ == "__main__":
    main()

"""EML (Exp-Minus-Log) Sheffer operator for elementary functions.

Implements eml(x, y) = exp(x) - ln(y) over complex numbers with IEEE 754
extended-real semantics (ln 0 = -inf, exp(-inf) = 0).

Reference: Odrzywolek 2026, arXiv:2603.21852v2, Eq. (3).
"""

import numpy as np


def eml(x, y):
    """EML Sheffer operator: eml(x, y) = exp(x) - ln(y).

    Operates over complex128 internally.  Real inputs are promoted; the
    result is cast back to real when the imaginary part is negligible.

    Uses numpy for proper inf/-inf handling (pure-Python math raises on
    special floats).

    Branch convention: the internal log uses the *conjugate branch*,
    i.e. log_eml(y) = conj(log(conj(y))).  For positive reals this is
    identical to the standard principal log.  For negative reals it gives
    Im = -pi instead of +pi, which compensates for the 1/z inversion in
    the ln_eml chain (Eq. 5) and makes ln_eml agree with the standard
    principal branch.  See paper Sect. 4.1: "redefine the branch for EML
    itself in such a way that ln z [...] follows standard implementation
    of principal branch."
    """
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    return np.exp(x) - np.conj(np.log(np.conj(y)))


# ---------------------------------------------------------------------------
# Bootstrapping chain: derive standard primitives from {1, eml}
# Order follows the paper's Fig. 1 spiral (innermost first).
# ---------------------------------------------------------------------------

# --- Constants ---

def const_e():
    """e = eml(1, 1) = exp(1) - ln(1) = e - 0."""
    return eml(1, 1)


def const_zero():
    """0 = eml(1, eml(1, 1))  [i.e. ln(e) via the ln identity below, = 1,
    then use 1 - 1 ... but simplest: 0 = exp(1) - e = e - e.]

    Actually: 0 via ln(1).  ln(1) = eml(1, eml(eml(1, 1), 1)).
    eml(eml(1, 1), 1) = exp(e) - ln(1) = exp(e).
    eml(1, exp(e)) = exp(1) - ln(exp(e)) = e - e = 0.
    """
    return ln_eml(1)


# --- Unary functions ---

def exp_eml(x):
    """exp(x) = eml(x, 1) = exp(x) - ln(1) = exp(x) - 0."""
    return eml(x, 1)


def ln_eml(z):
    """ln(z) = eml(1, eml(eml(1, z), 1)).  Eq. (5) in the paper.

    Derivation:
      inner = eml(1, z) = exp(1) - ln(z) = e - ln(z)
      mid   = eml(inner, 1) = exp(e - ln(z)) - ln(1) = exp(e - ln(z))
            = exp(e) * exp(-ln(z)) = exp(e) / z
      outer = eml(1, mid) = exp(1) - ln(exp(e)/z) = e - (e - ln(z)) = ln(z)
    """
    inner = eml(1, z)
    mid = eml(inner, 1)
    return eml(1, mid)


def minus_eml(x):
    """Negation: -x.

    From the bootstrapping chain (via subtraction and zero):
      x - y = eml(ln(exp(x) - y), 1)  ... but we need subtraction first.

    A pure-EML derivation uses multiple layers.  We build it from
    the known sub-results:
      0 = ln_eml(1)  (const_zero)
      subtraction: 0 - x = -x

    For now, implement via sub_eml(0, x).
    """
    return sub_eml(const_zero(), x)


def inv_eml(x):
    """Reciprocal: 1/x.

    1/x = exp(-ln(x)).
    In EML: exp_eml(minus_eml(ln_eml(x))) is one route, but that's deep.

    Shorter: eml(eml(1, eml(eml(1, x), 1)), eml(eml(1, x), 1))
    ... but let's compose from the building blocks we have:
    1/x = exp(-ln(x))
    """
    return exp_eml(minus_eml(ln_eml(x)))


# --- Binary operations ---

def sub_eml(x, y):
    """Subtraction: x - y = eml(ln(x), exp(y)).

    Derivation:
      eml(ln(x), exp(y)) = exp(ln(x)) - ln(exp(y)) = x - y.

    WARNING — overflow limitation: this routes through exp(y) as an
    intermediate.  IEEE 754 double overflows at |y| > ~710, producing
    inf and giving wrong results.  This matches the paper's "EML Compiler"
    column (Table 4, K=83 for subtraction).  The direct-search K=11 form
    would avoid this but its RPN decomposition is not given in the paper.
    For the purposes of verifying the algebraic identities, inputs should
    be kept in the range |x|, |y| < 700.
    """
    return eml(ln_eml(x), exp_eml(y))


def add_eml(x, y):
    """Addition: x + y = x - (-y) = sub_eml(x, minus_eml(y)).

    Or directly: eml(ln_eml(x), exp_eml(minus_eml(y)))
    = exp(ln(x)) - ln(exp(-y)) = x - (-y) = x + y.
    """
    return sub_eml(x, minus_eml(y))


def mul_eml(x, y):
    """Multiplication: x * y = exp(ln(x) + ln(y)).

    In EML: exp_eml(add_eml(ln_eml(x), ln_eml(y))).
    """
    return exp_eml(add_eml(ln_eml(x), ln_eml(y)))


def div_eml(x, y):
    """Division: x / y = exp(ln(x) - ln(y)).

    In EML: exp_eml(sub_eml(ln_eml(x), ln_eml(y))).
    """
    return exp_eml(sub_eml(ln_eml(x), ln_eml(y)))


def pow_eml(x, y):
    """Exponentiation: x^y = exp(y * ln(x)).

    In EML: exp_eml(mul_eml(y, ln_eml(x))).
    """
    return exp_eml(mul_eml(y, ln_eml(x)))


# --- Constants derived from operations ---

def const_neg1():
    """-1 = minus_eml(1)."""
    return minus_eml(1)


def const_2():
    """2 = 1 + 1."""
    return add_eml(1, 1)


def const_i():
    """i = sqrt(-1) = (-1)^(1/2) = exp(ln(-1)/2).

    ln(-1) = iπ (principal branch).
    exp(iπ/2) = cos(π/2) + i·sin(π/2) = i.

    In EML: pow_eml(-1, 1/2) would be circular (needs 1/2 first).
    Direct route: exp(ln(-1) * inv(2))
    = exp_eml(mul_eml(ln_eml(const_neg1()), inv_eml(const_2())))

    But the paper notes a branch-cut issue: the EML-derived ln may give
    -iπ for ln(-1) instead of +iπ.  The compiler corrects i sign.
    """
    lnm1 = ln_eml(const_neg1())
    half = inv_eml(const_2())
    return exp_eml(mul_eml(lnm1, half))


def const_pi():
    """π = -i * ln(-1).

    ln(-1) = iπ, so -i * iπ = -i²π = π.
    In EML: mul_eml(minus_eml(const_i()), ln_eml(const_neg1())).
    """
    mi = minus_eml(const_i())
    lnm1 = ln_eml(const_neg1())
    return mul_eml(mi, lnm1)


# --- Trigonometric (via Euler's formula) ---

def cos_eml(x):
    """cos(x) = Re(exp(ix)) = (exp(ix) + exp(-ix)) / 2.

    Note: minus_eml(ix) internally wraps when |x| > π (branch-cut
    limitation of sub_eml on imaginary axis), but exp is 2πi-periodic,
    so exp_eml(minus_eml(ix)) = exp(-ix) regardless.  Correct for all x.
    """
    ix = mul_eml(const_i(), x)
    mix = minus_eml(ix)
    return div_eml(add_eml(exp_eml(ix), exp_eml(mix)), const_2())


def sin_eml(x):
    """sin(x) = Im(exp(ix)) = (exp(ix) - exp(-ix)) / (2i)."""
    ix = mul_eml(const_i(), x)
    mix = minus_eml(ix)
    two_i = mul_eml(const_2(), const_i())
    return div_eml(sub_eml(exp_eml(ix), exp_eml(mix)), two_i)


def tan_eml(x):
    """tan(x) = sin(x) / cos(x)."""
    return div_eml(sin_eml(x), cos_eml(x))


# --- Hyperbolic (via exp) ---

def cosh_eml(x):
    """cosh(x) = (exp(x) + exp(-x)) / 2."""
    return div_eml(add_eml(exp_eml(x), exp_eml(minus_eml(x))), const_2())


def sinh_eml(x):
    """sinh(x) = (exp(x) - exp(-x)) / 2."""
    return div_eml(sub_eml(exp_eml(x), exp_eml(minus_eml(x))), const_2())


def tanh_eml(x):
    """tanh(x) = sinh(x) / cosh(x)."""
    return div_eml(sinh_eml(x), cosh_eml(x))


# --- Inverse trigonometric ---

def arcsin_eml(x):
    """arcsin(x) = -i * ln(ix + sqrt(1 - x²))."""
    i = const_i()
    mi = minus_eml(i)
    ix = mul_eml(i, x)
    one_minus_x2 = sub_eml(1, mul_eml(x, x))
    sqrt_term = pow_eml(one_minus_x2, inv_eml(const_2()))
    return mul_eml(mi, ln_eml(add_eml(ix, sqrt_term)))


def arccos_eml(x):
    """arccos(x) = π/2 - arcsin(x)."""
    pi_half = div_eml(const_pi(), const_2())
    return sub_eml(pi_half, arcsin_eml(x))


def arctan_eml(x):
    """arctan(x) = arcsin(x / sqrt(1 + x²)).

    This avoids the (1/(2i)) * ln((1+ix)/(1-ix)) formula, which fails
    for |x| > π because sub_eml(1, ix) wraps through exp(ix) and the
    principal branch of ln(exp(z)) strips multiples of 2πi when |Im(z)| > π.

    The arcsin-based formula keeps all sub_eml arguments real-valued,
    avoiding the branch-cut wrapping entirely.
    """
    return arcsin_eml(div_eml(x, sqrt_eml(add_eml(1, sqr_eml(x)))))


# --- Inverse hyperbolic ---

def arsinh_eml(x):
    """arsinh(x) = ln(x + sqrt(x² + 1))."""
    return ln_eml(add_eml(x, pow_eml(add_eml(mul_eml(x, x), 1),
                                      inv_eml(const_2()))))


def arcosh_eml(x):
    """arcosh(x) = ln(x + sqrt(x² - 1))."""
    return ln_eml(add_eml(x, pow_eml(sub_eml(mul_eml(x, x), 1),
                                      inv_eml(const_2()))))


def artanh_eml(x):
    """artanh(x) = (1/2) * ln((1 + x) / (1 - x))."""
    return mul_eml(inv_eml(const_2()),
                   ln_eml(div_eml(add_eml(1, x), sub_eml(1, x))))


# --- Additional unary functions (Table 1) ---

def half_eml(x):
    """half(x) = x / 2."""
    return div_eml(x, const_2())


def sqr_eml(x):
    """sqr(x) = x²."""
    return mul_eml(x, x)


def sqrt_eml(x):
    """sqrt(x) = x^(1/2)."""
    return pow_eml(x, inv_eml(const_2()))


def sigmoid_eml(x):
    """σ(x) = 1 / (1 + exp(-x))."""
    return inv_eml(add_eml(1, exp_eml(minus_eml(x))))


# --- Additional binary operations (Table 1) ---

def log_base_eml(x, y):
    """log_x(y) = ln(y) / ln(x)."""
    return div_eml(ln_eml(y), ln_eml(x))


def avg_eml(x, y):
    """avg(x, y) = (x + y) / 2."""
    return div_eml(add_eml(x, y), const_2())


def hypot_eml(x, y):
    """hypot(x, y) = sqrt(x² + y²)."""
    return sqrt_eml(add_eml(sqr_eml(x), sqr_eml(y)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_real(z, tol=1e-8):
    """Extract real part, asserting imaginary part is negligible.

    Raises ValueError if |Im(z)| > tol * max(|z|, 1).
    """
    z = np.asarray(z, dtype=np.complex128)
    scale = np.maximum(np.abs(z), 1.0)
    if np.any(np.abs(np.imag(z)) > tol * scale):
        bad = np.abs(np.imag(z)).max()
        raise ValueError(
            f"to_real: imaginary part {bad:.2e} exceeds tolerance "
            f"{tol:.0e} (relative to scale {scale.max():.2e})"
        )
    return np.real(z)


def approx_eq(a, b, tol=1e-10):
    """Check approximate equality for complex or real values."""
    a = np.asarray(a, dtype=np.complex128).ravel()
    b = np.asarray(b, dtype=np.complex128).ravel()
    return np.allclose(a, b, atol=tol, rtol=tol)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Verify core identities from Odrzywolek 2026."""
    print("=== EML self-test ===\n")
    ok = 0
    fail = 0

    def check(name, got, expected, tol=1e-10):
        nonlocal ok, fail
        g = np.asarray(got, dtype=np.complex128)
        e = np.asarray(expected, dtype=np.complex128)
        passed = approx_eq(g, e, tol)
        status = "PASS" if passed else "FAIL"
        if not passed:
            fail += 1
            print(f"  [{status}] {name}: got {g}, expected {e}")
        else:
            ok += 1
            print(f"  [{status}] {name}")

    # Constants
    check("e = eml(1,1)", const_e(), np.e)
    check("0 = ln_eml(1)", const_zero(), 0.0)
    check("-1", to_real(const_neg1()), -1.0)
    check("2", to_real(const_2()), 2.0)

    # exp and ln
    for x in [0.5, 1.0, 2.0, -1.0, 3.7]:
        check(f"exp({x})", to_real(exp_eml(x)), np.exp(x))

    for x in [0.5, 1.0, 2.0, np.e, 7.3]:
        check(f"ln({x})", to_real(ln_eml(x)), np.log(x))

    # Subtraction
    for x, y in [(5, 3), (1, 1), (0.5, 2.7), (10, -3)]:
        check(f"{x} - {y}", to_real(sub_eml(x, y)), x - y)

    # Addition
    for x, y in [(1, 1), (2.5, 3.5), (-1, 1)]:
        check(f"{x} + {y}", to_real(add_eml(x, y)), x + y)

    # Negation
    for x in [1.0, 2.5, -3.0, 0.7]:
        check(f"-({x})", to_real(minus_eml(x)), -x)

    # Reciprocal
    for x in [1.0, 2.0, 0.5, 3.0, -1.0]:
        check(f"1/({x})", to_real(inv_eml(x)), 1.0 / x)

    # Multiplication
    for x, y in [(2, 3), (0.5, 4), (np.e, 1), (-2, 3)]:
        check(f"{x} * {y}", to_real(mul_eml(x, y)), x * y, tol=1e-8)

    # Division
    for x, y in [(6, 3), (1, 3), (np.e, 2), (10, -2)]:
        check(f"{x} / {y}", to_real(div_eml(x, y)), x / y, tol=1e-8)

    # Exponentiation
    for x, y in [(2, 3), (np.e, 2), (2, 0.5)]:
        check(f"{x}^{y}", to_real(pow_eml(x, y)), x ** y, tol=1e-8)

    # Complex constants
    i_val = const_i()
    check("i", i_val, 1j, tol=1e-8)

    pi_val = const_pi()
    check("pi", to_real(pi_val), np.pi, tol=1e-6)

    # Trigonometric
    for x in [0.0, 0.5, 1.0, np.pi / 4]:
        check(f"cos({x:.4f})", to_real(cos_eml(x)), np.cos(x), tol=1e-6)
        check(f"sin({x:.4f})", to_real(sin_eml(x)), np.sin(x), tol=1e-6)

    # --- Branch-cut regression tests (Adversary finding #1) ---
    # ln_eml on negative reals must match standard principal branch
    check("ln(-1) = iπ", ln_eml(-1.0), 1j * np.pi, tol=1e-10)
    check("ln(-2)", ln_eml(-2.0), np.log(2.0) + 1j * np.pi, tol=1e-10)
    check("ln(-0.5)", ln_eml(-0.5), np.log(0.5) + 1j * np.pi, tol=1e-10)

    # --- Extended functions (Table 1 completeness) ---
    # tan
    for x in [0.3, 0.7, np.pi / 6]:
        check(f"tan({x:.4f})", to_real(tan_eml(x)), np.tan(x), tol=1e-4)

    # Hyperbolic
    for x in [0.5, 1.0, 2.0]:
        check(f"cosh({x})", to_real(cosh_eml(x)), np.cosh(x), tol=1e-6)
        check(f"sinh({x})", to_real(sinh_eml(x)), np.sinh(x), tol=1e-6)
        check(f"tanh({x})", to_real(tanh_eml(x)), np.tanh(x), tol=1e-6)

    # Inverse trig
    for x in [0.3, 0.5, 0.8]:
        check(f"arcsin({x})", to_real(arcsin_eml(x)), np.arcsin(x), tol=1e-4)
        check(f"arccos({x})", to_real(arccos_eml(x)), np.arccos(x), tol=1e-4)
        check(f"arctan({x})", to_real(arctan_eml(x)), np.arctan(x), tol=1e-4)
    # arctan regression: must work for |x| > π (Adversary finding)
    # Limited to |x| < ~26 due to sub_eml overflow (x² > 710 → exp overflow)
    for x in [5.0, 10.0, 20.0]:
        check(f"arctan({x})", to_real(arctan_eml(x)), np.arctan(x), tol=1e-3)

    # Inverse hyperbolic
    for x in [0.5, 1.0, 2.0]:
        check(f"arsinh({x})", to_real(arsinh_eml(x)), np.arcsinh(x), tol=1e-4)
    for x in [1.5, 2.0, 3.0]:
        check(f"arcosh({x})", to_real(arcosh_eml(x)), np.arccosh(x), tol=1e-4)
    for x in [0.3, 0.5, 0.8]:
        check(f"artanh({x})", to_real(artanh_eml(x)), np.arctanh(x), tol=1e-4)

    # Additional unary: half, sqr, sqrt, sigmoid
    for x in [2.0, 4.0, 7.0]:
        check(f"half({x})", to_real(half_eml(x)), x / 2.0, tol=1e-8)
        check(f"sqr({x})", to_real(sqr_eml(x)), x ** 2, tol=1e-6)
        check(f"sqrt({x})", to_real(sqrt_eml(x)), np.sqrt(x), tol=1e-6)
    for x in [-2.0, 0.0, 1.0, 5.0]:
        check(f"σ({x})", to_real(sigmoid_eml(x)),
              1.0 / (1.0 + np.exp(-x)), tol=1e-6)

    # Additional binary: log_base, avg, hypot
    check("log_2(8)", to_real(log_base_eml(2, 8)), 3.0, tol=1e-6)
    check("log_10(100)", to_real(log_base_eml(10, 100)), 2.0, tol=1e-6)
    for xv, yv in [(1.0, 3.0), (2.5, 3.5)]:
        check(f"avg({xv},{yv})", to_real(avg_eml(xv, yv)),
              (xv + yv) / 2, tol=1e-6)
        check(f"hypot({xv},{yv})", to_real(hypot_eml(xv, yv)),
              np.hypot(xv, yv), tol=1e-4)

    print(f"\n{ok} passed, {fail} failed out of {ok + fail} tests.")
    return fail == 0


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", "divide by zero", RuntimeWarning)
    _self_test()

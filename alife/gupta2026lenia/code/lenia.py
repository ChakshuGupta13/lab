"""Faithful Asymptotic Lenia (and standard Lenia) simulator.

Equations are grounded in the primary sources (R7-read), using Davis (2024,
arXiv:2407.21086) conventions, cross-checked against Kojima-Yevenko-Ikegami
(2025, arXiv:2508.04167):

  Convolution potential   U = K (*) A          (periodic, FFT)
  Gaussian ring kernel    K(r) = exp(-((r/kr - muK)/sigmaK)^2),  sum-normalised
  Standard Lenia growth   G(x) = 2*exp(-((x-muG)/sigmaG)^2) - 1   (range [-1,1])
  Asymptotic Lenia target T(x) =   exp(-((x-muT)/sigmaT)^2)       (range [0,1])

  Standard update   A <- clip(A + dt*G(U), 0, 1)
  Asymptotic update A <-      A + dt*(T(U) - A)            (no clipping needed)

Note: Davis's Gaussians use ((x-mu)/sigma)^2 WITHOUT a 1/2 factor; we follow
that convention exactly so the published glider parameters transfer directly.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)  # solitons are sensitive; use f64


@dataclass(frozen=True)
class Rule:
    """An (Asymptotic) Lenia rule. Defaults = Davis 2024 Fig.1 glider."""
    grid: int = 128
    kr: float = 13.0          # native kernel radius in pixels
    muK: float = 0.5          # kernel ring peak (relative radius)
    sigmaK: float = 0.15      # kernel ring width
    muT: float = 0.12         # target/growth centre
    sigmaT: float = 0.005     # target/growth width
    dt: float = 0.2
    asymptotic: bool = True   # True: AL target form; False: standard growth
    kernel_core: str = "gauss"  # "gauss": Gaussian ring; "poly": (4r(1-r))^4 bump (Orbium)
    half_growth: bool = False   # gauss growth: True = Chan 1/2-factor exp(-d^2/2); False = Davis exp(-d^2)
    growth_core: str = "gauss"  # "gauss": bell growth; "poly": Chan quad4 delta 2*max(0,1-d^2/9)^4-1 (Orbium)
    peaks: tuple = (1.0,)       # poly kernel ring heights; len>1 = Chan multi-ring banded kernel
    # ---- SmoothLife (Rafler 2011) ----  active when rule_type="smoothlife"
    rule_type: str = "lenia"    # "lenia" | "smoothlife"
    r_i: float = 4.0            # SmoothLife inner-disk radius (px)
    r_o: float = 12.0           # SmoothLife outer-ring radius (px), r_o > r_i
    b1: float = 0.278           # SmoothLife birth interval lower threshold
    b2: float = 0.365           # SmoothLife birth interval upper threshold
    d1: float = 0.267           # SmoothLife survival interval lower threshold
    d2: float = 0.445           # SmoothLife survival interval upper threshold
    alpha_n: float = 0.028      # SmoothLife outer-density sigma steepness
    alpha_m: float = 0.147      # SmoothLife inner-aliveness sigma steepness
    # ---- Glaberish (Davis & Bongard 2022) ----  active when rule_type="glaberish"
    muB: float = 0.18           # birth Gaussian centre
    sigmaB: float = 0.015       # birth Gaussian width
    muD: float = 0.25           # death Gaussian centre
    sigmaD: float = 0.025       # death Gaussian width


def _radial(grid: int) -> jnp.ndarray:
    """Distance (in pixels) from the grid centre, fft-centred."""
    ax = jnp.arange(grid) - grid // 2
    xx, yy = jnp.meshgrid(ax, ax, indexing="ij")
    return jnp.sqrt(xx.astype(jnp.float64) ** 2 + yy.astype(jnp.float64) ** 2)


def kernel(rule: Rule) -> jnp.ndarray:
    """Sum-normalised kernel on the full grid, centre at middle.

    kernel_core="gauss": Gaussian ring exp(-((r-muK)/sigmaK)^2).
    kernel_core="poly":  Chan banded kernel of B=len(peaks) equal rings, each a
                         quad4 bump (4u(1-u))^4 weighted by peaks; peaks=(1,) is
                         the single Orbium bump.
    """
    r = _radial(rule.grid) / rule.kr
    if rule.kernel_core == "poly":
        B = len(rule.peaks)
        rm = jnp.clip(r, 0.0, 1.0) * B
        idx = jnp.clip(jnp.floor(rm).astype(jnp.int32), 0, B - 1)
        u = rm - jnp.floor(rm)
        core = (4.0 * u * (1.0 - u)) ** 4
        K = jnp.asarray(rule.peaks)[idx] * core
    else:
        K = jnp.exp(-(((r - rule.muK) / rule.sigmaK) ** 2))
    K = jnp.where(r <= 1.0, K, 0.0)        # support within the native radius
    return K / jnp.sum(K)


def kernel_fft(rule: Rule):
    """Precomputed FFT of the ifftshifted kernel (origin-centred convolution).

    For rule_type="smoothlife" returns a (Kin_fft, Kout_fft) tuple corresponding
    to the inner disk (radius<r_i) and outer annulus (r_i<=radius<r_o), each
    sum-normalised.
    """
    if rule.rule_type == "smoothlife":
        R = _radial(rule.grid)
        inner = jnp.where(R < rule.r_i, 1.0, 0.0)
        outer = jnp.where((R >= rule.r_i) & (R < rule.r_o), 1.0, 0.0)
        inner = inner / jnp.sum(inner)
        outer = outer / jnp.sum(outer)
        Kin = jnp.fft.rfft2(jnp.fft.ifftshift(inner))
        Kout = jnp.fft.rfft2(jnp.fft.ifftshift(outer))
        return (Kin, Kout)
    K = kernel(rule)
    return jnp.fft.rfft2(jnp.fft.ifftshift(K))


def potential(A: jnp.ndarray, Kf: jnp.ndarray) -> jnp.ndarray:
    return jnp.fft.irfft2(jnp.fft.rfft2(A) * Kf, s=A.shape)


def _bell(U: jnp.ndarray, mu: float, sigma: float, half: bool) -> jnp.ndarray:
    """Gaussian bell. half=False: exp(-d^2) (Davis); half=True: exp(-d^2/2) (Chan)."""
    d2 = ((U - mu) / sigma) ** 2
    return jnp.exp(-0.5 * d2) if half else jnp.exp(-d2)


def _growth(U: jnp.ndarray, rule: Rule) -> jnp.ndarray:
    """Standard-Lenia growth in [-1,1]. poly: Chan quad4 delta (canonical Orbium)."""
    if rule.growth_core == "poly":
        q = jnp.clip(1.0 - (U - rule.muT) ** 2 / (9.0 * rule.sigmaT ** 2), 0.0, None)
        return 2.0 * q ** 4 - 1.0
    return 2.0 * _bell(U, rule.muT, rule.sigmaT, rule.half_growth) - 1.0


def target(U: jnp.ndarray, rule: Rule) -> jnp.ndarray:
    return _bell(U, rule.muT, rule.sigmaT, rule.half_growth)


def _sl_sigma(x, a, alpha):
    """SmoothLife smooth step: 1/(1+exp(-(x-a)*4/alpha))."""
    return 1.0 / (1.0 + jnp.exp(-(x - a) * 4.0 / alpha))


@functools.partial(jax.jit, static_argnums=(2,))
def step(A: jnp.ndarray, Kf, rule: Rule) -> jnp.ndarray:
    if rule.rule_type == "smoothlife":
        Kin, Kout = Kf
        Afft = jnp.fft.rfft2(A)
        M = jnp.fft.irfft2(Afft * Kin, s=A.shape)  # inner-disk density (current state)
        N = jnp.fft.irfft2(Afft * Kout, s=A.shape) # outer-ring density (neighbours)
        m_gate = _sl_sigma(M, 0.5, rule.alpha_m)
        lo = rule.b1 * (1 - m_gate) + rule.d1 * m_gate
        hi = rule.b2 * (1 - m_gate) + rule.d2 * m_gate
        s = _sl_sigma(N, lo, rule.alpha_n) * (1 - _sl_sigma(N, hi, rule.alpha_n))
        return jnp.clip(A + rule.dt * (2.0 * s - 1.0), 0.0, 1.0)
    if rule.rule_type == "glaberish":
        # Davis-Bongard 2022 Eq. 3: state-dependent genesis/persistence split,
        #   A <- A + dt*[(1-A)*Ggen(U) + A*P(U)],  Ggen,P stretched to [-1,1].
        # muB,sigmaB parametrise genesis; muD,sigmaD persistence. The canonical,
        # validated s613 rule (3-ring kernel, Chan 1/2-Gaussian) lives in the
        # self-contained scratch/glaberish_s613.py.
        U = potential(A, Kf)
        Ggen = 2.0 * jnp.exp(-(((U - rule.muB) / rule.sigmaB) ** 2)) - 1.0
        Pers = 2.0 * jnp.exp(-(((U - rule.muD) / rule.sigmaD) ** 2)) - 1.0
        return jnp.clip(A + rule.dt * ((1.0 - A) * Ggen + A * Pers), 0.0, 1.0)
    U = potential(A, Kf)
    if rule.asymptotic:
        return A + rule.dt * (target(U, rule) - A)
    return jnp.clip(A + rule.dt * _growth(U, rule), 0.0, 1.0)


@functools.partial(jax.jit, static_argnums=(2, 3))
def rollout(A0: jnp.ndarray, Kf: jnp.ndarray, rule: Rule, n: int):
    """Return (final state, stacked states) after n steps."""
    def body(A, _):
        An = step(A, Kf, rule)
        return An, An
    Afinal, traj = jax.lax.scan(body, A0, None, length=n)
    return Afinal, traj


# ---- diagnostics -----------------------------------------------------------

def mass(A: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum(A)


def center_of_mass(A: jnp.ndarray) -> jnp.ndarray:
    """Periodic-aware centre of mass (circular mean), in pixel coords."""
    grid = A.shape[0]
    ax = jnp.arange(grid)
    ang = 2 * jnp.pi * ax / grid
    m = jnp.sum(A) + 1e-12
    cx = jnp.array([jnp.sum(A * jnp.cos(ang)[:, None]), jnp.sum(A * jnp.sin(ang)[:, None])])
    cy = jnp.array([jnp.sum(A * jnp.cos(ang)[None, :]), jnp.sum(A * jnp.sin(ang)[None, :])])
    px = (jnp.arctan2(cx[1], cx[0]) % (2 * jnp.pi)) * grid / (2 * jnp.pi)
    py = (jnp.arctan2(cy[1], cy[0]) % (2 * jnp.pi)) * grid / (2 * jnp.pi)
    return jnp.array([px, py])


def recenter(A: jnp.ndarray) -> jnp.ndarray:
    """Integer-roll the pattern so its CoM sits at the grid centre."""
    grid = A.shape[0]
    com = center_of_mass(A)
    shift = (jnp.round(grid // 2 - com)).astype(jnp.int32)
    return jnp.roll(A, (int(shift[0]), int(shift[1])), axis=(0, 1))

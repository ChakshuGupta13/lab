"""Windmill-family figure for the paper's tile (website) and this README.

Paper: "Sharp bounds between the saturation number and the harmonic index"
       Chakshu Gupta, arXiv:2606.15761 (math.CO).

Motivation
----------
The personal website (haveli) shows each paper as a tile; this generates that
tile's image, and the same image heads this repo's README. It is made from the
paper's own object -- the friendship graph F_k -- not stock art.

What it shows
-------------
The friendship graph F_k is k triangles sharing one hub vertex (a windmill /
pinwheel). It is the paper's signature family: mu*(F_k) = k grows linearly while
H(F_k) = 2k/(k+1) + k/2 grows only like k/2, so the ratio mu*/H climbs from
below 1 to its limit 2, crossing the TxGraffiti threshold mu* = H exactly once.

Four windmills F_2, F_3, F_4, F_5 are drawn left to right. Each windmill's COLOUR
encodes its actual ratio mu*(F_k)/H(F_k) on the magma colormap: cool where the
conjecture mu* <= H holds, hot where it fails. The crossover is the whole story:

    F_2 : mu* = 2, H = 7/3,  ratio = 0.857  -> conjecture HOLDS  (cool)
    F_3 : mu* = 3, H = 3,    ratio = 1.000  -> equality          (threshold)
    F_4 : mu* = 4, H = 18/5, ratio = 1.111  -> conjecture FAILS  (hot)
    F_5 : mu* = 5, H = 25/6, ratio = 1.200  -> conjecture FAILS  (hotter)

F_4 -- a smallest counterexample, on nine vertices (the paper's wording; other
order-9 counterexamples may exist) -- is drawn largest and ringed.

Honesty (this is PUBLIC and sits next to the paper):
  * Vertex POSITIONS are decorative (a radial pinwheel layout); only the EDGES
    and the per-windmill COLOUR carry paper content. The edge set is exactly
    friendship_graph(k) from graph_utils.py; the colour is exactly mu*/H.
  * This is an emblem, not a figure from the paper. The paper's theorems are the
    two-sided bound 1/4 H(T) < mu*(T) < 3/2 H(T) on trees; this family lives off
    the tree case (F_k has triangles) and illustrates only that the GENERAL ratio
    is unbounded above the conjecture -- the motivation, not the main result.

Provenance (all cross-checked by verify_friendship.py, this directory):
  * F_k construction: friendship_graph(k) in graph_utils.py (k triangles, one hub).
  * mu*(F_k) = k: brute force, verify_friendship.py.
  * H(F_k) = 2k/(k+1) + k/2: exact Fraction, H_friendship_closed, verify_friendship.py.
  * ratios 0.857, 1.000, 1.111, 1.200 and "FAILS at k=4": verify_friendship.py output.

Rendering recipe (dark ground, layered-alpha neon strokes, a gaussian bloom pass,
a perceptual sequential colormap) follows the aesthetic of the companion
viz_landscape.py in math/gupta2026architecture. Self-contained: NumPy +
Matplotlib + SciPy + NetworkX only.
"""

import argparse
from fractions import Fraction

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter

from graph_utils import friendship_graph, harmonic_index, mu_star_bruteforce


# --- the paper's object and the quantity it bounds (see Provenance) ---------
def windmill_with_ratio(k):
    """Return (G, ratio) for F_k, ratio = mu*(F_k)/H(F_k), both exact."""
    G = friendship_graph(k)
    ratio = float(Fraction(mu_star_bruteforce(G)) / harmonic_index(G))
    return G, ratio


def layout(G, k, center, radius, rot=0.0, spread=0.62):
    """Radial pinwheel positions for friendship_graph(k).

    Node 0 is the hub; blade i uses nodes (2i+1, 2i+2) placed symmetrically
    about the blade axis 2*pi*i/k.  Positions are decorative (see Honesty).
    """
    cx, cy = center
    pos = {0: (cx, cy)}
    half = (np.pi / k) * spread
    for i in range(k):
        theta = 2.0 * np.pi * i / k + rot
        a, b = 2 * i + 1, 2 * i + 2
        pos[a] = (cx + radius * np.cos(theta - half), cy + radius * np.sin(theta - half))
        pos[b] = (cx + radius * np.cos(theta + half), cy + radius * np.sin(theta + half))
    return pos


def _stroke(ax, p, q, color, core_w, glow_w, glow_a):
    """One neon edge: a wide soft glow under a thin bright core."""
    x, y = [p[0], q[0]], [p[1], q[1]]
    ax.plot(x, y, color=color, lw=glow_w, alpha=glow_a, solid_capstyle="round", zorder=2)
    ax.plot(x, y, color=color, lw=glow_w * 0.45, alpha=glow_a * 1.4, solid_capstyle="round", zorder=3)
    ax.plot(x, y, color="white", lw=core_w, alpha=0.95, solid_capstyle="round", zorder=4)


def render(res_w=2100, res_h=900, palette="magma", rmin=0.80, rmax=1.26):
    """Compose the four windmills on a 21:9 dark ground and bloom them."""
    aspect = res_w / res_h
    fig = plt.figure(figsize=(aspect * 4.5, 4.5), dpi=res_h / 4.5)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    halfh = 4.5
    ax.set_xlim(-halfh * aspect, halfh * aspect)
    ax.set_ylim(-halfh, halfh)
    ax.set_aspect("equal")
    fig.patch.set_facecolor("#0b0410")
    ax.set_facecolor("#0b0410")

    cmap = colormaps[palette]
    ks = [2, 3, 4, 5]
    centers_x = [-7.7, -3.0, 2.2, 7.3]
    radii = [1.6, 1.9, 2.55, 2.0]         # F_4 largest (the hero counterexample)
    rotations = [0.30, 0.10, 0.0, 0.22]

    for k, cx, R, rot in zip(ks, centers_x, radii, rotations):
        G, ratio = windmill_with_ratio(k)
        t = float(np.clip((ratio - rmin) / (rmax - rmin), 0.0, 1.0))
        # keep even the "cool" windmill visible on near-black ground
        col = cmap(0.30 + 0.62 * t)[:3]
        pos = layout(G, k, (cx, 0.0), R, rot=rot)

        core_w = 2.1 + 0.6 * t
        glow_w = 13.0 + 5.0 * t
        glow_a = 0.16 + 0.10 * t
        for u, v in G.edges():
            _stroke(ax, pos[u], pos[v], col, core_w, glow_w, glow_a)

        # nodes: soft halo + bright core; hub brightest
        for n, (x, y) in pos.items():
            hub = n == 0
            ax.scatter([x], [y], s=(420 if hub else 200), color=col, alpha=0.28, zorder=5, edgecolors="none")
            ax.scatter([x], [y], s=(120 if hub else 60), color="white", alpha=0.95, zorder=6, edgecolors="none")

        # ring F_4, a smallest counterexample, on nine vertices
        if k == 4:
            ring = plt.Circle((cx, 0.0), R + 0.85, fill=False, lw=1.4,
                              color=col, alpha=0.5, zorder=1)
            ax.add_patch(ring)

    fig.canvas.draw()
    rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(np.float64) / 255.0
    plt.close(fig)

    # bloom: two-scale additive glow for a luminous, dimensional look
    bloom = 0.55 * gaussian_filter(rgb, (7, 7, 0)) + 0.30 * gaussian_filter(rgb, (22, 22, 0))
    out = np.clip(rgb + bloom, 0.0, 1.0)
    out = out ** 0.92                                   # mild lift
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="windmill.png")
    p.add_argument("--width", type=int, default=2100)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--palette", default="magma")
    args = p.parse_args()

    img = render(res_w=args.width, res_h=args.height, palette=args.palette)
    mpimg.imsave(args.out, np.clip(img, 0.0, 1.0))

    lum = img.mean(axis=2)
    lit = float((lum > 0.12).mean())
    print(f"wrote {args.out}  ({img.shape[1]}x{img.shape[0]}, lit fraction={lit:.3f})")


if __name__ == "__main__":
    main()

"""Loss-landscape figure for the paper's tile (website) and this README.

Paper: "Why Architecture Choice Matters in Symbolic Regression"
       Chakshu Gupta & Theodore J. LaGrow, arXiv:2604.23256 (MLSP 2026).

Motivation
----------
The personal website (haveli) shows each paper as a tile; this generates that
tile's image, and the same image heads this repo's README. It is made from the
paper's own operator and training data, not stock art.

What it shows (and what it does NOT)
------------------------------------
The paper's finding: for parameterized symbolic-regression trees, WHICH input
an active subtree routes a variable through governs whether gradient descent
recovers the target -- because the operator's two inputs carry asymmetric
gradients (the paper's factor F4).

This figure is a DECORATIVE TOY that borrows the paper's operator and data, not
a figure from the paper and not its mechanism. It renders the MSE loss
landscape of a single EML node whose two inputs are blended between x and y:

    target(x, y) = eml(x, y) = exp(x) - ln(y)              # the paper's operator
    a, b         : routing logits;  s(.) = logistic sigmoid
    inL          = s(a) * x + (1 - s(a)) * y               # left input  (-> exp)
    inR          = s(b) * x + (1 - s(b)) * y               # right input (-> ln)
    model(x, y)  = eml(inL, inR)
    loss(a, b)   = mean over x, y in [1, 3]  of  (model - target)^2

The (a, b) plane is the routing space; its global minimum sits where the left
input selects x (into exp) and the right selects y (into ln).

Honesty (this is PUBLIC and sits next to the paper):
  * This routing is NOT the paper's. The paper uses a 3-way softmax selection
    over {1, x, child} inside multi-level trees; this is a 2-way continuous
    blend between x and y, with no constant channel and no tree. It is
    "inspired by", not "a reduction of", the paper's mechanism.
  * The visible large-scale asymmetry is dominated by the operator's VALUE
    asymmetry (exp of a wrong input explodes; ln stays bounded on [1, 3]), not
    by the gradient asymmetry (F4) the paper studies -- F4 governs only the
    fine basin curvature near the optimum, not resolvable at this scale.
  * It is NOT the paper's full tree and NOT the 0-100% recovery-rate result.
    It is meant to be pretty and made from real ingredients, nothing more.

Provenance
----------
- eml(a, b) = exp(a) - ln(b): verbatim from reproduce.py (def _eml, line 78).
- training range x, y in [1, 3] step 0.1: matches reproduce.py
  TRAIN_LO/TRAIN_HI/TRAIN_STEP (line 65).
- EML is the exp-minus-log operator of Odrzywolek 2026 (arXiv:2603.21852),
  named "EML" there; this paper studies it.

Rendering recipe (rank/CDF equalization for contrast, a perceptual sequential
colormap, mild gamma) follows the aesthetic notes distilled while building the
companion plotting toolkit. This script is intentionally self-contained:
NumPy + Matplotlib + SciPy only.
"""

import argparse

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import colormaps
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter


# --- the paper's operator and data (see Provenance above) -----------------
def eml(a, b):
    return np.exp(a) - np.log(b)


def training_data():
    xs = np.arange(1.0, 3.0 + 0.05, 0.1)          # [1, 3] step 0.1
    X, Y = np.meshgrid(xs, xs)
    return X.ravel(), Y.ravel()


X, Y = training_data()
TARGET = eml(X, Y)


def _sigmoid(p):
    return 1.0 / (1.0 + np.exp(-np.clip(p, -50.0, 50.0)))


def loss_row(a, b):
    """MSE over the data for 1-D routing-logit arrays a, b (equal shape)."""
    sA = _sigmoid(a)[:, None]
    sB = _sigmoid(b)[:, None]
    inL = sA * X + (1.0 - sA) * Y
    inR = sB * X + (1.0 - sB) * Y
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        err = eml(inL, inR) - TARGET
        mse = np.mean(err ** 2, axis=-1)
    return np.nan_to_num(mse, nan=1e18, posinf=1e18)


def _equalize(field):
    """Rank/CDF equalize to [0, 1]; robust to the loss's heavy tail."""
    flat = field.ravel().astype(np.float64)
    ok = np.isfinite(flat)
    out = np.zeros_like(flat)
    v = flat[ok]
    sv = np.sort(v)
    r = np.searchsorted(sv, v, side="right").astype(np.float64)
    out[ok] = r / r.max()
    return out.reshape(field.shape)


def render(res=1500, lim=10.0, gamma=1.2, blur=1.0, palette="magma"):
    """Compute and color the loss landscape over (a, b) in [-lim, lim]^2.

    The grid is evaluated one row at a time so memory stays bounded
    regardless of `res`.
    """
    ax = np.linspace(-lim, lim, res)
    aa, bb = np.meshgrid(ax, ax)
    loss = np.empty((res, res))
    for i in range(res):
        loss[i] = loss_row(aa[i], bb[i])
    # low loss -> bright: invert the (log-compressed) loss before equalizing
    val = _equalize(-np.log1p(loss))
    val = gaussian_filter(val, blur) ** gamma
    rgb = colormaps[palette](val)[..., :3]
    return np.flipud(rgb)                            # +b upward


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="landscape.png")
    p.add_argument("--res", type=int, default=1500)
    p.add_argument("--lim", type=float, default=10.0)
    p.add_argument("--palette", default="magma")
    args = p.parse_args()

    img = render(res=args.res, lim=args.lim, palette=args.palette)
    mpimg.imsave(args.out, np.clip(img, 0.0, 1.0))

    lum = img.mean(axis=2)
    mid = float(((lum >= 0.2) & (lum <= 0.7)).mean())
    print(f"wrote {args.out}  ({args.res}x{args.res}, midtones={mid:.3f})")


if __name__ == "__main__":
    main()

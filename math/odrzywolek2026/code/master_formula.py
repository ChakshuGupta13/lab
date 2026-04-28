"""EML Master Formula — parameterized EML trees for symbolic regression.

Implements the parameterized level-n EML tree from Section 4.3 of
Odrzywolek 2026 (arXiv:2603.21852v2).

Each EML node input is a soft selection among {1, x, f_child}:
    input_i = softmax(α_i, β_i, γ_i) · [1, x, f_child]

At the leaf level (bottom), only {1, x} are available (no γ).

The level-n master formula has 5×2^n − 6 parameters.
Level 2: 14 params, Level 3: 34 params, Level 4: 74 params.

Uses PyTorch complex128 for autodiff-compatible forward pass with the
conjugate-branch EML convention.
"""

import torch
import torch.nn as nn
import numpy as np

DTYPE = torch.complex128
EXP_CLAMP = 500.0  # clamp exp arguments to avoid overflow


def safe_exp(z):
    """Clamped exp to prevent overflow in deeply nested trees."""
    real = torch.clamp(z.real, -EXP_CLAMP, EXP_CLAMP)
    return torch.exp(torch.complex(real, z.imag))


def eml_torch(x, y):
    """EML with conjugate-branch convention, torch version."""
    return safe_exp(x) - torch.conj(torch.log(torch.conj(y)))


def safe_sinh(z):
    """Clamped sinh to prevent overflow in deeply nested trees."""
    real = torch.clamp(z.real, -EXP_CLAMP, EXP_CLAMP)
    return torch.sinh(torch.complex(real, z.imag))


def sml_torch(x, y):
    """SML (Sinh-Minus-arcTan): structurally different asymmetric operator.

    sml(a, b) = sinh(a) - arctan(b)
    Gradient asymmetry: d/da = cosh(a) (exponential), d/db = -1/(1+b^2) (bounded).
    Used for generality testing of architecture-target interaction.
    """
    return safe_sinh(x) - torch.atan(y)


def rml_torch(x, y):
    """RML (aRctan-Minus-sinL): right-amplified asymmetric operator.

    rml(a, b) = arctan(a) - sinh(b)
    Gradient asymmetry: d/da = 1/(1+a^2) (bounded), d/db = -cosh(b) (exponential).
    Mirror of SML — right input is amplified, left is attenuated.
    """
    return torch.atan(x) - safe_sinh(y)


class EmlMasterFormula(nn.Module):
    """Level-n parameterized EML tree.

    Architecture:
        - Full binary tree of depth n
        - Each internal node applies eml(left_input, right_input)
        - Each input is softmax(α, β, γ) · [1, x, f_child]
        - Leaf-level inputs use softmax(α, β) · [1, x]  (no child)

    Parameters:
        depth: tree depth (n). Level-1 = single eml node.
        n_vars: number of input variables (1 for univariate, 2 for bivariate).
        intermediate_clamp: clamp magnitude for intermediate EML outputs.
            Prevents NaN cascades in deep trees.  Default 1e6 keeps MSE
            loss (~1e12) within float64 range while not restricting
            converged solutions.
    """

    def __init__(self, depth, n_vars=1, intermediate_clamp=1e6, op_fn=None):
        super().__init__()
        self.depth = depth
        self.n_vars = n_vars
        self.intermediate_clamp = intermediate_clamp
        self.op_fn = op_fn or eml_torch

        # Build parameter lists for each level
        # Level 0 (root) has 1 eml node → 2 inputs
        # Level k has 2^k eml nodes → 2^(k+1) inputs
        # Total internal nodes = 2^depth - 1
        # Total inputs = 2 * (2^depth - 1) = 2^(depth+1) - 2
        # Leaf inputs (level depth-1's children) = 2^depth
        # Internal inputs (levels 0 to depth-2) = 2^(depth+1) - 2 - 2^depth
        #                                       = 2^depth - 2

        # At each internal node's input: 3 logits (α, β, γ) for {1, x, child}
        # At each leaf input: 2 logits (α, β) for {1, x}
        # For n_vars > 1: leaf logits have (1 + n_vars) entries

        # Simpler: store logits per input position, indexed by tree location
        # We'll use a flat parameter tensor and index into it.

        # Number of inputs at each level of the tree:
        # The root eml node has 2 inputs.
        # Each input either receives a child eml (internal) or is a leaf.
        # For a depth-n tree, inputs at positions in the last level are leaves.

        # Internal input logits: 3 per input (α, β, γ)
        #   Count: 2 * (2^0 + 2^1 + ... + 2^(depth-2)) = 2 * (2^(depth-1) - 1)
        #        = 2^depth - 2  (for depth >= 2)
        # Leaf input logits: (1 + n_vars) per input
        #   Count: 2 * 2^(depth-1) = 2^depth

        if depth < 1:
            raise ValueError("depth must be >= 1")

        n_leaf_choices = 1 + n_vars  # {1, x} for univariate; {1, x, y} for bivariate

        if depth == 1:
            # Single eml node, both inputs are leaves
            self.n_internal = 0
            self.n_leaf = 2
            self.internal_logits = nn.Parameter(torch.empty(0))
            self.leaf_logits = nn.Parameter(
                torch.randn(2, n_leaf_choices, dtype=torch.float64) * 0.1)
        else:
            self.n_internal = 2**depth - 2
            self.n_leaf = 2**depth
            self.internal_logits = nn.Parameter(
                torch.randn(self.n_internal, 3, dtype=torch.float64) * 0.1)
            self.leaf_logits = nn.Parameter(
                torch.randn(self.n_leaf, n_leaf_choices, dtype=torch.float64) * 0.1)

        self.n_leaf_choices = n_leaf_choices

    def param_count(self):
        """Total trainable parameters."""
        return (self.n_internal * 3 + self.n_leaf * self.n_leaf_choices)

    def forward(self, x, y=None, tau_leaf=1.0, tau_gate=1.0):
        """Evaluate the master formula at input x (and optionally y).

        x: real tensor of shape (batch,) or scalar
        y: real tensor of shape (batch,) or scalar (if bivariate)
        tau_leaf: temperature for leaf softmax
        tau_gate: temperature for internal softmax (shared with leaf here)

        Returns: (pred, leaf_probs, internal_probs) to match EMLTreeV16
            interface.  pred is complex128 tensor of shape (batch,).
        """
        tau = tau_leaf  # Eq.6 uses the same softmax everywhere
        x = torch.as_tensor(x, dtype=DTYPE)
        if x.dim() == 0:
            x = x.unsqueeze(0)

        vars_list = [torch.ones_like(x), x]
        if self.n_vars >= 2:
            if y is None:
                raise ValueError(
                    f"Model has n_vars={self.n_vars} but y was not provided")
            y = torch.as_tensor(y, dtype=DTYPE)
            if y.dim() == 0:
                y = y.unsqueeze(0)
            vars_list.append(y)

        # Recursive evaluation from leaves up
        # Use a list to hold mutable counters (avoids instance-level mutation)
        state = [0, 0]  # [internal_idx, leaf_idx]
        self._tau = tau
        result = self._eval_node(0, vars_list, state)

        # Return probs IN the computation graph so penalty gradients flow
        leaf_probs = torch.softmax(self.leaf_logits / tau, dim=1)
        int_probs = (torch.softmax(self.internal_logits / tau, dim=1)
                     if self.n_internal > 0
                     else torch.empty(0, 3))
        return result, leaf_probs, int_probs

    def _eval_node(self, level, vars_list, state):
        """Recursively evaluate a node at given level.

        state: [internal_idx, leaf_idx] mutable counters.
        Returns the eml output for this node.
        """
        if level == self.depth:
            raise RuntimeError("Exceeded tree depth")

        left_input = self._get_input(level, vars_list, level + 1, state)
        right_input = self._get_input(level, vars_list, level + 1, state)
        return self._scrub(self.op_fn(left_input, right_input))

    def _scrub(self, z):
        """Replace NaN→0, clamp Inf and large values to ±intermediate_clamp.

        Applied after each EML node to prevent NaN/Inf cascades in deep
        trees.  Matches the per-layer scrubbing in the author's v16 trainer.
        """
        c = self.intermediate_clamp
        real = torch.nan_to_num(
            z.real, nan=0.0, posinf=c, neginf=-c).clamp(-c, c)
        imag = torch.nan_to_num(
            z.imag, nan=0.0, posinf=c, neginf=-c).clamp(-c, c)
        return torch.complex(real, imag)

    def _get_input(self, parent_level, vars_list, child_level, state):
        """Compute one input to an eml node.

        If child_level == depth: this is a leaf → softmax over {1, x, [y]}
        Otherwise: this is internal → softmax over {1, x, child_eml_output}
        state: [internal_idx, leaf_idx] mutable counters.
        """
        tau = getattr(self, '_tau', 1.0)
        if child_level == self.depth:
            # Leaf input
            idx = state[1]
            state[1] += 1
            logits = self.leaf_logits[idx]
            weights = torch.softmax(logits / tau, dim=0).to(DTYPE)
            # vars_list = [ones, x, (y)]
            result = sum(w * v for w, v in zip(weights, vars_list))
            return result
        else:
            # Internal input: compute child eml first
            child_output = self._eval_node(child_level, vars_list, state)
            idx = state[0]
            state[0] += 1
            logits = self.internal_logits[idx]
            weights = torch.softmax(logits / tau, dim=0).to(DTYPE)
            # choices: [1, x, child_output]  (for univariate)
            ones = vars_list[0]
            x = vars_list[1]
            result = weights[0] * ones + weights[1] * x + weights[2] * child_output
            return result

    def snap_weights(self, threshold=0.8):
        """Snap softmax weights to one-hot by setting logits to \u00b150.

        Modifies logits in-place (like V16's snap_weights).
        Returns snapped index arrays for inspection.
        """
        snapped = {}
        with torch.no_grad():
            if self.n_internal > 0:
                idx = torch.argmax(self.internal_logits, dim=1)
                new_int = torch.full_like(self.internal_logits, -50.0)
                new_int[torch.arange(self.n_internal), idx] = 50.0
                self.internal_logits.copy_(new_int)
                snapped['internal'] = idx.cpu().numpy()
            if self.n_leaf > 0:
                idx = torch.argmax(self.leaf_logits, dim=1)
                new_leaf = torch.full_like(self.leaf_logits, -50.0)
                new_leaf[torch.arange(self.n_leaf), idx] = 50.0
                self.leaf_logits.copy_(new_leaf)
                snapped['leaf'] = idx.cpu().numpy()
        return snapped

    def describe(self):
        """Human-readable description of the current (snapped) tree."""
        snapped = self.snap_weights()
        choices_internal = ['1', 'x', 'f']
        choices_leaf = ['1', 'x'] + (['y'] if self.n_vars >= 2 else [])

        lines = [f"EML Master Formula (depth={self.depth}, "
                 f"params={self.param_count()})"]

        if 'internal' in snapped:
            lines.append(f"  Internal routing: "
                         + ', '.join(choices_internal[i]
                                     for i in snapped['internal']))
        if 'leaf' in snapped:
            lines.append(f"  Leaf routing:     "
                         + ', '.join(choices_leaf[i]
                                     for i in snapped['leaf']))
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# V16 architecture — matches the author's actual PyTorch v16_final code
# ---------------------------------------------------------------------------

_BYPASS_THR = 1.0 - torch.finfo(torch.float64).eps

INIT_STRATEGIES = ["biased", "uniform", "xy_biased", "random_hot"]


class EMLTreeV16(nn.Module):
    """Full binary tree of EML nodes, matching the author's v16_final code.

    Architecture (differs from paper's Eq. 6):
        - Leaves: softmax over {1, x, y} — 3 logits per leaf
        - Internal nodes: sigmoid blend {1, child} — 2 logits per node
          s=1 → constant 1, s=0 → child value
        - x can ONLY enter at leaves (no direct x at internal nodes)
        - Bottom-up vectorized evaluation

    This matches the code that produced the paper's ~25% rates at depth 3-4.
    """

    def __init__(self, depth, init_scale=1.0, init_strategy="biased",
                 eml_clamp=1e300, op_fn=None):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_internal = self.n_leaves - 1
        self.eml_clamp = eml_clamp
        self.op_fn = op_fn or eml_torch

        if init_strategy == "manual":
            leaf_init = torch.zeros(self.n_leaves, 3, dtype=torch.float64)
            gate_init = torch.zeros(self.n_internal, 2, dtype=torch.float64)
        elif init_strategy == "biased":
            leaf_init = (torch.randn(self.n_leaves, 3, dtype=torch.float64)
                         * init_scale)
            leaf_init[:, 0] += 2.0
            gate_init = (torch.randn(self.n_internal, 2, dtype=torch.float64)
                         * init_scale + 4.0)
        elif init_strategy == "uniform":
            leaf_init = (torch.randn(self.n_leaves, 3, dtype=torch.float64)
                         * init_scale)
            gate_init = (torch.randn(self.n_internal, 2, dtype=torch.float64)
                         * init_scale + 4.0)
        elif init_strategy == "xy_biased":
            leaf_init = (torch.randn(self.n_leaves, 3, dtype=torch.float64)
                         * init_scale)
            leaf_init[:, 1] += 1.0
            leaf_init[:, 2] += 1.0
            gate_init = (torch.randn(self.n_internal, 2, dtype=torch.float64)
                         * init_scale + 4.0)
        elif init_strategy == "random_hot":
            leaf_init = (torch.randn(self.n_leaves, 3, dtype=torch.float64)
                         * init_scale)
            hot_idx = torch.randint(0, 3, (self.n_leaves,))
            leaf_init[torch.arange(self.n_leaves), hot_idx] += 3.0
            gate_init = (torch.randn(self.n_internal, 2, dtype=torch.float64)
                         * init_scale + 3.0)
            open_mask = torch.rand(self.n_internal, 2) < 0.25
            gate_init[open_mask] -= 6.0
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        self.leaf_logits = nn.Parameter(leaf_init)
        self.blend_logits = nn.Parameter(gate_init)

    def param_count(self):
        """Total trainable parameters."""
        return self.n_leaves * 3 + self.n_internal * 2

    def forward(self, x, y, tau_leaf=1.0, tau_gate=1.0):
        """Bottom-up vectorized forward pass.

        x, y: real tensors of shape (batch,)
        tau_leaf, tau_gate: temperature for leaf softmax / gate sigmoid
        """
        x = x.to(DTYPE)
        y = y.to(DTYPE)
        batch_size = x.shape[0]

        # Leaf level: softmax over {1, x, y}
        leaf_probs = torch.softmax(self.leaf_logits / tau_leaf, dim=1)
        weights = leaf_probs.to(DTYPE)
        ones = torch.ones(batch_size, dtype=DTYPE)
        candidates = torch.stack([ones, x, y], dim=1)  # (batch, 3)
        current_level = torch.matmul(candidates, weights.T)  # (batch, n_leaves)

        # Bottom-up: merge pairs with EML + sigmoid gate
        node_idx = 0
        while current_level.shape[1] > 1:
            n_pairs = current_level.shape[1] // 2
            left_children = current_level[:, 0::2]
            right_children = current_level[:, 1::2]

            s = torch.sigmoid(
                self.blend_logits[node_idx:node_idx + n_pairs] / tau_gate)

            # Blend: input = s*1 + (1-s)*child
            # Separate real/imag to avoid 0*Inf=NaN in complex multiplication
            s_left = s[:, 0].unsqueeze(0)   # (1, n_pairs)
            s_right = s[:, 1].unsqueeze(0)
            bypass_left = s_left > _BYPASS_THR
            bypass_right = s_right > _BYPASS_THR
            oml = 1.0 - s_left
            omr = 1.0 - s_right

            lr = torch.where(bypass_left, 1.0,
                             s_left + oml * left_children.real)
            li = torch.where(bypass_left, 0.0,
                             oml * left_children.imag)
            rr = torch.where(bypass_right, 1.0,
                             s_right + omr * right_children.real)
            ri = torch.where(bypass_right, 0.0,
                             omr * right_children.imag)

            left_input = torch.complex(lr, li)
            right_input = torch.complex(rr, ri)

            current_level = self.op_fn(left_input, right_input)

            # NaN scrub + clamp (per-layer)
            c = self.eml_clamp
            current_level = torch.complex(
                torch.nan_to_num(current_level.real,
                                 nan=0.0, posinf=c, neginf=-c
                                 ).clamp(-c, c),
                torch.nan_to_num(current_level.imag,
                                 nan=0.0, posinf=c, neginf=-c
                                 ).clamp(-c, c),
            )

            node_idx += n_pairs

        gate_probs = torch.sigmoid(self.blend_logits)
        return (current_level.squeeze(1), leaf_probs, gate_probs)

    def snap_weights(self):
        """Snap to hard 0/1 choices. Returns snapped copy."""
        with torch.no_grad():
            lc = torch.argmax(self.leaf_logits, dim=1)
            new_leaf = torch.full_like(self.leaf_logits, -50.0)
            new_leaf[torch.arange(self.n_leaves), lc] = 50.0
            self.leaf_logits.copy_(new_leaf)

            gc = (self.blend_logits >= 0).to(self.blend_logits.dtype)
            new_gate = torch.where(
                gc > 0.5,
                torch.full_like(self.blend_logits, 50.0),
                torch.full_like(self.blend_logits, -50.0),
            )
            self.blend_logits.copy_(new_gate)

    def describe(self):
        """Human-readable description of the current (snapped) tree."""
        with torch.no_grad():
            leaf_probs = torch.softmax(self.leaf_logits, dim=1)
            leaf_choices = torch.argmax(leaf_probs, dim=1).cpu().numpy()
            gate_probs = torch.sigmoid(self.blend_logits)
            gate_choices = (gate_probs > 0.5).cpu().numpy()

        names = {0: '1', 1: 'x', 2: 'y'}
        lines = [f"EMLTreeV16 (depth={self.depth}, "
                 f"params={self.param_count()})"]
        lines.append(f"  Leaves: "
                     + ', '.join(names[c] for c in leaf_choices))
        lines.append(f"  Gates:  "
                     + ', '.join(f"({int(g[0])},{int(g[1])})"
                                 for g in gate_choices))
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Hybrid architecture — V16 body + softmax root
# ---------------------------------------------------------------------------

class EMLTreeHybrid(nn.Module):
    """V16 architecture with softmax {1, x, y, f} at the root level.

    Motivation: Eq.6 (x at all internals) gives 100% at depth 3 but 0% at
    depth 4 because the optimizer collapses the tree to a constant via
    internal x-shortcuts.  V16 (sigmoid {1, f} everywhere) prevents collapse
    but limits expressiveness (23% at depth 3).

    This hybrid allows x/y access ONLY at the root, preventing deep collapse
    while preserving shallow expressiveness.

    Architecture:
        - Leaves: softmax over {1, x, y} — same as V16
        - Internal nodes (all but root): sigmoid {1, f} — same as V16
        - Root node: softmax {1, x, y, f} for each of its 2 inputs
    """

    def __init__(self, depth, init_scale=1.0, init_strategy="biased",
                 eml_clamp=1e300, op_fn=None):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        # Internal nodes: 2^depth - 1 total, but root's 2 gates are separate
        self.n_sub_internal = self.n_leaves - 1 - 1  # all except root
        self.n_internal = self.n_leaves - 1
        self.eml_clamp = eml_clamp
        self.op_fn = op_fn or eml_torch

        # Leaf logits: same as V16 (3-way softmax per leaf)
        if init_strategy == "biased":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=torch.float64) * init_scale
            leaf_init[:, 0] += 2.0
            gate_init = torch.randn(max(self.n_sub_internal, 0), 2,
                                    dtype=torch.float64) * init_scale + 4.0
            root_init = torch.randn(2, 4, dtype=torch.float64) * init_scale
            root_init[:, 0] += 2.0  # bias toward 1
        elif init_strategy == "uniform":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=torch.float64) * init_scale
            gate_init = torch.randn(max(self.n_sub_internal, 0), 2,
                                    dtype=torch.float64) * init_scale + 4.0
            root_init = torch.randn(2, 4, dtype=torch.float64) * init_scale
        elif init_strategy == "xy_biased":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=torch.float64) * init_scale
            leaf_init[:, 1] += 1.0
            leaf_init[:, 2] += 1.0
            gate_init = torch.randn(max(self.n_sub_internal, 0), 2,
                                    dtype=torch.float64) * init_scale + 4.0
            root_init = torch.randn(2, 4, dtype=torch.float64) * init_scale
        elif init_strategy == "random_hot":
            leaf_init = torch.randn(self.n_leaves, 3, dtype=torch.float64) * init_scale
            hot_idx = torch.randint(0, 3, (self.n_leaves,))
            leaf_init[torch.arange(self.n_leaves), hot_idx] += 3.0
            gate_init = torch.randn(max(self.n_sub_internal, 0), 2,
                                    dtype=torch.float64) * init_scale + 3.0
            open_mask = torch.rand(max(self.n_sub_internal, 0), 2) < 0.25
            gate_init[open_mask] -= 6.0
            root_init = torch.randn(2, 4, dtype=torch.float64) * init_scale
            root_hot = torch.randint(0, 4, (2,))
            root_init[torch.arange(2), root_hot] += 3.0
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        self.leaf_logits = nn.Parameter(leaf_init)
        self.blend_logits = nn.Parameter(gate_init)
        # Root: 2 inputs × 4 choices {1, x, y, f_child}
        self.root_logits = nn.Parameter(root_init)

    def param_count(self):
        return self.n_leaves * 3 + max(self.n_sub_internal, 0) * 2 + 2 * 4

    def forward(self, x, y, tau_leaf=1.0, tau_gate=1.0):
        x = x.to(DTYPE)
        y = y.to(DTYPE)
        batch_size = x.shape[0]

        # Leaf level: softmax over {1, x, y}
        leaf_probs = torch.softmax(self.leaf_logits / tau_leaf, dim=1)
        weights = leaf_probs.to(DTYPE)
        ones = torch.ones(batch_size, dtype=DTYPE)
        candidates = torch.stack([ones, x, y], dim=1)
        current_level = torch.matmul(candidates, weights.T)

        # Bottom-up: sigmoid {1, f} for sub-root levels
        node_idx = 0
        while current_level.shape[1] > 2:
            n_pairs = current_level.shape[1] // 2
            left_children = current_level[:, 0::2]
            right_children = current_level[:, 1::2]

            s = torch.sigmoid(
                self.blend_logits[node_idx:node_idx + n_pairs] / tau_gate)

            s_left = s[:, 0].unsqueeze(0)
            s_right = s[:, 1].unsqueeze(0)
            bypass_left = s_left > _BYPASS_THR
            bypass_right = s_right > _BYPASS_THR
            oml = 1.0 - s_left
            omr = 1.0 - s_right

            lr = torch.where(bypass_left, 1.0,
                             s_left + oml * left_children.real)
            li = torch.where(bypass_left, 0.0, oml * left_children.imag)
            rr = torch.where(bypass_right, 1.0,
                             s_right + omr * right_children.real)
            ri = torch.where(bypass_right, 0.0, omr * right_children.imag)

            left_input = torch.complex(lr, li)
            right_input = torch.complex(rr, ri)
            current_level = self.op_fn(left_input, right_input)

            c = self.eml_clamp
            current_level = torch.complex(
                torch.nan_to_num(current_level.real,
                                 nan=0.0, posinf=c, neginf=-c).clamp(-c, c),
                torch.nan_to_num(current_level.imag,
                                 nan=0.0, posinf=c, neginf=-c).clamp(-c, c),
            )
            node_idx += n_pairs

        # Root level: softmax {1, x, y, f_child}
        left_child = current_level[:, 0]   # (batch,)
        right_child = current_level[:, 1]  # (batch,)

        root_probs = torch.softmax(self.root_logits / tau_gate, dim=1)
        # root_probs shape: (2, 4) — [left_input, right_input] × [1, x, y, f]
        rp = root_probs.to(DTYPE)  # (2, 4)

        left_input = (rp[0, 0] * ones + rp[0, 1] * x + rp[0, 2] * y
                      + rp[0, 3] * left_child)
        right_input = (rp[1, 0] * ones + rp[1, 1] * x + rp[1, 2] * y
                       + rp[1, 3] * right_child)

        result = self.op_fn(left_input, right_input)
        c = self.eml_clamp
        result = torch.complex(
            torch.nan_to_num(result.real,
                             nan=0.0, posinf=c, neginf=-c).clamp(-c, c),
            torch.nan_to_num(result.imag,
                             nan=0.0, posinf=c, neginf=-c).clamp(-c, c),
        )

        gate_probs = torch.sigmoid(self.blend_logits) if self.n_sub_internal > 0 \
            else torch.empty(0, 2)
        return result, leaf_probs, gate_probs

    def snap_weights(self):
        with torch.no_grad():
            lc = torch.argmax(self.leaf_logits, dim=1)
            new_leaf = torch.full_like(self.leaf_logits, -50.0)
            new_leaf[torch.arange(self.n_leaves), lc] = 50.0
            self.leaf_logits.copy_(new_leaf)

            if self.n_sub_internal > 0:
                gc = (self.blend_logits >= 0).to(self.blend_logits.dtype)
                new_gate = torch.where(
                    gc > 0.5,
                    torch.full_like(self.blend_logits, 50.0),
                    torch.full_like(self.blend_logits, -50.0),
                )
                self.blend_logits.copy_(new_gate)

            rc = torch.argmax(self.root_logits, dim=1)
            new_root = torch.full_like(self.root_logits, -50.0)
            new_root[torch.arange(2), rc] = 50.0
            self.root_logits.copy_(new_root)

    def describe(self):
        with torch.no_grad():
            leaf_probs = torch.softmax(self.leaf_logits, dim=1)
            leaf_choices = torch.argmax(leaf_probs, dim=1).cpu().numpy()
            root_probs = torch.softmax(self.root_logits, dim=1)
            root_choices = torch.argmax(root_probs, dim=1).cpu().numpy()

        names_leaf = {0: '1', 1: 'x', 2: 'y'}
        names_root = {0: '1', 1: 'x', 2: 'y', 3: 'f'}
        lines = [f"EMLTreeHybrid (depth={self.depth}, "
                 f"params={self.param_count()})"]
        lines.append(f"  Leaves: "
                     + ', '.join(names_leaf[c] for c in leaf_choices))
        lines.append(f"  Root:   ({names_root[root_choices[0]]}, "
                     f"{names_root[root_choices[1]]})")
        return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test():
    """Verify master formula properties and known parameter settings."""
    print("=== Master Formula Test ===\n")
    ok = 0
    fail = 0

    def check(name, got, expected, tol=1e-8):
        nonlocal ok, fail
        err = abs(complex(got) - complex(expected))
        if err < tol:
            ok += 1
            print(f"  [PASS] {name}")
        else:
            fail += 1
            print(f"  [FAIL] {name}: got {got}, expected {expected}, err={err:.2e}")

    # --- Parameter count formula: 5 * 2^n - 6 ---
    for n in [1, 2, 3, 4, 5]:
        model = EmlMasterFormula(depth=n, n_vars=1)
        expected_params = 5 * 2**n - 6 if n >= 2 else 4  # depth=1: 2 leaves * 2 logits = 4
        actual = model.param_count()
        check(f"depth={n} param_count={actual}", actual, expected_params, tol=0.5)

    # --- Recover exp(x) at depth 2 ---
    # exp(x) = eml(x, 1)
    # Set: α1=0, β1=1, γ1=0 (left root input = x)
    #       α2=1, β2=0, γ2=0 (right root input = 1)
    # Children don't matter since their weight (γ) = 0
    model = EmlMasterFormula(depth=2, n_vars=1)
    with torch.no_grad():
        # Set internal logits to route: left=x, right=1
        # internal_logits[0] → left root input: want β=1 → large β logit
        model.internal_logits[0] = torch.tensor([0.0, 100.0, 0.0])  # → x
        # internal_logits[1] → right root input: want α=1 → large α logit
        model.internal_logits[1] = torch.tensor([100.0, 0.0, 0.0])  # → 1

    test_x = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float64)
    result, _, _ = model(test_x)
    for i, xv in enumerate(test_x):
        check(f"exp({xv.item()}) via master",
              result[i].real.item(), np.exp(xv.item()), tol=1e-6)

    # --- Recover constant e at depth 2 ---
    # e = eml(1, 1)
    model2 = EmlMasterFormula(depth=2, n_vars=1)
    with torch.no_grad():
        model2.internal_logits[0] = torch.tensor([100.0, 0.0, 0.0])  # → 1
        model2.internal_logits[1] = torch.tensor([100.0, 0.0, 0.0])  # → 1

    result2, _, _ = model2(test_x)
    for i in range(len(test_x)):
        check(f"e (constant) via master",
              result2[i].real.item(), np.e, tol=1e-6)

    # --- Recover exp(exp(x)) at depth 2 ---
    # exp(exp(x)) = eml(eml(x, 1), 1)
    # Root left: γ1=1 → child eml output
    # Root right: α2=1 → 1
    # Child left: needs to be eml(x, 1)
    #   child left input: β=1 → x
    #   child right input: α=1 → 1
    model3 = EmlMasterFormula(depth=2, n_vars=1)
    with torch.no_grad():
        # Left root input: route to child (γ=1)
        model3.internal_logits[0] = torch.tensor([0.0, 0.0, 100.0])  # → f
        # Right root input: route to 1 (α=1)
        model3.internal_logits[1] = torch.tensor([100.0, 0.0, 0.0])  # → 1
        # Left child's leaves: left=x, right=1
        model3.leaf_logits[0] = torch.tensor([0.0, 100.0])  # → x
        model3.leaf_logits[1] = torch.tensor([100.0, 0.0])  # → 1
        # Right child's leaves: don't matter (weight=0), but set to something
        model3.leaf_logits[2] = torch.tensor([100.0, 0.0])  # → 1
        model3.leaf_logits[3] = torch.tensor([100.0, 0.0])  # → 1

    result3, _, _ = model3(test_x)
    for i, xv in enumerate(test_x):
        expected = np.exp(np.exp(xv.item()))
        check(f"exp(exp({xv.item()})) via master",
              result3[i].real.item(), expected, tol=1e-4)

    # --- Forward pass shape test ---
    model4 = EmlMasterFormula(depth=3, n_vars=1)
    batch = torch.randn(10, dtype=torch.float64)
    out, _, _ = model4(batch)
    check("depth=3 output shape", out.shape[0], 10, tol=0.5)

    # --- Bivariate test ---
    model5 = EmlMasterFormula(depth=2, n_vars=2)
    check(f"bivariate param_count", model5.param_count(), 5 * 4 - 6 + 4, tol=0.5)
    # 5*2^2 - 6 = 14 for univariate; bivariate adds 1 logit per leaf = +4 → 18
    # Actually: internal still 3 logits each (2 internal inputs),
    # leaves have 3 logits each (4 leaves) → 2*3 + 4*3 = 18
    out5, _, _ = model5(torch.tensor([1.0]), torch.tensor([2.0]))
    check("bivariate forward runs", len(out5), 1, tol=0.5)

    # --- Bivariate model must reject missing y ---
    try:
        model5(torch.tensor([1.0]))
        ok_fail = False
    except ValueError:
        ok_fail = True
    check("bivariate rejects missing y", ok_fail, True, tol=0.5)

    # --- V16 architecture tests ---
    print("\n--- EMLTreeV16 tests ---")

    # V16 parameter count: n_leaves*3 + n_internal*2
    for n in [2, 3, 4]:
        v16 = EMLTreeV16(depth=n)
        expected = (2**n) * 3 + (2**n - 1) * 2
        check(f"V16 depth={n} params={v16.param_count()}", v16.param_count(),
              expected, tol=0.5)

    # V16 forward pass runs
    v16_2 = EMLTreeV16(depth=2)
    xv = torch.tensor([1.0, 2.0], dtype=torch.float64)
    yv = torch.tensor([1.5, 2.5], dtype=torch.float64)
    out_v16, lp, gp = v16_2(xv, yv)
    check("V16 forward shape", out_v16.shape[0], 2, tol=0.5)
    check("V16 leaf_probs shape", lp.shape[0], 4, tol=0.5)
    check("V16 gate_probs shape", gp.shape[0], 3, tol=0.5)

    # V16 with blend=1 everywhere → all inputs are 1 → eml(1,1) = e
    v16_const = EMLTreeV16(depth=2, init_strategy="manual")
    with torch.no_grad():
        v16_const.leaf_logits[:, 0] = 50.0  # select 1 at all leaves
        v16_const.blend_logits.fill_(50.0)   # all gates → 1
    out_c, _, _ = v16_const(xv, yv)
    for i in range(len(xv)):
        check(f"V16 all-1 gives e", out_c[i].real.item(), np.e, tol=1e-6)

    # V16 depth-2 target: eml(1, eml(y, x)) = e - log(exp(y) - log(x))
    # Tree: root=eml(1, child_right), child_right=eml(y, x)
    # Root: left gate=1 (use 1), right gate=0 (use child)
    # Bottom-right: left gate=0 (use leaf=y), right gate=0 (use leaf=x)
    # Bottom-left: doesn't matter (root ignores it)
    v16_d2 = EMLTreeV16(depth=2, init_strategy="manual")
    with torch.no_grad():
        # All leaves start at 1
        v16_d2.leaf_logits[:, 0] = 50.0
        # Bottom-right node uses leaf[2]=y, leaf[3]=x
        v16_d2.leaf_logits[2, :] = torch.tensor([-50.0, -50.0, 50.0])  # y
        v16_d2.leaf_logits[3, :] = torch.tensor([-50.0, 50.0, -50.0])  # x
        # Nodes bottom-up: [0]=bottom-left, [1]=bottom-right, [2]=root
        # Bottom-left: both gates closed (doesn't matter), set to 1
        v16_d2.blend_logits[0, :] = torch.tensor([50.0, 50.0])
        # Bottom-right: both gates open (use leaves y, x)
        v16_d2.blend_logits[1, :] = torch.tensor([-50.0, -50.0])
        # Root: left=1 (gate closed), right=child (gate open)
        v16_d2.blend_logits[2, :] = torch.tensor([50.0, -50.0])

    test_x = torch.tensor([1.5, 2.0], dtype=torch.float64)
    test_y = torch.tensor([1.2, 1.8], dtype=torch.float64)
    out_d2, _, _ = v16_d2(test_x, test_y)
    for i in range(len(test_x)):
        expected = np.exp(1.0) - np.log(
            np.exp(test_y[i].item()) - np.log(test_x[i].item()))
        check(f"V16 eml(1,eml(y,x)) at ({test_x[i]:.1f},{test_y[i]:.1f})",
              out_d2[i].real.item(), expected.real, tol=1e-6)

    # V16 init strategies don't crash
    for strat in INIT_STRATEGIES:
        torch.manual_seed(42)
        v = EMLTreeV16(depth=3, init_strategy=strat)
        o, _, _ = v(xv, yv)
        check(f"V16 init={strat} runs", o.shape[0], 2, tol=0.5)

    print(f"\n{ok} passed, {fail} failed out of {ok + fail} tests.")
    return fail == 0


if __name__ == "__main__":
    _self_test()

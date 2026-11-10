"""Mixing of the positive and negative regression losses.

Both objectives have the form (Sec. 3.2)
    L_i = lam * [q_i L_i^+ + (1 - q_i) L_i^-]
        = lam/2 (L_i^+ + L_i^-) + psi_i/2 (L_i^+ - L_i^-),   psi_i = lam (2 q_i - 1),
and both weights lam q_i and lam (1 - q_i) are non-negative iff
lam >= max_i |psi_i|.

L_i^+ and L_i^- are the per-sample regression losses of DiffusionNFT
(``positive_loss`` and ``negative_loss`` in its training script); they are not
changed.
"""

import math

import torch


def nft_mixing(batch_std_advantage, clip=5.0):
    """DiffusionNFT (Sec. 3.3): lam = C, q_i = 1/2 + clip(A~_i, -C, C) / (2C)."""
    clipped = torch.clamp(batch_std_advantage, -clip, clip)
    return clip, 0.5 + clipped / (2.0 * clip)


def anchor_coefficient(group_size):
    """C_G = sqrt(G - 1), the bound on |A_i| (Proposition 1)."""
    return math.sqrt(group_size - 1)


def proposed_mixing(group_advantage, gate, group_size):
    """Proposed objective (Sec. 4.4).

    lam = C_G,  q_i = 1/2 + tanh(A_i / C_G) / 2,  q_i^eff = 1/2 + g_i (q_i - 1/2),
    i.e. psi_i = g_i C_G tanh(A_i / C_G) and |psi_i| <= C_G.
    """
    c_g = anchor_coefficient(group_size)
    q = 0.5 + 0.5 * torch.tanh(group_advantage / c_g)
    return c_g, 0.5 + gate * (q - 0.5)


def mixture_loss(positive_loss, negative_loss, lam, q):
    """Batch mean of lam * [q L^+ + (1 - q) L^-]."""
    return (lam * (q * positive_loss + (1.0 - q) * negative_loss)).mean()

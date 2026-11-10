"""Numerical checks of the statements in Sec. 4 (run: python -m pytest tests)."""

import math

import numpy as np
import torch

from advantage_bound import (
    anchor_coefficient,
    group_standardized_advantage,
    nft_mixing,
    proposed_mixing,
    sign_stability_gate,
)

GROUP_SIZES = (6, 12, 16, 24)


def _random_groups(rng, group_size, num_groups=200, binary=False):
    if binary:
        rewards = rng.integers(0, 2, size=(num_groups, group_size)).astype(float)
    else:
        rewards = rng.random((num_groups, group_size))
    prompts = np.repeat(np.arange(num_groups), group_size)
    return prompts, rewards.reshape(-1)


def test_advantage_bound():
    """Proposition 1: |A_i| <= sqrt(G - 1), with equality for one outlier."""
    rng = np.random.default_rng(0)
    for g in GROUP_SIZES:
        for binary in (False, True):
            prompts, rewards = _random_groups(rng, g, binary=binary)
            a = group_standardized_advantage(prompts, rewards, eps=0.0 if not binary else 1e-4)
            assert np.all(np.abs(a) <= math.sqrt(g - 1) + 1e-9)
        outlier = np.zeros(g)
        outlier[0] = 1.0
        a = group_standardized_advantage(np.zeros(g), outlier, eps=0.0)
        assert math.isclose(a[0], math.sqrt(g - 1), rel_tol=1e-12)


def test_self_omission_keeps_sign():
    """Proposition 2: dropping sample i itself never flips sign(A_i)."""
    rng = np.random.default_rng(1)
    for g in GROUP_SIZES:
        group = rng.random(g)
        centered = group - group.mean()
        for i in range(g):
            loo = np.delete(group, i)
            assert np.sign(group[i] - loo.mean()) == np.sign(centered[i])


def test_gate_range():
    """rho_G <= g_i <= 1, and g_i = 1 when every omission keeps the sign."""
    rng = np.random.default_rng(2)
    for g in GROUP_SIZES:
        for binary in (False, True):
            prompts, rewards = _random_groups(rng, g, binary=binary)
            gate = sign_stability_gate(prompts, rewards)
            assert np.all(gate >= 2.0 / (g + 2.0) - 1e-12)
            assert np.all(gate <= 1.0 + 1e-12)
        well_separated = np.r_[np.zeros(g // 2), np.ones(g - g // 2)]
        assert np.allclose(sign_stability_gate(np.zeros(g), well_separated), 1.0)


def test_weights_non_negative():
    """lam >= |psi_i| for the proposed objective, for any group (Sec. 4.4)."""
    rng = np.random.default_rng(3)
    for g in GROUP_SIZES:
        prompts, rewards = _random_groups(rng, g)
        a = torch.as_tensor(group_standardized_advantage(prompts, rewards))
        gate = torch.as_tensor(sign_stability_gate(prompts, rewards))
        lam, q = proposed_mixing(a, gate, g)
        assert lam == anchor_coefficient(g)
        assert torch.all(lam * q >= 0) and torch.all(lam * (1 - q) >= 0)


def test_nft_clip_inactive_on_group_advantage():
    """sqrt(G - 1) < 5 for the group sizes used, so the clip at C = 5 is inactive."""
    for g in GROUP_SIZES:
        a = torch.linspace(-math.sqrt(g - 1), math.sqrt(g - 1), 101, dtype=torch.float64)
        lam, q = nft_mixing(a)
        assert torch.allclose(lam * (2 * q - 1), a)

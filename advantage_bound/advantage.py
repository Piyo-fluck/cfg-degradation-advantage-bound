"""Advantages and the sign-stability gate.

Section numbers refer to the paper (see README).

* ``batch_std_advantage``: the advantage of the DiffusionNFT implementation
  (group mean, standard deviation over all rewards of the update), Sec. 3.3.
* ``group_standardized_advantage``: A_i = (r_i - mean) / (std + eps) within a
  group, Sec. 3.1. Its magnitude is at most sqrt(G - 1) (Proposition 1).
* ``sign_stability_gate``: g_i = rho_G + (1 - rho_G) s_i, Sec. 4.3.

All standard deviations use the number of samples as the denominator, as in
the DiffusionNFT implementation.
"""

import numpy as np

EPS = 1e-4
# |value| at or below this counts as sign 0 when comparing signs.
SIGN_MARGIN = 1e-6


def _groups(prompts):
    prompts = np.asarray(prompts)
    for prompt in np.unique(prompts):
        yield np.flatnonzero(prompts == prompt)


def batch_std_advantage(prompts, rewards, eps=EPS):
    """(r_i - group mean) / (std of all rewards in the batch + eps)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    batch_std = rewards.std()
    advantages = np.zeros_like(rewards)
    for idx in _groups(prompts):
        advantages[idx] = (rewards[idx] - rewards[idx].mean()) / (batch_std + eps)
    return advantages


def _standardize(values, reference, eps):
    return (values - reference.mean()) / (reference.std() + eps)


def group_standardized_advantage(prompts, rewards, eps=EPS):
    """(r_i - group mean) / (group std + eps)."""
    rewards = np.asarray(rewards, dtype=np.float64)
    advantages = np.zeros_like(rewards)
    for idx in _groups(prompts):
        advantages[idx] = _standardize(rewards[idx], rewards[idx], eps)
    return advantages


def _sign(x, margin=SIGN_MARGIN):
    return np.where(np.abs(x) > margin, np.sign(x), 0.0)


def sign_stability_gate(prompts, rewards, eps=EPS):
    """Gate g_i in [rho_G, 1] from the leave-one-out stability of sign(A_i).

    For each j, the group statistics are recomputed without sample j and every
    sample i (including i = j) is standardized with them, giving A_i^(-j).
    k_i counts the j with sign(A_i^(-j)) = sign(A_i). The match at j = i always
    holds (Proposition 2) and is not counted as evidence:
        s_i = (k_i - 1) / (G - 1),  g_i = rho_G + (1 - rho_G) s_i,
        rho_G = 2 / (G + 2).
    A sample whose reward equals the group mean has A_i = 0 and gets
    g_i = rho_G; its directional coefficient is zero either way.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    gate = np.ones_like(rewards)
    for idx in _groups(prompts):
        group = rewards[idx]
        size = len(group)
        if size < 2:
            raise ValueError("the gate needs at least 2 samples per group")
        base_sign = _sign(group - group.mean())
        matches = np.zeros(size)
        for j in range(size):
            loo_advantage = _standardize(group, np.delete(group, j), eps)
            matches += (_sign(loo_advantage) == base_sign) & (base_sign != 0.0)
        rho = 2.0 / (size + 2.0)
        stability = np.clip((matches - 1.0) / (size - 1.0), 0.0, 1.0)
        gate[idx] = rho + (1.0 - rho) * stability
    return gate

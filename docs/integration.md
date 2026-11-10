# Applying the proposed objective to DiffusionNFT

The experiments modified the training script of the DiffusionNFT implementation,
<https://github.com/NVlabs/DiffusionNFT> (`scripts/train_nft_sd3.py`). Line
numbers below refer to commit `cbb14f8` of that repository. Only two places
change; the sampler, the rewards, the regression losses `positive_loss` /
`negative_loss`, and the penalty on the deviation from the reference policy
(`config.train.beta * kl_div_loss`) are unchanged.

## 1. Advantages and the gate (after the reward gather, around line 750)

DiffusionNFT computes the advantage with
`PerPromptStatTracker(config.sample.global_std)`. The default
`global_std=True` divides by the standard deviation of all rewards of the update
(`advantage_bound.batch_std_advantage`).

For the proposed objective, compute the group-standardized advantage and the
gate from the same gathered prompts and rewards. The rewards are repeated over
the training timesteps (line 723), so compute on one column and repeat:

```python
from advantage_bound import group_standardized_advantage, sign_stability_gate

rewards = gathered_rewards_dict["avg"]            # (num_samples, num_train_timesteps)
adv = group_standardized_advantage(prompts_all_decoded, rewards[:, 0])
gate = sign_stability_gate(prompts_all_decoded, rewards[:, 0])
advantages = np.repeat(adv[:, None], rewards.shape[1], axis=1)
gate = np.repeat(gate[:, None], rewards.shape[1], axis=1)
```

Distribute `gate` to the processes exactly like `advantages` (lines 774-783),
for example as `collated_samples["gate"]`. It is then shuffled together with the
other per-sample tensors.

## 2. Mixing of the two losses (lines 928-929 and 959-960)

DiffusionNFT mixes the losses with `r = clip(A, -C, C) / (2C) + 1/2` and scales
by `C = config.train.adv_clip_max = 5` (`advantage_bound.nft_mixing`). Replace the
mixing ratio and the scale:

```python
from advantage_bound import proposed_mixing

lam, r = proposed_mixing(
    train_sample_batch["advantages"][:, j_idx],
    train_sample_batch["gate"][:, j_idx],
    config.sample.num_image_per_prompt,
)
ori_policy_loss = r * positive_loss / config.beta + (1.0 - r) * negative_loss / config.beta
policy_loss = (ori_policy_loss * lam).mean()
```

In the experiments the advantage was still passed through
`clamp(-adv_clip_max, adv_clip_max)` before this step. With
`adv_clip_max = 5` and G <= 24 this has no effect, because |A_i| <= sqrt(G - 1) < 5.

## Correspondence with the paper

| paper | code |
|---|---|
| A_i (Sec. 3.1) | `group_standardized_advantage` |
| A~_i, C = 5 (Sec. 3.3) | `batch_std_advantage`, `nft_mixing` |
| C_G = sqrt(G - 1) (Sec. 4.1-4.2) | `anchor_coefficient` |
| k_i, s_i, rho_G, g_i (Sec. 4.3) | `sign_stability_gate` |
| q_i, q_i^eff, Eq. (5) (Sec. 4.4) | `proposed_mixing`, `mixture_loss` |

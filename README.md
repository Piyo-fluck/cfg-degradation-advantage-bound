# Mitigating Performance Degradation under Inference-Time CFG Based on the Upper Bound of the Group-Standardized Advantage

画像生成 RL における群内標準化アドバンテージの上限に基づく推論時 CFG 下での性能低下の抑制

Shiyu Nagao, Akihiro Inokuchi (Kwansei Gakuin University)
[JSAI SIG-KBS](https://ai-gakkai.or.jp/sig-kbs/), 2026 (in Japanese)

This repository contains the implementation of the proposed objective and the
information needed to reproduce the experiments of the paper. The experiments
modified the DiffusionNFT implementation (<https://github.com/NVlabs/DiffusionNFT>);
this repository does not contain a copy of it.

## Summary

Online RL post-training of text-to-image diffusion models, such as DiffusionNFT,
trains and evaluates at classifier-free guidance (CFG) scale w = 1, while
inference commonly uses a larger scale (4.5 for Stable Diffusion 3.5). Evaluating
across scales, the scores at larger scales fall as training proceeds, even while
the score at w = 1 keeps improving.

The DiffusionNFT loss is a weighted sum of two regression losses,
L_i = lam/2 (L_i^+ + L_i^-) + psi_i/2 (L_i^+ - L_i^-). Both weights are
non-negative iff the anchor coefficient lam is at least |psi_i|. The DiffusionNFT
implementation normalizes the advantage by the batch-wide reward standard
deviation and uses C = 5 as both the clipping threshold and lam. For a group of
G samples, the group-standardized advantage A_i satisfies |A_i| <= sqrt(G - 1).
The proposed objective therefore

* standardizes the advantage within each group,
* sets the anchor coefficient to C_G = sqrt(G - 1),
* saturates the directional coefficient, psi_i = g_i C_G tanh(A_i / C_G),
* gates it by the leave-one-out stability of the sign of the advantage,
  g_i = rho_G + (1 - rho_G) s_i with rho_G = 2 / (G + 2).

C_G and rho_G follow from G; no tuned coefficient is introduced.

GenEval score (x100) at checkpoint 200, change from w = 1.0 to w = 4.5
(GenEval as the reward, Stable Diffusion 3.5 Medium):

| G | DiffusionNFT | proposed |
|---|---|---|
| 24 | -15.37 | +1.18 |
| 12 | -85.17 | +1.94 |
| 6 | -37.24 | +1.01 |

With PickScore and HPSv2.1 as the reward, the score at w = 4.5 fell for the
proposed objective as well.

## Contents

| path | contents |
|---|---|
| `group_bound/advantage.py` | DiffusionNFT advantage, group-standardized advantage, sign-stability gate |
| `group_bound/loss.py` | mixing of the two regression losses for DiffusionNFT and the proposed objective |
| `group_bound/evaluation.py` | evaluation seeds and standard errors |
| `tests/test_properties.py` | numerical checks of Propositions 1 and 2 and of the non-negative weights |
| `docs/integration.md` | where the two changes go in the DiffusionNFT training script |
| `docs/training.md` | training settings of every run |
| `docs/evaluation.md` | evaluation protocol |

`group_bound` returns the same advantages, gate and mixing weights as the code
that produced the results.

## Requirements

Python 3.10 or later, NumPy and PyTorch. To run the checks:

```bash
pip install numpy torch pytest
python -m pytest tests
```

## License

Apache License 2.0 (see `LICENSE` and `NOTICE`). DiffusionNFT is licensed under
the Apache License 2.0. The prompt sets, model weights and reward models used in
the experiments are not included and are subject to their own licenses.

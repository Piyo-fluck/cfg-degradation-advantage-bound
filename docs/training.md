# Training settings

All runs used 4 GPUs. The baseline (DiffusionNFT) and the proposed objective
share every setting below and differ only in the advantage, the anchor
coefficient and the directional coefficient (docs/integration.md).

## Common settings

| item | value |
|---|---|
| base model | Stable Diffusion 3.5 Medium (`stabilityai/stable-diffusion-3.5-medium`) |
| adapter | LoRA, rank 32, alpha 64, Gaussian initialization, on `attn.to_q`, `attn.to_k`, `attn.to_v`, `attn.to_out.0`, `attn.add_q_proj`, `attn.add_k_proj`, `attn.add_v_proj`, `attn.to_add_out` |
| resolution | 512 x 512 |
| training sampler | 10 steps, guidance scale w = 1, DPM2 solver, `noise_level=0.7`, `deterministic=True` |
| precision | fp16, TF32 allowed |
| optimizer | AdamW, learning rate 3e-4, betas (0.9, 0.999), weight decay 1e-4, eps 1e-8, gradient clipping 1.0 |
| penalty on deviation from the reference policy | 1e-4 (`train.beta`) |
| timestep fraction | 0.99 |
| EMA | on |
| inner epochs | 1 |
| advantage eps | 1e-4 |
| DiffusionNFT clip threshold C | 5 (`train.adv_clip_max`) |
| DiffusionNFT advantage normalization | standard deviation of all rewards of the update (`sample.global_std=True`) |
| seed | 42 (all runs) |

## Per-run settings

`batch/GPU`, `batches/epoch` and `grad. accum.` are `sample.train_batch_size`,
`sample.num_batches_per_epoch` and `train.gradient_accumulation_steps`.
`NFT beta` is `config.beta` and `decay` is `config.decay_type`; both are the
DiffusionNFT settings for the old policy.
`fixed noise` means that the initial noise of each training image is fixed and
the two methods use the same sequence of initial noise.

| reward | G | prompts/epoch | images/epoch | batch/GPU | batches/epoch | grad. accum. | epochs | checkpoint every | NFT beta | decay | fixed noise |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GenEval | 24 | 48 | 1152 | 24 | 12 | 12 | 200 | 20 | 1.0 | 1 | yes |
| GenEval | 12 | 96 | 1152 | 24 | 12 | 12 | 200 | 20 | 1.0 | 1 | no |
| GenEval | 6 | 192 | 1152 | 24 | 12 | 12 | 200 | 20 | 1.0 | 1 | no |
| PickScore | 24 | 48 | 1152 | 24 | 12 | 12 | 200 | 20 | 1.0 | 1 | yes |
| HPSv2.1 | 24 | 48 | 1152 | 24 | 12 | 12 | 200 | 20 | 1.0 | 1 | yes |
| OCR | 24 | 48 | 1152 | 24 | 12 | 6 | 100 | 20 | 0.1 | 2 | yes |
| GenEval2 | 16 | 64 | 1024 | 8 | 32 | 32 | 600 | 50 | 1.0 | 1 | no |

## Prompts and rewards

| reward | training prompts | reward |
|---|---|---|
| GenEval | `dataset/geneval/train_metadata.jsonl` of the DiffusionNFT repository | binary GenEval judgement |
| PickScore | `dataset/pickscore/train.txt` of the DiffusionNFT repository | PickScore |
| HPSv2.1 | `dataset/pickscore/train.txt` of the DiffusionNFT repository | HPSv2.1 |
| OCR | `dataset/ocr/train.txt` of the DiffusionNFT repository | OCR |
| GenEval2 | `datasets/geneval2/synthetic/train.jsonl` (20,000 prompts) of the Flow-DPPO repository, <https://github.com/Tencent-Hunyuan/UniRL> | GenEval2 Soft-TIFA (Qwen3-VL-8B-Instruct) |

The prompt files are not redistributed here.

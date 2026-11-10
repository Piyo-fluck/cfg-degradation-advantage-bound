# Evaluation protocol

Every saved checkpoint is evaluated with the LoRA adapter applied to
Stable Diffusion 3.5 Medium, at guidance scales w in {1.0, 2.0, 3.0, 4.5}.
Differences between guidance scales are always measured on the same checkpoint.

## GenEval (Sec. 5.2, 6.2)

| item | value |
|---|---|
| prompts | the official 553 prompts x 4 images = 2212 images (`dataset/geneval/test_metadata.jsonl` of the DiffusionNFT repository, one row per image) |
| sampler | 40 steps, 512 x 512, fp16 |
| guidance scales | w = 1.0 and 4.5 at every saved checkpoint (20, 40, ..., 200); w = 2.0 and 3.0 at checkpoint 200 |
| initial noise | `advantage_bound.evaluation.eval_seed(20260929, row, "geneval")`, where `row` is the row index (0-2211); independent of batch size and processing order |
| scorer | the GenEval scorer used for the training reward |
| metric | overall = macro average of the six category scores (`single_object`, `two_object`, `counting`, `colors`, `position`, `color_attr`) |
| standard error | the prompt as the cluster (`advantage_bound.evaluation.geneval_overall`) |
| paired differences | per-prompt differences between two conditions on the same prompts, aggregated with the same function |

Degradation criterion, fixed before the evaluation: on the same checkpoint, the
w = 4.5 score is lower than the w = 1 score by more than twice the standard error
of the prompt-paired difference.

## PickScore and HPSv2.1 (Sec. 6.3)

| item | value |
|---|---|
| prompts | the 2048 evaluation prompts of Pick-a-Pic (`dataset/pickscore/test.txt` of the DiffusionNFT repository), one image per prompt |
| sampler | as for GenEval |
| initial noise | `eval_seed(20260929, row, "pickscore")` for both rewards |
| scorer | the reward function used in training |
| standard error | over prompts (`advantage_bound.evaluation.mean_and_se`) |

## GenEval2 (Sec. 6.4)

| item | value |
|---|---|
| prompts | the official 800 prompts (`datasets/geneval2/synthetic/test.jsonl` of the Flow-DPPO repository, <https://github.com/Tencent-Hunyuan/UniRL>), one image per prompt |
| checkpoints | 450 and 600, at all four guidance scales |
| sampler | 40 steps, 512 x 512, fp16, sigmas `linspace(1, 1/40, 40)` before the scheduler shift, `max_sequence_length=256` |
| initial noise | `advantage_bound.evaluation.geneval2_seed(prompt)` (CRC32 of the prompt text) |
| scorer | Qwen3-VL-8B-Instruct. Each question of the prompt is asked as "`<question>` Answer in one word."; the score of a question is the probability of the expected answer, summed over its surface forms, in the full-vocabulary softmax of the first generated token |
| per-prompt score | geometric mean over the prompt's questions (Soft-TIFA GM) |
| metric | mean over the 800 prompts, x100 |

Check of the evaluation pipeline: the published Flow-DPPO model evaluated with
this protocol at w = 1 gives 84.23 (reported value 84.1).

Surface forms of the expected answer: "Yes", "yes", " Yes", " yes" for yes/no
questions; for "How many ..." questions, the number word in lower case and
capitalized, each with and without a leading space, and the digit with and
without a leading space.

"""Seeds and standard errors used in the evaluation (Sec. 5.2, 6.2-6.4).

See docs/evaluation.md for the full protocol.
"""

import hashlib
import math
import zlib

GENEVAL_CATEGORIES = (
    "single_object", "two_object", "counting", "colors", "position", "color_attr",
)


def eval_seed(base_seed, sample_id, tag="geneval"):
    """Seed of the initial noise of one evaluation sample.

    It depends only on the base seed and the sample's row in the prompt file,
    not on batch size or processing order. The paper used base_seed=20260929,
    tag="geneval" for GenEval and tag="pickscore" for PickScore / HPSv2.1.
    """
    payload = f"{int(base_seed)}|{int(sample_id)}|{tag}-eval-v1".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31 - 1)


def geneval2_seed(prompt):
    """Seed of the single GenEval2 sample of a prompt."""
    return zlib.crc32(prompt.encode("utf-8")) & 0x7FFFFFFF


def geneval_overall(per_prompt, category):
    """GenEval overall (x100) and its prompt-clustered standard error.

    per_prompt: {prompt: mean correctness over its images}
    category:   {prompt: one of GENEVAL_CATEGORIES}
    The overall score is the macro average of the six category means. With the
    prompt as the cluster, its variance is sum_c var_c / (n_c * 36), where var_c
    is the sample variance of the per-prompt values in category c.
    Passing per-prompt differences between two conditions (same prompts) gives
    the paired difference and its standard error.
    """
    mean, var = 0.0, 0.0
    k = len(GENEVAL_CATEGORIES)
    for c in GENEVAL_CATEGORIES:
        xs = [v for p, v in per_prompt.items() if category[p] == c]
        n = len(xs)
        m = sum(xs) / n
        var_c = sum((x - m) ** 2 for x in xs) / (n - 1)
        mean += m / k
        var += var_c / n / k**2
    return 100 * mean, 100 * math.sqrt(var)


def mean_and_se(values):
    """Mean and standard error over prompts (one image per prompt)."""
    n = len(values)
    m = sum(values) / n
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    return m, math.sqrt(var / n)

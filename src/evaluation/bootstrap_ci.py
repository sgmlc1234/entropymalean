"""Bootstrap CIs on dataset row indices.

Intervals resample dataset row indices, not individual repeats.

The caller must hand the per-row vector in a deterministic order. The resample
draws indices, so two orderings of the same rows are two different draws from a
seeded generator, and the bound moves --- which is a property of the run rather
than of the data, and not something a reported interval may depend on.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple


def _resample_rows(per_row: Sequence[float], rng: random.Random) -> List[float]:
    n = len(per_row)
    return [per_row[rng.randrange(n)] for _ in range(n)]


def _resample_means(per_row: Sequence[float], rng: random.Random, iterations: int) -> List[float]:
    """Means of `iterations` row-resamples of `per_row`.

    For a 0/1 vector --- which is what a Pass@K indicator is --- drawing n rows
    with replacement and averaging is exactly Binomial(n, k/n)/n, so the shortcut
    is the same estimator rather than an approximation of it. It matters because
    the direct form costs O(iterations * n) list indexing, which held the run to
    a thousand resamples, and at a thousand the interval bounds carried about a
    point of Monte Carlo noise --- enough to decide on its own whether a bound
    sat above zero. Anything not 0/1 falls back to the direct form.
    """
    if all(v in (0.0, 1.0) for v in per_row):
        n = len(per_row)
        p = sum(per_row) / n
        return [rng.binomialvariate(n, p) / n for _ in range(iterations)]
    return [sum(_resample_rows(per_row, rng)) / len(per_row) for _ in range(iterations)]


def bootstrap_ci(
    per_row_accuracy: Sequence[float],
    *,
    iterations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) for a single arm.

    `per_row_accuracy[i]` is the row-level accuracy (0..1) for problem i,
    typically averaged over the three repeats.
    """
    if not per_row_accuracy:
        return (0.0, 0.0, 0.0)
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(iterations):
        sample = _resample_rows(per_row_accuracy, rng)
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(iterations * (alpha / 2))]
    hi = means[int(iterations * (1 - alpha / 2)) - 1]
    mean = sum(per_row_accuracy) / len(per_row_accuracy)
    return (mean, lo, hi)


def bootstrap_drop_ci(
    control_acc: Sequence[float],
    treatment_acc: Sequence[float],
    *,
    iterations: int = 200_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Return (drop_mean_pp, lo_pp, hi_pp) for control-minus-treatment.

    The control and treatment arms are resampled independently because they
    contain different row indices; this matches the per-cell bootstrap used in
    the EntropyMaG-1 paper.
    """
    if not control_acc or not treatment_acc:
        return (0.0, 0.0, 0.0)
    rng_c = random.Random(seed)
    rng_t = random.Random(seed + 1)
    cs = _resample_means(control_acc, rng_c, iterations)
    ts = _resample_means(treatment_acc, rng_t, iterations)
    drops: List[float] = [100.0 * (c - t) for c, t in zip(cs, ts)]
    drops.sort()
    lo = drops[int(iterations * (alpha / 2))]
    hi = drops[int(iterations * (1 - alpha / 2)) - 1]
    mean = 100.0 * (
        sum(control_acc) / len(control_acc) - sum(treatment_acc) / len(treatment_acc)
    )
    return (mean, lo, hi)

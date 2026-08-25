from __future__ import annotations

from typing import TypedDict

import numpy as np

# Fixed seed so a given set of values always yields the same interval; reports
# must be reproducible from the same inputs.
_BOOTSTRAP_SEED = 20240501


class Interval(TypedDict):
    n: int
    mean: float | None
    std: float | None
    median: float | None
    p95: float | None
    min: float | None
    max: float | None
    ci_low: float | None
    ci_high: float | None
    ci_level: float


def summarize_values(
    values: list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
) -> Interval:
    """Central tendency, dispersion, and a bootstrap CI for the mean."""
    if not values:
        return Interval(
            n=0,
            mean=None,
            std=None,
            median=None,
            p95=None,
            min=None,
            max=None,
            ci_low=None,
            ci_high=None,
            ci_level=confidence,
        )
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    ci_low, ci_high = bootstrap_ci(arr, confidence=confidence, resamples=resamples)
    return Interval(
        n=int(arr.size),
        mean=mean,
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        median=float(np.median(arr)),
        p95=float(np.percentile(arr, 95)),
        min=float(arr.min()),
        max=float(arr.max()),
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=confidence,
    )


def bootstrap_ci(
    values: np.ndarray | list[float],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean.

    Non-parametric: makes no normality assumption, which suits skewed and
    bounded TTS metrics (WER, similarity, silence ratio). With fewer than two
    samples the interval collapses to the point estimate.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (0.0, 0.0)
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    samples = rng.choice(arr, size=(resamples, arr.size), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low = float(np.percentile(means, 100 * alpha))
    high = float(np.percentile(means, 100 * (1.0 - alpha)))
    return (low, high)


def intervals_separated(a: Interval, b: Interval) -> bool:
    """True when two CIs do not overlap (a conservative significance signal)."""
    if a["ci_low"] is None or b["ci_low"] is None:
        return False
    return a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]

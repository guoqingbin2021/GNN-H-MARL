from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import scipy
from scipy.stats import t, wilcoxon


def mean_ci95(values: Iterable[float]) -> dict[str, float | int | list[float]]:
    """Return mean, sample std, sample count, and a two-sided 95% t interval."""
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("mean_ci95 requires at least one value")
    mean = float(array.mean())
    if array.size == 1:
        std = 0.0
        lower = upper = mean
    else:
        std = float(array.std(ddof=1))
        half_width = float(
            t.ppf(0.975, df=array.size - 1) * std / np.sqrt(array.size)
        )
        lower, upper = mean - half_width, mean + half_width
    return {
        "mean": mean,
        "std": std,
        "n": int(array.size),
        "ci95": [float(lower), float(upper)],
    }


def paired_signed_rank(differences: Iterable[float]) -> dict:
    """Predeclared two-sided paired test; never select tails from observed signs.

    Each value is one independent seed's full-minus-control hypervolume.
    Signed-rank inference assumes independent, approximately symmetric paired
    differences. The t interval describes their mean, not the signed-rank target.
    """
    values = np.asarray(list(differences), dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Paired comparisons require nonempty finite seed differences")
    summary = mean_ci95(values)
    n = len(values)
    std = float(summary["std"])
    dz = float(values.mean() / std) if n >= 2 and std > 0 else None
    if dz is not None and not np.isfinite(dz):
        dz = None
    statistic = pvalue = None
    method = "not_tested"
    status = "not_tested_single_seed"
    if n >= 2:
        status = "tested" if n >= 5 else "descriptive_only_fewer_than_five_seeds"
        if np.all(values == 0):
            statistic, pvalue, method = 0.0, 1.0, "all_zero_differences"
        else:
            # Exact signed-rank distribution requires nonzero, distinct ranks.
            method = (
                "exact"
                if n <= 50 and np.all(values != 0) and len(np.unique(np.abs(values))) == n
                else "auto"
            )
            result = wilcoxon(values, alternative="two-sided", zero_method="wilcox", method=method)
            statistic, pvalue = float(result.statistic), float(result.pvalue)
            if not np.isfinite(pvalue):
                raise ValueError("Wilcoxon returned a non-finite p-value")
    return {
        "difference_summary": summary,
        "median_difference": float(np.median(values)),
        "minimum_difference": float(values.min()),
        "maximum_difference": float(values.max()),
        "paired_dz": dz,
        "paired_dz_definition": "mean(full-control) / sample_SD(full-control); undefined with fewer than two seeds or zero variance",
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "losses": int(np.sum(values < 0)),
        "wilcoxon": {
            "status": status, "alternative": "two-sided", "statistic": statistic,
            "p_uncorrected": pvalue, "method": method, "zero_method": "wilcox",
            "scipy_version": scipy.__version__,
            "assumptions": "Independent paired seeds and approximately symmetric differences; review exported seed deltas before interpreting inference.",
        },
        "mean_ci95_caution": "Two-sided t interval for mean paired difference; a single seed cannot estimate uncertainty.",
    }


def holm_adjust(pvalues: Iterable[float | None]) -> list[float | None]:
    """Holm family-wise correction preserving declared but untested comparisons."""
    values = list(pvalues)
    tested = []
    for index, value in enumerate(values):
        if value is None:
            continue
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Holm correction requires p-values between zero and one")
        tested.append((index, float(value)))
    adjusted: list[float | None] = [None] * len(values)
    previous = 0.0
    for rank, (index, value) in enumerate(sorted(tested, key=lambda item: item[1])):
        previous = max(previous, min(1.0, (len(values) - rank) * value))
        adjusted[index] = previous
    return adjusted


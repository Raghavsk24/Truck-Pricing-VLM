"""The metric. Not MAE.

Coverage: on held-out sales, does the stated 80% interval actually contain the
true price 80% of the time? Almost nobody building these measures it, and it
is the number that decides whether anyone trusts the output twice.

Track it separately for well-photographed and poorly-photographed inputs. The
second bucket is where these systems quietly fail: the range looks impressively
tight and is wrong far more often than it claims.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CoverageReport:
    nominal: float
    empirical: float
    n: int
    median_width: float
    median_relative_width: float
    median_absolute_error: float
    below: float = 0.0   # truth fell under the interval
    above: float = 0.0   # truth fell over the interval
    median_bias: float = 0.0

    def __str__(self) -> str:
        gap = self.empirical - self.nominal
        verdict = ("well calibrated" if abs(gap) < 0.03
                   else "overconfident" if gap < 0 else "underconfident")
        # Symmetric misses mean the interval is too NARROW: widen it.
        # Lopsided misses mean it is mis-CENTRED: the fix is a bias, not width.
        tail = max(self.below, self.above)
        shape = ("too narrow" if tail < 0.65 * (self.below + self.above + 1e-9)
                 else ("mis-centred low" if self.above > self.below
                       else "mis-centred high"))
        return (
            f"n={self.n}  nominal={self.nominal:.0%}  "
            f"empirical={self.empirical:.1%}  ({verdict})\n"
            f"  misses            {self.below:.1%} below / {self.above:.1%} above"
            f"   -> {shape}\n"
            f"  median bias       {self.median_bias:+,.0f}\n"
            f"  median width      {self.median_width:,.0f}\n"
            f"  median rel. width {self.median_relative_width:.1%}\n"
            f"  median abs error  {self.median_absolute_error:,.0f}"
        )


def coverage(
    truth: np.ndarray,
    low: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
    nominal: float = 0.80,
) -> CoverageReport:
    truth, low, mid, high = map(np.asarray, (truth, low, mid, high))
    inside = (truth >= low) & (truth <= high)
    width = high - low
    return CoverageReport(
        nominal=nominal,
        empirical=float(inside.mean()),
        n=int(truth.size),
        median_width=float(np.median(width)),
        median_relative_width=float(np.median(width / np.clip(mid, 1e-9, None))),
        median_absolute_error=float(np.median(np.abs(truth - mid))),
        below=float((truth < low).mean()),
        above=float((truth > high).mean()),
        median_bias=float(np.median(mid - truth)),
    )


def reliability_curve(
    truth: np.ndarray,
    samples: np.ndarray,
    levels: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
) -> pd.DataFrame:
    """Empirical coverage at several nominal levels.

    `samples` is (n_cases, n_draws) from run_uncertainty. A straight diagonal
    means your intervals mean what they say at every width, not just at 80%.
    """
    rows = []
    for lvl in levels:
        lo = np.percentile(samples, (1 - lvl) / 2 * 100, axis=1)
        hi = np.percentile(samples, (1 + lvl) / 2 * 100, axis=1)
        inside = (truth >= lo) & (truth <= hi)
        rows.append({
            "nominal": lvl,
            "empirical": float(inside.mean()),
            "median_width": float(np.median(hi - lo)),
        })
    return pd.DataFrame(rows)


def coverage_by_bucket(
    truth: np.ndarray,
    low: np.ndarray,
    mid: np.ndarray,
    high: np.ndarray,
    bucket: np.ndarray,
    nominal: float = 0.80,
) -> pd.DataFrame:
    """Split coverage by e.g. photo count. This is the diagnostic that matters."""
    rows = []
    for b in pd.unique(bucket):
        m = bucket == b
        if m.sum() < 5:
            continue
        rep = coverage(truth[m], low[m], mid[m], high[m], nominal)
        rows.append({
            "bucket": b,
            "n": rep.n,
            "empirical": rep.empirical,
            "median_width": rep.median_width,
            "median_rel_width": rep.median_relative_width,
        })
    return pd.DataFrame(rows).sort_values("bucket")


def pinball_loss(truth: np.ndarray, pred: np.ndarray, q: float) -> float:
    """Train-time metric for a single quantile model."""
    d = truth - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))

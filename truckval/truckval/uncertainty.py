"""Stage 4. Where the range actually comes from.

Two distinct sources of uncertainty, and they must be composed rather than
picked between:

  MODEL uncertainty     — even with perfect information, identical trucks sell
                          for different prices. This is what the p10/p50/p90
                          quantile models capture.
  INFORMATION uncertainty — we could not see the tires, the odometer or three
                          of the panels. This is what the Monte Carlo over
                          priors captures.

Naively running the Monte Carlo through the median model alone gives a
fully-photographed truck a zero-width range, which is wrong. So on every draw
we also sample a quantile level u ~ U(0,1) and invert the model's own
predictive distribution at that level. The two sources compose, and the range
behaves correctly at both extremes: twelve good photos gives a tight window,
two blurry side shots gives a wide one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from scipy.stats import norm

from .features import build_row
from .ledger import build_ledger, expected_condition_cost, sample_ledger_total
from .pricing import QuantilePricer
from .priors import DEFAULT_PRIORS, PriorSet, with_claims
from .schema import Evidence, PriceRange, VarianceContribution

DEFAULT_DRAWS = 2_000
ABLATION_DRAWS = 500

# What to ask the seller for, per unknown field.
PHOTO_ASKS = {
    "condition.odometer_km": "the odometer, straight on, ignition in accessory",
    "condition.tires": "the driver-side tires from about a metre back",
    "condition.rust": "the rocker panel and frame rail behind the front wheel",
    "condition.seat_wear": "the driver's seat and steering wheel",
    "condition.dash_condition": "the dashboard from the passenger seat",
    "condition.glass_cracked": "the windscreen from directly in front",
    "condition.paint_fade": "the roof or hood in daylight",
    "configuration.box_length_ft": "the full side of the box with a wheel in frame",
    "configuration.axle_count": "the whole vehicle from the side",
    "configuration.gvwr_lb": "the door-jamb sticker",
    "identity.vin": "the VIN plate on the driver's door jamb",
    "identity.year": "the VIN plate or the badge on the side of the cab",
    "identity.make": "the badge on the front or side of the cab",
    "identity.model": "the model badge on the side of the cab",
    "equipment.liftgate": "the rear of the vehicle",
    "equipment.reefer_unit": "the front wall of the box, above the cab",
    "equipment.crane": "the deck behind the cab",
    "equipment.fifth_wheel": "the frame behind the cab",
    "equipment.pto": "under the chassis behind the transmission",
    "_hidden_damage_count": "the two sides you haven't shown yet",
}


def invert_quantiles(levels: list[float], values: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample from a distribution defined by a handful of quantiles.

    Interpolates in (normal-score, LOG price) space. Both halves matter:

      normal score  keeps the mapping monotone and gives sane tails beyond p10
                    and p90 without fitting more quantile models.
      log price     because prices are right-skewed. Interpolating linearly in
                    price space compresses the upper tail and extrapolates past
                    p90 on a slope that is too shallow, which shows up
                    downstream as intervals that sit systematically low —
                    misses land almost entirely ABOVE the range. In log space
                    the reconstruction is exact for a lognormal, which is very
                    nearly what a used-vehicle price is.

    `values` is (n_rows, n_levels) of prices; `u` is (n_rows,) in (0, 1).
    """
    z_known = np.asarray([norm.ppf(l) for l in levels], dtype=float)
    z_want = norm.ppf(np.clip(u, 1e-4, 1 - 1e-4))
    logv = np.log(np.clip(values, 1e-6, None))

    # Vectorised piecewise-linear interpolation across all rows at once.
    idx = np.clip(np.searchsorted(z_known, z_want) - 1, 0, len(z_known) - 2)
    z0, z1 = z_known[idx], z_known[idx + 1]
    rows = np.arange(values.shape[0])
    y0, y1 = logv[rows, idx], logv[rows, idx + 1]
    t = (z_want - z0) / (z1 - z0)
    return np.exp(y0 + t * (y1 - y0))


@dataclass
class UncertaintyResult:
    price: PriceRange
    baseline_median: float
    samples: np.ndarray
    sampled_fields: list[str]
    floor_width: float = 0.0  # width if every unknown were resolved

    @property
    def excess_ratio(self) -> float:
        """How much wider than perfect information. 0.0 means at the floor.

        This, not raw relative width, is what should gate a price. Used
        commercial trucks genuinely trade in a ~50%-wide band at 80%
        confidence; a threshold on absolute width would suppress every
        estimate you ever make.
        """
        if self.floor_width <= 0:
            return 0.0
        return max(0.0, (self.price.width - self.floor_width) / self.floor_width)


def _draw_unknowns(
    ev: Evidence,
    unknowns: list[str],
    priors: PriorSet,
    rng: np.random.Generator,
    pinned: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    draw: dict[str, Any] = dict(pinned or {})
    for path in unknowns:
        if path in draw:
            continue
        if priors.covers(path):
            draw[path] = priors.sample(path, ev, rng)
    if "_hidden_damage_count" not in draw and priors.covers("_hidden_damage_count"):
        draw["_hidden_damage_count"] = priors.sample("_hidden_damage_count", ev, rng)
    return draw


def run_uncertainty(
    ev: Evidence,
    pricer: QuantilePricer,
    priors: PriorSet = DEFAULT_PRIORS,
    n_draws: int = DEFAULT_DRAWS,
    seed: int = 0,
    pinned: Optional[dict[str, Any]] = None,
    contradicted: frozenset[str] = frozenset(),
) -> UncertaintyResult:
    rng = np.random.default_rng(seed)
    # Typed claims become claim-anchored samplers, not pinned values. A field
    # the cross-check flagged as disputed gets a much flatter one.
    claims = ev.claimed_values()
    if claims:
        priors = with_claims(priors, claims, contradicted)
    unknowns = [p for p in ev.unknown_fields() if priors.covers(p)]

    draws = [_draw_unknowns(ev, unknowns, priors, rng, pinned) for _ in range(n_draws)]
    rows = [build_row(ev, d) for d in draws]

    q_values = pricer.predict(rows)  # (n_draws, n_quantiles)
    u = rng.random(n_draws)
    baseline_samples = invert_quantiles(pricer.quantile_levels, q_values, u)

    # Net against what the comps already price in, or ordinary wear gets
    # charged twice. See ledger.expected_condition_cost.
    cohort = expected_condition_cost(ev, priors, rng)
    deductions = np.array([
        sample_ledger_total(build_ledger(ev, d), rng) - cohort for d in draws
    ])

    samples = np.clip(baseline_samples + deductions, 0.0, None)
    lo, mid, hi = np.percentile(samples, [10, 50, 90])

    # Irreducible spread: what the range would be with every field resolved.
    # One row at the modal draw, straight model quantiles, no sampling noise.
    modal = _modal_draw(ev, unknowns, priors, seed)
    floor_q = pricer.predict([build_row(ev, modal)])[0]
    floor_width = float(floor_q[-1] - floor_q[0])

    return UncertaintyResult(
        price=PriceRange(low=float(lo), mid=float(mid), high=float(hi),
                         n_samples=n_draws),
        baseline_median=float(np.median(baseline_samples)),
        samples=samples,
        sampled_fields=unknowns,
        floor_width=floor_width,
    )


def _modal_draw(
    ev: Evidence, unknowns: list[str], priors: PriorSet, seed: int, trials: int = 48
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 999)
    out: dict[str, Any] = {}
    for path in unknowns:
        vals = [priors.sample(path, ev, rng) for _ in range(trials)]
        vals = [v for v in vals if v is not None]
        if not vals:
            out[path] = None
            continue
        nums = [v for v in vals if isinstance(v, (int, float, np.number))
                and not isinstance(v, bool)]
        if nums:
            out[path] = float(np.median(nums))
        else:
            u, c = np.unique([str(v) for v in vals], return_counts=True)
            mode = u[int(c.argmax())]
            out[path] = {"True": True, "False": False}.get(str(mode), mode)
    return out


def attribute_variance(
    ev: Evidence,
    pricer: QuantilePricer,
    priors: PriorSet = DEFAULT_PRIORS,
    n_draws: int = ABLATION_DRAWS,
    seed: int = 0,
    top_k: int = 3,
) -> list[VarianceContribution]:
    """Leave-one-out: how much narrower would the range be if we knew X?

    Pins each unknown field to its median draw and re-runs. The reduction in
    interval width is what that one photo is worth. This is what powers
    'a photo of the odometer would narrow this by $2,300'.
    """
    unknowns = [p for p in ev.unknown_fields() if priors.covers(p)]
    if not unknowns:
        return []

    base = run_uncertainty(ev, pricer, priors, n_draws=n_draws, seed=seed)
    base_width = base.price.width
    rng = np.random.default_rng(seed + 1)

    out: list[VarianceContribution] = []
    for path in unknowns:
        # A representative value: the median of many prior draws.
        trials = [priors.sample(path, ev, rng) for _ in range(64)]
        numeric = [t for t in trials if isinstance(t, (int, float, np.number))]
        if numeric:
            pin: Any = float(np.median(numeric))
        else:
            vals, counts = np.unique([str(t) for t in trials], return_counts=True)
            pin = vals[int(counts.argmax())]

        pinned = run_uncertainty(
            ev, pricer, priors, n_draws=n_draws, seed=seed, pinned={path: pin}
        )
        gain = base_width - pinned.price.width
        if gain > 0:
            out.append(VarianceContribution(
                field=path,
                dollars=float(gain),
                ask=PHOTO_ASKS.get(path, "a clearer photo of this area"),
            ))

    out.sort(key=lambda c: c.dollars, reverse=True)
    return out[:top_k]

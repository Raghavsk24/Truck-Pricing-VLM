"""Priors for the fields perception could not see.

This is where a guess is *allowed* — because it is sampled, not asserted, and
the spread of the samples becomes the width of the price range. A field that
perception invented would silently narrow the range; a field sampled here
correctly widens it.

Priors are conditioned on what was observed. Not seeing the odometer on a
2012 truck is a different distribution from not seeing it on a 2023 truck.
Replace these with empirical distributions fitted to your comps table as soon
as you have one — `fit_priors_from_comps` does exactly that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from .schema import Evidence

Sampler = Callable[[Evidence, np.random.Generator], Any]

# Annual distance by body type, km. Wide on purpose.
ANNUAL_KM = {
    "tractor": (95_000, 45_000),
    "reefer": (80_000, 38_000),
    "box_truck": (35_000, 22_000),
    "flatbed": (40_000, 25_000),
    "dump": (22_000, 15_000),
    "service": (25_000, 16_000),
    "stake": (30_000, 20_000),
    "tanker": (60_000, 30_000),
    "pickup": (24_000, 14_000),
}
DEFAULT_ANNUAL_KM = (38_000, 26_000)

TREAD_STATES = ["new", "good", "worn", "legal_minimum", "below_minimum"]
TREAD_PRIOR = [0.06, 0.34, 0.36, 0.18, 0.06]

RUST_STATES = ["none", "surface", "scaling", "perforation"]
# Rust is strongly regional. Northern salt-belt states are a different world.
RUST_BY_REGION = {
    "salt_belt": [0.06, 0.42, 0.34, 0.18],
    "coastal": [0.12, 0.46, 0.30, 0.12],
    "arid": [0.46, 0.40, 0.11, 0.03],
    "default": [0.22, 0.46, 0.24, 0.08],
}

WEAR_STATES = ["light", "moderate", "heavy", "destroyed"]

BOX_LENGTHS = [12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]
BOX_LENGTH_PRIOR = [0.04, 0.07, 0.17, 0.09, 0.13, 0.07, 0.28, 0.11, 0.04]


@dataclass
class PriorSet:
    samplers: dict[str, Sampler]

    def sample(self, path: str, ev: Evidence, rng: np.random.Generator) -> Any:
        fn = self.samplers.get(path)
        return fn(ev, rng) if fn else None

    def covers(self, path: str) -> bool:
        return path in self.samplers


# --------------------------------------------------------------------------
# Individual samplers
# --------------------------------------------------------------------------


def _age(ev: Evidence, now_year: int = 2026) -> float:
    y = ev.identity.year.value
    return max(1.0, now_year - int(y)) if y else 12.0


def sample_odometer(ev: Evidence, rng: np.random.Generator) -> float:
    body = ev.configuration.body_type.value or "box_truck"
    mu, sd = ANNUAL_KM.get(str(body), DEFAULT_ANNUAL_KM)
    age = _age(ev)
    # Lognormal on total distance keeps it positive and right-skewed.
    # Spread grows with age: you know far less about a 20-year-old truck's
    # total distance than a 3-year-old's, and the earlier version cancelled
    # that term out to 1 by accident.
    total_mu = np.log(max(mu * age, 5_000.0))
    total_sd = np.sqrt(np.log1p((sd / mu) ** 2) + 0.06 * np.log1p(age))
    return float(np.clip(rng.lognormal(total_mu, max(total_sd, 0.40)), 1_000, 2_500_000))


def sample_tread(ev: Evidence, rng: np.random.Generator) -> str:
    """Older, higher-distance trucks skew toward worn rubber."""
    p = np.array(TREAD_PRIOR, dtype=float)
    odo = ev.condition.odometer_km.value
    if odo and float(odo) > 400_000:
        p = p * np.array([0.5, 0.8, 1.2, 1.4, 1.6])
    p /= p.sum()
    return str(rng.choice(TREAD_STATES, p=p))


def sample_rust(ev: Evidence, rng: np.random.Generator) -> str:
    p = np.array(RUST_BY_REGION.get(ev.region or "default",
                                    RUST_BY_REGION["default"]), dtype=float)
    age = _age(ev)
    if age > 12:
        p = p * np.array([0.4, 0.9, 1.3, 1.7])
    p /= p.sum()
    return str(rng.choice(RUST_STATES, p=p))


def sample_seat_wear(ev: Evidence, rng: np.random.Generator) -> str:
    odo = ev.condition.odometer_km.value or sample_odometer(ev, rng)
    odo = float(odo)
    if odo < 150_000:
        p = [0.55, 0.33, 0.10, 0.02]
    elif odo < 400_000:
        p = [0.18, 0.50, 0.28, 0.04]
    elif odo < 800_000:
        p = [0.05, 0.32, 0.51, 0.12]
    else:
        p = [0.02, 0.18, 0.53, 0.27]
    return str(rng.choice(WEAR_STATES, p=p))


def sample_box_length(ev: Evidence, rng: np.random.Generator) -> Optional[float]:
    body = ev.configuration.body_type.value
    if body not in ("box_truck", "reefer", None):
        return None
    return float(rng.choice(BOX_LENGTHS, p=BOX_LENGTH_PRIOR))


def sample_axles(ev: Evidence, rng: np.random.Generator) -> int:
    body = str(ev.configuration.body_type.value or "box_truck")
    if body == "tractor":
        return int(rng.choice([2, 3], p=[0.35, 0.65]))
    if body in ("dump", "tanker"):
        return int(rng.choice([2, 3, 4], p=[0.45, 0.40, 0.15]))
    return int(rng.choice([2, 3], p=[0.86, 0.14]))


def sample_bool(prob_true: float) -> Sampler:
    def fn(ev: Evidence, rng: np.random.Generator) -> bool:
        return bool(rng.random() < prob_true)
    return fn


def sample_damage_count(ev: Evidence, rng: np.random.Generator) -> int:
    """Unseen panels may hide damage. Absence of evidence is not evidence."""
    unseen = max(0, 6 - ev.usable_exterior_angles)
    return int(rng.poisson(0.35 * unseen))


DEFAULT_PRIORS = PriorSet(samplers={
    "condition.odometer_km": sample_odometer,
    "condition.rust": sample_rust,
    "condition.seat_wear": sample_seat_wear,
    "condition.dash_condition": sample_seat_wear,
    "condition.glass_cracked": sample_bool(0.14),
    "condition.paint_fade": sample_seat_wear,
    "condition.tires": sample_tread,
    "configuration.box_length_ft": sample_box_length,
    "configuration.axle_count": sample_axles,
    "equipment.liftgate": sample_bool(0.31),
    "equipment.reefer_unit": sample_bool(0.09),
    "equipment.crane": sample_bool(0.04),
    "equipment.fifth_wheel": sample_bool(0.06),
    "equipment.pto": sample_bool(0.11),
    "_hidden_damage_count": sample_damage_count,
})


# --------------------------------------------------------------------------
# Fitting priors from your own data — do this once you have comps
# --------------------------------------------------------------------------


def fit_priors_from_comps(comps: pd.DataFrame) -> PriorSet:
    """Replace hand-authored priors with empirical ones. Do this immediately.

    Hand-authored priors are how you become quietly overconfident. The shipped
    defaults said liftgate probability 0.31 unconditionally; the data says
    ~0.46 for box trucks and 0 for tractors. Wrong value AND wrong
    conditioning, and the damage shows up as undercoverage in exactly the
    bucket where most fields are hidden.

    Everything here conditions on body_type where the data supports it, and
    falls back to the marginal, then to DEFAULT_PRIORS.
    """
    samplers = dict(DEFAULT_PRIORS.samplers)
    if "body_type" not in comps.columns:
        return PriorSet(samplers=samplers)

    # --- distance per year, by body type
    if {"age_years", "odometer_km"} <= set(comps.columns):
        rates = (
            comps.assign(kpy=comps.odometer_km / comps.age_years.clip(lower=1))
            .groupby("body_type")["kpy"].agg(["mean", "std"]).dropna()
        )
        table = {str(k): (float(v["mean"]), float(max(v["std"], v["mean"] * 0.35)))
                 for k, v in rates.iterrows()}

        def empirical_odo(ev: Evidence, rng: np.random.Generator) -> float:
            body = str(ev.configuration.body_type.value or "box_truck")
            mu, sd = table.get(body, DEFAULT_ANNUAL_KM)
            age = _age(ev)
            draw = rng.normal(mu, sd) * age * rng.lognormal(0, 0.18)
            return float(np.clip(draw, 1_000, 2_500_000))

        samplers["condition.odometer_km"] = empirical_odo

    # --- equipment, conditioned on body type
    for col in ("liftgate", "reefer_unit", "crane", "fifth_wheel", "pto"):
        if col not in comps.columns:
            continue
        rate = comps.groupby("body_type")[col].mean().to_dict()
        marginal = float(comps[col].mean())

        def make_bool(rate=rate, marginal=marginal) -> Sampler:
            def fn(ev: Evidence, rng: np.random.Generator) -> bool:
                body = str(ev.configuration.body_type.value or "")
                return bool(rng.random() < float(rate.get(body, marginal)))
            return fn

        samplers[f"equipment.{col}"] = make_bool()

    # --- numeric configuration, conditioned on body type
    for col, path in (("box_length_ft", "configuration.box_length_ft"),
                      ("axle_count", "configuration.axle_count"),
                      ("gvwr_lb", "configuration.gvwr_lb")):
        if col not in comps.columns:
            continue
        by_body = {
            str(b): g[col].dropna().to_numpy()
            for b, g in comps.groupby("body_type") if g[col].notna().sum() >= 20
        }
        pooled = comps[col].dropna().to_numpy()
        is_int = col == "axle_count"

        def make_num(by_body=by_body, pooled=pooled, is_int=is_int) -> Sampler:
            def fn(ev: Evidence, rng: np.random.Generator):
                body = str(ev.configuration.body_type.value or "")
                arr = by_body.get(body)
                if arr is None or len(arr) == 0:
                    if len(pooled) == 0:
                        return None
                    arr = pooled
                v = float(rng.choice(arr))
                return int(round(v)) if is_int else v
            return fn

        samplers[path] = make_num()

    # --- condition categoricals, if the table carries them
    for col, path in (("rust", "condition.rust"),
                      ("seat_wear", "condition.seat_wear")):
        if col in comps.columns:
            counts = comps[col].value_counts(normalize=True)
            states, probs = list(counts.index), counts.to_numpy(dtype=float)

            def make_cat(states=states, probs=probs) -> Sampler:
                def fn(ev: Evidence, rng: np.random.Generator) -> Any:
                    return states[int(rng.choice(len(states), p=probs))]
                return fn

            samplers[path] = make_cat()

    return PriorSet(samplers=samplers)


# --------------------------------------------------------------------------
# Soft evidence: what a seller TYPED is not what a photo SHOWS
# --------------------------------------------------------------------------

# Probability a seller's unverified typed claim is roughly true. Mileage is the
# most-misstated field on any used vehicle listing, and understatement is the
# direction of the error, so the lie model is deliberately asymmetric.
CLAIM_HONESTY = 0.80
CLAIM_HONESTY_CONTRADICTED = 0.25
CLAIM_TIGHTNESS = 0.06  # relative sd around a claim believed to be honest


def claim_anchored(
    path: str, claimed: Any, honesty: float = CLAIM_HONESTY
) -> Sampler:
    """Sample around a typed claim instead of trusting it outright.

    Merging a claim as a hard observation means an unverified number typed by
    someone with a financial interest in a high price gets the same weight as
    an odometer read off the dash. This samples a mixture instead: mostly near
    the claim, sometimes from the unconditional prior. The range widens exactly
    as much as the claim is doubtful, which is the honest answer to 'sellers
    sometimes lie'.
    """
    fallback = DEFAULT_PRIORS.samplers.get(path)

    def fn(ev: Evidence, rng: np.random.Generator) -> Any:
        if rng.random() > honesty and fallback is not None:
            return fallback(ev, rng)
        if isinstance(claimed, (int, float)) and not isinstance(claimed, bool):
            # Understating distance is the common direction of the lie.
            skew = 1.0 + 0.35 * CLAIM_TIGHTNESS if "odometer" in path else 1.0
            v = float(claimed) * skew * rng.lognormal(0, CLAIM_TIGHTNESS)
            return int(round(v)) if isinstance(claimed, int) else v
        return claimed

    return fn


def with_claims(
    base: PriorSet, claims: dict[str, Any], contradicted: set[str] = frozenset()
) -> PriorSet:
    """Layer claim-anchored samplers over a fitted prior set."""
    samplers = dict(base.samplers)
    for path, value in claims.items():
        if value is None:
            continue
        honesty = (CLAIM_HONESTY_CONTRADICTED if path in contradicted
                   else CLAIM_HONESTY)
        samplers[path] = claim_anchored(path, value, honesty)
    return PriorSet(samplers=samplers)

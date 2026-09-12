"""The calibration check that actually matters.

train_pricing.py measures the model on fully-observed rows. Production never
sees fully-observed rows. This script takes held-out sales, hides a random
subset of fields to simulate real seller photography, runs the whole
uncertainty pass, and asks whether the stated 80% interval covers the truth 80%
of the time — bucketed by how much was hidden.

If coverage holds at 0 hidden fields but collapses at 8, your priors are too
narrow and you are quietly overconfident exactly where sellers actually live.

    python scripts/eval_calibration.py --n 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truckval.calibration import coverage, coverage_by_bucket, reliability_curve  # noqa: E402
from truckval.pricing import QuantilePricer  # noqa: E402
from truckval.priors import DEFAULT_PRIORS, fit_priors_from_comps  # noqa: E402
from truckval.schema import (  # noqa: E402
    Condition,
    Configuration,
    Equipment,
    Evidence,
    Identity,
    Observation,
    Visibility,
)
from truckval.uncertainty import run_uncertainty  # noqa: E402

HIDEABLE = [
    ("condition", "odometer_km"),
    ("configuration", "box_length_ft"),
    ("configuration", "axle_count"),
    ("configuration", "gvwr_lb"),
    ("equipment", "liftgate"),
    ("equipment", "reefer_unit"),
    ("equipment", "crane"),
    ("equipment", "fifth_wheel"),
    ("equipment", "pto"),
]


def obs(v) -> Observation:
    return Observation(value=v, visibility=Visibility.OBSERVED, confidence=0.95,
                       photo_index=0)


def row_to_evidence(r: pd.Series, hide: list[tuple[str, str]]) -> Evidence:
    """Build a fully-observed Evidence from a comps row, then hide fields."""
    ev = Evidence(
        identity=Identity(
            # round, do NOT truncate: int() here silently ages every truck
            # by ~0.5y on average, which at 14%/yr depreciation reads as a
            # -7% price bias and looks exactly like a broken model.
            year=obs(int(round(2026 - r.age_years))),
            make=obs(r.make), model=obs(r.model),
        ),
        configuration=Configuration(
            body_type=obs(r.body_type), cab_type=obs(r.cab_type),
            axle_count=obs(int(r.axle_count)),
            box_length_ft=obs(None if pd.isna(r.box_length_ft)
                              else float(r.box_length_ft)),
            gvwr_lb=obs(int(r.gvwr_lb)),
        ),
        equipment=Equipment(
            liftgate=obs(bool(r.liftgate)), reefer_unit=obs(bool(r.reefer_unit)),
            crane=obs(bool(r.crane)), fifth_wheel=obs(bool(r.fifth_wheel)),
            pto=obs(bool(r.pto)),
        ),
        condition=Condition(odometer_km=obs(float(r.odometer_km))),
        usable_exterior_angles=6,
        region=r.region,
    )
    for group, name in hide:
        setattr(getattr(ev, group), name, Observation(visibility=Visibility.NOT_VISIBLE))
    return ev


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comps", default="data/comps.parquet")
    ap.add_argument("--pricer", default="data/pricer.pkl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--draws", type=int, default=600)
    ap.add_argument("--hand-priors", action="store_true",
                    help="use the hand-authored priors instead of fitted ones")
    a = ap.parse_args()

    df = (pd.read_parquet(a.comps) if a.comps.endswith(".parquet")
          else pd.read_csv(a.comps))
    rng = np.random.default_rng(0)
    test = df[rng.random(len(df)) >= 0.82].sample(
        n=min(a.n, int((rng.random(len(df)) >= 0.82).sum())), random_state=1
    )
    pricer = QuantilePricer.load(a.pricer)
    priors = (DEFAULT_PRIORS if a.hand_priors
              else fit_priors_from_comps(df))
    print('priors:', 'hand-authored' if a.hand_priors else 'fitted from comps')

    lows, mids, highs, truths, hidden_counts, all_samples = [], [], [], [], [], []
    for i, (_, r) in enumerate(test.iterrows()):
        k = int(rng.integers(0, len(HIDEABLE) + 1))
        hide = [HIDEABLE[j] for j in rng.choice(len(HIDEABLE), size=k, replace=False)]
        ev = row_to_evidence(r, hide)
        res = run_uncertainty(ev, pricer, priors, n_draws=a.draws, seed=i)
        lows.append(res.price.low)
        mids.append(res.price.mid)
        highs.append(res.price.high)
        truths.append(float(r.sale_price))
        hidden_counts.append(k)
        all_samples.append(res.samples)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(test)}", flush=True)

    lows, mids, highs = map(np.array, (lows, mids, highs))
    truths = np.array(truths)
    hidden_counts = np.array(hidden_counts)

    print("\n=== overall ===")
    print(coverage(truths, lows, mids, highs, nominal=0.80))

    print("\n=== by number of hidden fields ===")
    bucket = np.where(hidden_counts <= 2, "0-2",
                      np.where(hidden_counts <= 5, "3-5", "6-9"))
    print(coverage_by_bucket(truths, lows, mids, highs, bucket).to_string(index=False))

    print("\n=== reliability curve ===")
    n_draws = min(len(s) for s in all_samples)
    samples = np.vstack([s[:n_draws] for s in all_samples])
    print(reliability_curve(truths, samples).to_string(index=False))

    print(
        "\nRead the bucket table first. Coverage that holds at 0-2 hidden\n"
        "fields but drops in the 6-9 row means the priors are too narrow and\n"
        "the system is overconfident precisely where real sellers live."
    )


if __name__ == "__main__":
    main()

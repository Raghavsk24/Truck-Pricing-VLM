"""End-to-end demo. No API key, no network, no real photos.

Runs five scenarios through the real pipeline with a stubbed detector and a
stubbed perception stage, so you can watch the behaviour that matters:

  1. wrong object          -> Tier 0, names what it saw
  2. unusable capture      -> Tier 0
  3. no identity           -> Tier 1, condition report but no price
  4. well photographed     -> Tier 2, tight range
  5. same truck, 2 photos  -> much wider range, or Tier 1 if too wide

    python scripts/make_synth_comps.py
    python scripts/train_pricing.py
    python scripts/demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truckval.gate import DetectorResult, PhotoInput  # noqa: E402
from truckval.pipeline import Pipeline  # noqa: E402
from truckval.pricing import QuantilePricer  # noqa: E402
from truckval.report import render  # noqa: E402
from truckval.schema import (  # noqa: E402
    Condition,
    Configuration,
    DamageInstance,
    Equipment,
    Evidence,
    Identity,
    Observation,
    SellerClaims,
    TireObservation,
    Visibility,
)


# --------------------------------------------------------------------------
# Fake photos
# --------------------------------------------------------------------------


def sharp_photo(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(70, 190, (256, 256, 3), dtype=np.uint8)
    return base


def dark_photo() -> np.ndarray:
    rng = np.random.default_rng(1)
    return (rng.integers(0, 24, (256, 256, 3))).astype(np.uint8)


def blurry_photo() -> np.ndarray:
    y = np.linspace(80, 160, 256)
    grad = np.repeat(y[:, None], 256, axis=1)
    return np.stack([grad] * 3, axis=-1).astype(np.uint8)


def photos(n: int, seed: int = 0) -> list[PhotoInput]:
    return [PhotoInput(index=i, image=sharp_photo(seed + i)) for i in range(n)]


# --------------------------------------------------------------------------
# Stubs
# --------------------------------------------------------------------------


def make_detector(cls: str, coverage: float = 0.55):
    def detector(photo: PhotoInput) -> DetectorResult:
        return DetectorResult(top_class=cls, confidence=0.93,
                              frame_coverage=coverage)
    return detector


def obs(value, photo_index=0, conf=0.9) -> Observation:
    return Observation(value=value, visibility=Visibility.OBSERVED,
                       confidence=conf, photo_index=photo_index)


def hidden(reason=Visibility.NOT_VISIBLE) -> Observation:
    return Observation(visibility=reason)


def well_photographed() -> Evidence:
    return Evidence(
        identity=Identity(
            year=obs(2016), make=obs("hino"), model=obs("268"),
            vin=hidden(),
        ),
        configuration=Configuration(
            body_type=obs("box_truck"), cab_type=obs("standard"),
            axle_count=obs(2), box_length_ft=obs(24.0), gvwr_lb=obs(25_950),
        ),
        equipment=Equipment(
            liftgate=obs(True), reefer_unit=obs(False), crane=obs(False),
            fifth_wheel=obs(False), pto=obs(False),
        ),
        condition=Condition(
            odometer_km=obs(311_000, photo_index=6),
            tires=[
                TireObservation(position="steer_left", tread=obs("worn", 2)),
                TireObservation(position="steer_right", tread=obs("worn", 3)),
                TireObservation(position="drive_left_outer", tread=obs("good", 2)),
                TireObservation(position="drive_right_outer", tread=obs("good", 3)),
            ],
            damage=[
                DamageInstance(panel="rear roll door", kind="dent",
                               severity="moderate", photo_index=4, confidence=0.88),
                DamageInstance(panel="passenger fender", kind="scratch",
                               severity="minor", photo_index=3, confidence=0.81),
            ],
            rust=obs("surface", 5), seat_wear=obs("moderate", 7),
            dash_condition=obs("moderate", 7), glass_cracked=obs(False, 1),
            paint_fade=obs("moderate", 0),
        ),
        usable_exterior_angles=6,
        region="midwest",
    )


def barely_photographed() -> Evidence:
    """Same truck, two blurry side shots. Everything else unseen."""
    return Evidence(
        identity=Identity(year=obs(2016), make=obs("hino"), model=obs("268")),
        configuration=Configuration(
            body_type=obs("box_truck"), cab_type=hidden(),
            axle_count=obs(2), box_length_ft=hidden(), gvwr_lb=hidden(),
        ),
        equipment=Equipment(),
        condition=Condition(),
        usable_exterior_angles=2,
        region="midwest",
    )


def no_identity() -> Evidence:
    ev = barely_photographed()
    ev.identity = Identity()
    return ev


def make_perception(evidence: Evidence):
    def perceive(_: Sequence[PhotoInput]) -> Evidence:
        return evidence.model_copy(deep=True)
    return perceive


# --------------------------------------------------------------------------


def scenario(title: str, pipe: Pipeline, ph, claims=None) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    v = pipe.run(ph, claims=claims)
    print(render(v))


def main() -> None:
    pricer_path = Path("data/pricer.pkl")
    if not pricer_path.exists():
        raise SystemExit(
            "Train first:\n"
            "  python scripts/make_synth_comps.py\n"
            "  python scripts/train_pricing.py"
        )
    pricer = QuantilePricer.load(pricer_path)

    scenario(
        "1. Seller uploads a photo of a motorcycle",
        Pipeline(detector=make_detector("motorcycle"),
                 perceive=make_perception(well_photographed()), pricer=pricer),
        photos(2),
    )

    scenario(
        "2. Photos taken at night in an unlit yard",
        Pipeline(detector=make_detector("truck"),
                 perceive=make_perception(well_photographed()), pricer=pricer),
        [PhotoInput(index=0, image=dark_photo()),
         PhotoInput(index=1, image=blurry_photo())],
    )

    scenario(
        "3. Clear truck photos, no badge or VIN legible",
        Pipeline(detector=make_detector("truck"),
                 perceive=make_perception(no_identity()), pricer=pricer),
        photos(3, seed=20),
    )

    scenario(
        "4. Twelve good photos, odometer and tires visible",
        Pipeline(detector=make_detector("truck"),
                 perceive=make_perception(well_photographed()), pricer=pricer),
        photos(8, seed=40),
        SellerClaims(odometer_km=311_000, region="midwest"),
    )

    scenario(
        "5. Same truck, two side shots and nothing else",
        Pipeline(detector=make_detector("truck"),
                 perceive=make_perception(barely_photographed()), pricer=pricer),
        photos(2, seed=60),
    )

    scenario(
        "6. Seller states 90,000 km; interior says otherwise",
        Pipeline(detector=make_detector("truck"),
                 perceive=make_perception(_liar_evidence()), pricer=pricer),
        photos(6, seed=80),
        SellerClaims(odometer_km=90_000, year=2016, make="hino"),
    )


def _liar_evidence() -> Evidence:
    ev = well_photographed()
    ev.condition.odometer_km = hidden()  # dash not photographed
    ev.condition.seat_wear = obs("heavy", 7)
    return ev




def compare_widths() -> None:
    """The behaviour the whole architecture exists to produce."""
    from truckval.uncertainty import run_uncertainty
    pricer = QuantilePricer.load("data/pricer.pkl")
    print("\n" + "=" * 74)
    print("Range width vs. photo coverage (same truck, same market)")
    print("=" * 74)
    print(f"{'evidence':<34}{'low':>11}{'high':>11}{'width':>11}{'excess':>9}")
    for label, ev in (("12 photos, everything visible", well_photographed()),
                      ("2 side shots, nothing else", barely_photographed())):
        r = run_uncertainty(ev, pricer, n_draws=2000)
        print(f"{label:<34}{r.price.low:>11,.0f}{r.price.high:>11,.0f}"
              f"{r.price.width:>11,.0f}{r.excess_ratio:>8.0%}")
    print("\nThe floor is market spread. Everything above it is missing photos.")


if __name__ == "__main__":
    main()
    compare_widths()

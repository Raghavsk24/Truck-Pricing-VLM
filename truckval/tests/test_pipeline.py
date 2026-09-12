"""The properties worth locking down. Run with: python -m pytest tests/ -q

The second test is the one that matters. If `test_range_widens_when_blind`
ever goes green-to-red, the product is broken even if nothing throws.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truckval.gate import (  # noqa: E402
    DetectorResult, PhotoInput, capture_usable, run_gate, same_vehicle,
)
from truckval.identity import vin_check_digit_ok  # noqa: E402
from truckval.ledger import build_ledger, ledger_total  # noqa: E402
from truckval.pricing import QuantilePricer  # noqa: E402
from truckval.schema import (  # noqa: E402
    Condition, Configuration, Evidence, GateReason, Identity, Observation,
    Tier, TireObservation, Visibility,
)
from truckval.uncertainty import invert_quantiles, run_uncertainty  # noqa: E402

PRICER = Path("data/pricer.pkl")


def obs(v, i=0):
    return Observation(value=v, visibility=Visibility.OBSERVED, confidence=0.9,
                       photo_index=i)


def sharp(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(70, 190, (128, 128, 3), dtype=np.uint8)


def det(cls, cov=0.5):
    return lambda p: DetectorResult(top_class=cls, confidence=0.95, frame_coverage=cov)


# --------------------------------------------------------------------- gate


def test_motorcycle_is_refused_and_named():
    g = run_gate([PhotoInput(0, sharp())], det("motorcycle"))
    assert g.tier == Tier.NOTHING
    assert GateReason.WRONG_OBJECT in g.reasons
    assert g.detected_object == "motorcycle"


def test_dark_photo_refused():
    dark = np.zeros((128, 128, 3), dtype=np.uint8)
    g = run_gate([PhotoInput(0, dark)], det("truck"))
    assert g.tier == Tier.NOTHING
    assert GateReason.UNUSABLE_CAPTURE in g.reasons


def test_distant_truck_refused():
    g = run_gate([PhotoInput(0, sharp())], det("truck", cov=0.02))
    assert GateReason.FRAME_COVERAGE_LOW in g.reasons


def test_conflicting_plates_beat_colour_evidence():
    ok, _ = same_vehicle([PhotoInput(0, sharp(1)), PhotoInput(1, sharp(1))],
                         plates=["ABC1234", "XYZ9876"])
    assert not ok


def test_blur_detection():
    smooth = np.full((128, 128, 3), 128, dtype=np.uint8)
    assert not capture_usable(PhotoInput(0, smooth))[0]
    assert capture_usable(PhotoInput(1, sharp()))[0]


# ----------------------------------------------------------------- identity


def test_vin_check_digit():
    assert vin_check_digit_ok("1HGCM82633A004352")
    assert not vin_check_digit_ok("1HGCM82633A004353")
    assert not vin_check_digit_ok("TOOSHORT")


# ------------------------------------------------------------------- ledger


def test_worn_tires_cost_more_than_good_ones():
    def ev_with(state):
        return Evidence(condition=Condition(tires=[
            TireObservation(position="steer_left", tread=obs(state))
        ]))
    worn = ledger_total(build_ledger(ev_with("worn")))
    good = ledger_total(build_ledger(ev_with("good")))
    assert worn < good < 0


def test_every_ledger_line_is_a_deduction_or_zero():
    ev = Evidence(condition=Condition(
        rust=obs("perforation"), seat_wear=obs("heavy"), glass_cracked=obs(True),
    ))
    for line in build_ledger(ev):
        assert line.amount <= 0
        assert line.amount_sd >= 0


def test_observed_lines_cite_evidence():
    ev = Evidence(condition=Condition(rust=obs("scaling", i=4)))
    lines = [l for l in build_ledger(ev) if "Corrosion" in l.label]
    assert lines and lines[0].evidence_ref == "photo 4"


# -------------------------------------------------------------- uncertainty


def test_quantile_inversion_is_monotone():
    vals = np.array([[10_000.0, 20_000.0, 34_000.0]])
    us = np.linspace(0.01, 0.99, 40)
    out = [invert_quantiles([0.1, 0.5, 0.9], vals, np.array([u]))[0] for u in us]
    assert all(b >= a - 1e-6 for a, b in zip(out, out[1:]))


def test_quantile_inversion_recovers_the_median():
    vals = np.array([[10_000.0, 20_000.0, 34_000.0]])
    assert invert_quantiles([0.1, 0.5, 0.9], vals, np.array([0.5]))[0] == pytest.approx(
        20_000.0, rel=1e-6
    )


# ------------------------------------------------- the one that really matters


def _seen() -> Evidence:
    return Evidence(
        identity=Identity(year=obs(2016), make=obs("hino"), model=obs("268")),
        configuration=Configuration(
            body_type=obs("box_truck"), cab_type=obs("standard"),
            axle_count=obs(2), box_length_ft=obs(24.0), gvwr_lb=obs(25_950),
        ),
        condition=Condition(odometer_km=obs(311_000)),
        usable_exterior_angles=6,
        region="midwest",
    )


def _blind() -> Evidence:
    return Evidence(
        identity=Identity(year=obs(2016), make=obs("hino"), model=obs("268")),
        configuration=Configuration(body_type=obs("box_truck")),
        usable_exterior_angles=2,
        region="midwest",
    )


@pytest.mark.skipif(not PRICER.exists(), reason="run scripts/train_pricing.py first")
def test_range_widens_when_blind():
    """Missing photos must cost precision. This is the whole product."""
    p = QuantilePricer.load(PRICER)
    seen = run_uncertainty(_seen(), p, n_draws=800, seed=3)
    blind = run_uncertainty(_blind(), p, n_draws=800, seed=3)
    assert blind.price.width > seen.price.width * 1.15
    assert blind.excess_ratio > seen.excess_ratio


@pytest.mark.skipif(not PRICER.exists(), reason="run scripts/train_pricing.py first")
def test_fully_observed_sits_near_the_floor():
    p = QuantilePricer.load(PRICER)
    r = run_uncertainty(_seen(), p, n_draws=800, seed=3)
    assert r.excess_ratio < 0.25


@pytest.mark.skipif(not PRICER.exists(), reason="run scripts/train_pricing.py first")
def test_conformal_widens_never_narrows():
    p = QuantilePricer.load(PRICER)
    from truckval.features import build_row
    rows = [build_row(_seen())]
    raw = p.predict(rows, conformal=False)
    cal = p.predict(rows, conformal=True)
    assert cal[0, 0] <= raw[0, 0] + 1e-6
    assert cal[0, -1] >= raw[0, -1] - 1e-6

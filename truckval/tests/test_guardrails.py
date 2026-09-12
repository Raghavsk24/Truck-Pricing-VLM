"""Guardrails, soft evidence, and the anchor-leak rule.

Everything here locks down a defect that was real at some point, not a
hypothetical. Each test names the failure it prevents.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from truckval.identity import merge_claims  # noqa: E402
from truckval.perception import PerceptionRejected, parse_evidence  # noqa: E402
from truckval.pipeline import _assert_no_anchor_leak  # noqa: E402
from truckval.pricing import QuantilePricer  # noqa: E402
from truckval.priors import DEFAULT_PRIORS, claim_anchored, with_claims  # noqa: E402
from truckval.schema import (  # noqa: E402
    Condition, Configuration, Evidence, GateResult, Identity, Observation,
    PriceRange, SellerClaims, Tier, Valuation, Visibility,
)
from truckval.uncertainty import invert_quantiles, run_uncertainty  # noqa: E402
from truckval.validate import LEGAL_KEYS, validate  # noqa: E402

PRICER = Path("data/pricer.pkl")


def ob(v=None, vis="not_visible", c=0.0, i=None):
    return {"value": v, "visibility": vis, "confidence": c, "photo_index": i}


def payload():
    return {
        "identity": {"vin": ob(), "year": ob(2016, "observed", .9, 0),
                     "make": ob("hino", "observed", .9, 0),
                     "model": ob("268", "observed", .8, 0), "trim": ob()},
        "configuration": {"body_type": ob("box_truck", "observed", .9, 0),
                          "cab_type": ob(), "axle_count": ob(2, "observed", .9, 1),
                          "box_length_ft": ob(), "gvwr_lb": ob()},
        "equipment": {k: ob() for k in
                      ("liftgate", "reefer_unit", "crane", "fifth_wheel", "pto")},
        "condition": {"odometer_km": ob(), "engine_hours": ob(), "rust": ob(),
                      "seat_wear": ob(), "dash_condition": ob(),
                      "glass_cracked": ob(), "paint_fade": ob(),
                      "tires": [], "damage": []},
        "usable_exterior_angles": 3,
    }


# ------------------------------------------------------------------ layer 2


def test_clean_payload_passes():
    r = validate(payload(), 3)
    assert not r.rejected and not r.issues


def test_value_without_visibility_is_quarantined():
    """The exact hallucination the NOT_VISIBLE design exists to prevent."""
    p = payload()
    p["condition"]["rust"] = ob("surface", "not_visible", .8, 0)
    r = validate(p, 3)
    assert r.payload["condition"]["rust"]["value"] is None
    assert "condition.rust" in r.quarantined


def test_observed_without_provenance_is_quarantined():
    p = payload()
    p["condition"]["seat_wear"] = ob("heavy", "observed", .9, None)
    assert validate(p, 3).payload["condition"]["seat_wear"]["value"] is None


def test_photo_index_out_of_range_is_quarantined():
    p = payload()
    p["identity"]["make"] = ob("hino", "observed", .9, 7)
    assert validate(p, 3).payload["identity"]["make"]["value"] is None


def test_duplicate_wheel_position_dropped():
    p = payload()
    p["condition"]["tires"] = [
        {"position": "steer_left", "tread": ob("worn", "observed", .8, 1)},
        {"position": "steer_left", "tread": ob("good", "observed", .7, 2)},
    ]
    assert len(validate(p, 3).payload["condition"]["tires"]) == 1


def test_implausible_odometer_for_year_is_quarantined():
    p = payload()
    p["condition"]["odometer_km"] = ob(3_100_000, "observed", .9, 2)
    assert validate(p, 3).payload["condition"]["odometer_km"]["value"] is None


def test_pricing_key_rejects_whole_payload():
    p = payload()
    p["estimated_market_value"] = 24_500
    assert validate(p, 3).rejected


def test_money_in_free_text_rejects():
    p = payload()
    p["condition"]["rust"] = {**ob("surface", "observed", .8, 1),
                              "note": "repair approx $1,200"}
    assert validate(p, 3).rejected


def test_structural_keys_are_not_mistaken_for_pricing():
    """`value` and `confidence` are contract keys, not appraisals."""
    assert {"value", "confidence", "photo_index"} <= LEGAL_KEYS
    assert not validate(payload(), 3).rejected


def test_bad_field_does_not_destroy_good_ones():
    """A guardrail failure degrades the FIELD, not the response."""
    p = payload()
    p["condition"]["odometer_km"] = ob(311_000, "observed", .9, 1)
    p["condition"]["damage"] = [
        {"panel": "door", "kind": "dent", "severity": "moderate",
         "photo_index": 9, "confidence": .8},
        {"panel": "fender", "kind": "scratch", "severity": "minor",
         "photo_index": 1, "confidence": .7},
    ]
    r = validate(p, 3)
    assert len(r.payload["condition"]["damage"]) == 1
    assert r.payload["condition"]["odometer_km"]["value"] == 311_000


def test_parse_evidence_raises_on_rejected_payload():
    p = payload()
    p["appraised_value_usd"] = 30_000
    with pytest.raises(PerceptionRejected):
        parse_evidence(p, n_photos=3)


# -------------------------------------------------------------- soft evidence


def test_typed_claims_are_claimed_not_observed():
    """A number typed by someone with a financial interest is not a photo."""
    ev = Evidence()
    ev = merge_claims(ev, SellerClaims(odometer_km=90_000, year=2016))
    assert ev.condition.odometer_km.visibility == Visibility.CLAIMED
    assert not ev.condition.odometer_km.hard
    assert ev.condition.odometer_km.known


def test_claimed_fields_are_sampled_not_pinned():
    ev = merge_claims(Evidence(), SellerClaims(odometer_km=90_000))
    assert "condition.odometer_km" in ev.unknown_fields()
    assert ev.claimed_values()["condition.odometer_km"] == 90_000


def test_photo_beats_claim():
    ev = Evidence(condition=Condition(odometer_km=Observation(
        value=311_000, visibility=Visibility.OBSERVED, confidence=.9, photo_index=1)))
    ev = merge_claims(ev, SellerClaims(odometer_km=90_000))
    assert ev.condition.odometer_km.value == 311_000


def test_contradicted_claim_widens_more_than_trusted_one():
    """Disputed claims must cost precision, or cross-check is decoration."""
    rng = np.random.default_rng(0)
    ev = merge_claims(Evidence(configuration=Configuration()),
                      SellerClaims(odometer_km=90_000))
    trusted = with_claims(DEFAULT_PRIORS, {"condition.odometer_km": 90_000})
    disputed = with_claims(DEFAULT_PRIORS, {"condition.odometer_km": 90_000},
                           contradicted={"condition.odometer_km"})
    a = np.std([trusted.sample("condition.odometer_km", ev, rng) for _ in range(800)])
    b = np.std([disputed.sample("condition.odometer_km", ev, rng) for _ in range(800)])
    assert b > a * 1.5


# ------------------------------------------------------------ anchor leakage


def _val(tier, **kw):
    return Valuation(tier=tier, gate=GateResult(tier=tier), evidence=Evidence(), **kw)


def test_tier0_with_a_price_raises():
    with pytest.raises(AssertionError):
        _assert_no_anchor_leak(_val(
            Tier.NOTHING, price=PriceRange(low=1, mid=2, high=3, n_samples=1)))


def test_tier1_message_with_a_ballpark_raises():
    with pytest.raises(AssertionError):
        _assert_no_anchor_leak(_val(
            Tier.CONDITION_ONLY,
            message="Trucks like this usually go for $25,000 to $60,000."))


def test_tier1_clean_message_passes():
    _assert_no_anchor_leak(_val(
        Tier.CONDITION_ONLY,
        message="I can't price this yet. Send the VIN plate on the door jamb."))


def test_tier2_without_a_price_raises():
    with pytest.raises(AssertionError):
        _assert_no_anchor_leak(_val(Tier.PRICED))


# ---------------------------------------------------------------- tail shape


def test_quantile_inversion_is_lognormal_exact():
    """Linear-in-price interpolation compressed the right tail. Log space fixes it."""
    mu, sig = np.log(30_000), 0.25
    from scipy.stats import norm as N
    q = np.array([[np.exp(mu + sig * N.ppf(l)) for l in (0.1, 0.5, 0.9)]])
    for u in (0.2, 0.35, 0.65, 0.8):
        got = invert_quantiles([0.1, 0.5, 0.9], q, np.array([u]))[0]
        assert got == pytest.approx(np.exp(mu + sig * N.ppf(u)), rel=1e-6)


@pytest.mark.skipif(not PRICER.exists(), reason="run scripts/train_pricing.py first")
def test_unverified_claim_widens_versus_photographed_fact():
    p = QuantilePricer.load(PRICER)
    base = dict(identity=Identity(
        year=Observation(value=2016, visibility=Visibility.OBSERVED, confidence=.9, photo_index=0),
        make=Observation(value="hino", visibility=Visibility.OBSERVED, confidence=.9, photo_index=0),
        model=Observation(value="268", visibility=Visibility.OBSERVED, confidence=.9, photo_index=0)),
        configuration=Configuration(body_type=Observation(
            value="box_truck", visibility=Visibility.OBSERVED, confidence=.9, photo_index=0)),
        usable_exterior_angles=4, region="midwest")
    seen = Evidence(**copy.deepcopy(base), condition=Condition(odometer_km=Observation(
        value=311_000, visibility=Visibility.OBSERVED, confidence=.9, photo_index=2)))
    told = Evidence(**copy.deepcopy(base), condition=Condition(odometer_km=Observation(
        value=311_000, visibility=Visibility.CLAIMED, confidence=.55)))
    a = run_uncertainty(seen, p, n_draws=700, seed=5).price.width
    b = run_uncertainty(told, p, n_draws=700, seed=5).price.width
    assert b > a

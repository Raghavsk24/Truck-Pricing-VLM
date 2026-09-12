"""Stage 2. Pin down what the vehicle is, and check what the seller said.

Two jobs:
  1. A legible VIN replaces five uncertain visual guesses with ground truth.
     vPIC is free, needs no key, and covers trucks, buses, trailers and
     incomplete vehicles since 1981. Pull the standalone DB for production so
     this is a table lookup rather than a network call inside your hot path.
  2. Typed details are claims. Cross-check them and surface disagreements
     rather than silently trusting or silently overriding.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Optional

from .schema import Contradiction, Evidence, Observation, SellerClaims, Visibility

VPIC_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevinvalues/{vin}?format=json"

VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")
_TRANSLIT = {
    **{c: i for i, c in enumerate("0123456789")},
    **dict(zip("ABCDEFGH", range(1, 9))),
    **dict(zip("JKLMN", range(1, 6))),
    "P": 7, "R": 9,
    **dict(zip("STUVWXYZ", range(2, 10))),
}
_WEIGHTS = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]


def vin_check_digit_ok(vin: str) -> bool:
    """Position 9 is a check digit computed from the other 16 characters.

    A VIN read off a photo that fails this is almost always an OCR error
    (0 vs Q, 1 vs I or L) rather than a tampered plate. Either way: do not
    trust it, mark it occluded and ask for a clearer shot.
    """
    vin = vin.upper()
    if not VIN_RE.match(vin):
        return False
    total = sum(_TRANSLIT[c] * w for c, w in zip(vin, _WEIGHTS))
    rem = total % 11
    expected = "X" if rem == 10 else str(rem)
    return vin[8] == expected


def decode_vin(vin: str, timeout: float = 10.0) -> Optional[dict[str, Any]]:
    """Live vPIC lookup. Swap for a local table read in production."""
    if not VIN_RE.match(vin.upper()):
        return None
    url = VPIC_URL.format(vin=urllib.parse.quote(vin.upper()))
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            payload = json.loads(r.read().decode())
    except Exception:
        return None
    results = payload.get("Results") or []
    return results[0] if results else None


def apply_vin_decode(evidence: Evidence, decoded: dict[str, Any]) -> Evidence:
    """vPIC outranks visual inference on every field it covers."""
    def setobs(group: str, name: str, value: Any) -> None:
        if value in (None, "", "Not Applicable"):
            return
        setattr(
            getattr(evidence, group),
            name,
            Observation(
                value=value,
                visibility=Visibility.OBSERVED,
                confidence=0.99,
                note="vPIC VIN decode",
            ),
        )

    def num(key: str) -> Optional[float]:
        raw = decoded.get(key)
        try:
            return float(str(raw).strip())
        except (TypeError, ValueError):
            return None

    year = num("ModelYear")
    setobs("identity", "year", int(year) if year else None)
    setobs("identity", "make", (decoded.get("Make") or "").title() or None)
    setobs("identity", "model", (decoded.get("Model") or "").title() or None)

    gvwr = decoded.get("GVWR") or ""
    lo = re.search(r"([\d,]+)\s*lb", gvwr)
    if lo:
        setobs("configuration", "gvwr_lb", int(lo.group(1).replace(",", "")))

    axles = num("Axles")
    setobs("configuration", "axle_count", int(axles) if axles else None)
    return evidence


# --------------------------------------------------------------------------
# Claim cross-checking
# --------------------------------------------------------------------------

# Rough interior-wear expectations. Calibrate these on your own labelled set;
# the point is the shape of the check, not these particular numbers.
WEAR_KM_RANGES = {
    "light": (0, 180_000),
    "moderate": (120_000, 450_000),
    "heavy": (350_000, 1_200_000),
    "destroyed": (700_000, 3_000_000),
}


def cross_check(evidence: Evidence, claims: SellerClaims) -> list[Contradiction]:
    out: list[Contradiction] = []

    if claims.year and evidence.identity.year.known:
        seen = int(evidence.identity.year.value)
        if abs(seen - claims.year) >= 2:
            out.append(Contradiction(
                field="year",
                claimed=claims.year,
                evidence_suggests=seen,
                basis=evidence.identity.year.note or "photo evidence",
                severity="major" if abs(seen - claims.year) >= 4 else "warn",
            ))

    if claims.make and evidence.identity.make.known:
        if claims.make.strip().lower() != str(evidence.identity.make.value).lower():
            out.append(Contradiction(
                field="make",
                claimed=claims.make,
                evidence_suggests=evidence.identity.make.value,
                basis=evidence.identity.make.note or "badge or VIN",
                severity="major",
            ))

    # Odometer vs interior wear. This is the classic one.
    if claims.odometer_km and evidence.condition.seat_wear.known:
        wear = str(evidence.condition.seat_wear.value)
        lo, hi = WEAR_KM_RANGES.get(wear, (0, 10**9))
        if claims.odometer_km < lo * 0.6:
            out.append(Contradiction(
                field="odometer_km",
                claimed=claims.odometer_km,
                evidence_suggests=f"{lo:,}+ km",
                basis=f"{wear} interior wear (photo "
                      f"{evidence.condition.seat_wear.photo_index})",
                severity="major",
            ))

    # Odometer read off the dash beats anything typed.
    if claims.odometer_km and evidence.condition.odometer_km.known:
        seen = float(evidence.condition.odometer_km.value)
        if seen > 0 and abs(seen - claims.odometer_km) / seen > 0.10:
            out.append(Contradiction(
                field="odometer_km",
                claimed=claims.odometer_km,
                evidence_suggests=seen,
                basis=f"odometer legible in photo "
                      f"{evidence.condition.odometer_km.photo_index}",
                severity="major",
            ))

    if claims.box_length_ft and evidence.configuration.box_length_ft.known:
        seen = float(evidence.configuration.box_length_ft.value)
        if abs(seen - claims.box_length_ft) > 3.0:
            out.append(Contradiction(
                field="box_length_ft",
                claimed=claims.box_length_ft,
                evidence_suggests=seen,
                basis="measured against wheel diameter",
                severity="warn",
            ))

    return out


def merge_claims(evidence: Evidence, claims: SellerClaims) -> Evidence:
    """Fill gaps from typed input as CLAIMED. Never overwrite an observation.

    Earlier this marked claims OBSERVED at confidence 0.55, which meant the
    uncertainty pass pinned them — an unverified number typed by someone with
    a financial interest in a high price carried the same weight as an
    odometer read off the dash. Confidence was recorded and then ignored.

    CLAIMED is a distinct visibility, so `unknown_fields()` returns these and
    the Monte Carlo samples around them instead of trusting them.
    """
    def fill(group: str, name: str, value: Any) -> None:
        if value is None:
            return
        obs = getattr(getattr(evidence, group), name)
        if obs.hard:
            return
        setattr(
            getattr(evidence, group),
            name,
            Observation(
                value=value,
                visibility=Visibility.CLAIMED,
                confidence=0.55,
                note="seller-stated, unverified",
            ),
        )

    fill("identity", "year", claims.year)
    fill("identity", "make", claims.make)
    fill("identity", "model", claims.model)
    fill("condition", "odometer_km", claims.odometer_km)
    fill("configuration", "box_length_ft", claims.box_length_ft)
    if claims.region and not evidence.region:
        evidence.region = claims.region
    return evidence

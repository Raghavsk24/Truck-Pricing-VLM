"""Condition adjustments as a ledger of line items.

Two rules that keep this honest:

1. The ledger and the feature set must PARTITION the world. Anything the
   pricing model already sees as a feature (age, distance, body type, axles,
   mounted equipment) must never appear as a ledger line, or you double count.
   The ledger covers condition only: rubber, damage, corrosion, glass, interior.

2. Every line carries an `amount_sd` and an `evidence_ref`. The sd feeds the
   Monte Carlo so that uncertain deductions widen the range. The evidence_ref
   is what turns the ledger into the explanation — the report is generated
   from these lines, so a claim with no line behind it cannot be printed.

The numbers below are reconditioning-cost placeholders. Calibrate them against
Copart / IAA lots, where the same vehicle carries both a damage classification
and a realized sale price.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .schema import Evidence, LedgerLine

# Installed cost per tire by position class. Steer tires cost more and wear
# faster; drive positions are usually duals.
TIRE_COST = {"steer": 620.0, "drive": 480.0, "tag": 460.0}
TIRE_COST_SD = 90.0

# Fraction of a new tire's cost you lose at each tread state.
TREAD_PENALTY = {
    "new": 0.0,
    "good": 0.15,
    "worn": 0.55,
    "legal_minimum": 0.85,
    "below_minimum": 1.10,  # over 100%: below legal is a sale blocker
}

DAMAGE_COST = {
    ("dent", "minor"): (240.0, 90.0),
    ("dent", "moderate"): (780.0, 260.0),
    ("dent", "severe"): (2_400.0, 900.0),
    ("scratch", "minor"): (110.0, 45.0),
    ("scratch", "moderate"): (380.0, 140.0),
    ("scratch", "severe"): (950.0, 380.0),
    ("crack", "minor"): (200.0, 80.0),
    ("crack", "moderate"): (650.0, 240.0),
    ("crack", "severe"): (1_800.0, 700.0),
    ("glass_shatter", "minor"): (420.0, 130.0),
    ("glass_shatter", "moderate"): (620.0, 180.0),
    ("glass_shatter", "severe"): (900.0, 280.0),
    ("lamp_broken", "minor"): (130.0, 50.0),
    ("lamp_broken", "moderate"): (260.0, 90.0),
    ("lamp_broken", "severe"): (480.0, 160.0),
    ("tire_flat", "minor"): (180.0, 60.0),
    ("tire_flat", "moderate"): (400.0, 130.0),
    ("tire_flat", "severe"): (600.0, 200.0),
    ("panel_missing", "minor"): (500.0, 200.0),
    ("panel_missing", "moderate"): (1_400.0, 550.0),
    ("panel_missing", "severe"): (3_200.0, 1_300.0),
    ("deformation", "minor"): (600.0, 250.0),
    ("deformation", "moderate"): (2_100.0, 850.0),
    ("deformation", "severe"): (6_500.0, 2_800.0),
}
DEFAULT_DAMAGE = (700.0, 400.0)

# Rust is not a paint problem. Perforation on a frame rail or rocker is
# structural and buyers price it as such.
RUST_COST = {
    "none": (0.0, 0.0),
    "surface": (350.0, 200.0),
    "scaling": (1_600.0, 700.0),
    "perforation": (5_200.0, 2_600.0),
}

INTERIOR_COST = {
    "light": (0.0, 0.0),
    "moderate": (280.0, 120.0),
    "heavy": (900.0, 350.0),
    "destroyed": (2_100.0, 800.0),
}

GLASS_CRACK_COST = (520.0, 170.0)


def _position_class(position: str) -> str:
    if position.startswith("steer"):
        return "steer"
    if position.startswith("tag"):
        return "tag"
    return "drive"


def _ref(photo_index: Optional[int]) -> Optional[str]:
    return f"photo {photo_index}" if photo_index is not None else None


def build_ledger(
    ev: Evidence,
    draw: Optional[dict[str, Any]] = None,
    total_wheel_positions: int = 6,
) -> list[LedgerLine]:
    """Deterministic given evidence + one draw. Amounts are means, not samples."""
    draw = draw or {}
    lines: list[LedgerLine] = []

    # ------------------------------------------------------------ rubber
    observed = {t.position: t for t in ev.condition.tires if t.tread.known}
    sampled_tread = draw.get("condition.tires")

    for pos, tire in observed.items():
        state = str(tire.tread.value)
        pen = TREAD_PENALTY.get(state, 0.4)
        if pen <= 0:
            continue
        cost = TIRE_COST[_position_class(pos)] * pen
        lines.append(LedgerLine(
            label=f"Tire, {pos.replace('_', ' ')} ({state.replace('_', ' ')})",
            amount=-cost,
            amount_sd=TIRE_COST_SD * pen,
            evidence_ref=_ref(tire.tread.photo_index),
            basis="installed replacement cost x remaining-life penalty",
        ))

    unseen_positions = max(0, total_wheel_positions - len(observed))
    if unseen_positions and sampled_tread:
        pen = TREAD_PENALTY.get(str(sampled_tread), 0.4)
        cost = TIRE_COST["drive"] * pen * unseen_positions
        lines.append(LedgerLine(
            label=f"Tires, {unseen_positions} position(s) not visible",
            amount=-cost,
            amount_sd=TIRE_COST_SD * max(pen, 0.4) * unseen_positions * 1.6,
            evidence_ref=None,
            basis="sampled from tread prior; photo would replace this",
        ))

    # ------------------------------------------------------------ damage
    for d in ev.condition.damage:
        mean, sd = DAMAGE_COST.get((d.kind, d.severity), DEFAULT_DAMAGE)
        lines.append(LedgerLine(
            label=f"{d.severity.title()} {d.kind.replace('_', ' ')}, {d.panel}",
            amount=-mean,
            amount_sd=sd,
            evidence_ref=_ref(d.photo_index),
            basis="reconditioning estimate",
        ))

    hidden = int(draw.get("_hidden_damage_count", 0) or 0)
    if hidden:
        mean, sd = DEFAULT_DAMAGE
        lines.append(LedgerLine(
            label=f"Possible damage on {hidden} unphotographed panel(s)",
            amount=-mean * hidden,
            amount_sd=sd * hidden * 1.4,
            evidence_ref=None,
            basis="sampled; absence of a photo is not absence of damage",
        ))

    # -------------------------------------------------------------- rust
    rust = ev.condition.rust.value or draw.get("condition.rust")
    if rust:
        mean, sd = RUST_COST.get(str(rust), (0.0, 0.0))
        if mean:
            lines.append(LedgerLine(
                label=f"Corrosion: {rust}",
                amount=-mean,
                amount_sd=sd,
                evidence_ref=_ref(ev.condition.rust.photo_index),
                basis="structural where perforated, cosmetic otherwise",
            ))

    # ---------------------------------------------------------- interior
    wear = ev.condition.seat_wear.value or draw.get("condition.seat_wear")
    if wear:
        mean, sd = INTERIOR_COST.get(str(wear), (0.0, 0.0))
        if mean:
            lines.append(LedgerLine(
                label=f"Cab interior: {wear} wear",
                amount=-mean,
                amount_sd=sd,
                evidence_ref=_ref(ev.condition.seat_wear.photo_index),
                basis="seat, dash and trim refresh",
            ))

    # ------------------------------------------------------------- glass
    glass = ev.condition.glass_cracked.value
    if glass is None:
        glass = draw.get("condition.glass_cracked")
    if glass:
        mean, sd = GLASS_CRACK_COST
        lines.append(LedgerLine(
            label="Cracked glass",
            amount=-mean,
            amount_sd=sd,
            evidence_ref=_ref(ev.condition.glass_cracked.photo_index),
            basis="windscreen replacement",
        ))

    return lines


def ledger_total(lines: list[LedgerLine]) -> float:
    return float(sum(line.amount for line in lines))


def sample_ledger_total(lines: list[LedgerLine], rng: np.random.Generator) -> float:
    """One draw of the total, respecting each line's own uncertainty."""
    if not lines:
        return 0.0
    means = np.array([line.amount for line in lines])
    sds = np.array([line.amount_sd for line in lines])
    draws = rng.normal(means, np.maximum(sds, 1e-9))
    # Deductions cannot flip sign and become bonuses.
    draws = np.minimum(draws, 0.0)
    return float(draws.sum())


def expected_condition_cost(ev, priors, rng, n: int = 256) -> float:
    """What the comparable sales ALREADY price in.

    A hammer price from an auction is the price of a truck in whatever
    condition it happened to be in. So the comps baseline is not "this model in
    perfect condition" — it is "this model in average condition for its age and
    distance". Subtracting absolute reconditioning cost on top of that charges
    the seller twice for ordinary wear.

    The fix is to net the ledger against the cohort mean, so an average truck
    gets a zero net adjustment, a rough one gets a deduction, and a clean one
    gets a premium. That is also how appraisers actually work.

    The cohort mean is estimated by running the ledger under the priors with
    every condition field unknown — self-consistent, and needs no extra data.
    Netting removes the bias while leaving the spread intact, which is exactly
    what you want: uncertainty should widen the range, not move it.
    """
    from .schema import Condition

    blank = ev.model_copy(deep=True)
    blank.condition = Condition()
    if ev.condition.odometer_km.known:
        blank.condition.odometer_km = ev.condition.odometer_km

    paths = [p for p in blank.unknown_fields()
             if p.startswith("condition.") and priors.covers(p)]
    totals = []
    for _ in range(n):
        draw = {p: priors.sample(p, blank, rng) for p in paths}
        draw["_hidden_damage_count"] = priors.sample("_hidden_damage_count", blank, rng)
        totals.append(ledger_total(build_ledger(blank, draw)))
    return float(np.mean(totals))

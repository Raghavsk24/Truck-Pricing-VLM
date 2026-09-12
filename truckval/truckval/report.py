"""Stage 5. Render everything the seller sees.

Refusals and prices go through the same path on purpose. A refusal is not an
error state, it is a lower tier of the same output, and it should feel like the
system working rather than the system breaking.

Four rules encoded here:
  - Name what was seen. "That looks like a motorcycle" beats a generic error.
  - Never leak an anchor. No ballpark inside a Tier 0 or Tier 1 response.
  - Give the exact next photo: distance, angle, subject.
  - Don't accuse. Reused photos are usually a mistake.
"""

from __future__ import annotations

from .schema import (
    Contradiction,
    Evidence,
    GateReason,
    GateResult,
    LedgerLine,
    PriceRange,
    Tier,
    Valuation,
    VarianceContribution,
)

# A price is suppressed when missing information has inflated the range well
# beyond the market's own irreducible spread — not when the range is merely
# wide. See UncertaintyResult.excess_ratio.
MAX_EXCESS_RATIO = 0.60


def money(x: float) -> str:
    return f"${x:,.0f}"


# --------------------------------------------------------------------------
# Refusal copy
# --------------------------------------------------------------------------


def gate_message(gate: GateResult) -> str:
    r = gate.reasons[0] if gate.reasons else None

    if r == GateReason.WRONG_OBJECT:
        obj = gate.detected_object or "something else"
        article = "an" if obj[0] in "aeiou" else "a"
        return (
            f"That looks like {article} {obj} rather than a truck. Send a photo "
            "of the vehicle you're listing and I'll take another look."
        )
    if r == GateReason.OBJECT_UNCLEAR:
        return (
            "I can't make out a vehicle in these. One shot of the whole truck "
            "from the side would be enough to start."
        )
    if r == GateReason.UNUSABLE_CAPTURE:
        detail = gate.detail or "not usable"
        return (
            f"There's something there, but the photos are {detail}. One shot "
            "outdoors in daylight with the whole vehicle in frame would fix it."
        )
    if r == GateReason.FRAME_COVERAGE_LOW:
        return (
            "The truck is too far away to assess. Move in until it fills most "
            "of the frame and take one from the side."
        )
    if r == GateReason.MULTIPLE_VEHICLES:
        return (
            "These look like two different vehicles — "
            f"{gate.detail or 'the details don\'t match across photos'}. Send "
            "photos of just the one you're listing."
        )
    return "I can't assess these photos. Send a few clear shots of the truck."


def tier1_message(
    evidence: Evidence,
    gate: GateResult,
    next_photos: list[VarianceContribution],
) -> str:
    reasons = set(gate.reasons)
    parts: list[str] = []

    if GateReason.DUPLICATE_LISTING_PHOTOS in reasons:
        parts.append(
            "These photos also appear in a listing elsewhere. If that's yours, "
            "no problem — I'd just need one fresh photo of the truck before I "
            "can price it."
        )
    elif not evidence.identity.resolved:
        missing = [n for n in ("year", "make", "model")
                   if not getattr(evidence.identity, n).known]
        parts.append(
            f"I can't put a number on this yet. I don't know the "
            f"{' or '.join(missing)}, and everything downstream depends on "
            "that. The VIN plate on the driver's door jamb is the fastest fix; "
            "the badge on the side of the cab also works."
        )
    elif GateReason.TOO_FEW_ANGLES in reasons:
        parts.append(
            "I've got a condition read, but not enough angles to price it. "
            "Two more exterior shots — the opposite side and the rear — would "
            "get me there."
        )
    elif GateReason.INTERVAL_TOO_WIDE in reasons:
        parts.append(
            "I can price this, but missing information has stretched the "
            "window well past what the market spread alone would give, and "
            "the number wouldn't be useful to you."
        )

    if next_photos:
        asks = "; ".join(f"{c.ask}" for c in next_photos[:2])
        parts.append(f"Most useful next: {asks}.")

    return " ".join(parts)


def next_photo_lines(next_photos: list[VarianceContribution]) -> list[str]:
    return [
        f"{c.ask} — would narrow the range by about {money(c.dollars)}"
        for c in next_photos
    ]


# --------------------------------------------------------------------------
# Condition report
# --------------------------------------------------------------------------


def condition_report(ev: Evidence) -> str:
    lines: list[str] = []

    if ev.identity.resolved:
        i = ev.identity
        head = f"{i.year.value} {i.make.value} {i.model.value}"
        if i.trim.known:
            head += f" {i.trim.value}"
        lines.append(head)
    else:
        lines.append("Vehicle not identified")

    cfg = ev.configuration
    spec = []
    if cfg.body_type.known:
        spec.append(str(cfg.body_type.value).replace("_", " "))
    if cfg.box_length_ft.known:
        spec.append(f"{cfg.box_length_ft.value:g} ft box")
    if cfg.axle_count.known:
        spec.append(f"{cfg.axle_count.value} axles")
    if cfg.gvwr_lb.known:
        spec.append(f"{int(cfg.gvwr_lb.value):,} lb GVWR")
    if spec:
        lines.append("Configuration: " + ", ".join(spec))

    if ev.condition.odometer_km.known:
        src = ev.condition.odometer_km.note or "read from photo"
        lines.append(
            f"Odometer: {float(ev.condition.odometer_km.value):,.0f} km ({src})"
        )
    else:
        lines.append("Odometer: not visible")

    if ev.condition.tires:
        for t in ev.condition.tires:
            state = (str(t.tread.value).replace("_", " ")
                     if t.tread.known else t.tread.visibility.value.replace("_", " "))
            lines.append(f"Tire {t.position.replace('_', ' ')}: {state}")
    else:
        lines.append("Tires: no wheel positions assessable")

    if ev.condition.damage:
        for d in ev.condition.damage:
            lines.append(
                f"Damage: {d.severity} {d.kind.replace('_', ' ')} on {d.panel} "
                f"(photo {d.photo_index})"
            )
    else:
        lines.append("Damage: none visible in the supplied angles")

    for label, obs in (
        ("Corrosion", ev.condition.rust),
        ("Cab interior", ev.condition.seat_wear),
        ("Glass", ev.condition.glass_cracked),
    ):
        if obs.known:
            lines.append(f"{label}: {obs.value}")
        else:
            lines.append(f"{label}: {obs.visibility.value.replace('_', ' ')}")

    eq = [n for n in ("liftgate", "reefer_unit", "crane", "fifth_wheel", "pto")
          if getattr(ev.equipment, n).known and getattr(ev.equipment, n).value]
    lines.append("Equipment: " + (", ".join(e.replace("_", " ") for e in eq)
                                  if eq else "none identified"))

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Reasoning — generated FROM the ledger, never alongside it
# --------------------------------------------------------------------------


def reasoning(
    baseline: float,
    ledger: list[LedgerLine],
    price: PriceRange,
    evidence: Evidence,
    sampled_fields: list[str],
    contradictions: list[Contradiction],
) -> str:
    out: list[str] = []
    i = evidence.identity
    out.append(
        f"Comparable {i.year.value} {i.make.value} {i.model.value} sales put a "
        f"baseline around {money(baseline)} before condition."
    )

    if ledger:
        out.append("\nCondition adjustments:")
        for line in sorted(ledger, key=lambda l: l.amount):
            ref = f"  [{line.evidence_ref}]" if line.evidence_ref else "  [inferred]"
            out.append(f"  {money(line.amount):>10}  {line.label}{ref}")
        total = sum(l.amount for l in ledger)
        out.append(f"  {money(total):>10}  total")
    else:
        out.append("\nNo condition deductions applied.")

    out.append(
        f"\nRange {money(price.low)} to {money(price.high)}, midpoint "
        f"{money(price.mid)}, from {price.n_samples:,} simulations."
    )

    if sampled_fields:
        pretty = ", ".join(f.split(".", 1)[1].replace("_", " ")
                           for f in sampled_fields[:6])
        out.append(
            f"The window is this wide because {len(sampled_fields)} field(s) "
            f"weren't visible and had to be sampled: {pretty}."
        )
    else:
        out.append(
            "Every priced field was directly observed, so the width here is "
            "market spread rather than missing information."
        )

    if contradictions:
        out.append("\nClaims that don't match the photos:")
        for c in contradictions:
            out.append(
                f"  {c.field}: you said {c.claimed}; the photos suggest "
                f"{c.evidence_suggests} ({c.basis})."
            )

    return "\n".join(out)


def render(v: Valuation) -> str:
    """Human-readable dump of a Valuation. Your API would return the object."""
    parts = [f"TIER {v.tier.value}"]
    if v.message:
        parts.append(v.message)
    if v.tier != Tier.NOTHING:
        parts.append("\n--- CONDITION ---\n" + v.condition_report)
    if v.price:
        parts.append(
            f"\n--- PRICE ---\n{money(v.price.low)} to {money(v.price.high)}  "
            f"(midpoint {money(v.price.mid)})"
        )
    if v.reasoning:
        parts.append("\n--- REASONING ---\n" + v.reasoning)
    if v.next_photos:
        parts.append("\n--- NEXT PHOTOS ---\n" +
                     "\n".join("  " + s for s in next_photo_lines(v.next_photos)))
    return "\n".join(parts)

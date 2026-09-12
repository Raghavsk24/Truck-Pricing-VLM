"""Stage 1. Turn photos into structured evidence. Never into a price.

The schema below is the whole trick. Every field must come back with a
visibility and a confidence, and `not_visible` is an explicit allowed value.
Models hallucinate when the output format punishes silence; this one rewards it.

Set ANTHROPIC_API_KEY and call `extract_evidence`. Without a key, use
`StubPerception` from scripts/demo.py.
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any, Optional, Sequence

from .validate import HARDENED_EVIDENCE_SCHEMA, validate
from .schema import (
    Condition,
    Configuration,
    DamageInstance,
    Equipment,
    Evidence,
    Identity,
    Observation,
    TireObservation,
    Visibility,
)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """\
You are a commercial vehicle inspector reading seller photographs. You extract \
observable facts only.

Rules you must not break:
1. You never estimate, appraise or mention a monetary value. Not even a range.
2. For every field, if the photos do not show it, return visibility \
"not_visible". If something is present but blocked by mud, snow, a tarp, glare \
or a crop, return "occluded". Do not infer a value from what is typical.
3. Every observed field must cite the photo index it came from.
4. Confidence is your own calibrated probability that the value is correct, \
not how typical the value is.
5. Read text literally. A VIN plate, a door sticker, an odometer or a badge is \
worth more than any visual inference. If characters are ambiguous, mark the \
field occluded rather than guessing a character.

Output a single JSON object matching the supplied schema. No prose, no \
markdown fences.
"""

# Kept as a dict so it can be passed straight into a structured-output /
# tool-use call, or pasted into the prompt for models without that support.
# Superseded by validate.HARDENED_EVIDENCE_SCHEMA, which closes every
# vocabulary, types `value` per field, caps array lengths and forbids extra
# properties. Kept only for models without structured-output support that need
# the shape pasted into a prompt.
EVIDENCE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["identity", "configuration", "equipment", "condition",
                 "usable_exterior_angles"],
    "properties": {
        "identity": {
            "type": "object",
            "properties": {
                f: {"$ref": "#/$defs/obs"}
                for f in ("vin", "year", "make", "model", "trim")
            },
        },
        "configuration": {
            "type": "object",
            "properties": {
                f: {"$ref": "#/$defs/obs"}
                for f in ("body_type", "cab_type", "axle_count",
                          "box_length_ft", "gvwr_lb")
            },
        },
        "equipment": {
            "type": "object",
            "properties": {
                f: {"$ref": "#/$defs/obs"}
                for f in ("liftgate", "reefer_unit", "crane",
                          "fifth_wheel", "pto")
            },
        },
        "condition": {
            "type": "object",
            "properties": {
                **{
                    f: {"$ref": "#/$defs/obs"}
                    for f in ("odometer_km", "engine_hours", "rust",
                              "seat_wear", "dash_condition", "glass_cracked",
                              "paint_fade")
                },
                "tires": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["position", "tread"],
                        "properties": {
                            "position": {"type": "string"},
                            "tread": {"$ref": "#/$defs/obs"},
                            "sidewall_damage": {"$ref": "#/$defs/obs"},
                            "mismatched_brand": {"$ref": "#/$defs/obs"},
                        },
                    },
                },
                "damage": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["panel", "kind", "severity",
                                     "photo_index", "confidence"],
                        "properties": {
                            "panel": {"type": "string"},
                            "kind": {"type": "string"},
                            "severity": {
                                "type": "string",
                                "enum": ["minor", "moderate", "severe"],
                            },
                            "photo_index": {"type": "integer"},
                            "confidence": {"type": "number"},
                        },
                    },
                },
            },
        },
        "usable_exterior_angles": {"type": "integer"},
    },
    "$defs": {
        "obs": {
            "type": "object",
            "required": ["visibility", "confidence"],
            "properties": {
                "value": {},
                "visibility": {
                    "type": "string",
                    "enum": ["observed", "not_visible", "occluded"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "photo_index": {"type": ["integer", "null"]},
                "note": {"type": ["string", "null"]},
            },
        }
    },
}

USER_INSTRUCTION = """\
Inspect the attached photographs of a single commercial vehicle and fill in \
the schema. Photos are indexed from 0 in the order attached.

Pay particular attention to:
- Any legible VIN plate, door-jamb sticker, or cab badge.
- Tread on each visible wheel position. Duals: report outer and inner \
separately, and mark the inner one occluded if the outer wheel hides it.
- Box or container length. Estimate it against wheel diameter as a scale \
reference and say so in the note. If no wheel is visible in the same frame as \
the box, mark it not_visible.
- Rust: distinguish surface oxidation from scaling and from perforation. \
Perforation means you can see a hole through the metal.
- Mounted equipment: liftgate, reefer unit, crane, fifth wheel, PTO.

Return JSON only.
"""


def _obs(raw: Optional[dict[str, Any]]) -> Observation[Any]:
    if not raw:
        return Observation(visibility=Visibility.NOT_VISIBLE)
    vis = Visibility(raw.get("visibility", "not_visible"))
    value = raw.get("value") if vis == Visibility.OBSERVED else None
    return Observation(
        value=value,
        visibility=vis,
        confidence=float(raw.get("confidence", 0.0)),
        photo_index=raw.get("photo_index"),
        note=raw.get("note"),
    )


class PerceptionRejected(ValueError):
    """The payload violated a whole-payload guardrail. Do not use any of it."""


def parse_evidence(
    payload: dict[str, Any],
    region: Optional[str] = None,
    n_photos: Optional[int] = None,
    strict: bool = True,
) -> Evidence:
    """Map raw model JSON onto the typed Evidence bundle.

    Runs the layer-2 guardrails first when `n_photos` is supplied. Fields that
    fail an invariant are quarantined to NOT_VISIBLE rather than dropping the
    whole extraction, which has a useful side effect: a quarantined field is
    indistinguishable downstream from a genuinely unseen one, so the
    uncertainty pass widens the range for it automatically. Guardrail
    violations make the answer vaguer, not wrong.
    """
    if n_photos is not None:
        result = validate(payload, n_photos)
        if result.rejected and strict:
            raise PerceptionRejected(
                "; ".join(f"{i.path}: {i.problem}"
                          for i in result.issues if i.severity == "reject")
            )
        payload = result.payload

    ident_raw = payload.get("identity", {})
    conf_raw = payload.get("configuration", {})
    equip_raw = payload.get("equipment", {})
    cond_raw = payload.get("condition", {})

    identity = Identity(**{f: _obs(ident_raw.get(f))
                           for f in ("vin", "year", "make", "model", "trim")})
    configuration = Configuration(**{
        f: _obs(conf_raw.get(f))
        for f in ("body_type", "cab_type", "axle_count", "box_length_ft", "gvwr_lb")
    })
    equipment = Equipment(**{
        f: _obs(equip_raw.get(f))
        for f in ("liftgate", "reefer_unit", "crane", "fifth_wheel", "pto")
    })

    tires = [
        TireObservation(
            position=t["position"],
            tread=_obs(t.get("tread")),
            sidewall_damage=_obs(t.get("sidewall_damage")),
            mismatched_brand=_obs(t.get("mismatched_brand")),
        )
        for t in cond_raw.get("tires", [])
    ]
    damage = [DamageInstance(**d) for d in cond_raw.get("damage", [])]

    condition = Condition(
        odometer_km=_obs(cond_raw.get("odometer_km")),
        engine_hours=_obs(cond_raw.get("engine_hours")),
        tires=tires,
        damage=damage,
        rust=_obs(cond_raw.get("rust")),
        seat_wear=_obs(cond_raw.get("seat_wear")),
        dash_condition=_obs(cond_raw.get("dash_condition")),
        glass_cracked=_obs(cond_raw.get("glass_cracked")),
        paint_fade=_obs(cond_raw.get("paint_fade")),
    )

    return Evidence(
        identity=identity,
        configuration=configuration,
        equipment=equipment,
        condition=condition,
        usable_exterior_angles=int(payload.get("usable_exterior_angles", 0)),
        region=region,
    )


def extract_evidence(
    image_bytes: Sequence[bytes],
    media_types: Sequence[str],
    region: Optional[str] = None,
    model: str = MODEL,
) -> Evidence:
    """Live VLM call. Requires `anthropic` and ANTHROPIC_API_KEY."""
    from anthropic import Anthropic  # imported lazily so the repo runs without it

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    content: list[dict[str, Any]] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mt,
                "data": base64.b64encode(b).decode(),
            },
        }
        for b, mt in zip(image_bytes, media_types)
    ]
    content.append({
        "type": "text",
        "text": USER_INSTRUCTION
        + "\n\nSchema:\n"
        + json.dumps(HARDENED_EVIDENCE_SCHEMA),
    })

    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return parse_evidence(json.loads(text), region=region,
                          n_photos=len(image_bytes))

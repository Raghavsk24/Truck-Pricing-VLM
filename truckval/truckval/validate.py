"""Guardrails for the one JSON contract a language model authors.

Everything else in the pipeline is internal, external, or user input. This is
the only boundary where a model writes the JSON, so it is the only one that
needs real hardening.

TWO LAYERS, because JSON Schema cannot express the invariants that matter most.

  Layer 1 — HARDENED_EVIDENCE_SCHEMA. Structure, closed vocabularies, ranges,
            array caps, no extra properties. Pass this to structured output so
            violations are impossible rather than caught.

  Layer 2 — validate(). Cross-field invariants, provenance, plausibility, and
            the semantic guardrail that matters more than all the structural
            ones combined: no pricing anywhere in the payload.

ONE DESIGN RULE: a guardrail failure degrades the FIELD, not the response.
A model that hallucinates one tread value should not cost you the odometer read
in the same payload. Bad fields are quarantined to NOT_VISIBLE and logged; the
rest survives. Only whole-payload violations (pricing leakage, wrong shape)
reject outright.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from .schema import (
    BodyType,
    CabType,
    DamageKind,
    RustKind,
    TreadState,
    WearLevel,
    WheelPosition,
)

# Vocabularies are derived from schema.py rather than restated, so the schema
# and the validator cannot drift apart.
TREAD = list(get_args(TreadState))
BODY = list(get_args(BodyType))
CAB = list(get_args(CabType))
RUST = list(get_args(RustKind))
WEAR = list(get_args(WearLevel))
DAMAGE = list(get_args(DamageKind))
WHEELS = list(get_args(WheelPosition))
SEVERITY = ["minor", "moderate", "severe"]
VISIBILITY = ["observed", "not_visible", "occluded"]

MAX_DAMAGE_INSTANCES = 24
MAX_TIRE_ENTRIES = 12
MAX_NOTE_CHARS = 200
MIN_MODEL_YEAR = 1970
MAX_MODEL_YEAR = 2027
MAX_ODOMETER_KM = 2_500_000
MAX_ENGINE_HOURS = 60_000


# --------------------------------------------------------------------------
# Layer 1 — the schema
# --------------------------------------------------------------------------


def _obs(value_schema: dict[str, Any]) -> dict[str, Any]:
    """One observation. `value` is typed per field, never a free-form blob."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["visibility", "confidence", "value", "photo_index"],
        "properties": {
            "value": {"oneOf": [value_schema, {"type": "null"}]},
            "visibility": {"type": "string", "enum": VISIBILITY},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "photo_index": {"type": ["integer", "null"], "minimum": 0},
            "note": {"type": ["string", "null"], "maxLength": MAX_NOTE_CHARS},
        },
    }


_STR = {"type": "string", "maxLength": 60}
_BOOL = {"type": "boolean"}


def _enum(vals: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": vals}


def _num(lo: float, hi: float) -> dict[str, Any]:
    return {"type": "number", "minimum": lo, "maximum": hi}


HARDENED_EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["identity", "configuration", "equipment", "condition",
                 "usable_exterior_angles"],
    "properties": {
        "identity": {
            "type": "object",
            "additionalProperties": False,
            "required": ["vin", "year", "make", "model", "trim"],
            "properties": {
                # Pattern excludes I, O and Q, which never appear in a VIN.
                "vin": _obs({"type": "string",
                             "pattern": "^[A-HJ-NPR-Z0-9]{17}$"}),
                "year": _obs({"type": "integer",
                              "minimum": MIN_MODEL_YEAR,
                              "maximum": MAX_MODEL_YEAR}),
                "make": _obs(_STR),
                "model": _obs(_STR),
                "trim": _obs(_STR),
            },
        },
        "configuration": {
            "type": "object",
            "additionalProperties": False,
            "required": ["body_type", "cab_type", "axle_count",
                         "box_length_ft", "gvwr_lb"],
            "properties": {
                "body_type": _obs(_enum(BODY)),
                "cab_type": _obs(_enum(CAB)),
                "axle_count": _obs({"type": "integer", "minimum": 2,
                                    "maximum": 6}),
                "box_length_ft": _obs(_num(8, 60)),
                "gvwr_lb": _obs({"type": "integer", "minimum": 6_000,
                                 "maximum": 120_000}),
            },
        },
        "equipment": {
            "type": "object",
            "additionalProperties": False,
            "required": ["liftgate", "reefer_unit", "crane", "fifth_wheel", "pto"],
            "properties": {k: _obs(_BOOL) for k in
                           ("liftgate", "reefer_unit", "crane",
                            "fifth_wheel", "pto")},
        },
        "condition": {
            "type": "object",
            "additionalProperties": False,
            "required": ["odometer_km", "engine_hours", "tires", "damage",
                         "rust", "seat_wear", "dash_condition",
                         "glass_cracked", "paint_fade"],
            "properties": {
                "odometer_km": _obs(_num(0, MAX_ODOMETER_KM)),
                "engine_hours": _obs(_num(0, MAX_ENGINE_HOURS)),
                "rust": _obs(_enum(RUST)),
                "seat_wear": _obs(_enum(WEAR)),
                "dash_condition": _obs(_enum(WEAR)),
                "glass_cracked": _obs(_BOOL),
                "paint_fade": _obs(_enum(WEAR)),
                "tires": {
                    "type": "array",
                    "maxItems": MAX_TIRE_ENTRIES,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["position", "tread"],
                        "properties": {
                            "position": _enum(WHEELS),
                            "tread": _obs(_enum(TREAD)),
                            "sidewall_damage": _obs(_BOOL),
                            "mismatched_brand": _obs(_BOOL),
                        },
                    },
                },
                "damage": {
                    "type": "array",
                    "maxItems": MAX_DAMAGE_INSTANCES,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["panel", "kind", "severity",
                                     "photo_index", "confidence"],
                        "properties": {
                            "panel": _STR,
                            "kind": _enum(DAMAGE),
                            "severity": _enum(SEVERITY),
                            "photo_index": {"type": "integer", "minimum": 0},
                            "confidence": _num(0, 1),
                        },
                    },
                },
            },
        },
        "usable_exterior_angles": {"type": "integer", "minimum": 0, "maximum": 24},
    },
}


# --------------------------------------------------------------------------
# Layer 2 — invariants JSON Schema cannot express
# --------------------------------------------------------------------------

Severity = Literal["quarantine", "reject"]

# Any key or string value hinting at money. The perception stage is forbidden
# from valuing anything, and a model that starts appraising has stopped doing
# the job the rest of the pipeline depends on it doing.
PRICE_KEY = re.compile(
    r"price|value|worth|cost|estimat|apprais|msrp|resale|market|usd|dollar|"
    r"valuation|book|retail|wholesale|nada|kbb",
    re.I,
)
PRICE_VALUE = re.compile(r"[$£€]\s?\d|\d[\d,]{2,}\s?(usd|dollars?|eur|gbp)\b", re.I)


@dataclass
class Issue:
    path: str
    problem: str
    severity: Severity = "quarantine"


@dataclass
class Result:
    payload: dict[str, Any]
    issues: list[Issue] = field(default_factory=list)

    @property
    def rejected(self) -> bool:
        return any(i.severity == "reject" for i in self.issues)

    @property
    def quarantined(self) -> list[str]:
        return [i.path for i in self.issues if i.severity == "quarantine"]


def _walk_keys(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append((f"{path}.{k}".lstrip("."), k))
            out += _walk_keys(v, f"{path}.{k}".lstrip("."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _walk_keys(v, f"{path}[{i}]")
    return out


def _walk_values(obj: Any, path: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _walk_values(v, f"{path}.{k}".lstrip("."))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _walk_values(v, f"{path}[{i}]")
    else:
        out.append((path, obj))
    return out


def _legal_keys(schema: dict[str, Any]) -> set[str]:
    """Every property name the contract defines, harvested from the schema.

    Derived rather than restated so it cannot drift. `additionalProperties:
    false` already defines this universe when the model honours the schema;
    this is for when it does not.
    """
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            props = node.get("properties")
            if isinstance(props, dict):
                out.update(props.keys())
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return out


LEGAL_KEYS = _legal_keys(HARDENED_EVIDENCE_SCHEMA)


def assert_no_pricing(payload: dict[str, Any]) -> list[Issue]:
    """The guardrail that matters more than every structural one combined.

    Only keys the contract does NOT define are scanned — `value` and
    `confidence` are structural, not appraisals. An undefined key that looks
    monetary rejects the whole payload: a perception stage that has started
    appraising is not doing its job, and anything else it said in the same
    pass is suspect. An undefined key that looks harmless is merely stripped.

    Free text is scanned separately, because a model told not to emit a price
    field will sometimes put the number in a note instead.
    """
    issues: list[Issue] = []
    for path, key in _walk_keys(payload):
        k = str(key)
        if k in LEGAL_KEYS:
            continue
        if PRICE_KEY.search(k):
            issues.append(Issue(path, f"pricing-adjacent key '{k}'", "reject"))
        else:
            issues.append(Issue(path, f"undefined key '{k}'", "quarantine"))
    for path, val in _walk_values(payload):
        if isinstance(val, str) and PRICE_VALUE.search(val):
            issues.append(Issue(path, "monetary amount in free text", "reject"))
    return issues


def _quarantine(obs: dict[str, Any]) -> None:
    obs["value"] = None
    obs["visibility"] = "not_visible"
    obs["confidence"] = 0.0
    obs["photo_index"] = None


def _check_observation(
    obs: Any, path: str, n_photos: int, issues: list[Issue]
) -> None:
    if not isinstance(obs, dict):
        issues.append(Issue(path, "not an observation object"))
        return

    vis = obs.get("visibility")
    val = obs.get("value")
    idx = obs.get("photo_index")

    # A value asserted without visibility is the exact hallucination the
    # NOT_VISIBLE design exists to prevent.
    if vis != "observed" and val is not None:
        issues.append(Issue(path, f"value present but visibility={vis}"))
        _quarantine(obs)
        return

    if vis == "observed":
        if val is None:
            issues.append(Issue(path, "visibility=observed with null value"))
            _quarantine(obs)
            return
        # Provenance is not optional. An observation with no photo behind it
        # cannot be cited in the report, so it cannot be used.
        if idx is None:
            issues.append(Issue(path, "observed with no photo_index"))
            _quarantine(obs)
            return
        if not isinstance(idx, int) or not 0 <= idx < n_photos:
            issues.append(Issue(path, f"photo_index {idx} out of range"))
            _quarantine(obs)
            return
        # High confidence on an occluded read is a contradiction the model
        # will produce under pressure to be helpful.
        conf = obs.get("confidence", 0.0)
        if not isinstance(conf, (int, float)) or not 0.0 <= conf <= 1.0:
            issues.append(Issue(path, f"confidence {conf} out of bounds"))
            _quarantine(obs)

    if vis == "occluded" and (obs.get("confidence") or 0) > 0.5:
        issues.append(Issue(path, "occluded but confidence > 0.5"))
        obs["confidence"] = 0.0


def validate(payload: dict[str, Any], n_photos: int) -> Result:
    """Run every invariant. Mutates a copy; returns it with issues attached."""
    import copy

    p = copy.deepcopy(payload)
    issues = assert_no_pricing(p)
    if any(i.severity == "reject" for i in issues):
        return Result(p, issues)

    for group in ("identity", "configuration", "equipment", "condition"):
        block = p.get(group)
        if not isinstance(block, dict):
            issues.append(Issue(group, "missing or malformed group", "reject"))
            return Result(p, issues)
        for name, obs in block.items():
            if name in ("tires", "damage"):
                continue
            _check_observation(obs, f"{group}.{name}", n_photos, issues)

    cond = p.get("condition", {})

    # Tires: one entry per wheel position, no duplicates.
    seen: set[str] = set()
    kept = []
    for i, t in enumerate(cond.get("tires", []) or []):
        pos = t.get("position")
        if pos in seen:
            issues.append(Issue(f"condition.tires[{i}]", f"duplicate position {pos}"))
            continue
        if pos not in WHEELS:
            issues.append(Issue(f"condition.tires[{i}]", f"unknown position {pos}"))
            continue
        seen.add(pos)
        for k in ("tread", "sidewall_damage", "mismatched_brand"):
            if k in t:
                _check_observation(t[k], f"condition.tires[{i}].{k}", n_photos, issues)
        kept.append(t)
    cond["tires"] = kept

    # Damage: every instance must cite a real photo, or it cannot be shown.
    kept_dmg = []
    for i, d in enumerate(cond.get("damage", []) or []):
        idx = d.get("photo_index")
        if not isinstance(idx, int) or not 0 <= idx < n_photos:
            issues.append(Issue(f"condition.damage[{i}]", f"photo_index {idx} invalid"))
            continue
        if d.get("kind") not in DAMAGE or d.get("severity") not in SEVERITY:
            issues.append(Issue(f"condition.damage[{i}]", "unknown kind or severity"))
            continue
        kept_dmg.append(d)
    cond["damage"] = kept_dmg

    # Plausibility. Cheap, and catches OCR that read 31100 as 311000.
    year = (p.get("identity", {}).get("year") or {}).get("value")
    odo = (cond.get("odometer_km") or {}).get("value")
    if isinstance(year, int) and isinstance(odo, (int, float)):
        age = max(1, MAX_MODEL_YEAR - year)
        if odo / age > 250_000:
            issues.append(Issue("condition.odometer_km",
                                f"{odo:,.0f} km over {age}y is implausible"))
            _quarantine(cond["odometer_km"])

    angles = p.get("usable_exterior_angles", 0)
    if isinstance(angles, int) and angles > n_photos:
        issues.append(Issue("usable_exterior_angles",
                            f"{angles} claimed, {n_photos} photos supplied"))
        p["usable_exterior_angles"] = n_photos

    return Result(p, issues)

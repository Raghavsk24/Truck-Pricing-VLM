"""Data contracts for the whole pipeline.

The single most important design decision lives here: `Observation.visibility`.
Every perceived field is OBSERVED, NOT_VISIBLE, OCCLUDED or CLAIMED. A model that is
allowed to say "I can't see it" stops inventing values, and the downstream
uncertainty pass has something concrete to widen the price range around.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------


class Visibility(str, Enum):
    OBSERVED = "observed"
    NOT_VISIBLE = "not_visible"  # no photo covers this
    OCCLUDED = "occluded"  # a photo covers it but mud/tarp/snow/glare blocks it
    CLAIMED = "claimed"  # the seller typed it; no photo corroborates it


class BBox(BaseModel):
    x: float
    y: float
    w: float
    h: float


class Observation(BaseModel, Generic[T]):
    """One perceived field, always carrying its own provenance.

    `value` is None whenever visibility != OBSERVED. Never fill it with a guess:
    the guess belongs in priors.py, where it is sampled rather than asserted.
    """

    value: Optional[T] = None
    visibility: Visibility = Visibility.NOT_VISIBLE
    confidence: float = 0.0  # perception model's own confidence, 0..1
    photo_index: Optional[int] = None
    bbox: Optional[BBox] = None
    note: Optional[str] = None

    @property
    def hard(self) -> bool:
        """Corroborated by a photo. The only kind of evidence pricing pins."""
        return self.visibility == Visibility.OBSERVED and self.value is not None

    @property
    def known(self) -> bool:
        """Has a value, from a photo OR from the seller. Report-facing."""
        return (
            self.visibility in (Visibility.OBSERVED, Visibility.CLAIMED)
            and self.value is not None
        )

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


def unknown(reason: Visibility = Visibility.NOT_VISIBLE) -> Observation[Any]:
    return Observation(visibility=reason)


# --------------------------------------------------------------------------
# Perceived vocabulary
# --------------------------------------------------------------------------

TreadState = Literal["new", "good", "worn", "legal_minimum", "below_minimum"]
BodyType = Literal[
    "box_truck", "flatbed", "dump", "tractor", "stake", "reefer",
    "service", "tanker", "chassis_cab", "pickup", "other",
]
CabType = Literal["day_cab", "sleeper", "crew", "extended", "standard"]
RustKind = Literal["none", "surface", "scaling", "perforation"]
WearLevel = Literal["light", "moderate", "heavy", "destroyed"]
DamageKind = Literal[
    "dent", "scratch", "crack", "glass_shatter", "lamp_broken",
    "tire_flat", "panel_missing", "deformation",
]
WheelPosition = Literal[
    "steer_left", "steer_right",
    "drive_left_outer", "drive_left_inner",
    "drive_right_outer", "drive_right_inner",
    "tag_left", "tag_right",
]


class DamageInstance(BaseModel):
    panel: str
    kind: DamageKind
    severity: Literal["minor", "moderate", "severe"]
    photo_index: int
    bbox: Optional[BBox] = None
    confidence: float = 0.0


class TireObservation(BaseModel):
    position: WheelPosition
    tread: Observation[TreadState] = Field(default_factory=unknown)
    sidewall_damage: Observation[bool] = Field(default_factory=unknown)
    mismatched_brand: Observation[bool] = Field(default_factory=unknown)


# --------------------------------------------------------------------------
# Evidence bundle — the only thing the pricing stage is allowed to read
# --------------------------------------------------------------------------


class Identity(BaseModel):
    vin: Observation[str] = Field(default_factory=unknown)
    year: Observation[int] = Field(default_factory=unknown)
    make: Observation[str] = Field(default_factory=unknown)
    model: Observation[str] = Field(default_factory=unknown)
    trim: Observation[str] = Field(default_factory=unknown)

    @property
    def resolved(self) -> bool:
        """Hard gate for any price at all. No identity, no comp set.

        A seller-typed year/make/model counts here — it is enough to pick a
        comp set — but it is still sampled rather than pinned downstream.
        """
        return self.year.known and self.make.known and self.model.known


class Configuration(BaseModel):
    body_type: Observation[BodyType] = Field(default_factory=unknown)
    cab_type: Observation[CabType] = Field(default_factory=unknown)
    axle_count: Observation[int] = Field(default_factory=unknown)
    box_length_ft: Observation[float] = Field(default_factory=unknown)
    gvwr_lb: Observation[int] = Field(default_factory=unknown)


class Equipment(BaseModel):
    liftgate: Observation[bool] = Field(default_factory=unknown)
    reefer_unit: Observation[bool] = Field(default_factory=unknown)
    crane: Observation[bool] = Field(default_factory=unknown)
    fifth_wheel: Observation[bool] = Field(default_factory=unknown)
    pto: Observation[bool] = Field(default_factory=unknown)


class Condition(BaseModel):
    odometer_km: Observation[float] = Field(default_factory=unknown)
    engine_hours: Observation[float] = Field(default_factory=unknown)
    tires: list[TireObservation] = Field(default_factory=list)
    damage: list[DamageInstance] = Field(default_factory=list)
    rust: Observation[RustKind] = Field(default_factory=unknown)
    seat_wear: Observation[WearLevel] = Field(default_factory=unknown)
    dash_condition: Observation[WearLevel] = Field(default_factory=unknown)
    glass_cracked: Observation[bool] = Field(default_factory=unknown)
    paint_fade: Observation[WearLevel] = Field(default_factory=unknown)


class Evidence(BaseModel):
    """Everything perception extracted. Pricing reads this and nothing else."""

    identity: Identity = Field(default_factory=Identity)
    configuration: Configuration = Field(default_factory=Configuration)
    equipment: Equipment = Field(default_factory=Equipment)
    condition: Condition = Field(default_factory=Condition)
    usable_exterior_angles: int = 0
    region: Optional[str] = None

    def unknown_fields(self) -> list[str]:
        """Dotted paths of every field the uncertainty pass must sample."""
        out: list[str] = []
        for group in ("identity", "configuration", "equipment", "condition"):
            obj = getattr(self, group)
            for name, _ in type(obj).model_fields.items():
                val = getattr(obj, name)
                if isinstance(val, Observation) and not val.hard:
                    out.append(f"{group}.{name}")
        if not self.condition.tires:
            out.append("condition.tires")
        return out

    def claimed_values(self) -> dict[str, Any]:
        """Typed-but-uncorroborated fields, for claim-anchored sampling."""
        out: dict[str, Any] = {}
        for group in ("identity", "configuration", "equipment", "condition"):
            obj = getattr(self, group)
            for name in type(obj).model_fields:
                val = getattr(obj, name)
                if (isinstance(val, Observation)
                        and val.visibility == Visibility.CLAIMED
                        and val.value is not None):
                    out[f"{group}.{name}"] = val.value
        return out

    def get(self, path: str) -> Any:
        group, name = path.split(".", 1)
        obs = getattr(getattr(self, group), name, None)
        return obs.value if isinstance(obs, Observation) else obs


# --------------------------------------------------------------------------
# Seller claims (typed input) — treated as claims, never as facts
# --------------------------------------------------------------------------


class SellerClaims(BaseModel):
    year: Optional[int] = None
    make: Optional[str] = None
    model: Optional[str] = None
    odometer_km: Optional[float] = None
    box_length_ft: Optional[float] = None
    region: Optional[str] = None
    asking_price: Optional[float] = None


class Contradiction(BaseModel):
    field: str
    claimed: Any
    evidence_suggests: Any
    basis: str
    severity: Literal["note", "warn", "major"]


# --------------------------------------------------------------------------
# Gate
# --------------------------------------------------------------------------


class GateReason(str, Enum):
    WRONG_OBJECT = "wrong_object"
    OBJECT_UNCLEAR = "object_unclear"
    FRAME_COVERAGE_LOW = "frame_coverage_low"
    UNUSABLE_CAPTURE = "unusable_capture"
    MULTIPLE_VEHICLES = "multiple_vehicles"
    NO_IDENTITY = "no_identity"
    TOO_FEW_ANGLES = "too_few_angles"
    DUPLICATE_LISTING_PHOTOS = "duplicate_listing_photos"
    INTERVAL_TOO_WIDE = "interval_too_wide"


class Tier(int, Enum):
    NOTHING = 0  # no condition report, no price
    CONDITION_ONLY = 1  # condition report, no price
    PRICED = 2  # price with range, possibly flagged


class GateResult(BaseModel):
    tier: Tier
    reasons: list[GateReason] = Field(default_factory=list)
    detected_object: Optional[str] = None
    detail: Optional[str] = None


# --------------------------------------------------------------------------
# Pricing output
# --------------------------------------------------------------------------


class LedgerLine(BaseModel):
    label: str
    amount: float  # negative = deduction
    amount_sd: float = 0.0  # uncertainty on this line
    evidence_ref: Optional[str] = None  # "photo 3, bbox 0.4/0.2/0.1/0.1"
    basis: Optional[str] = None


class PriceRange(BaseModel):
    low: float  # p10
    mid: float  # p50
    high: float  # p90
    n_samples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    @property
    def relative_width(self) -> float:
        return self.width / self.mid if self.mid else float("inf")


class VarianceContribution(BaseModel):
    field: str
    dollars: float  # how much interval width this unknown accounts for
    ask: str  # what photo would resolve it


class Valuation(BaseModel):
    tier: Tier
    gate: GateResult
    evidence: Evidence
    price: Optional[PriceRange] = None
    baseline: Optional[float] = None
    ledger: list[LedgerLine] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    next_photos: list[VarianceContribution] = Field(default_factory=list)
    message: str = ""
    condition_report: str = ""
    reasoning: str = ""

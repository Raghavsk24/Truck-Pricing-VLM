"""Stage 0. Decide whether we are looking at a truck at all.

Three independent checks, all cheap, all before any pricing logic runs. Two of
them need no VLM: capture quality is signal processing, and the object class
question is answered by any COCO-trained detector, which already ships `truck`,
`car`, `motorcycle`, `bus` and `bicycle` as native classes.

Everything here returns reason codes rather than prose. The prose lives in
report.py so that a refusal and a price go through the same rendering path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence

import numpy as np

from .schema import GateReason, GateResult, Tier

# Objects that get an immediate Tier 0 with a named class in the message.
NAMEABLE_NON_TRUCKS = {
    "motorcycle", "car", "bicycle", "bus", "boat", "airplane",
    "train", "person", "dog", "trailer_only",
}

MIN_FRAME_COVERAGE = 0.15
MIN_BLUR_VARIANCE = 60.0  # Laplacian variance; tune on your own capture data
MIN_MEAN_LUMA = 28.0
MAX_MEAN_LUMA = 232.0
MIN_USABLE_ANGLES = 2


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass
class PhotoInput:
    index: int
    image: np.ndarray  # HxWx3 uint8
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = hashlib.sha256(self.image.tobytes()).hexdigest()


@dataclass
class DetectorResult:
    """What an object detector returns for one photo."""

    top_class: str
    confidence: float
    frame_coverage: float  # fraction of frame occupied by the top box
    all_classes: dict[str, float] = field(default_factory=dict)


class Detector(Protocol):
    """Swap in YOLO / DETR / Grounding DINO. See scripts/demo.py for a stub."""

    def __call__(self, photo: PhotoInput) -> DetectorResult: ...


# --------------------------------------------------------------------------
# Capture quality — pure signal processing, no model needed
# --------------------------------------------------------------------------


def laplacian_variance(gray: np.ndarray) -> float:
    k = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    h, w = gray.shape
    if h < 3 or w < 3:
        return 0.0
    win = np.lib.stride_tricks.sliding_window_view(gray, (3, 3))
    return float((win * k).sum(axis=(-2, -1)).var())


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float64)
    return img[..., :3].astype(np.float64) @ np.array([0.299, 0.587, 0.114])


def capture_usable(photo: PhotoInput) -> tuple[bool, str]:
    gray = to_gray(photo.image)
    luma = float(gray.mean())
    if luma < MIN_MEAN_LUMA:
        return False, "too dark"
    if luma > MAX_MEAN_LUMA:
        return False, "blown out"
    lv = laplacian_variance(gray)
    if lv < MIN_BLUR_VARIANCE:
        return False, "out of focus"
    return True, "ok"


# --------------------------------------------------------------------------
# Cross-photo consistency — are all these photos the same vehicle?
# --------------------------------------------------------------------------


def dominant_hue_signature(img: np.ndarray, bins: int = 12) -> np.ndarray:
    """Coarse colour histogram. Cheap proxy for 'same paint'."""
    arr = img[..., :3].astype(np.float64) / 255.0
    mx = arr.max(axis=-1)
    mn = arr.min(axis=-1)
    delta = mx - mn
    hue = np.zeros_like(mx)
    mask = delta > 1e-6
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    idx = mask & (mx == r)
    hue[idx] = ((g - b)[idx] / delta[idx]) % 6
    idx = mask & (mx == g)
    hue[idx] = ((b - r)[idx] / delta[idx]) + 2
    idx = mask & (mx == b)
    hue[idx] = ((r - g)[idx] / delta[idx]) + 4
    hue = hue / 6.0
    sat_mask = delta > 0.15
    if sat_mask.sum() < 64:
        return np.ones(bins) / bins
    hist, _ = np.histogram(hue[sat_mask], bins=bins, range=(0.0, 1.0))
    total = hist.sum()
    return hist / total if total else np.ones(bins) / bins


def chi2(a: np.ndarray, b: np.ndarray) -> float:
    denom = a + b
    denom[denom == 0] = 1.0
    return float(0.5 * (((a - b) ** 2) / denom).sum())


def same_vehicle(
    photos: Sequence[PhotoInput],
    plates: Optional[Sequence[Optional[str]]] = None,
    threshold: float = 0.45,
) -> tuple[bool, str]:
    """A hard disagreement on a legible plate outranks any colour evidence."""
    if plates:
        seen = {p.strip().upper() for p in plates if p and len(p.strip()) >= 5}
        if len(seen) > 1:
            return False, f"two distinct plates read: {', '.join(sorted(seen))}"
    if len(photos) < 2:
        return True, "single photo"
    sigs = [dominant_hue_signature(p.image) for p in photos]
    worst = max(chi2(sigs[0], s) for s in sigs[1:])
    if worst > threshold:
        return False, "paint colour differs across photos"
    return True, "consistent"


# --------------------------------------------------------------------------
# Duplicate / stolen photo check
# --------------------------------------------------------------------------


def phash(img: np.ndarray, size: int = 8) -> int:
    """Average hash. Replace with a real pHash (DCT) before production."""
    gray = to_gray(img)
    h, w = gray.shape
    ys = np.linspace(0, h - 1, size).astype(int)
    xs = np.linspace(0, w - 1, size).astype(int)
    small = gray[np.ix_(ys, xs)]
    bits = (small > small.mean()).flatten()
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def matches_known_listing(
    photos: Sequence[PhotoInput],
    corpus_hashes: Sequence[int],
    max_distance: int = 6,
) -> bool:
    for p in photos:
        h = phash(p.image)
        if any(hamming(h, c) <= max_distance for c in corpus_hashes):
            return True
    return False


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


def run_gate(
    photos: Sequence[PhotoInput],
    detector: Detector,
    plates: Optional[Sequence[Optional[str]]] = None,
    corpus_hashes: Sequence[int] = (),
) -> GateResult:
    if not photos:
        return GateResult(
            tier=Tier.NOTHING,
            reasons=[GateReason.OBJECT_UNCLEAR],
            detail="no photos supplied",
        )

    reasons: list[GateReason] = []

    usable = [p for p in photos if capture_usable(p)[0]]
    if not usable:
        _, why = capture_usable(photos[0])
        return GateResult(
            tier=Tier.NOTHING,
            reasons=[GateReason.UNUSABLE_CAPTURE],
            detail=why,
        )

    dets = [detector(p) for p in usable]
    truck_dets = [d for d in dets if d.top_class == "truck"]

    if not truck_dets:
        best = max(dets, key=lambda d: d.confidence)
        if best.top_class in NAMEABLE_NON_TRUCKS and best.confidence > 0.7:
            return GateResult(
                tier=Tier.NOTHING,
                reasons=[GateReason.WRONG_OBJECT],
                detected_object=best.top_class,
            )
        return GateResult(
            tier=Tier.NOTHING,
            reasons=[GateReason.OBJECT_UNCLEAR],
            detected_object=best.top_class,
        )

    if max(d.frame_coverage for d in truck_dets) < MIN_FRAME_COVERAGE:
        return GateResult(
            tier=Tier.NOTHING,
            reasons=[GateReason.FRAME_COVERAGE_LOW],
            detected_object="truck",
            detail="truck occupies too little of the frame",
        )

    ok, why = same_vehicle(usable, plates)
    if not ok:
        return GateResult(
            tier=Tier.NOTHING,
            reasons=[GateReason.MULTIPLE_VEHICLES],
            detected_object="truck",
            detail=why,
        )

    if corpus_hashes and matches_known_listing(usable, corpus_hashes):
        reasons.append(GateReason.DUPLICATE_LISTING_PHOTOS)

    if len(truck_dets) < MIN_USABLE_ANGLES:
        reasons.append(GateReason.TOO_FEW_ANGLES)

    tier = Tier.CONDITION_ONLY if reasons else Tier.PRICED
    return GateResult(tier=tier, reasons=reasons, detected_object="truck")

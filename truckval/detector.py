"""YOLO-backed detector for the gate. Replaces the stub in scripts/demo.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

from truckval.gate import DetectorResult, PhotoInput

RELEVANT = {
    "truck", "car", "motorcycle", "bus", "bicycle",
    "boat", "train", "airplane", "person", "dog",
}

TRAILER_HINT_RATIO = 3.4
DEFAULT_WEIGHTS = "yolov8n.pt"


@dataclass
class YoloDetector:
    weights: str = DEFAULT_WEIGHTS
    conf: float = 0.25
    imgsz: int = 640
    device: Optional[str] = None
    trailer_check: bool = True

    @cached_property
    def _model(self):
        from ultralytics import YOLO
        return YOLO(self.weights)

    def __call__(self, photo: PhotoInput) -> DetectorResult:
        res = self._model.predict(
            photo.image, conf=self.conf, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )[0]

        h, w = photo.image.shape[:2]
        frame_area = float(h * w) or 1.0
        names = res.names

        best = {}
        truck_boxes = []

        for box in res.boxes:
            label = names[int(box.cls)]
            score = float(box.conf)
            key = label if label in RELEVANT else "other"
            best[key] = max(best.get(key, 0.0), score)
            if label == "truck":
                truck_boxes.append(tuple(float(v) for v in box.xyxy[0]))

        if not best:
            return DetectorResult(top_class="other", confidence=0.0,
                                  frame_coverage=0.0, all_classes={})

        top = max(best, key=lambda k: best[k])
        coverage = _union_coverage(truck_boxes, frame_area) if truck_boxes else 0.0

        if top == "truck" and self.trailer_check and _looks_like_trailer(truck_boxes):
            return DetectorResult(top_class="trailer_only",
                                  confidence=best["truck"],
                                  frame_coverage=coverage, all_classes=best)

        if top != "truck":
            coverage = _largest_box_coverage(res, top, frame_area)

        return DetectorResult(top_class=top, confidence=best[top],
                              frame_coverage=coverage, all_classes=best)


def _union_coverage(boxes, frame_area):
    """Union area of truck boxes. A close-up truck is often two detections
    (cab, box); the largest single box would understate coverage and trip
    FRAME_COVERAGE_LOW on a perfectly good photo."""
    if not boxes:
        return 0.0
    if len(boxes) == 1:
        x0, y0, x1, y1 = boxes[0]
        return float(max(0.0, x1 - x0) * max(0.0, y1 - y0) / frame_area)
    xs = sorted({v for b in boxes for v in (b[0], b[2])})
    ys = sorted({v for b in boxes for v in (b[1], b[3])})
    area = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes):
                area += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    return float(area / frame_area)


def _largest_box_coverage(res, label, frame_area):
    names = res.names
    areas = [
        float((b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]))
        for b in res.boxes if names[int(b.cls)] == label
    ]
    return max(areas) / frame_area if areas else 0.0


def _looks_like_trailer(boxes):
    """COCO has no trailer class, so unhitched trailers come back as `truck`.
    Top near-domain false accept in this product."""
    if len(boxes) != 1:
        return False
    x0, y0, x1, y1 = boxes[0]
    w, h = x1 - x0, y1 - y0
    return h > 0 and (w / h) > TRAILER_HINT_RATIO


@dataclass
class CachedDetector:
    inner: object
    _cache: dict = field(default_factory=dict)

    def __call__(self, photo: PhotoInput) -> DetectorResult:
        if photo.sha256 not in self._cache:
            self._cache[photo.sha256] = self.inner(photo)
        return self._cache[photo.sha256]

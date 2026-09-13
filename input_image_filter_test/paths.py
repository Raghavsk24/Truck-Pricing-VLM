"""Shared paths and data shapes for the Class 7/8 input-image filter eval."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent
FILTER_DIR = Path(__file__).resolve().parent
DATA_DIR = FILTER_DIR / "data"

DATASET_ROOT = (
    REPO_ROOT
    / "image_scraper"
    / "truckpaper_scraped_images_dataset"
)
POSITIVE_DIR = DATASET_ROOT / "positive (class 7 - 8)"
NEGATIVE_DIR = DATASET_ROOT / "negative (class 2 - 6)"

INSTRUCTIONS_PATH = (
    REPO_ROOT
    / "vlm_instructions"
    / "filter_invalid_input_images_instructions.json"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

# Claude Sonnet standard vision long-edge cap. Staging downscales to this
# so attachments stay small and the model is not given unused pixels.
STAGE_MAX_LONG_EDGE = 1568
STAGE_JPEG_QUALITY = 85

DEFAULT_SEED = 42
N_POSITIVE = 25
N_NEGATIVE = 25
ACCURACY_THRESHOLD = 0.90

EVAL_SAMPLE_PATH = DATA_DIR / "eval_sample.json"
BLIND_MANIFEST_PATH = DATA_DIR / "blind_manifest.json"
PREDICTIONS_PATH = DATA_DIR / "predictions.json"
METRICS_PATH = DATA_DIR / "metrics.json"


class SampleItem(TypedDict):
    id: str
    path: str
    label: str  # "positive" | "negative"
    is_valid_class_7_8: bool
    category: str
    listing_id: str


class BlindItem(TypedDict):
    id: str
    path: str


VALID_TRUCK_TYPES = ("day-cab-truck", "dump-truck", "sleeper-truck")
TRUCK_TYPES = (*VALID_TRUCK_TYPES, "none")

CATEGORY_TO_TRUCK_TYPE = {
    "day-cab-trucks": "day-cab-truck",
    "dump-trucks": "dump-truck",
    "sleeper-trucks": "sleeper-truck",
}


class PredictionOutput(TypedDict):
    truck_type: str
    user_message: str


class PredictionPayload(TypedDict):
    reasoning: str
    output: PredictionOutput

"""Predict a truck price RANGE from master-schema VLM output (or an image).

Features used (from the image via Cursor Sonnet VLM):
  - truck_type: day_cab | sleeper | dump
  - brand
  - condition (total_penalty_percent), if the fitted winner uses it

Range is center ± error_bound, where error_bound is the model's measured
prediction error (LOO APE percentile), not a wide brand-spread band.

Usage:
    python pricing/predict.py --image path/to/truck.jpg
    python pricing/predict.py --vlm-json path/to/vlm_output.json
    python pricing/predict.py --truck-type day_cab --brand FREIGHTLINER --penalty 4.0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL_PATH = ROOT / "price_range_model.json"
SCHEMA_PATH = REPO / "vlm_instructions" / "truck_feature_extraction_master_instructions.json"
STAGED_DIR = ROOT / "staged_images"
LONG_EDGE = 1568
DEFAULT_MODEL = os.environ.get("CURSOR_MODEL", "claude-sonnet-5-thinking-high")

sys.path.insert(0, str(ROOT))
from label_sample import (  # noqa: E402
    extract_fields,
    is_priceable,
    resize_long_edge,
)
from fit_price_range import brand_cell, normalize_truck_type  # noqa: E402
from vlm_client import call_vlm_cursor  # noqa: E402


def load_model(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Model not found: {path}. Run fit_price_range.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def lookup_mu(tables: dict, truck_type: str, brand_cell_name: str, use_brand: bool) -> float:
    if use_brand:
        key = f"{truck_type}|{brand_cell_name}"
        cell = tables.get("cells", {}).get(key)
        if cell is not None:
            return cell["mu"]
    t = tables.get("types", {}).get(truck_type)
    if t is not None:
        return t["mu"]
    return tables["global_mu"]


def predict_from_features(
    model: dict,
    truck_type: str,
    brand: str | None,
    penalty: float | None,
) -> dict:
    winner = model["winner"]
    tables = model["tables"]
    beta_info = model["beta"]
    error_bound = winner["error_bound"]

    t = normalize_truck_type(truck_type)
    bcell = brand_cell(brand) if winner.get("use_brand") and brand else model.get("other_brand", "OTHER")

    mu = lookup_mu(tables, t, bcell, use_brand=bool(winner.get("use_brand")))
    if winner.get("use_condition") and penalty is not None:
        center = math.exp(mu + beta_info["beta"] * (penalty - beta_info["penalty_bar"]))
    else:
        center = math.exp(mu)

    low = max(0.0, center * (1.0 - error_bound))
    high = center * (1.0 + error_bound)
    return {
        "truck_type": t,
        "brand_input": brand,
        "brand_cell": bcell if winner.get("use_brand") else None,
        "total_penalty_percent": penalty,
        "center": round(center, 2),
        "low": round(low, 2),
        "high": round(high, 2),
        "range": [round(low, 2), round(high, 2)],
        "error_bound": error_bound,
        "error_bound_pct": round(100 * error_bound, 1),
        "variant": winner["id"],
        "use_brand": bool(winner.get("use_brand")),
        "use_condition": bool(winner.get("use_condition")),
        "beta": beta_info.get("beta"),
    }


def vlm_from_image(image: Path, schema_path: Path, model_name: str) -> dict:
    """Optional API path. Prefer --vlm-json from an in-chat agent labeling pass."""
    if not schema_path.exists():
        raise SystemExit(f"Master schema not found: {schema_path}")
    staged = STAGED_DIR / f"predict_{image.stem}.jpg"
    STAGED_DIR.mkdir(parents=True, exist_ok=True)
    resize_long_edge(image, staged, long_edge=LONG_EDGE)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    try:
        return call_vlm_cursor(staged, schema, model=model_name, cwd=REPO)
    except SystemExit as exc:
        raise SystemExit(
            f"{exc}\n\n"
            "Or label the image in Cursor chat with the master schema and pass:\n"
            "  python pricing/predict.py --vlm-json path/to/prediction.json"
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="fitted model JSON")
    parser.add_argument("--image", type=Path, help="truck image to price")
    parser.add_argument("--vlm-json", type=Path, help="precomputed master-schema VLM output")
    parser.add_argument("--truck-type", type=str, help="day_cab | sleeper | dump")
    parser.add_argument("--brand", type=str, help="override / direct brand")
    parser.add_argument("--penalty", type=float, help="override / direct total_penalty_percent")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_MODEL,
        help="Cursor Sonnet model id for --image",
    )
    args = parser.parse_args()

    model = load_model(args.model)
    fields = None

    if args.truck_type is not None or args.brand is not None or args.penalty is not None:
        if not args.truck_type:
            raise SystemExit("Provide --truck-type (day_cab | sleeper | dump)")
        truck_type = args.truck_type
        brand = args.brand
        penalty = args.penalty
        if model["winner"].get("use_brand") and not brand:
            raise SystemExit("This model uses brand; provide --brand")
        if model["winner"].get("use_condition") and penalty is None:
            raise SystemExit("This model uses condition; provide --penalty")
    elif args.vlm_json or args.image:
        if args.vlm_json:
            raw = json.loads(args.vlm_json.read_text(encoding="utf-8"))
        else:
            raw = vlm_from_image(args.image, args.schema, args.vlm_model)
        fields = extract_fields(raw)
        if not is_priceable(fields):
            result = {
                "status": "rejected",
                "user_message": (
                    fields.get("validity_user_message")
                    or fields.get("primary_user_message")
                    or "Image is not a priceable Class 7/8 front or side view."
                ),
                "fields": fields,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            raise SystemExit(2)

        truck_type = fields.get("truck_type")
        if not truck_type:
            result = {
                "status": "needs_truck_type",
                "user_message": (
                    "Could not tell if this is a day cab, sleeper, or dump truck. "
                    "Re-run with --truck-type day_cab|sleeper|dump."
                ),
                "fields": fields,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            raise SystemExit(4)

        brand = fields.get("vlm_brand")
        if model["winner"].get("use_brand") and (
            not brand or fields.get("needs_user_input")
        ):
            result = {
                "status": "needs_brand",
                "user_message": (
                    "Could not read a brand from the image. "
                    "Provide --brand (and --penalty if needed)."
                ),
                "fields": fields,
                "truck_type": truck_type,
                "partial_penalty": fields.get("total_penalty_percent"),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            raise SystemExit(3)

        penalty = fields.get("total_penalty_percent")
        if model["winner"].get("use_condition") and penalty is None:
            raise SystemExit("VLM did not return total_penalty_percent")
    else:
        raise SystemExit(
            "Provide --image, --vlm-json, or --truck-type/--brand/--penalty"
        )

    pred = predict_from_features(model, truck_type, brand, penalty)
    result = {"status": "ok", **pred}
    if fields is not None:
        result["vlm_fields"] = {
            "primary_subject": fields.get("primary_subject"),
            "vehicle_cue": fields.get("vehicle_cue"),
            "truck_type": fields.get("truck_type"),
            "overall_condition_label": fields.get("overall_condition_label"),
            "overall_score": fields.get("overall_score"),
            "brand_confidence": fields.get("brand_confidence"),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

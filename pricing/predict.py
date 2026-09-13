"""Predict a truck price RANGE from master-schema VLM output (or an image).

Image-only product path:
  1. Stage image to JPEG with long edge 1568px
  2. Call VLM with master schema
  3. Gate on Class 7/8 + primary_subject in {front, side}
  4. Look up brand cell + apply fitted (beta, z) -> [low, high]

You can also pass a precomputed VLM JSON with --vlm-json (no API call).

Usage:
    python pricing/predict.py --image path/to/truck.jpg
    python pricing/predict.py --vlm-json path/to/vlm_output.json
    python pricing/predict.py --brand FREIGHTLINER --penalty 4.0
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
SCHEMA_PATH = REPO / "vlm instructions" / "truck_feature_extraction_master_instructions.json"
STAGED_DIR = ROOT / "staged_images"
LONG_EDGE = 1568
PRICEABLE_SUBJECTS = frozenset({"front", "side"})
DEFAULT_MODEL = "claude-sonnet-4-20250514"

# Reuse helpers from sibling modules
sys.path.insert(0, str(ROOT))
from label_sample import (  # noqa: E402
    call_vlm,
    extract_fields,
    is_priceable,
    resize_long_edge,
)
from fit_price_range import brand_cell  # noqa: E402


def load_model(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Model not found: {path}. Run fit_price_range.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def predict_from_features(
    model: dict,
    brand: str | None,
    penalty: float | None,
) -> dict:
    winner = model["winner"]
    bands = model["bands"]
    beta_info = model["beta"]
    z = winner["z"]

    if winner.get("use_brand", True) and brand:
        cell_name = brand_cell(brand)
        cell = bands["cells"].get(cell_name)
        if cell is None:
            mu, sd, cell_n = bands["global_mu"], bands["global_sd"], 0
            cell_name = model.get("other_cell", "OTHER")
        else:
            mu, sd, cell_n = cell["mu"], cell["sd"], cell["n"]
    else:
        cell_name = "GLOBAL"
        mu, sd, cell_n = bands["global_mu"], bands["global_sd"], bands["population_n"]

    if winner.get("use_condition") and penalty is not None:
        center_log = mu + beta_info["beta"] * (penalty - beta_info["penalty_bar"])
    else:
        center_log = mu

    center = math.exp(center_log)
    low = center * math.exp(-z * sd)
    high = center * math.exp(z * sd)
    return {
        "brand_input": brand,
        "brand_cell": cell_name,
        "total_penalty_percent": penalty,
        "center": round(center, 2),
        "low": round(low, 2),
        "high": round(high, 2),
        "range": [round(low, 2), round(high, 2)],
        "z": z,
        "variant": winner["id"],
        "use_condition": bool(winner.get("use_condition")),
        "cell_n": cell_n,
        "beta": beta_info["beta"],
    }


def vlm_from_image(image: Path, schema_path: Path, model_name: str) -> dict:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY to predict from an image")
    if not schema_path.exists():
        raise SystemExit(f"Master schema not found: {schema_path}")
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("Install anthropic: pip install anthropic") from exc

    staged = STAGED_DIR / f"predict_{image.stem}.jpg"
    resize_long_edge(image, staged, long_edge=LONG_EDGE)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    client = anthropic.Anthropic(api_key=api_key)
    return call_vlm(client, model_name, schema, staged)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="fitted model JSON")
    parser.add_argument("--image", type=Path, help="truck image to price")
    parser.add_argument("--vlm-json", type=Path, help="precomputed master-schema VLM output")
    parser.add_argument("--brand", type=str, help="override / direct brand (skip VLM)")
    parser.add_argument("--penalty", type=float, help="override / direct total_penalty_percent")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument(
        "--vlm-model",
        default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
    )
    args = parser.parse_args()

    model = load_model(args.model)
    reject_message = None
    fields = None

    if args.brand is not None or args.penalty is not None:
        brand = args.brand
        penalty = args.penalty
        if brand is None or penalty is None:
            # Allow brand-only when winner does not use condition
            if brand is None:
                raise SystemExit("Provide --brand (and --penalty if the model uses condition)")
            if model["winner"].get("use_condition") and penalty is None:
                raise SystemExit("This model uses condition; provide --penalty")
    elif args.vlm_json or args.image:
        if args.vlm_json:
            raw = json.loads(args.vlm_json.read_text(encoding="utf-8"))
        else:
            raw = vlm_from_image(args.image, args.schema, args.vlm_model)
        fields = extract_fields(raw)
        if not is_priceable(fields):
            reject_message = (
                fields.get("validity_user_message")
                or fields.get("primary_user_message")
                or "Image is not a priceable Class 7/8 front or side view."
            )
            result = {
                "status": "rejected",
                "user_message": reject_message,
                "fields": fields,
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            raise SystemExit(2)

        brand = fields.get("vlm_brand")
        if not brand or fields.get("needs_user_input"):
            result = {
                "status": "needs_brand",
                "user_message": (
                    "Could not read a brand from the image. "
                    "Provide the brand and re-run with --brand and --penalty."
                ),
                "fields": fields,
                "partial_penalty": fields.get("total_penalty_percent"),
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            raise SystemExit(3)

        penalty = fields.get("total_penalty_percent")
        if model["winner"].get("use_condition") and penalty is None:
            raise SystemExit("VLM did not return total_penalty_percent")
    else:
        raise SystemExit("Provide --image, --vlm-json, or --brand/--penalty")

    pred = predict_from_features(model, brand, penalty)
    result = {"status": "ok", **pred}
    if fields is not None:
        result["vlm_fields"] = {
            "primary_subject": fields.get("primary_subject"),
            "overall_condition_label": fields.get("overall_condition_label"),
            "overall_score": fields.get("overall_score"),
            "brand_confidence": fields.get("brand_confidence"),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

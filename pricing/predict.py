"""Predict a truck price RANGE from image-derived features.

Features used: truck type, brand, era (age bucket), condition score — all of
which the master VLM schema returns for a single photo.

Uses the weights saved in price_range_model.json by fit_price_range.py.

Usage:
    python pricing/predict.py --truck-type dump --brand MACK --era 2021_plus --score 4
    python pricing/predict.py --vlm-json path/to/prediction.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MODEL_PATH = ROOT / "price_range_model.json"
SCHEMA_PATH = REPO / "vlm_instructions" / "truck_feature_extraction_master_instructions.json"

sys.path.insert(0, str(ROOT))
from fit_price_range import (  # noqa: E402
    UNKNOWN_BRAND,
    UNKNOWN_ERA,
    canonical_brand,
    era_posterior,
    normalize_era,
    normalize_truck_type,
)
from label_sample import extract_fields, is_priceable  # noqa: E402


def load_model(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Model not found: {path}. Run fit_price_range.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _mu_from_tables(model: dict, truck_type: str, brand: str, era: str) -> float:
    """Walk the saved hierarchy: (type,brand,era) -> (type,brand) -> type -> global."""
    cells = model["cells"]
    if era != UNKNOWN_ERA:
        hit = cells["type_brand_era"].get(f"{truck_type}|{brand}|{era}")
        if hit:
            return float(hit["mu"])
    hit = cells["type_brand"].get(f"{truck_type}|{brand}")
    if hit:
        return float(hit["mu"])
    hit = cells["types"].get(truck_type)
    if hit:
        return float(hit["mu"])
    return float(cells["global_mu"])


def _mu_soft(model: dict, truck_type: str, brand: str, era: str, conf: str | None) -> float:
    if conf is None:
        return _mu_from_tables(model, truck_type, brand, era)
    post = era_posterior(era, conf)
    if not post:
        return _mu_from_tables(model, truck_type, brand, UNKNOWN_ERA)
    return sum(w * _mu_from_tables(model, truck_type, brand, e) for e, w in post.items())


def _error_bound(model: dict, truck_type: str, era: str, conf: str | None) -> float:
    table = model["error_table"]

    def flat(e: str) -> float | None:
        hit = table.get("type_era", {}).get(f"{truck_type}|{e}")
        return float(hit["bound"]) if hit else None

    if conf is not None:
        post = era_posterior(era, conf)
        if post:
            num = den = 0.0
            for e, w in post.items():
                b = flat(e)
                if b is not None:
                    num += w * b
                    den += w
            if den > 0:
                return num / den
    else:
        direct = flat(era)
        if direct is not None:
            return direct

    hit = table.get("types", {}).get(truck_type)
    return float(hit["bound"]) if hit else float(table["global"]["bound"])


def predict_from_features(
    model: dict,
    truck_type: str,
    brand: str | None,
    era: str | None = None,
    era_confidence: str | None = None,
    overall_score: float | None = None,
) -> dict:
    t = normalize_truck_type(truck_type)
    b = canonical_brand(brand) if brand else UNKNOWN_BRAND
    e = normalize_era(era)

    mu = _mu_soft(model, t, b, e, era_confidence)

    cond = model["condition"]
    effect = 0.0
    if overall_score is not None and int(cond.get("n", 0)) >= 3:
        effect = (
            float(cond["condition_scale"])
            * float(cond["beta"])
            * (float(overall_score) - float(cond["score_bar"]))
        )

    center = math.exp(mu + effect)
    bound = _error_bound(model, t, e, era_confidence)
    low = max(0.0, center * (1.0 - bound))
    high = center * (1.0 + bound)

    return {
        "truck_type": t,
        "brand": b,
        "era": e,
        "era_confidence": era_confidence,
        "overall_score": overall_score,
        "center": round(center, 2),
        "low": round(low, 2),
        "high": round(high, 2),
        "range": [round(low, 2), round(high, 2)],
        "error_bound_pct": round(100 * bound, 1),
        "condition_effect_pct": round(100 * (math.exp(effect) - 1), 1),
        "model_family": model.get("model_family"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--vlm-json", type=Path, help="a filled master-schema prediction")
    parser.add_argument("--truck-type", type=str, help="day_cab | sleeper | dump")
    parser.add_argument("--brand", type=str)
    parser.add_argument("--era", type=str, help="pre_2010 | 2010_2015 | 2016_2020 | 2021_plus")
    parser.add_argument("--era-confidence", type=str, default="medium",
                        help="high | medium | low (how sure the era call is)")
    parser.add_argument("--score", type=float, help="condition overall_score, 1-5")
    args = parser.parse_args()

    model = load_model(args.model)
    fields = None

    if args.vlm_json:
        raw = json.loads(args.vlm_json.read_text(encoding="utf-8"))
        fields = extract_fields(raw)
        if not is_priceable(fields):
            print(json.dumps({
                "status": "rejected",
                "user_message": (
                    fields.get("validity_user_message")
                    or fields.get("primary_user_message")
                    or "Image is not a priceable Class 7/8 front or side view."
                ),
            }, indent=2, ensure_ascii=False))
            raise SystemExit(2)
        truck_type = fields.get("truck_type")
        brand = fields.get("vlm_brand")
        era = fields.get("era")
        era_conf = fields.get("era_confidence") or "medium"
        score = fields.get("overall_score")
        if not brand or fields.get("needs_user_input"):
            print(json.dumps({
                "status": "needs_brand",
                "user_message": "Could not read the brand from the photo — re-run with --brand.",
                "truck_type": truck_type,
            }, indent=2, ensure_ascii=False))
            raise SystemExit(3)
    elif args.truck_type:
        truck_type = args.truck_type
        brand = args.brand
        era = args.era
        era_conf = args.era_confidence if args.era else None
        score = args.score
    else:
        raise SystemExit("Provide --vlm-json, or --truck-type with --brand/--era/--score")

    pred = predict_from_features(model, truck_type, brand, era, era_conf, score)
    result = {"status": "ok", **pred}
    if fields is not None:
        result["vlm_fields"] = {
            "primary_subject": fields.get("primary_subject"),
            "model_series": fields.get("model_series"),
            "era_evidence": fields.get("era_evidence"),
            "overall_condition_label": fields.get("overall_condition_label"),
        }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

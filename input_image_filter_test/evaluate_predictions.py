"""Score Sonnet 5 filter predictions against the stratified eval sample.

Expects:
  - data/eval_sample.json   from sample_eval_set.py
  - data/predictions.json   filled after the VLM assessment

predictions.json shape:

{
  "items": [
    {
      "id": "pos_001",
      "prediction": {
        "reasoning": "...",
        "output": {
          "truck_type": "day-cab-truck",
          "user_message": ""
        }
      }
    }
  ]
}

Validity is implied: truck_type none = invalid; any other allowed type = valid.

Example:

    python input_image_filter_test/evaluate_predictions.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from paths import (
    ACCURACY_THRESHOLD,
    CATEGORY_TO_TRUCK_TYPE,
    EVAL_SAMPLE_PATH,
    METRICS_PATH,
    PREDICTIONS_PATH,
    TRUCK_TYPES,
    VALID_TRUCK_TYPES,
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_truck_type(truth: dict) -> str:
    if not truth.get("is_valid_class_7_8"):
        return "none"
    return CATEGORY_TO_TRUCK_TYPE.get(truth.get("category", ""), "none")


def validate_prediction(pred: dict) -> list[str]:
    errors: list[str] = []
    if "reasoning" not in pred or not str(pred.get("reasoning", "")).strip():
        errors.append("missing/empty reasoning")
    output = pred.get("output")
    if not isinstance(output, dict):
        errors.append("missing output object")
        return errors

    truck_type = output.get("truck_type")
    msg = output.get("user_message")
    if truck_type not in TRUCK_TYPES:
        errors.append(f"truck_type must be one of {list(TRUCK_TYPES)}")
        return errors
    if truck_type in VALID_TRUCK_TYPES:
        if msg != "":
            errors.append("valid truck_type requires user_message to be empty string")
    elif not isinstance(msg, str) or not msg.strip():
        errors.append("truck_type none requires non-empty user_message")
    return errors


def evaluate(sample: dict, predictions_doc: dict) -> dict:
    truth_by_id = {it["id"]: it for it in sample["items"]}
    pred_items = predictions_doc.get("items")
    if not isinstance(pred_items, list):
        raise ValueError("predictions.json must contain an 'items' list")

    pred_by_id = {}
    for row in pred_items:
        if not isinstance(row, dict) or "id" not in row:
            raise ValueError("each prediction row needs an 'id'")
        pred_by_id[row["id"]] = row

    missing = sorted(set(truth_by_id) - set(pred_by_id))
    extra = sorted(set(pred_by_id) - set(truth_by_id))

    rows = []
    correct = 0
    type_correct = 0
    schema_ok = 0
    confusion = {
        "true_positive": 0,
        "true_negative": 0,
        "false_positive": 0,
        "false_negative": 0,
    }

    for item_id, truth in truth_by_id.items():
        expected_type = expected_truck_type(truth)
        row = {
            "id": item_id,
            "path": truth["path"],
            "category": truth["category"],
            "label": truth["label"],
            "ground_truth_is_valid": truth["is_valid_class_7_8"],
            "ground_truth_truck_type": expected_type,
        }
        if item_id not in pred_by_id:
            row["status"] = "missing_prediction"
            row["correct"] = False
            row["type_correct"] = False
            rows.append(row)
            continue

        pred = pred_by_id[item_id].get("prediction")
        if not isinstance(pred, dict):
            row["status"] = "invalid_prediction_shape"
            row["correct"] = False
            row["type_correct"] = False
            rows.append(row)
            continue

        schema_errors = validate_prediction(pred)
        row["prediction"] = pred
        row["schema_errors"] = schema_errors
        row["schema_ok"] = len(schema_errors) == 0
        if row["schema_ok"]:
            schema_ok += 1

        truck_type = pred.get("output", {}).get("truck_type")
        predicted_valid = truck_type in VALID_TRUCK_TYPES
        row["predicted_truck_type"] = truck_type
        row["predicted_is_valid"] = predicted_valid if truck_type in TRUCK_TYPES else None

        if truck_type in TRUCK_TYPES:
            is_correct = predicted_valid == truth["is_valid_class_7_8"]
            is_type_correct = truck_type == expected_type
            row["correct"] = is_correct
            row["type_correct"] = is_type_correct
            if is_correct:
                correct += 1
                if predicted_valid:
                    confusion["true_positive"] += 1
                else:
                    confusion["true_negative"] += 1
            elif predicted_valid:
                confusion["false_positive"] += 1
            else:
                confusion["false_negative"] += 1
            if is_type_correct:
                type_correct += 1
            row["status"] = "scored"
        else:
            row["correct"] = False
            row["type_correct"] = False
            row["status"] = "missing_truck_type"
        rows.append(row)

    n = len(truth_by_id)
    accuracy = (correct / n) if n else 0.0
    type_accuracy = (type_correct / n) if n else 0.0
    threshold = float(sample.get("accuracy_threshold", ACCURACY_THRESHOLD))
    metrics = {
        "n_total": n,
        "n_scored_correct": correct,
        "accuracy": round(accuracy, 4),
        "accuracy_percent": round(accuracy * 100, 2),
        "accuracy_threshold": threshold,
        "passes_threshold": accuracy >= threshold,
        "n_type_correct": type_correct,
        "type_accuracy": round(type_accuracy, 4),
        "type_accuracy_percent": round(type_accuracy * 100, 2),
        "n_schema_ok": schema_ok,
        "schema_ok_rate": round((schema_ok / n) if n else 0.0, 4),
        "missing_prediction_ids": missing,
        "extra_prediction_ids": extra,
        "confusion": confusion,
        "rows": rows,
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=EVAL_SAMPLE_PATH)
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--out", type=Path, default=METRICS_PATH)
    args = parser.parse_args()

    sample = load_json(args.sample)
    predictions = load_json(args.predictions)
    metrics = evaluate(sample, predictions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"Validity accuracy: {metrics['accuracy_percent']}% "
        f"({metrics['n_scored_correct']}/{metrics['n_total']})"
    )
    print(
        f"Type accuracy: {metrics['type_accuracy_percent']}% "
        f"({metrics['n_type_correct']}/{metrics['n_total']})"
    )
    print(f"Threshold: {metrics['accuracy_threshold'] * 100:.0f}%")
    print(f"Passes: {metrics['passes_threshold']}")
    print(f"Schema-ok: {metrics['n_schema_ok']}/{metrics['n_total']}")
    print(f"Confusion: {metrics['confusion']}")
    if metrics["missing_prediction_ids"]:
        print(f"Missing predictions: {metrics['missing_prediction_ids']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

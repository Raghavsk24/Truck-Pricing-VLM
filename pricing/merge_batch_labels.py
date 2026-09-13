"""Merge in-chat VLM batch predictions into pricing/labels.jsonl.

After the Cursor agent fills pricing/batches/batch_XX.json, run:

    python pricing/merge_batch_labels.py

Produces resume-friendly labels.jsonl for fit_price_range.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BATCHES_DIR = ROOT / "batches"
MANIFEST_PATH = ROOT / "staged_manifest.json"
LABELS_PATH = ROOT / "labels.jsonl"

# Reuse field normalizers from label_sample
from label_sample import extract_fields, is_priceable  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches-dir", type=Path, default=BATCHES_DIR)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path, default=LABELS_PATH)
    args = parser.parse_args()

    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}. Run label_sample.py first.")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    by_opaque = {it["opaque_id"]: it for it in manifest["items"]}

    predictions: dict[str, dict] = {}
    for path in sorted(args.batches_dir.glob("batch_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc.get("items", []):
            opaque = row["opaque_id"]
            pred = row.get("prediction")
            if pred is None:
                continue
            if opaque not in by_opaque:
                raise KeyError(f"{path.name}: unknown opaque_id {opaque}")
            predictions[opaque] = pred

    missing = sorted(set(by_opaque) - set(predictions))
    if missing:
        print(f"Warning: {len(missing)} images still unlabeled (will skip):")
        print(" ", ", ".join(missing[:20]) + (" ..." if len(missing) > 20 else ""))

    ok = rejected = 0
    rows_out: list[dict] = []
    for opaque_id, meta in sorted(by_opaque.items(), key=lambda kv: kv[0]):
        if opaque_id not in predictions:
            continue
        raw = predictions[opaque_id]
        fields = extract_fields(raw)
        priceable = is_priceable(fields)
        truck_type = fields.get("truck_type") or meta.get("truck_type")
        if priceable:
            status = "ok"
            ok += 1
        else:
            status = "rejected"
            rejected += 1
        rows_out.append(
            {
                "listing_id": meta["listing_id"],
                "opaque_id": opaque_id,
                "listing_brand": meta["listing_brand"],
                "brand_cell": meta["brand_cell"],
                "truck_type": truck_type,
                "price": meta["price"],
                "category": meta.get("category"),
                "image_rel": meta.get("image_rel"),
                "staged_path": meta.get("path"),
                "status": status,
                **fields,
                "truck_type": truck_type,
                "raw": raw,
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in rows_out:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Merged {len(rows_out)} labels -> {args.out}")
    print(f"  ok (priceable)={ok}  rejected={rejected}  missing={len(missing)}")
    if ok:
        print("Next: python pricing/fit_price_range.py")


if __name__ == "__main__":
    main()

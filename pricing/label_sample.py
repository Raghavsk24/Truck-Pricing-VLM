"""Stage sample images and write empty batch templates for in-chat VLM labeling.

This matches the input_image_filter_test workflow: Python does NOT call any API.
A Cursor / Sonnet agent in chat reads the master schema + staged images and
fills pricing/batches/batch_XX.json. Then run merge_batch_labels.py.

Steps:
  1. python pricing/build_sample.py          # if sample.json missing
  2. python pricing/label_sample.py          # stage + empty batches
  3. In Cursor chat: ask the agent to label batches using the master schema
  4. python pricing/merge_batch_labels.py    # -> labels.jsonl

Usage:
    python pricing/label_sample.py
    python pricing/label_sample.py --batch-size 10 --limit 40
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SAMPLE_PATH = ROOT / "sample.json"
SCHEMA_PATH = REPO / "vlm_instructions" / "truck_feature_extraction_master_instructions.json"
STAGED_DIR = ROOT / "staged_images"
BATCHES_DIR = ROOT / "batches"
MANIFEST_PATH = ROOT / "staged_manifest.json"
ID_MAP_PATH = ROOT / "opaque_id_map.json"
AGENT_PROMPT_PATH = ROOT / "AGENT_LABEL_PROMPT.md"

LONG_EDGE = 1568
JPEG_QUALITY = 85
DEFAULT_BATCH_SIZE = 10
PRICEABLE_SUBJECTS = frozenset({"front", "side"})


def resize_long_edge(src: Path, dest: Path, long_edge: int = LONG_EDGE) -> tuple[int, int, int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        src_w, src_h = im.size
        longest = max(src_w, src_h)
        if longest > long_edge:
            scale = long_edge / longest
            im = im.resize(
                (max(1, round(src_w * scale)), max(1, round(src_h * scale))),
                Image.Resampling.LANCZOS,
            )
        dst_w, dst_h = im.size
        im.save(dest, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return src_w, src_h, dst_w, dst_h


def pick_candidate(listing: dict) -> dict | None:
    for cand in listing.get("candidate_images") or []:
        src = Path(cand["abs"])
        if src.exists():
            return cand
    return None


def write_agent_prompt(
    path: Path,
    *,
    schema: Path,
    manifest: Path,
    batches_dir: Path,
    n_batches: int,
    n_images: int,
) -> None:
    text = f"""# Pricing VLM labeling

Python staged the images. Fill each batch JSON with master-schema predictions.

## Schema

`{schema.as_posix()}`

Return one JSON object per image matching **TruckFeatureExtractionMaster**.

## Staged data

- Manifest: `{manifest.as_posix()}`
- Images: `pricing/staged_images/img_NNN.jpg` (long edge <= {LONG_EDGE}px)
- Batches: `{batches_dir.as_posix()}` ({n_batches} files, {n_images} images)

Classify from **image pixels only**. Do not use listing brand, price, or folder names when filling VLM fields.

## Batch format

Each item starts as:

```json
{{
  "opaque_id": "img_001",
  "path": ".../staged_images/img_001.jpg",
  "prediction": null
}}
```

Replace `prediction` with the full master-schema object (`reasoning` + `output` with truck_type, primary_subject, brand, condition).

Priceable images need `truck_type != none` and `primary_subject` in {{front, side}}. Otherwise still fill brand/condition with the schema skip placeholders.

## Workflow

1. Open one batch file.
2. For each item, inspect the image at `path`.
3. Write the full prediction.
4. Save the batch.
5. Repeat until every `prediction` is non-null.

Then:

```text
python pricing/merge_batch_labels.py
python pricing/fit_price_range.py
```
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--staged-dir", type=Path, default=STAGED_DIR)
    parser.add_argument("--batches-dir", type=Path, default=BATCHES_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--long-edge", type=int, default=LONG_EDGE)
    parser.add_argument("--limit", type=int, default=0, help="stage at most N listings (0=all)")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete existing staged_images/ and batches/ before writing",
    )
    args = parser.parse_args()

    if not args.sample.exists():
        raise SystemExit(f"Sample not found: {args.sample}. Run build_sample.py first.")
    if not args.schema.exists():
        raise SystemExit(f"Master schema not found: {args.schema}")

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    listings = sample["listings"]
    if args.limit > 0:
        listings = listings[: args.limit]

    if args.fresh:
        if args.staged_dir.exists():
            shutil.rmtree(args.staged_dir)
        if args.batches_dir.exists():
            shutil.rmtree(args.batches_dir)

    args.staged_dir.mkdir(parents=True, exist_ok=True)
    args.batches_dir.mkdir(parents=True, exist_ok=True)

    staged_items = []
    id_map: dict[str, str] = {}
    skipped = 0

    for i, listing in enumerate(listings, start=1):
        cand = pick_candidate(listing)
        if cand is None:
            skipped += 1
            continue
        opaque_id = f"img_{len(staged_items) + 1:03d}"
        dest = args.staged_dir / f"{opaque_id}.jpg"
        src_w, src_h, dst_w, dst_h = resize_long_edge(
            Path(cand["abs"]), dest, long_edge=args.long_edge
        )
        staged_items.append(
            {
                "opaque_id": opaque_id,
                "listing_id": listing["listing_id"],
                "listing_brand": listing["brand"],
                "brand_cell": listing["brand_cell"],
                "truck_type": listing.get("truck_type"),
                "price": listing["price"],
                "category": listing.get("category"),
                "image_rel": cand.get("rel"),
                "path": str(dest.resolve()),
                "source_size": [src_w, src_h],
                "staged_size": [dst_w, dst_h],
            }
        )
        id_map[opaque_id] = listing["listing_id"]

    n = len(staged_items)
    n_batches = max(1, math.ceil(n / args.batch_size)) if n else 0

    # Write empty batch templates (do not overwrite filled predictions unless --fresh)
    for b in range(n_batches):
        batch_path = args.batches_dir / f"batch_{b + 1:02d}.json"
        chunk = staged_items[b * args.batch_size : (b + 1) * args.batch_size]
        if batch_path.exists() and not args.fresh:
            existing = json.loads(batch_path.read_text(encoding="utf-8"))
            # Keep any already-filled predictions keyed by opaque_id
            filled = {
                it["opaque_id"]: it.get("prediction")
                for it in existing.get("items", [])
                if it.get("prediction") is not None
            }
        else:
            filled = {}
        items = []
        for it in chunk:
            items.append(
                {
                    "opaque_id": it["opaque_id"],
                    "path": it["path"],
                    "prediction": filled.get(it["opaque_id"]),
                }
            )
        batch_path.write_text(
            json.dumps({"items": items}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    manifest = {
        "instructions_schema": str(args.schema.resolve()),
        "max_long_edge": args.long_edge,
        "jpeg_quality": JPEG_QUALITY,
        "batch_size": args.batch_size,
        "n_images": n,
        "n_batches": n_batches,
        "note": (
            "Staged copies for in-chat Sonnet labeling (no API key). "
            "Fill pricing/batches/batch_XX.json prediction fields using the master schema. "
            "Do not infer VLM labels from listing brand/price — those are only for merge/fit."
        ),
        "items": staged_items,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ID_MAP_PATH.write_text(json.dumps(id_map, indent=2) + "\n", encoding="utf-8")
    write_agent_prompt(
        AGENT_PROMPT_PATH,
        schema=args.schema,
        manifest=MANIFEST_PATH,
        batches_dir=args.batches_dir,
        n_batches=n_batches,
        n_images=n,
    )

    filled_count = 0
    for path in sorted(args.batches_dir.glob("batch_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        filled_count += sum(1 for it in doc["items"] if it.get("prediction") is not None)

    print(f"Staged {n} images -> {args.staged_dir}")
    if skipped:
        print(f"Skipped {skipped} listings with no on-disk candidate image")
    print(f"Wrote {n_batches} batch templates -> {args.batches_dir}")
    print(f"Already filled predictions: {filled_count}/{n}")
    print(f"Wrote {MANIFEST_PATH}")
    print(f"Wrote {ID_MAP_PATH}")
    print(f"Wrote {AGENT_PROMPT_PATH}")
    print()
    print("Next: in Cursor chat, ask the agent to label the batches")
    print(f"  (follow {AGENT_PROMPT_PATH.name}), then run:")
    print("  python pricing/merge_batch_labels.py")


# --- helpers still imported by predict.py / merge ---
def extract_fields(vlm_output: dict) -> dict:
    """Normalize master-schema (or nested) output into flat pricing fields."""
    out = vlm_output.get("output", vlm_output)

    validity = out.get("validity") or out
    brand = out.get("brand") or out
    condition = out.get("condition") or out

    if isinstance(out.get("primary_subject"), str):
        primary_subject = out.get("primary_subject")
        primary_user_message = out.get("primary_subject_user_message", "") or ""
    elif isinstance(out.get("primary_subject"), dict):
        primary_subject = out["primary_subject"].get("primary_subject")
        primary_user_message = out["primary_subject"].get("user_message", "")
    else:
        primary_subject = out.get("primary_subject")
        primary_user_message = ""

    raw_truck_type = out.get("truck_type") or (
        validity.get("truck_type") if isinstance(validity, dict) else None
    ) or out.get("vehicle_cue")
    if isinstance(raw_truck_type, dict):
        raw_truck_type = raw_truck_type.get("truck_type")

    truck_type = _normalize_truck_type(
        raw_truck_type if isinstance(raw_truck_type, str) else None
    )

    is_valid = None
    if isinstance(validity, dict):
        is_valid = validity.get("is_valid_class_7_8")
    if is_valid is None:
        is_valid = out.get("is_valid_class_7_8")
    if is_valid is None:
        is_valid = (
            isinstance(raw_truck_type, str)
            and raw_truck_type.strip().lower() not in ("", "none")
            and truck_type is not None
        )

    vehicle_cue = raw_truck_type
    validity_msg = out.get("truck_type_user_message") or ""
    if isinstance(validity, dict) and validity.get("user_message"):
        validity_msg = validity_msg or validity.get("user_message") or ""

    has_brand = brand.get("has_brand") if isinstance(brand, dict) else out.get("has_brand")
    brand_name = brand.get("brand_name") if isinstance(brand, dict) else out.get("brand_name")
    brand_confidence = (
        brand.get("confidence") if isinstance(brand, dict) else out.get("confidence")
    )
    needs_user_input = (
        brand.get("needs_user_input") if isinstance(brand, dict) else out.get("needs_user_input")
    )

    if isinstance(condition, dict) and "categories" in condition:
        overall_score = condition.get("overall_score")
        overall_label = condition.get("overall_condition_label")
        penalty = condition.get("total_penalty_percent")
        explanation = condition.get("explanation")
    else:
        overall_score = out.get("overall_score")
        overall_label = out.get("overall_condition_label")
        penalty = out.get("total_penalty_percent")
        explanation = out.get("explanation")

    return {
        "is_valid_class_7_8": bool(is_valid) if is_valid is not None else None,
        "vehicle_cue": vehicle_cue,
        "truck_type": truck_type,
        "validity_user_message": validity_msg or "",
        "primary_subject": primary_subject,
        "primary_user_message": primary_user_message or "",
        "has_brand": has_brand,
        "vlm_brand": brand_name,
        "brand_confidence": brand_confidence,
        "needs_user_input": needs_user_input,
        "overall_score": overall_score,
        "overall_condition_label": overall_label,
        "total_penalty_percent": penalty,
        "condition_explanation": explanation,
    }


def _normalize_truck_type(value: str | None) -> str | None:
    if not value:
        return None
    v = value.strip().lower().replace(" ", "_")
    v_hyphen = v.replace("_", "-")
    mapping = {
        "day_cab": "day_cab",
        "day-cab": "day_cab",
        "day_cab_tractor": "day_cab",
        "day-cab-truck": "day_cab",
        "day_cab_truck": "day_cab",
        "sleeper": "sleeper",
        "sleeper_tractor": "sleeper",
        "sleeper-truck": "sleeper",
        "sleeper_truck": "sleeper",
        "dump": "dump",
        "dumper": "dump",
        "dump-truck": "dump",
        "dump_truck": "dump",
        "heavy_dump_truck": "dump",
        "heavy-dump-truck": "dump",
        "none": None,
    }
    if v in mapping:
        return mapping[v]
    if v_hyphen in mapping:
        return mapping[v_hyphen]
    if "dump" in v:
        return "dump"
    if "sleeper" in v:
        return "sleeper"
    if "day" in v and "cab" in v:
        return "day_cab"
    return None


def is_priceable(fields: dict) -> bool:
    return (
        fields.get("is_valid_class_7_8") is True
        and fields.get("primary_subject") in PRICEABLE_SUBJECTS
    )


if __name__ == "__main__":
    main()

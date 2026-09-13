"""Label sample listings with the master VLM schema.

Stages each image to JPEG with long edge = 1568px (downscale only; never
upsamples), then calls the VLM. Writes resume-friendly labels.jsonl.

Requires:
  - ANTHROPIC_API_KEY in the environment
  - vlm instructions/truck_feature_extraction_master_instructions.json
    (or pass --schema)

Usage:
    python pricing/label_sample.py
    python pricing/label_sample.py --concurrency 8 --limit 20
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SAMPLE_PATH = ROOT / "sample.json"
LABELS_PATH = ROOT / "labels.jsonl"
STAGED_DIR = ROOT / "staged_images"
SCHEMA_PATH = REPO / "vlm instructions" / "truck_feature_extraction_master_instructions.json"

LONG_EDGE = 1568
PRICEABLE_SUBJECTS = frozenset({"front", "side"})
DEFAULT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_CONCURRENCY = 8

_write_lock = threading.Lock()


def resize_long_edge(src: Path, dest: Path, long_edge: int = LONG_EDGE) -> Path:
    """Downscale so the longest side is `long_edge` px; write JPEG to dest."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit(
            "Pillow is required for image staging. Install with: pip install Pillow"
        ) from exc

    dest.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        longest = max(w, h)
        if longest > long_edge:
            scale = long_edge / longest
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
            im = im.resize(new_size, Image.Resampling.LANCZOS)
        im.save(dest, format="JPEG", quality=90, optimize=True)
    return dest


def image_to_data_url(path: Path) -> str:
    data = path.read_bytes()
    b64 = base64.standard_b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def load_done_ids(labels_path: Path) -> set[str]:
    done: set[str] = set()
    if not labels_path.exists():
        return done
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        lid = obj.get("listing_id")
        if lid and obj.get("status") in ("ok", "rejected", "exhausted"):
            done.add(lid)
    return done


def append_label(labels_path: Path, row: dict) -> None:
    with _write_lock:
        with labels_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_truck_type(value: str | None) -> str | None:
    """Map VLM vehicle_cue / truck_type strings to day_cab | sleeper | dump."""
    if not value:
        return None
    v = value.strip().lower().replace(" ", "_").replace("-", "_")
    mapping = {
        "day_cab": "day_cab",
        "day_cab_tractor": "day_cab",
        "sleeper": "sleeper",
        "sleeper_tractor": "sleeper",
        "dump": "dump",
        "dumper": "dump",
        "heavy_dump_truck": "dump",
        "dump_truck": "dump",
    }
    if v in mapping:
        return mapping[v]
    if "dump" in v:
        return "dump"
    if "sleeper" in v:
        return "sleeper"
    if "day" in v and "cab" in v:
        return "day_cab"
    return None


def extract_fields(vlm_output: dict) -> dict:
    """Normalize master-schema (or nested) output into flat pricing fields."""
    out = vlm_output.get("output", vlm_output)

    validity = out.get("validity") or out
    primary = out.get("primary_subject") or out
    brand = out.get("brand") or out
    condition = out.get("condition") or out
    truck = out.get("truck_type") or out.get("truck") or out

    if isinstance(primary, dict) and "primary_subject" in primary:
        primary_subject = primary.get("primary_subject")
        primary_user_message = primary.get("user_message", "")
    else:
        primary_subject = out.get("primary_subject")
        primary_user_message = out.get("user_message", "") if "primary_subject" in out else ""

    is_valid = validity.get("is_valid_class_7_8")
    if is_valid is None:
        is_valid = out.get("is_valid_class_7_8")

    vehicle_cue = validity.get("vehicle_cue") or out.get("vehicle_cue")
    raw_truck_type = None
    if isinstance(truck, dict):
        raw_truck_type = truck.get("truck_type") or truck.get("type") or truck.get("vehicle_cue")
    if raw_truck_type is None:
        raw_truck_type = out.get("truck_type") or vehicle_cue
    truck_type = _normalize_truck_type(
        raw_truck_type if isinstance(raw_truck_type, str) else None
    )

    validity_msg = validity.get("user_message")
    if validity_msg is None:
        validity_msg = ""

    has_brand = brand.get("has_brand") if isinstance(brand, dict) else out.get("has_brand")
    brand_name = brand.get("brand_name") if isinstance(brand, dict) else out.get("brand_name")
    brand_confidence = (
        brand.get("confidence") if isinstance(brand, dict) else out.get("confidence")
    )
    needs_user_input = (
        brand.get("needs_user_input") if isinstance(brand, dict) else out.get("needs_user_input")
    )

    if isinstance(condition, dict):
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


def is_priceable(fields: dict) -> bool:
    return (
        fields.get("is_valid_class_7_8") is True
        and fields.get("primary_subject") in PRICEABLE_SUBJECTS
    )


def call_vlm(
    client,
    model: str,
    schema: dict,
    staged_path: Path,
    max_tokens: int = 4096,
) -> dict:
    schema_text = json.dumps(schema, ensure_ascii=False)
    prompt = (
        "Inspect this truck image and return ONE JSON object that matches the "
        "provided JSON schema exactly. No markdown fences, no prose outside JSON.\n\n"
        f"JSON schema:\n{schema_text}"
    )
    data_url = image_to_data_url(staged_path)
    # Anthropic Messages API wants raw base64 + media_type, not data URLs
    b64 = data_url.split(",", 1)[1]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    text_parts = [b.text for b in response.content if getattr(b, "type", None) == "text"]
    text = "\n".join(text_parts).strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    return json.loads(text)


def label_one_listing(
    listing: dict,
    *,
    client,
    model: str,
    schema: dict,
    staged_dir: Path,
    long_edge: int,
) -> dict:
    listing_id = listing["listing_id"]
    last_error = None
    attempts = []

    for cand in listing.get("candidate_images") or []:
        src = Path(cand["abs"])
        if not src.exists():
            attempts.append({"rel": cand.get("rel"), "error": "missing_file"})
            continue
        staged = staged_dir / f"{listing_id}_{src.stem}.jpg"
        try:
            resize_long_edge(src, staged, long_edge=long_edge)
            raw = call_vlm(client, model, schema, staged)
            fields = extract_fields(raw)
            attempt = {
                "rel": cand.get("rel"),
                "staged": str(staged),
                "fields": fields,
                "raw": raw,
            }
            attempts.append(attempt)
            if is_priceable(fields):
                truck_type = fields.get("truck_type") or listing.get("truck_type")
                return {
                    "listing_id": listing_id,
                    "listing_brand": listing["brand"],
                    "brand_cell": listing["brand_cell"],
                    "truck_type": truck_type,
                    "price": listing["price"],
                    "category": listing.get("category"),
                    "image_rel": cand.get("rel"),
                    "staged_path": str(staged),
                    "status": "ok",
                    **fields,
                    "truck_type": truck_type,  # listing fallback after fields spread
                    "attempts": [
                        {
                            "rel": a.get("rel"),
                            "error": a.get("error"),
                            "primary_subject": (a.get("fields") or {}).get("primary_subject"),
                            "is_valid_class_7_8": (a.get("fields") or {}).get(
                                "is_valid_class_7_8"
                            ),
                            "truck_type": (a.get("fields") or {}).get("truck_type"),
                        }
                        for a in attempts
                    ],
                }
        except Exception as exc:  # noqa: BLE001 - resume-friendly labeling loop
            last_error = str(exc)
            attempts.append({"rel": cand.get("rel"), "error": last_error})
            continue

    # All candidates rejected or failed
    if attempts and any((a.get("fields") is not None) for a in attempts):
        status = "rejected"
    else:
        status = "exhausted"
    return {
        "listing_id": listing_id,
        "listing_brand": listing["brand"],
        "brand_cell": listing["brand_cell"],
        "price": listing["price"],
        "category": listing.get("category"),
        "status": status,
        "error": last_error,
        "attempts": [
            {
                "rel": a.get("rel"),
                "error": a.get("error"),
                "primary_subject": (a.get("fields") or {}).get("primary_subject"),
                "is_valid_class_7_8": (a.get("fields") or {}).get("is_valid_class_7_8"),
            }
            for a in attempts
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    parser.add_argument("--out", type=Path, default=LABELS_PATH)
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--staged-dir", type=Path, default=STAGED_DIR)
    parser.add_argument("--long-edge", type=int, default=LONG_EDGE)
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0, help="label at most N listings (0=all)")
    args = parser.parse_args()

    if not args.sample.exists():
        raise SystemExit(f"Sample not found: {args.sample}. Run build_sample.py first.")
    if not args.schema.exists():
        raise SystemExit(
            f"Master schema not found: {args.schema}\n"
            "Create truck_feature_extraction_master_instructions.json first."
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("Set ANTHROPIC_API_KEY before running label_sample.py")

    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("Install anthropic: pip install anthropic") from exc

    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    listings = sample["listings"]
    if args.limit > 0:
        listings = listings[: args.limit]

    done = load_done_ids(args.out)
    todo = [l for l in listings if l["listing_id"] not in done]
    print(f"Sample listings: {len(listings)} | already labeled: {len(done)} | todo: {len(todo)}")
    if not todo:
        print("Nothing to do.")
        return

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    client = anthropic.Anthropic(api_key=api_key)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.staged_dir.mkdir(parents=True, exist_ok=True)

    ok = rejected = exhausted = errors = 0
    t0 = time.time()

    def work(listing: dict) -> dict:
        return label_one_listing(
            listing,
            client=client,
            model=args.model,
            schema=schema,
            staged_dir=args.staged_dir,
            long_edge=args.long_edge,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(work, l): l["listing_id"] for l in todo}
        for fut in concurrent.futures.as_completed(futures):
            lid = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                row = {"listing_id": lid, "status": "error", "error": str(exc)}
                errors += 1
            else:
                if row["status"] == "ok":
                    ok += 1
                elif row["status"] == "rejected":
                    rejected += 1
                else:
                    exhausted += 1
            append_label(args.out, row)
            print(
                f"[{ok + rejected + exhausted + errors}/{len(todo)}] "
                f"{row['listing_id']} -> {row['status']}"
            )

    elapsed = time.time() - t0
    print(
        f"Done in {elapsed:.0f}s | ok={ok} rejected={rejected} "
        f"exhausted={exhausted} errors={errors} -> {args.out}"
    )


if __name__ == "__main__":
    main()

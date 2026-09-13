"""Build a stratified sample of priced Class 7/8 listings for VLM labeling.

Reads truckpaper_scraped_listings.json, keeps positive (class 7-8) listings with
USD price + brand, samples ~160 listings proportional to brand, and writes
sample.json with candidate image paths (_image_01, then _image_02).

Usage:
    python pricing/build_sample.py
    python pricing/build_sample.py --n 160 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
DATASET = REPO / "image_scraper" / "truckpaper_scraped_images_dataset"
LISTINGS_JSON = DATASET / "truckpaper_scraped_listings.json"
OUT_PATH = ROOT / "sample.json"

MAJOR_BRANDS = ("FREIGHTLINER", "INTERNATIONAL", "KENWORTH", "PETERBILT", "MACK")
OTHER = "OTHER"
DEFAULT_N = 160
CATEGORY_TO_TYPE = {
    "day-cab-trucks": "day_cab",
    "sleeper-trucks": "sleeper",
    "dump-trucks": "dump",
}


def brand_cell(brand: str | None) -> str:
    if not brand:
        return OTHER
    b = brand.strip().upper()
    return b if b in MAJOR_BRANDS else OTHER


def truck_type_from_category(category: str) -> str:
    return CATEGORY_TO_TYPE.get(category, "other")


def load_positive_listings(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for folder, listings in data.items():
        if not folder.startswith("positive"):
            continue
        category = folder.split("/", 1)[1] if "/" in folder else folder
        for listing in listings:
            price = listing.get("price")
            brand = listing.get("brand")
            images = listing.get("images") or []
            if price is None or not brand or not images:
                continue
            if listing.get("currency") not in (None, "USD"):
                continue
            abs_images = []
            for rel in images:
                p = DATASET / rel
                if p.exists():
                    abs_images.append({"rel": rel, "abs": str(p)})
            if not abs_images:
                continue
            candidates = abs_images[:2]
            rows.append(
                {
                    "listing_id": listing["id"],
                    "brand": brand.strip().upper(),
                    "brand_cell": brand_cell(brand),
                    "truck_type": truck_type_from_category(category),
                    "price": float(price),
                    "currency": listing.get("currency") or "USD",
                    "category": category,
                    "folder": folder,
                    "candidate_images": candidates,
                }
            )
    return rows


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell[r["brand_cell"]].append(r)

    rng = random.Random(seed)
    for cell in by_cell:
        rng.shuffle(by_cell[cell])

    counts = {cell: len(v) for cell, v in by_cell.items()}
    total = sum(counts.values())
    if total == 0:
        return []

    quotas: dict[str, int] = {}
    assigned = 0
    for cell, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        q = max(1, round(n * c / total)) if c else 0
        q = min(q, c)
        quotas[cell] = q
        assigned += q

    while assigned > n:
        cell = max(quotas, key=lambda k: quotas[k])
        if quotas[cell] > 0:
            quotas[cell] -= 1
            assigned -= 1
        else:
            break

    while assigned < n:
        leftovers = {
            cell: counts[cell] - quotas[cell]
            for cell in counts
            if counts[cell] - quotas[cell] > 0
        }
        if not leftovers:
            break
        cell = max(leftovers, key=leftovers.get)
        quotas[cell] += 1
        assigned += 1

    sample: list[dict] = []
    for cell, q in quotas.items():
        sample.extend(by_cell[cell][:q])
    rng.shuffle(sample)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=DEFAULT_N, help="target sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--listings", type=Path, default=LISTINGS_JSON)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    args = parser.parse_args()

    rows = load_positive_listings(args.listings)
    sample = stratified_sample(rows, args.n, args.seed)

    payload = {
        "seed": args.seed,
        "target_n": args.n,
        "population_n": len(rows),
        "sample_n": len(sample),
        "population_brand_counts": dict(Counter(r["brand_cell"] for r in rows)),
        "sample_brand_counts": dict(Counter(r["brand_cell"] for r in sample)),
        "listings": sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Population: {len(rows)} priced Class 7/8 listings with brand + images")
    print(f"Sample:     {len(sample)} -> {args.out}")
    print("Sample brand cells:", payload["sample_brand_counts"])


if __name__ == "__main__":
    main()

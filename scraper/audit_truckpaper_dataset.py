"""Audit and clean up the scraped TruckPaper dataset: drop unusable listings, remove
duplicate images, then renumber what's left.

Three checks run in this order, each dropping a full listing (folder, images, and its
manifest entry) or a single duplicate image:

1. Bad price - the listing's price is null (TruckPaper showed "Call for Price" or
   something non-positive). A price-prediction dataset needs a real price on every row.
2. Non-USD currency - the listing's currency isn't exactly "USD" (including missing
   currency). Keeps the price column comparable across the whole dataset.
3. Duplicate images - a hash shared with an image already kept: either a second copy
   inside the same listing, or the exact same photo turning up under a different
   listing (almost always a shared "no photo available" placeholder, not a real truck
   photo). Detected by exact content (SHA-256), scoped separately within each of the two
   top-level datasets (Positive and Negative) - a hash shared between the two datasets is
   left alone, since that's a much weaker signal of a placeholder than one repeating
   across dozens of listings in the same dataset. The first occurrence (scanning folders
   and listings in a fixed, repeatable order) is kept.

After dedup, each listing's surviving images are renamed to close the gap -
"sleeper_truck_07_image_01", "_image_02", ... - and the manifest is rewritten to match.
A listing left with zero images after dedup is dropped too.

This is a real, destructive cleanup (deletes files, renames files, rewrites the
manifest), so it defaults to a dry run:

    python scraper/audit_truckpaper_dataset.py              # report only, nothing changes
    python scraper/audit_truckpaper_dataset.py --apply      # actually delete/rename/rewrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

OUT_DIR = Path(__file__).resolve().parent.parent / "TruckPaper Scraped Images Dataset"
JSON_NAME = "truckpaper_listings.json"
GOOD_CURRENCY = "USD"
HASH_CHUNK = 1024 * 1024


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def drop_listing(data: dict[str, list], folder_key: str, listing: dict, out_dir: Path,
                 folders_to_delete: list[Path]) -> None:
    data[folder_key].remove(listing)
    folders_to_delete.append(out_dir / folder_key / listing["id"])


def audit_prices_and_currency(data: dict[str, list], out_dir: Path,
                              folders_to_delete: list[Path]) -> Counter:
    """Drop listings with no usable price or a non-USD currency. Runs before dedup so
    their images are never hashed at all."""
    stats = Counter()
    for folder_key in list(data):
        for listing in list(data[folder_key]):
            reason = None
            if listing.get("price") is None:
                reason = "bad_price"
            elif listing.get("currency") != GOOD_CURRENCY:
                reason = "bad_currency"
            if reason:
                stats[reason] += 1
                stats["images_removed_with_listing"] += len(listing["images"])
                drop_listing(data, folder_key, listing, out_dir, folders_to_delete)
    return stats


def audit_duplicate_images(data: dict[str, list], out_dir: Path, files_to_delete: list[Path],
                           renames: list[tuple[Path, Path]], folders_to_delete: list[Path]) -> Counter:
    """Remove duplicate images (by content hash) and plan the renumbering of what's left."""
    stats = Counter()
    groups: dict[str, list[str]] = {}
    for folder_key in sorted(data):
        groups.setdefault(folder_key.split("/", 1)[0], []).append(folder_key)

    total_images = sum(len(l["images"]) for keys in groups.values() for k in keys for l in data[k])
    bar = tqdm(total=total_images, unit="img", desc="Hashing") if tqdm else None

    for group, folder_keys in groups.items():
        seen: dict[str, tuple[str, str]] = {}  # content hash -> (folder_key, listing id) that kept it
        for folder_key in folder_keys:
            for listing in list(data[folder_key]):
                kept_paths: list[Path] = []
                for rel_path in listing["images"]:
                    abs_path = out_dir / rel_path
                    if bar:
                        bar.update(1)
                    if not abs_path.exists():
                        stats["already_missing"] += 1
                        continue
                    digest = sha256_of_file(abs_path)
                    owner = seen.get(digest)
                    if owner is None:
                        seen[digest] = (folder_key, listing["id"])
                        kept_paths.append(abs_path)
                    else:
                        files_to_delete.append(abs_path)
                        stats["dup_within_listing" if owner[1] == listing["id"] else "dup_cross_listing"] += 1

                stats["listings_seen"] += 1
                if not kept_paths:
                    stats["listings_emptied"] += 1
                    drop_listing(data, folder_key, listing, out_dir, folders_to_delete)
                    continue

                code = listing["id"]
                new_rel_paths = []
                for i, old_path in enumerate(kept_paths):
                    new_path = old_path.with_name(f"{code}_image_{i + 1:02d}{old_path.suffix}")
                    if new_path != old_path:
                        renames.append((old_path, new_path))
                    new_rel_paths.append(new_path.relative_to(out_dir).as_posix())
                listing["images"] = new_rel_paths
    if bar:
        bar.close()
    return stats


def run(out_dir: Path, apply: bool) -> None:
    json_path = out_dir / JSON_NAME
    data: dict[str, list] = json.loads(json_path.read_text(encoding="utf-8"))
    folders_to_delete: list[Path] = []
    files_to_delete: list[Path] = []
    renames: list[tuple[Path, Path]] = []

    price_stats = audit_prices_and_currency(data, out_dir, folders_to_delete)
    print(f"Bad price (null / \"Call for Price\"): {price_stats['bad_price']} listings dropped")
    print(f"Non-{GOOD_CURRENCY} currency: {price_stats['bad_currency']} listings dropped")
    print(f"  ({price_stats['images_removed_with_listing']} images removed along with those listings)")

    dedup_stats = audit_duplicate_images(data, out_dir, files_to_delete, renames, folders_to_delete)
    print(f"\nScanned {dedup_stats['listings_seen']} remaining listings, "
          f"{dedup_stats['already_missing']} images already missing on disk.")
    print(f"Duplicate images: {dedup_stats['dup_within_listing']} repeated inside their own listing, "
          f"{dedup_stats['dup_cross_listing']} repeated across different listings (likely placeholders)")
    print(f"Listings left with zero images after dedup: {dedup_stats['listings_emptied']}")
    print(f"Images to rename to close the gap: {len(renames)}")

    total_dropped = price_stats["bad_price"] + price_stats["bad_currency"] + dedup_stats["listings_emptied"]
    remaining = sum(len(v) for v in data.values())
    print(f"\nTotal listings dropped: {total_dropped}. Listings remaining: {remaining}.")

    if not apply:
        print("\nDry run only - nothing changed. Re-run with --apply to make these changes.")
        return

    for path in files_to_delete:
        path.unlink(missing_ok=True)
    # Safe to rename in this order: a kept image's new name is always an earlier slot than
    # its own old name, and strictly earlier than any later-processed image's old name (since
    # dedup only ever compacts the sequence), so a target name is never still occupied by a
    # file that hasn't been moved out of the way yet.
    for old_path, new_path in renames:
        os.replace(old_path, new_path)
    for folder in folders_to_delete:
        shutil.rmtree(folder, ignore_errors=True)
    data = {k: v for k, v in data.items() if v}  # drop categories left with no listings at all

    write_json_atomic(json_path, data)
    print(f"Applied: dropped {total_dropped} listings, deleted {len(files_to_delete)} duplicate images, "
          f"renamed {len(renames)}, rewrote {json_path.name}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="actually delete/rename/rewrite (default: dry run)")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="dataset directory (default: %(default)s)")
    args = parser.parse_args()
    run(args.out, args.apply)


if __name__ == "__main__":
    main()

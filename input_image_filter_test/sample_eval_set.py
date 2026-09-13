"""Build a stratified 50-image eval set (25 positive / 25 negative).

Does not call any VLM. Writes:
  - data/eval_sample.json   (paths + ground-truth labels)
  - data/blind_manifest.json (paths only, for the Sonnet 5 assessment)

Example (run only when approved):

    python input_image_filter_test/sample_eval_set.py
    python input_image_filter_test/sample_eval_set.py --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from paths import (
    ACCURACY_THRESHOLD,
    BLIND_MANIFEST_PATH,
    DATA_DIR,
    EVAL_SAMPLE_PATH,
    IMAGE_EXTENSIONS,
    INSTRUCTIONS_PATH,
    N_NEGATIVE,
    N_POSITIVE,
    NEGATIVE_DIR,
    POSITIVE_DIR,
    PREDICTIONS_PATH,
    DEFAULT_SEED,
    SampleItem,
)


def load_excluded_paths(exclude_from: Path | None) -> set[str]:
    if exclude_from is None:
        return set()
    doc = json.loads(exclude_from.read_text(encoding="utf-8"))
    return {str(Path(it["path"]).resolve()) for it in doc.get("items", [])}


def discover_images(root: Path, excluded: set[str] | None = None) -> dict[str, list[dict]]:
    """Map category folder name -> list of image records under that category."""
    excluded = excluded or set()
    by_category: dict[str, list[dict]] = defaultdict(list)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        category = category_dir.name
        for listing_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            listing_id = listing_dir.name
            for image_path in sorted(listing_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    resolved = str(image_path.resolve())
                    if resolved in excluded:
                        continue
                    by_category[category].append(
                        {
                            "path": image_path,
                            "category": category,
                            "listing_id": listing_id,
                        }
                    )
    if not by_category:
        raise RuntimeError(f"No images found under {root}")
    return dict(by_category)


def allocate_strata(n_total: int, strata: list[str], counts: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation across strata, capped by available counts."""
    eligible = [s for s in strata if counts.get(s, 0) > 0]
    if not eligible:
        raise RuntimeError("No strata with available images")
    if sum(counts[s] for s in eligible) < n_total:
        raise RuntimeError(
            f"Need {n_total} images but only {sum(counts[s] for s in eligible)} available"
        )

    raw = {s: (n_total * counts[s] / sum(counts[t] for t in eligible)) for s in eligible}
    base = {s: int(raw[s]) for s in eligible}
    for s in eligible:
        base[s] = min(base[s], counts[s])

    assigned = sum(base.values())
    remainders = sorted(
        eligible,
        key=lambda s: (raw[s] - int(raw[s]), counts[s], s),
        reverse=True,
    )
    idx = 0
    while assigned < n_total and idx < len(remainders) * (n_total + 1):
        s = remainders[idx % len(remainders)]
        if base[s] < counts[s]:
            base[s] += 1
            assigned += 1
        idx += 1

    if assigned < n_total:
        raise RuntimeError("Could not allocate enough images across strata")
    return base


def sample_from_category(
    records: list[dict],
    n: int,
    rng: random.Random,
) -> list[dict]:
    """Sample n images, preferring at most one image per listing when possible."""
    by_listing: dict[str, list[dict]] = defaultdict(list)
    for rec in records:
        by_listing[rec["listing_id"]].append(rec)

    listing_ids = list(by_listing.keys())
    rng.shuffle(listing_ids)

    chosen: list[dict] = []
    # First pass: one random image from each listing until n or listings exhausted.
    for listing_id in listing_ids:
        if len(chosen) >= n:
            break
        chosen.append(rng.choice(by_listing[listing_id]))

    if len(chosen) < n:
        remaining = [r for r in records if r not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: n - len(chosen)])

    if len(chosen) < n:
        raise RuntimeError(f"Category only has {len(chosen)} images, need {n}")
    return chosen[:n]


def build_split(
    root: Path,
    label: str,
    is_valid: bool,
    n: int,
    id_prefix: str,
    rng: random.Random,
    excluded: set[str] | None = None,
) -> list[SampleItem]:
    by_category = discover_images(root, excluded)
    strata = sorted(by_category.keys())
    counts = {s: len(by_category[s]) for s in strata}
    allocation = allocate_strata(n, strata, counts)

    picked: list[dict] = []
    for category, k in allocation.items():
        picked.extend(sample_from_category(by_category[category], k, rng))

    rng.shuffle(picked)
    items: list[SampleItem] = []
    for i, rec in enumerate(picked, start=1):
        items.append(
            {
                "id": f"{id_prefix}_{i:03d}",
                "path": str(rec["path"].resolve()),
                "label": label,
                "is_valid_class_7_8": is_valid,
                "category": rec["category"],
                "listing_id": rec["listing_id"],
            }
        )
    return items


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-positive", type=int, default=N_POSITIVE)
    parser.add_argument("--n-negative", type=int, default=N_NEGATIVE)
    parser.add_argument(
        "--exclude-from",
        type=Path,
        default=None,
        help="Prior eval_sample.json whose image paths must not be resampled.",
    )
    args = parser.parse_args()

    excluded = load_excluded_paths(args.exclude_from)
    rng = random.Random(args.seed)
    positive = build_split(
        POSITIVE_DIR, "positive", True, args.n_positive, "pos", rng, excluded
    )
    negative = build_split(
        NEGATIVE_DIR, "negative", False, args.n_negative, "neg", rng, excluded
    )
    items = positive + negative
    rng.shuffle(items)

    # Re-number display order after shuffle while keeping stable ids.
    sample = {
        "seed": args.seed,
        "n_positive": args.n_positive,
        "n_negative": args.n_negative,
        "accuracy_threshold": ACCURACY_THRESHOLD,
        "instructions_schema": str(INSTRUCTIONS_PATH.resolve()),
        "dataset_positive_dir": str(POSITIVE_DIR.resolve()),
        "dataset_negative_dir": str(NEGATIVE_DIR.resolve()),
        "excluded_prior_images": len(excluded),
        "items": items,
    }
    blind = {
        "instructions_schema": str(INSTRUCTIONS_PATH.resolve()),
        "note": (
            "Present each image to the Sonnet 5 VLM agent with the instructions "
            "schema. The agent must return JSON matching the schema. Do not "
            "reveal ground-truth labels from eval_sample.json."
        ),
        "items": [{"id": it["id"], "path": it["path"]} for it in items],
    }

    predictions_skeleton = {
        "instructions_schema": str(INSTRUCTIONS_PATH.resolve()),
        "note": (
            "Fill each item.prediction with the Sonnet 5 VLM JSON response that "
            "matches filter_invalid_input_images_instructions.json. Leave no nulls."
        ),
        "items": [
            {
                "id": it["id"],
                "path": it["path"],
                "prediction": None,
            }
            for it in items
        ],
    }

    write_json(EVAL_SAMPLE_PATH, sample)
    write_json(BLIND_MANIFEST_PATH, blind)
    write_json(PREDICTIONS_PATH, predictions_skeleton)

    pos_cats = sorted({it["category"] for it in positive})
    neg_cats = sorted({it["category"] for it in negative})
    print(f"Wrote {EVAL_SAMPLE_PATH} ({len(items)} items)")
    print(f"Wrote {BLIND_MANIFEST_PATH}")
    print(f"Wrote {PREDICTIONS_PATH} (empty prediction stubs)")
    print(f"Positive categories sampled: {pos_cats}")
    print(f"Negative categories sampled: {neg_cats}")
    print(f"Data dir: {DATA_DIR}")


if __name__ == "__main__":
    main()

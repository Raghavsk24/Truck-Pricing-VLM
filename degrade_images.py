import os
import json
import random
from pathlib import Path
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm

INPUT_JSON = "truck_dataset.json"          # File containing your original JSON list
OUTPUT_JSON = "degraded_truck_dataset.json" # Output JSON with degraded image paths
OUTPUT_IMAGE_DIR = "degraded_truck_images"  # Where corrupted images are saved

os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# 1. Lens Blur & Severe Defocus
heavy_blur_pipeline = A.Compose([
    A.OneOf([
        A.GaussianBlur(blur_limit=(35, 55), p=1.0),
        A.Defocus(radius=(12, 22), alias_blur=(0.3, 0.6), p=1.0),
        A.MotionBlur(blur_limit=(31, 51), p=1.0)
    ], p=1.0)
])

# 2. Severe Underexposure
dark_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(-0.85, -0.65), contrast_limit=(-0.4, -0.1), p=1.0),
    A.GaussNoise(var_limit=(50.0, 120.0), p=1.0)
])

# 3. Severe Overexposure / Sunlight Flare
glare_pipeline = A.Compose([
    A.RandomBrightnessContrast(brightness_limit=(0.6, 0.85), contrast_limit=(0.3, 0.6), p=1.0),
    A.RandomSunFlare(flare_roi=(0, 0, 1, 0.6), angle_lower=0.5, src_radius=150, p=0.8)
])

# 4. Severe Downsampling / Pixelation
pixelate_pipeline = A.Compose([
    A.Downscale(scale_range=(0.05, 0.12), interpolation_pair={"downscale": cv2.INTER_NEAREST, "upscale": cv2.INTER_NEAREST}, p=1.0)
])

# 5. Extreme JPEG Compression
compression_pipeline = A.Compose([
    A.ImageCompression(quality_range=(3, 8), compression_type="jpeg", p=1.0)
])

def apply_image_sharding(img: np.ndarray) -> np.ndarray:
    corrupted = img.copy()
    h, w, c = corrupted.shape
    num_shards = random.randint(3, 7)

    for _ in range(num_shards):
        shard_type = random.choice(["blackout", "noise", "channel_shift", "slice_drop"])
        if random.random() > 0.4:
            y1 = random.randint(0, max(0, h - 30))
            y2 = min(h, y1 + random.randint(20, max(30, h // 4)))
            x1, x2 = 0, w
        else:
            x1 = random.randint(0, max(0, w - 30))
            x2 = min(w, x1 + random.randint(20, max(30, w // 4)))
            y1, y2 = 0, h

        if shard_type == "blackout":
            corrupted[y1:y2, x1:x2] = 0
        elif shard_type == "noise":
            corrupted[y1:y2, x1:x2] = np.random.randint(0, 256, (y2 - y1, x2 - x1, c), dtype=np.uint8)
        elif shard_type == "channel_shift":
            corrupted[y1:y2, x1:x2, random.randint(0, c - 1)] = 255
        elif shard_type == "slice_drop":
            shift = random.randint(20, 80)
            corrupted[y1:y2, :] = np.roll(corrupted[y1:y2, :], shift, axis=1)

    return corrupted

DEGRADATION_MODES = [
    ("blur", lambda img: heavy_blur_pipeline(image=img)["image"]),
    ("sharded", apply_image_sharding),
    ("underexposed", lambda img: dark_pipeline(image=img)["image"]),
    ("overexposed", lambda img: glare_pipeline(image=img)["image"]),
    ("pixelated", lambda img: pixelate_pipeline(image=img)["image"]),
    ("compressed", lambda img: compression_pipeline(image=img)["image"])
]

def degrade_and_update_json():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Ensure we work with a list of listing objects
    records = data if isinstance(data, list) else [data]
    degraded_records = []

    for entry in tqdm(records, desc="Processing Listings"):
        new_entry = dict(entry)  # Preserve original metadata keys (id, brand, price, currency, etc.)
        image_key = "images" if "images" in new_entry else "image_urls"
        original_images = new_entry.get(image_key, [])
        new_image_paths = []

        for idx, img_path in enumerate(original_images):
            # Load local image
            img = cv2.imread(img_path)
            if img is None:
                # If path is missing or unreadable, retain path and continue
                new_image_paths.append(img_path)
                continue

            # Pick a corruption method at random
            mode_name, transform_fn = random.choice(DEGRADATION_MODES)
            corrupted_img = transform_fn(img)

            # Generate target corrupted file name
            orig_name = Path(img_path).stem
            dest_filename = f"{orig_name}_{mode_name}_{idx}.jpg"
            dest_path = Path(OUTPUT_IMAGE_DIR) / dest_filename

            cv2.imwrite(str(dest_path), corrupted_img)
            new_image_paths.append(dest_path.as_posix())

        new_entry[image_key] = new_image_paths
        degraded_records.append(new_entry)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(degraded_records if isinstance(data, list) else degraded_records[0], f, indent=2)

    print(f"\nDone. Updated JSON written to {OUTPUT_JSON} with exact matching schema.")

if __name__ == "__main__":
    degrade_and_update_json()
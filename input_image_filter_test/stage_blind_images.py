"""Copy eval images to opaque filenames and resize for Sonnet vision.

Each source image is written as img_NNN.jpg with the long edge capped at
STAGE_MAX_LONG_EDGE (1568 px), aspect ratio preserved. WebP and other
formats are converted to JPEG so VLM attachments stay readable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageOps

from paths import STAGE_JPEG_QUALITY, STAGE_MAX_LONG_EDGE

ROOT = Path(__file__).resolve().parent / "data"
BLIND = ROOT / "blind_manifest.json"
STAGE = ROOT / "blind_images"
OUT = ROOT / "blind_staged_manifest.json"
MAP = ROOT / "blind_id_map.json"


def resize_for_vlm(src: Path, dst: Path) -> tuple[int, int, int, int]:
    """Write a JPEG whose long edge is at most STAGE_MAX_LONG_EDGE.

    Returns (src_w, src_h, dst_w, dst_h). Does not upscale.
    """
    with Image.open(src) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        src_w, src_h = image.size
        long_edge = max(src_w, src_h)
        if long_edge > STAGE_MAX_LONG_EDGE:
            scale = STAGE_MAX_LONG_EDGE / long_edge
            image = image.resize(
                (max(1, round(src_w * scale)), max(1, round(src_h * scale))),
                Image.Resampling.LANCZOS,
            )
        dst_w, dst_h = image.size
        dst.parent.mkdir(parents=True, exist_ok=True)
        image.save(dst, format="JPEG", quality=STAGE_JPEG_QUALITY, optimize=True)
    return src_w, src_h, dst_w, dst_h


def main() -> None:
    blind = json.loads(BLIND.read_text(encoding="utf-8"))
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)

    staged_items = []
    id_map = {}
    for i, it in enumerate(blind["items"], start=1):
        opaque_id = f"img_{i:03d}"
        src = Path(it["path"])
        dst = STAGE / f"{opaque_id}.jpg"
        src_w, src_h, dst_w, dst_h = resize_for_vlm(src, dst)
        staged_items.append(
            {
                "id": it["id"],
                "opaque_id": opaque_id,
                "path": str(dst.resolve()),
                "source_size": [src_w, src_h],
                "staged_size": [dst_w, dst_h],
            }
        )
        id_map[opaque_id] = it["id"]

    out = {
        "instructions_schema": blind["instructions_schema"],
        "max_long_edge": STAGE_MAX_LONG_EDGE,
        "jpeg_quality": STAGE_JPEG_QUALITY,
        "note": (
            "Blind staged copies with opaque filenames, resized to the Sonnet "
            "standard vision long-edge cap. Classify from image pixels only. "
            "Do not infer labels from filenames."
        ),
        "items": staged_items,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    MAP.write_text(json.dumps(id_map, indent=2) + "\n", encoding="utf-8")
    print(f"Staged {len(staged_items)} images into {STAGE}")
    print(f"Long-edge cap: {STAGE_MAX_LONG_EDGE}px, JPEG q={STAGE_JPEG_QUALITY}")
    print(f"Wrote {OUT}")
    print(f"Wrote {MAP}")


if __name__ == "__main__":
    main()

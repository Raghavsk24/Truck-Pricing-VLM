"""Merge VLM batch outputs into predictions.json using the opaque id map."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "data"
BATCHES = ROOT / "batches"
ID_MAP = ROOT / "blind_id_map.json"
STAGED = ROOT / "blind_staged_manifest.json"
OUT = ROOT / "predictions.json"


def main() -> None:
    id_map = json.loads(ID_MAP.read_text(encoding="utf-8"))  # opaque -> sample id
    staged = {
        it["opaque_id"]: it
        for it in json.loads(STAGED.read_text(encoding="utf-8"))["items"]
    }

    by_opaque: dict[str, dict] = {}
    for path in sorted(BATCHES.glob("batch_*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        for row in doc["items"]:
            opaque = row["opaque_id"]
            if opaque not in id_map:
                raise KeyError(f"{path.name}: unknown opaque_id {opaque}")
            by_opaque[opaque] = row["prediction"]

    missing = sorted(set(id_map) - set(by_opaque))
    if missing:
        raise RuntimeError(f"Missing predictions for: {missing}")

    items = []
    for opaque_id, sample_id in sorted(id_map.items(), key=lambda kv: kv[0]):
        items.append(
            {
                "id": sample_id,
                "path": staged[opaque_id]["path"],
                "prediction": by_opaque[opaque_id],
            }
        )

    out = {
        "instructions_schema": json.loads(STAGED.read_text(encoding="utf-8"))[
            "instructions_schema"
        ],
        "items": items,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Merged {len(items)} predictions into {OUT}")


if __name__ == "__main__":
    main()

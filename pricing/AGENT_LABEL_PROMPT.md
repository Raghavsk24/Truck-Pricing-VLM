# Pricing VLM labeling

Python staged the images. Fill each batch JSON with master-schema predictions.

## Schema

`vlm_instructions/truck_feature_extraction_master_instructions.json`

Return one JSON object per image matching **TruckFeatureExtractionMaster**.

## Staged data

- Manifest: `pricing/staged_manifest.json`
- Images: `pricing/staged_images/img_NNN.jpg` (long edge ≤ 1568px)
- Batches: `pricing/batches/batch_XX.json`

Classify from **image pixels only**. Do not use listing brand, price, or folder names when filling VLM fields.

## Batch format

Each item starts as:

```json
{
  "opaque_id": "img_001",
  "path": ".../staged_images/img_001.jpg",
  "prediction": null
}
```

Replace `prediction` with the full master-schema object (`reasoning` + `output` with truck_type, primary_subject, brand, condition).

Priceable images need `truck_type != none` and `primary_subject` in `{front, side}`. Otherwise still fill brand/condition with the schema skip placeholders.

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

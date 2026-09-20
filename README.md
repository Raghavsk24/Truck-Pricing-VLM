# Truck-Pricing VLM

Kamion (YC S22) is a marketplace that connects truck sellers with buyers. However, truck sellers are responsible for setting the price of their own trucks, and many sellers lie about the quality of their truck and price it higher than it's market value. Consequently, many buyers are scammed as they end up buying overpriced trucks with degraded quality. We built _**Truck-Pricing-VLM**_ to solve this. We use a heavily fine-tuned Claude Sonnet 4.6 Vision Language Model to extract features from the image of a truck (condition, brand, type, primary subject). Each features is mapped to a value, which aggregates to form a vector. The vector is run on a quantile regression model, which estimates the price range of the truck. Additionally, an appraisal report with a condition assessment and a statement of reasoning is prepared to back up the pricing valuation. 


<p align="center">
  <em>Built by Raghav Senthil Kumar, Krishiv Nandakumar, Adhithya Kota and Akash.</em>
</p>

## The problem

Kamion matches sellers who have a used truck with buyers who will actually pay for it. Pricing is the bottleneck. Sellers often copy a number from a neighboring listing, or ask for what they owe on the truck, not what the market will clear. Buyers bounce when the ask is outside a believable band. Kamion needs a pricing algorithm that can look at the same photos a buyer sees and return a range that is specific to **what the truck is** (day cab, sleeper, dump), **who built it** (Freightliner, Kenworth, Peterbilt, International, Mack), and **what shape it is in** (frame rust, front-end damage, tread, cab/aero).

That is the only job of this engine: turn photos into a defensible `[low, high]` USD range so more trucks sell.

## Tech Stack

### Data collection

![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![curl_cffi](https://img.shields.io/badge/curl__cffi-000000?style=for-the-badge&logo=curl&logoColor=white)
![Pillow](https://img.shields.io/badge/Pillow-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TruckPaper](https://img.shields.io/badge/TruckPaper-1B4F72?style=for-the-badge&logo=databricks&logoColor=white)

### Vision

![Claude](https://img.shields.io/badge/Claude_Sonnet-D97706?style=for-the-badge&logo=anthropic&logoColor=white)
![JSON Schema](https://img.shields.io/badge/JSON_Schema-000000?style=for-the-badge&logo=json&logoColor=white)
![RLHF](https://img.shields.io/badge/RLHF-6B21A8?style=for-the-badge&logo=openai&logoColor=white)

### Pricing

![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=for-the-badge&logo=plotly&logoColor=white)
![JSON](https://img.shields.io/badge/price__range__model.json-000000?style=for-the-badge&logo=json&logoColor=white)

## System Architecture

```
  TruckPaper listings
  (Class 7/8 + Class 2–6)
            │
            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Dataset                                                │
  │    scrape  →  audit (price / USD / SHA-256 dedup)       │
  └─────────────────────────────────────────────────────────┘
            │
            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Supervised Class 7/8 filter                            │
  │    gold labels from scrape folders                      │
  │    VLM truck_type ∈ {day-cab, dump, sleeper, none}      │
  └─────────────────────────────────────────────────────────┘
            │  valid Class 7/8 only
            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  RLHF VLM tasks  (one master schema, four stages)       │
  │    1. truck type / validity                             │
  │    2. primary subject  (must be front or side)          │
  │    3. brand            (read off a badge, never guess)  │
  │    4. condition        (4 categories → penalty %)       │
  └─────────────────────────────────────────────────────────┘
            │  type + brand + category penalties
            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Price range model                                      │
  │    μ(type, brand) + condition effect  →  [low, high]    │
  └─────────────────────────────────────────────────────────┘
            │
            ▼
       USD asking range for the listing
```

## Dataset

Market comps come from [TruckPaper](https://www.truckpaper.com) listing photos, not from synthetic data. The scraper in `image_scraper/scripts/truckpaper_image_scraper.py` impersonates Chrome TLS with `curl_cffi` so Cloudflare does not serve a JS challenge, then walks allowlisted category searches.

### How it was collected

Every search is scoped to one category and one GVWR class. Positives are Class 7/8 (GVWR 26,001 lb and up). Negatives are Class 2–6 and exist so the filter has hard counterexamples.

| Split | Categories | Class mix | Image target |
| --- | --- | --- | --- |
| Positive (Class 7–8) | sleeper, day cab, dump | 15% Class 7, 85% Class 8 | 10,000 full / 200 test |
| Negative (Class 2–6) | 1-ton and 3/4-ton pickups, cargo vans, step vans, cutaway cubes, moving and cargo box trucks, stake, flatbed, service/utility | 45 / 22 / 18 / 10 / 5% across Classes 6→2 | 5,000 full / 100 test |

Inside each category the scrape is stratified again:

- **Year buckets:** 1900–2012, 2013–2017, 2018–2021, 2022–present
- **Price terciles:** 45% cheap / 30% mid / 25% high / 10% no listed price, so the model is not trained only on premium trucks
- **Brand cap:** no brand may take more than 30% of a stratum
- **Listing rules:** at least 5 photos, real asking price (auctions and "Call for Price" are skipped), trucks only (no trailers, bodies-only, or dismantled units)

Listings are named by category, not by TruckPaper's numeric ID (`sleeper_truck_07/sleeper_truck_07_image_03.jpg`), and a manifest (`truckpaper_scraped_listings.json`) stores `id`, `brand`, `price`, `currency`, and image paths. Re-runs resume from `.state/` so a `--full` job continues a test scrape instead of starting over.

The priced Class 7/8 population used by the model is **358 listings** after cleanup:

| Type | n | Typical log-price center |
| --- | ---: | ---: |
| Dump | 142 | ~$83k |
| Day cab | 130 | ~$31k |
| Sleeper | 86 | ~$36k |

| Brand cell | n |
| --- | ---: |
| Freightliner | 112 |
| International | 61 |
| Kenworth | 58 |
| Peterbilt | 52 |
| Mack | 42 |
| Other | 33 |

### How it was audited

`image_scraper/scripts/audit_truckpaper_image_dataset.py` is a destructive cleanup that defaults to a dry run. Three checks run in order, each dropping a listing or a single image:

1. **Bad price** — `price` is null (TruckPaper showed "Call for Price" or a non-positive number). A price model cannot train on a missing target.
2. **Non-USD currency** — anything other than exactly `USD` is dropped so the price column is comparable.
3. **Duplicate images** — SHA-256 of file bytes. A hash that repeats inside one listing, or across listings in the same top-level split (positive or negative), is treated as a placeholder or a copy. The first occurrence is kept. A hash shared *between* the two splits is left alone.

After dedup, surviving images are renamed to close gaps (`_image_01`, `_image_02`, …) and the manifest is rewritten. A listing left with zero images is dropped. Apply with `--apply`.

## Supervised filter, then RLHF on the VLM tasks

The vision stack is not one prompt. It is four tasks that were written as separate JSON schemas, scored against labels, rewritten when they failed, and finally fused into `vlm_instructions/truck_feature_extraction_master_instructions.json`.

### Stage 0 — supervised learning for the initial filter

The scrape folders *are* the gold labels: every image under `positive (class 7 - 8)/` is a valid Class 7/8 truck (or a part of one); every image under `negative (class 2 - 6)/` is not. That gives a supervised classification problem before any human has to sit down and tag photos.

`input_image_filter_test/` builds a stratified blind eval set, resizes the long edge to 1568 px, and hides folder names behind opaque IDs (`img_001.jpg`). A Sonnet agent fills `truck_type` from pixels only. Validity is implied:

| `truck_type` | Meaning |
| --- | --- |
| `day-cab-truck` | Valid Class 7/8 day cab or heavy conventional cab |
| `dump-truck` | Valid Class 7/8 dump / hopper / grain body |
| `sleeper-truck` | Valid Class 7/8 sleeper box or sleeper interior |
| `none` | Invalid — Class 2–6, non-truck, or a part that only those lighter vehicles have |

`evaluate_predictions.py` scores the agent against the gold folders (schema validity, Class 7/8 accuracy, exact type accuracy, confusion matrix). Those errors are the supervised signal. Instructions get rewritten — close-ups of duals and dump tailgates must count as valid, Super Duty chassis-cabs and enclosed box bodies must not — and the eval is re-run. That is the first filter every later task sits behind.

### Stages 1–3 — RLHF loops on the remaining VLM tasks

Once the filter is stable, three more schemas are refined the same way: draft a strict JSON contract, have the VLM label a blind batch, compare the output to human judgment, change the rules that caused the miss, repeat.

**Primary subject.** Center-weighted composition, not raw pixel area. Allowed labels: `front`, `side`, `back`, `engine`, `container`, `interior`, `dashboard`, `tires`, `unusable`. Background trucks in a yard are normal and do not make an image unusable. The pricing path only accepts `front` or `side` — those are the shots where type, badge, and condition are actually visible.

**Brand.** Read a logo, hood ornament, badge, or wordmark off the cab door, grille, roof cap, or steering wheel. Cross-check the string against US truck OEMs (Freightliner, Peterbilt, Kenworth, Mack, International/Navistar, Volvo, Western Star, plus vocational builders). Never guess from grille shape or paint. If nothing is readable, set `needs_user_input` and ask the seller — do not invent a brand.

**Condition.** Four categories, scored 1–5 from visible evidence only, each with a penalty percent:

| Category | Weight | What it inspects | Max penalty |
| --- | ---: | --- | ---: |
| Chassis and frame | 35% | Frame rails, leaf springs, hangers, U-bolts | 30% |
| Front end / hood / engine bay | 25% | Grille, bumper, hood, visible engine bay | 10% |
| Tires / wheels / suspension | 20% | Tread, rims, visible wear | 8% |
| Cab / sleeper / aero | 20% | Interior (only if shown) and fairings | 5% |

A part that is out of frame is `null` and dropped from the weighted average — the model does not invent rust it cannot see.

### One master call

The four tasks run in a fixed order inside a single Sonnet call (`reasoning`, then `output`). Brand and condition are skipped unless `truck_type ≠ none` and `primary_subject ∈ {front, side}`. That is the definition of a **priceable** image. Everything else is rejected with a short user message asking for a better photo.

Human corrections on failed batches are written back into the schemas. That is the RLHF loop: the reward is agreement with gold labels (filter) or with the written rubric (subject, brand, condition), and the policy that gets updated is the instruction JSON, not a weight file.

## Pricing algorithm

The VLM does not predict price. It predicts features. `pricing/fit_price_range.py` fits a log-price model on the 358 priced Class 7/8 listings, then uses the VLM labels to estimate how condition moves that price.

### Step 1 — type and brand tables

For every listing with a USD ask and a brand, take `log(price)`. Shrink each cell toward its parent so a rare `dump × Mack` row does not overfit:

```
μ_cell = (n / (n + k)) · raw_μ + (1 − n / (n + k)) · parent_μ
```

`k` (shrinkage) is grid-searched. The type table is the parent of each type×brand cell. Brands outside Freightliner, International, Kenworth, Peterbilt, and Mack collapse to `OTHER`.

### Step 2 — blend type and brand

```
μ = brand_blend · μ_(type, brand) + (1 − brand_blend) · μ_type
```

`brand_blend = 1` in the saved model: when a cell exists, use it; otherwise fall back to the type mean.

### Step 3 — condition effect

The four VLM penalties are combined with the rubric weights (chassis 0.35, front 0.25, tires 0.20, cab 0.20). Ordinary least squares then fits `β` on the labeled set:

```
log(price) − μ  =  intercept + β · (weighted_penalty − penalty_bar)
```

`β` is negative: a dirtier truck is cheaper than its type×brand peers. `condition_scale` stretches or shrinks that effect and is tuned with the other weights.

### Step 4 — point estimate and range

```
center = exp( μ + condition_scale · β · (weighted_penalty − penalty_bar) )
low    = center · (1 − error_bound)
high   = center · (1 + error_bound)
```

`error_bound` is the leave-one-out median absolute percentage error on the labeled trucks (about **±36%** in the current fit). The range is the number Kamion can show a seller: not a single lucky point, a band the market has actually cleared.

Weights (`shrink_k`, `brand_blend`, `condition_scale`, category mix) are chosen by grid search to minimize that LOO median APE. The winner is written to `pricing/price_range_model.json`.

### Step 5 — inference

`pricing/predict.py` accepts a photo, a filled VLM JSON, or explicit `--truck-type` / `--brand` / `--penalty` flags. Photos are resized and run through the master schema. Non-priceable images exit with a user message. Missing brand exits and asks the seller. Otherwise the model returns `center`, `low`, `high`, and the features that produced them.

```
python pricing/predict.py --image path/to/truck.jpg
python pricing/predict.py --truck-type day_cab --brand FREIGHTLINER --penalty 4.0
```

## Getting Started

### Prerequisites

- **Python 3.11+** — [python.org](https://www.python.org)
- **Cursor** (or any Claude Sonnet workspace) for in-chat VLM labeling — no separate API key is required for the batch workflow

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USER/54hackathon.git
cd 54hackathon
python -m venv .venv
```

**macOS / Linux:** `source .venv/bin/activate`

**Windows:** `.venv\Scripts\activate`

```bash
pip install -r image_scraper/requirements.txt
```

### 2. Scrape a dataset

```bash
python image_scraper/scripts/truckpaper_image_scraper.py            # 200 + 100 image test
python image_scraper/scripts/truckpaper_image_scraper.py --full     # 10k + 5k
python image_scraper/scripts/truckpaper_image_scraper.py plan       # print strata and quotas
```

### 3. Audit it

```bash
python image_scraper/scripts/audit_truckpaper_image_dataset.py           # dry run
python image_scraper/scripts/audit_truckpaper_image_dataset.py --apply
```

### 4. Evaluate the supervised filter

```bash
python input_image_filter_test/sample_eval_set.py
python input_image_filter_test/stage_blind_images.py
# fill data/predictions.json from the filter schema, then:
python input_image_filter_test/evaluate_predictions.py
```

### 5. Label priceable photos and fit the range

```bash
python pricing/build_sample.py
python pricing/label_sample.py
# fill pricing/batches/batch_XX.json with the master schema
python pricing/merge_batch_labels.py
python pricing/fit_price_range.py
```

### 6. Predict

```bash
python pricing/predict.py --image path/to/truck.jpg
```

## License

MIT

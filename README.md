# Truck-Pricing VLM

Kamion (YC S22) is a marketplace that connects truck sellers with buyers. However, truck sellers are responsible for setting the price of their own trucks, and many sellers lie about the quality of their truck and price it higher than it's market value. Consequently, many buyers are scammed as they end up buying overpriced trucks with degraded quality. We built _**Truck-Pricing-VLM**_ to solve this. We use a heavily fine-tuned Claude Sonnet 4.6 Vision Language Model to extract features from the image of a truck (condition, brand, type, primary subject). Each features is mapped to a value, which aggregates to form a vector. The vector is run on a quantile regression model, which estimates the price range of the truck. Additionally, an appraisal report with a condition assessment and a statement of reasoning is prepared to back up the pricing valuation. 

- **Live App:** https://truck-pricing-vlm.vercel.app/


<p align="center">
  <em>Built by Raghav Senthil Kumar, Krishiv Nandakumar, Adhithya Kota and Akash.</em>
</p>

## Tech Stack

### Frontend

[![Next.js](https://img.shields.io/badge/Next.js_16-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS_v4-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com)
[![Sonner](https://img.shields.io/badge/Sonner-000000?style=for-the-badge&logo=npm&logoColor=white)](https://sonner.emilkowal.ski)
[![shadcn/ui](https://img.shields.io/badge/shadcn/ui-000000?style=for-the-badge&logo=shadcnui&logoColor=white)](https://ui.shadcn.com)

### Backend

[![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)](https://supabase.com)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org)
[![Python](https://img.shields.io/badge/Python_3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![Vercel](https://img.shields.io/badge/Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)](https://git-scm.com)

### Data / AI / ML

[![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)
[![Claude](https://img.shields.io/badge/Claude_Sonnet_4.6_API-D97757?style=for-the-badge&logo=anthropic&logoColor=white)](https://anthropic.com)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/Raghavsk24/truckpaper_scraped_images_dataset)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)

## System Architecture

```
  TruckPaper listings
  (Class 7/8 + Class 2–6)
            │
            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Dataset                                                │
  │    scrape  →  audit (price / USD / SHA-256 dedup)       │
  │    published to Hugging Face                            │
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
  │  RLHF VLM tasks  (one master schema, four stages)        │
  │    1. truck type / validity                             │
  │    2. primary subject  (must be front or side)          │
  │    3. brand            (read off a badge, never guess)  │
  │    4. age               (era bucket, photo only)         │
  │    5. condition        (4 categories → 1–5 score)        │
  └─────────────────────────────────────────────────────────┘
            │  type + brand + era + condition score
            ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Price range model                                      │
  │    μ(type, brand, era) + condition effect → [low, high] │
  │    price_range_model.json (Python) — trained offline    │
  └─────────────────────────────────────────────────────────┘
            │
            ├──────────────────────────────┐
            ▼                              ▼
   pricing/predict.py (CLI)      web/ Next.js app (Vercel)
   photo or flags in → range out   upload → chat → report/PDF
```

## Dataset

Market comps come from [TruckPaper](https://www.truckpaper.com) listing photos, not from synthetic data. The full labeled image set — the same scrape this model is trained on — is published on Hugging Face: **[Raghavsk24/truckpaper_scraped_images_dataset](https://huggingface.co/datasets/Raghavsk24/truckpaper_scraped_images_dataset)**. The scraper in `image_scraper/scripts/truckpaper_image_scraper.py` impersonates Chrome TLS with `curl_cffi` so Cloudflare does not serve a JS challenge, then walks allowlisted category searches.

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

The vision stack is not one prompt. It is five tasks that were written as separate JSON schemas, scored against labels, rewritten when they failed, and finally fused into `vlm_instructions/truck_feature_extraction_master_instructions.json`.

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

### Stages 1–4 — RLHF loops on the remaining VLM tasks

Once the filter is stable, four more schemas are refined the same way: draft a strict JSON contract, have the VLM label a blind batch, compare the output to human judgment, change the rules that caused the miss, repeat.

**Primary subject.** Center-weighted composition, not raw pixel area. Allowed labels: `front`, `side`, `back`, `engine`, `container`, `interior`, `dashboard`, `tires`, `unusable`. Background trucks in a yard are normal and do not make an image unusable. The pricing path only accepts `front` or `side` — those are the shots where type, badge, and condition are actually visible.

**Brand.** Read a logo, hood ornament, badge, or wordmark off the cab door, grille, roof cap, or steering wheel. Cross-check the string against US truck OEMs (Freightliner, Peterbilt, Kenworth, Mack, International/Navistar, Volvo, Western Star, plus vocational builders). Never guess from grille shape or paint. If nothing is readable, set `needs_user_input` and ask the seller — do not invent a brand.

**Age / era.** The VLM buckets the truck into one of four eras from body style, badge design, and cab shape alone (`pre_2010`, `2010_2015`, `2016_2020`, `2021_plus`) and states its own confidence (`high` / `medium` / `low`). A photo era call is a guess, not a fact, so the price model never trusts it outright — see [Step 2](#step-2--era-as-a-soft-posterior-not-a-hard-bucket) below.

**Condition.** Four categories, scored 1–5 from visible evidence only, each with a penalty percent:

| Category | Weight | What it inspects | Max penalty |
| --- | ---: | --- | ---: |
| Chassis and frame | 35% | Frame rails, leaf springs, hangers, U-bolts | 30% |
| Front end / hood / engine bay | 25% | Grille, bumper, hood, visible engine bay | 10% |
| Tires / wheels / suspension | 20% | Tread, rims, visible wear | 8% |
| Cab / sleeper / aero | 20% | Interior (only if shown) and fairings | 5% |

A part that is out of frame is `null` and dropped from the weighted average — the model does not invent rust it cannot see. The four category scores also roll up into a single `overall_score` (1–5), which is what the price model actually regresses on (see [Step 3](#step-3--condition-effect)).

### One master call

The five tasks run in a fixed order inside a single Sonnet call (`reasoning`, then `output`). Brand, age, and condition are skipped unless `truck_type ≠ none` and `primary_subject ∈ {front, side}`. That is the definition of a **priceable** image. Everything else is rejected with a short user message asking for a better photo.

Human corrections on failed batches are written back into the schemas. That is the RLHF loop: the reward is agreement with gold labels (filter) or with the written rubric (subject, brand, age, condition), and the policy that gets updated is the instruction JSON, not a weight file.

## Pricing algorithm

The VLM does not predict price. It predicts features. `pricing/fit_price_range.py` fits a hierarchical log-price model on the 358 priced Class 7/8 listings, then uses the VLM labels to estimate how condition moves that price. The model is tuned in two stages so a handful of hand-labeled photos never gets to overfit the whole population.

### Step 1 — four-level shrinkage: global → type → brand → era

For every listing with a USD ask and a brand, take `log(price)`. Each level is pulled toward its parent so a rare `dump × Mack × 2021_plus` cell does not overfit on one listing:

```
μ_type  = shrink(type rows,             global,  k_type)
μ_brand = shrink((type, brand) rows,    μ_type,  k_brand)
μ_era   = shrink((type, brand, era) rows, μ_brand, k_era)

shrink(rows, parent, k) = w·mean(rows) + (1 − w)·parent,   w = n / (n + k)
```

A brand with one listing barely moves off its type baseline; a brand with sixty listings is trusted almost fully — that is what lets the model use the full brand list instead of a five-brands-plus-`OTHER` bucket. `k_type`, `k_brand`, and `k_era` are grid-searched (Stage A) on all 358 listings, since type, brand, and year all come from scrape metadata and need no VLM labels at all. The winning fit uses `k_type = k_brand = 8`, `k_era = 1`.

### Step 2 — era as a soft posterior, not a hard bucket

At **training** time, era comes from the real TruckPaper model year, so it is exact. At **inference** time, era comes from a VLM guess off a photo, which can miss the bucket. Rather than trust the call outright, the stated confidence is converted into a probability distribution over the named era and its two neighbors:

```
ERA_TRUST = { high: 0.80, medium: 0.55, low: 0.34 }
```

measured against how often each confidence level was actually correct on the labeled sample (`high` 5/5, `medium` 7/11, `low` 0/4 — but never off by more than one bucket). The remaining probability mass splits evenly across the adjacent eras, and `μ_era` is the probability-weighted average of the era cell over that posterior. A shaky era call therefore pulls the price gently toward its neighbors instead of committing hard to a bucket the VLM was not sure about.

### Step 3 — condition effect

Condition is fit *after* the cells, and only on the VLM-labeled rows (Stage B) — tuning everything on the small labeled set would overfit it. OLS regresses the residual log-price on the centered `overall_score` (1–5):

```
log(price) − μ_era  =  intercept + β · (overall_score − scorē)
```

`β` is fit once, then a `condition_scale` multiplier (grid-searched over `[0, 0.25, …, 2.0]`) rescales its effect against labeled leave-one-out error. In the current fit `β ≈ +0.299` (`n = 30` labeled trucks) — condition score and price move together, as expected, though at this sample size the effect is directionally right but not yet statistically significant.

### Step 4 — point estimate and adaptive range

```
center = exp( μ_era + condition_scale · β · (overall_score − scorē) )
low    = max(0, center · (1 − error_bound))
high   = center · (1 + error_bound)
```

`error_bound` is **adaptive**, not a single global number: it is the median leave-one-out absolute-percentage-error within the truck's own `(type, era)` cell, falling back to `type`, then the global pool, whenever a cell is too thin (`n < 20`) or wider than its parent. Every bound is clamped to `[15%, 40%]` so a noisy cell never claims unrealistic precision, and a wide one never collapses the low end to $0. Current published bounds: **day cab ±29%**, **dump ±25%**, **sleeper ±30%** (global fallback ±30%).

### Step 5 — what actually moves the number

Weights (`k_type`, `k_brand`, `k_era`, `condition_scale`) are chosen by grid search to minimize leave-one-out median APE, evaluated twice: once on the full 358-listing population (cells only) and once on the 30 labeled trucks (cells + condition), with an ablation printed for each feature in isolation:

| Feature set | Population LOO median APE |
| --- | ---: |
| Type only | highest |
| Type + brand | lower |
| Type + era | lower |
| Type + brand + era (shipped) | **~30%** |

The final model also reports how often the published range actually contains the real asking price on held-out labeled trucks (**~67% coverage** at `n = 30` — read as an early signal, not a guarantee, given the label count). The winning weights are written to `pricing/price_range_model.json`.

### Step 6 — inference

`pricing/predict.py` accepts a photo, a filled VLM JSON, or explicit `--truck-type` / `--brand` / `--penalty` flags. Photos are resized and run through the master schema. Non-priceable images exit with a user message. Missing brand exits and asks the seller. Otherwise the model returns `center`, `low`, `high`, and the features that produced them.

```bash
python pricing/predict.py --image path/to/truck.jpg
python pricing/predict.py --truck-type day_cab --brand FREIGHTLINER --penalty 4.0
```

The same formula is ported line-for-line to TypeScript in `web/lib/pricing.ts` so the deployed app can price a photo without shelling out to Python — see below.

## Web application

`web/` is a Next.js 16 / React 19 app that puts the whole pipeline behind a conversational, upload-to-report flow, deployed on Vercel at **[truck-pricing-vlm.vercel.app](https://truck-pricing-vlm.vercel.app)**. It does not retrain anything — it loads the same `price_range_model.json` the Python pipeline produced and the same master VLM schema, and calls Claude directly per request.

### The flow

1. **Landing** (`components/Landing.tsx`) — pitch screen with a "start an appraisal" CTA and a canned sample report so a first-time visitor can see the output before uploading anything.
2. **Chat** (`components/Conversation.tsx`) — the seller drags in photos one at a time. Each upload is resized client-side (`lib/resize-client.ts`, long edge 1568 px, matching the training pipeline) and POSTed to `/api/analyze`.
3. **Server-side inspection** (`app/api/analyze/route.ts` → `lib/vlm.ts`) — the server re-resizes with `sharp` (`lib/image.ts`), uploads the JPEG to Supabase Storage, and sends it to Claude (`@anthropic-ai/sdk`, model `claude-sonnet-4-5-20250929` by default) with the exact master schema from `vlm_instructions/truck_feature_extraction_master_instructions.json` as the system prompt. The response is parsed and defensively coerced back into typed `VlmOutput` (`lib/vlm.ts#parseVlmResult`) so a malformed field never crashes pricing.
4. **Branching on priceability** (`lib/pipeline.ts`) —
   - Not a Class 7/8 front/side shot → `rejected`, with the VLM's own rejection message shown back to the seller.
   - Priceable but no readable brand → `needs_brand`: the chat asks the seller directly, with OEM quick-reply chips (Freightliner, Kenworth, Peterbilt, Mack, International, Volvo), and re-prices via `/api/price` once they answer.
   - Priceable with a brand → priced immediately with `lib/pricing.ts#priceFromVlm`, a TypeScript port of the Python model (`predict_center`, hierarchical `μ` lookup, adaptive error bound) that reads `web/data/price_range_model.json` directly instead of shelling out to Python.
5. **Persistence** (`lib/supabase.ts`, `web/supabase/schema.sql`) — every analysis (rejected, needs-brand, or priced) is written to a Postgres `analyses` table with the full VLM JSON, the uploaded image path in a private Storage bucket, and the resulting range, so a report can be reloaded by ID later.
6. **Report** (`components/Report.tsx`) — the priced range as a labeled axis (`lib/appraisal.ts#axisBounds`), a per-category condition breakdown with USD contribution attribution (`lib/pricing.ts#splitConditionUsd` walks each category's pull away from the score bar), a photo checklist showing which shots were actually used vs. filtered, and a `recharts` contribution chart (`components/ContributionChart.tsx`) showing how type, brand, era, and each condition category moved the price off the population baseline.
7. **Condition report PDF** (`components/ConditionPdf*.tsx` → `lib/appraisal.ts`, rendered with `@react-pdf/renderer`) — the same report data exported as a print-ready PDF the seller can hand to a buyer.

### API surface

| Route | Method | Purpose |
| --- | --- | --- |
| `/api/analyze` | `POST` (multipart, `image`) | Resize → VLM extract → price (if brand known) → persist → return `AnalyzeResponse` |
| `/api/price` | `POST` (JSON, `{ analysisId, brand }`) | Re-price a `needs_brand` analysis once the seller supplies the brand |

Both routes return one of `AnalyzeOk`, `AnalyzeNeedsBrand`, `AnalyzeRejected`, or `AnalyzeError` (`lib/types.ts`), which the UI switches on directly — there is no separate error-handling path.

## Repo layout

```
image_scraper/          TruckPaper scraper + audit script + raw manifest
input_image_filter_test/  Stage-0 supervised Class 7/8 filter: sampling, staging, scoring
pricing/                 Label sampling, price model fit, CLI inference, saved model + plot
vlm_instructions/        The five VLM task schemas, fused into the master schema
web/                     Next.js appraisal app (UI, API routes, Supabase, PDF report)
```

## Getting Started

### Prerequisites

- **Python 3.11+** — [python.org](https://www.python.org)
- **Node.js 18+** and **npm** — [nodejs.org](https://nodejs.org) (for `web/`)
- **Cursor** (or any Claude Sonnet workspace) for in-chat VLM labeling during the offline pipeline — no separate API key is required for the batch workflow
- An **Anthropic API key** and a **Supabase project** if you want to run the web app (see below)

### 1. Clone

```bash
git clone https://github.com/Raghavsk24/Truck-Pricing-VLM.git
cd Truck-Pricing-VLM
```

### 2. Data pipeline (Python)

```bash
python -m venv .venv
```

**macOS / Linux:** `source .venv/bin/activate` · **Windows:** `.venv\Scripts\activate`

```bash
pip install -r image_scraper/requirements.txt
```

Scrape a dataset (or just use the [published Hugging Face dataset](https://huggingface.co/datasets/Raghavsk24/truckpaper_scraped_images_dataset) directly):

```bash
python image_scraper/scripts/truckpaper_image_scraper.py            # 200 + 100 image test
python image_scraper/scripts/truckpaper_image_scraper.py --full     # 10k + 5k
python image_scraper/scripts/truckpaper_image_scraper.py plan       # print strata and quotas
```

Audit it:

```bash
python image_scraper/scripts/audit_truckpaper_image_dataset.py           # dry run
python image_scraper/scripts/audit_truckpaper_image_dataset.py --apply
```

Evaluate the supervised filter:

```bash
python input_image_filter_test/sample_eval_set.py
python input_image_filter_test/stage_blind_images.py
# fill data/predictions.json from the filter schema, then:
python input_image_filter_test/evaluate_predictions.py
```

Label priceable photos and fit the range:

```bash
python pricing/build_sample.py
python pricing/label_sample.py
# fill pricing/batches/batch_XX.json with the master schema
python pricing/merge_batch_labels.py
python pricing/fit_price_range.py
```

Predict from the CLI:

```bash
python pricing/predict.py --image path/to/truck.jpg
```

### 3. Web app (Next.js)

```bash
cd web
npm install
cp .env.example .env.local
```

Fill in `.env.local`:

```
ANTHROPIC_API_KEY=
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929
NEXT_PUBLIC_SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
```

Create the Supabase table and storage bucket by running `web/supabase/schema.sql` once in the Supabase SQL editor, then start the dev server:

```bash
npm run dev
```

Or from the repo root (delegates into `web/`):

```bash
npm run dev
npm run build
npm run start
```

The app deploys to Vercel with `vercel.json` pointing the build at `web/`.

## License

MIT

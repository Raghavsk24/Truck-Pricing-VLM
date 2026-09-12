# truckval

Price a used commercial truck from seller photographs. Returns a range, a
condition report, and a reasoning trace — and refuses to price a photo of a
motorcycle instead of confidently guessing.

Runs end to end today on synthetic data, with no API key and no network.

```bash
pip install -r requirements.txt
python scripts/make_synth_comps.py     # 12k synthetic realized sales
python scripts/train_pricing.py        # quantile models + conformal calibration
python scripts/demo.py                 # six scenarios across all three tiers
python scripts/eval_calibration.py     # coverage with fields deliberately hidden
python -m pytest tests/ -q
```

---

## The one decision everything else follows from

**The vision model never outputs a price.**

Ask a VLM what a truck is worth and it will confidently say $32,000 for a photo
of a shed. So the system splits in two: a perception stage that extracts
*evidence*, and a pricing stage that turns evidence into money. Perception is
allowed to say "I can't see the tires." Pricing is allowed to respond "then the
range is wider."

That split is what makes the refusal requirement fall out for free rather than
being bolted on as a guard rail.

```
photos + typed details
        │
        ▼
   ┌─────────┐   wrong object / unusable / two vehicles
   │  gate   │──────────────────────────────────► Tier 0: refuse, name what you saw
   └────┬────┘
        ▼
  ┌────────────┐  structured evidence, every field carries
  │ perception │  visibility + confidence + source photo
  └─────┬──────┘
        ▼
  ┌────────────┐  VIN decode, claim cross-check
  │  identity  │──────────────────────────────────► Tier 1: condition report, no price
  └─────┬──────┘   (no identity → no comp set → no price, hard gate)
        ▼
  ┌────────────┐  LightGBM quantile regression at p10/p50/p90
  │  pricing   │  on realized auction prices, conformally calibrated
  └─────┬──────┘
        ▼
  ┌─────────────┐ Monte Carlo over the unknowns, composed with
  │ uncertainty │ the model's own spread ─────────► Tier 1 if excess too high
  └─────┬───────┘
        ▼
  ┌────────────┐  rendered from the adjustment ledger; every
  │   report   │  claim points back to a photo
  └────────────┘  Tier 2: price + condition + reasoning + next photo to take
```

---

## Three things that are not obvious until you build it

### 1. Quantile models come out overconfident. Conformalize them.

Nominal 80% intervals covered **63.5%** on held-out data. Gradient-boosted
quantile regression systematically shrinks the outer quantiles toward the
median, and tuning does not reliably fix it. Conformalized quantile regression
(Romano et al., 2019) on a third data split took it to **79.5%** with a
finite-sample marginal coverage guarantee and no distributional assumptions.

The offset lands around 0.071 in log space — a 7.3% multiplicative widening on
price, which is the right shape when a $200k tractor and a $9k box truck share
one model. See `QuantilePricer.calibrate`.

The calibration split must be data used for **neither** training nor final
evaluation, or the guarantee is worthless and you have simply overfitted your
error bars.

### 2. An absolute width threshold suppresses every price you would ever make.

"Don't show a price if the interval exceeds 40% of the midpoint" sounds
reasonable and is wrong. Used commercial trucks genuinely trade in a ~52% band
at 80% confidence. That is market spread, not ignorance, and no amount of
photography narrows it.

The gate measures **excess over the irreducible floor** instead — the width the
range would have if every unknown field were resolved. See
`UncertaintyResult.excess_ratio`.

```
evidence                                  low       high      width   excess
12 photos, everything visible          14,767     27,468     12,701      1%
2 side shots, nothing else             10,755     26,757     16,002     25%
```

Same truck, same market. The difference is entirely missing photographs. That
table is the product.

### 3. The ledger double-charges for ordinary wear unless you net it.

An auction hammer price is the price of a truck in whatever condition it
happened to be in. So the comps baseline is "this model in *average* condition
for its age and distance," not "this model in perfect condition." Deducting
absolute reconditioning cost on top charges the seller twice.

`ledger.expected_condition_cost` estimates the cohort mean by running the ledger
under the priors with every condition field unknown — self-consistent, needs no
extra data. Netting removes the bias and leaves the spread intact, which is
exactly right: uncertainty should widen the range, not move it.

Effect: overall coverage 66% → **76.7%**, median absolute error $8,095 → $5,917.

### 4. Typed claims must be sampled, not pinned.

Merging a seller's typed odometer as an `OBSERVED` field at confidence 0.55
recorded the doubt and then ignored it — `unknown_fields()` skipped it, so the
Monte Carlo pinned it. An unverified number typed by someone with a financial
interest in a high price carried the same weight as an odometer read off the
dash.

`Visibility.CLAIMED` is now a distinct level. Claimed fields are returned by
`unknown_fields()` and sampled from a claim-anchored mixture: mostly near the
claim, sometimes from the unconditional prior. A field the cross-check flagged
as disputed drops from 80% to 25% honesty and the range widens accordingly.
That connects `identity.cross_check` to the pricing math instead of leaving it
as a warning message.

### 5. Guardrails degrade the field, not the response.

`validate.py` runs two layers on the one JSON contract a model authors. Layer 1
is the schema — closed vocabularies derived via `get_args()` from `schema.py`
so they cannot drift, typed values, array caps, `additionalProperties: false`.
Layer 2 is the invariants schema cannot express: value must be null unless
observed, observed requires a photo index in range, unique wheel positions,
odometer plausible for the model year, and no pricing anywhere in the payload.

One hallucinated tread value does not cost you the odometer read from the same
pass. Bad fields are quarantined to `NOT_VISIBLE`; only pricing leakage and
structural collapse reject outright. Useful side effect: a quarantined field is
indistinguishable downstream from a genuinely unseen one, so the range widens
for it automatically. Guardrail violations make the answer vaguer, not wrong.

---

## Two composed sources of uncertainty

This is the part most implementations get wrong by picking one.

| source | what it is | captured by |
|---|---|---|
| **model** | identical trucks sell for different prices | p10/p50/p90 quantile models |
| **information** | we could not see the tires, the odometer, three panels | Monte Carlo over priors |

Running the Monte Carlo through the median model alone gives a
fully-photographed truck a zero-width range, which is nonsense. So every draw
also samples a quantile level `u ~ U(0,1)` and inverts the model's own
predictive distribution at that level, interpolating in normal-score space.
The two compose and the range behaves correctly at both extremes.

See `uncertainty.invert_quantiles` and `uncertainty.run_uncertainty`.

---

## Refusal is three tiers, not a binary

| | what the seller gets |
|---|---|
| **Tier 0** | nothing. Wrong object, unusable capture, two different vehicles. No price, no hedged number, no ballpark. |
| **Tier 1** | full condition report, no price. Usually: no identity resolved, too few angles, or excess width. Still valuable. |
| **Tier 2** | price with range, condition report, reasoning, and the next photo worth taking. |

Copy rules encoded in `report.py`:

- **Name what you saw.** "That looks like a motorcycle" proves the system is
  working. A generic error makes the seller assume your product is broken.
- **Never leak an anchor.** No ballparks inside a Tier 0 or Tier 1 response. The
  moment a refusal says "trucks like this usually go for $25k–60k," you have
  priced it, and worse than your model would have.
- **Give the exact next photo.** Distance, angle, subject, and what it is worth:
  *"the odometer, straight on — would narrow the range by about $3,588."*
  Ranked by variance contribution in `uncertainty.attribute_variance`.
- **Don't accuse.** Reused photos and wrong-vehicle uploads are usually mistakes.

Two production concerns handled in `pipeline.py`:

- **Sticky abstain.** Sellers re-upload the same image with a new filename until
  they get a number. Inputs are hashed; a rejected set returns the same answer
  rather than rerolling VLM nondeterminism.
- **Reason-code logging.** Every refusal is labelled data. Tier 0 false
  positives — real trucks the gate rejected — are the highest-value training
  examples you will ever collect, and the ones that quietly kill retention
  because those sellers never come back to tell you.

---

## Data

No public dataset has (commercial truck photos, condition labels, realized
prices). You assemble it from four layers.

| layer | source | status |
|---|---|---|
| identity spine | **NHTSA vPIC** — free, no key, trucks/buses/trailers since 1981, returns GVWR. Standalone DB downloadable. | `ingest/vpic.py`, working |
| realized prices | **Ritchie Bros** — 1M+ past auction results, last 24 months, free. Unreserved auctions, so hammer prices are real clearing prices. | `ingest/comps.py`, normalizer only |
| damage perception | **CarDD** — 4,000 images, 9,163 instances, six categories, annotated to insurance claim standards. `cardd-ustc.github.io` | not wired |
| damage → dollars | **Copart / IAA** — damage type, condition code, repair cost and sale outcome on the same vehicle. | not wired |

**Ask prices are not sale prices.** Classified listings on TruckPaper and
Commercial Truck Trader are aspirational and will bias a quantile model high at
every level. If you mix sources, carry a `price_kind` column and never train on
them interchangeably — `ingest.comps.validate` fails loudly if you do.

**On scraping.** Ritchie Bros and Copart both restrict automated access in their
terms. The third-party actors that scrape them operate in a grey zone; at least
one openly states it is not affiliated with the site. Survivable for a class
project, not survivable for a funded company in diligence. Recommended order:

1. Hand-collect a few hundred rows from the free Ritchie Bros UI. Enough to fit
   a first model, costs an afternoon.
2. Request a MarketCheck quote in parallel — their Heavy Equipments API is the
   licensed path, though they flag it as having narrower coverage than their
   Cars API.
3. Automate only against sources whose terms permit it, or under agreement.

**DVM-CAR** (1.4M images, 335k used car ads with prices, viewpoint and
quality-check flags already labelled) is the only public dataset with the exact
`(photos, attributes, price)` shape. UK passenger cars and released for
non-commercial use — good for validating the architecture, cannot ship on it.

---

## What is stubbed, and how to un-stub it

| component | stub | replacement |
|---|---|---|
| object detector | `demo.make_detector` returns a fixed class | any COCO-trained detector; `truck`, `car`, `motorcycle`, `bus` are native classes, so the motorcycle case is nearly free |
| perception | `demo.make_perception` returns preset evidence | `perception.extract_evidence` — set `ANTHROPIC_API_KEY`, it is written and untested against a live key |
| comps table | `scripts/make_synth_comps.py` | `ingest.comps.normalize` + real hammer prices |
| priors | hand-authored in `priors.py` | `priors.fit_priors_from_comps` once you have data |
| pHash | average hash in `gate.phash` | a real DCT-based pHash before production |
| vPIC | live API call | restore the standalone DB, use `ingest.vpic.build_cache` |

Nothing downstream changes when you swap these. The interfaces are the point.

---

## The metric

Not MAE. **Coverage**: on held-out sales, does the stated 80% interval contain
the true price 80% of the time? `scripts/eval_calibration.py` runs it through
the full uncertainty pass with fields deliberately hidden, bucketed by how much
was hidden — because production never sees fully-observed rows.

Current state on synthetic data (n=260, nominal 80%):

```
overall  82.3%   misses 8.5% below / 9.2% above   median bias -$51

bucket    n   empirical   median_rel_width
   0-2   78       0.795              0.593
   3-5   74       0.770              0.585
   6-9  108       0.880              0.661
```

The `below / above` split is the diagnostic that matters most and it is worth
reading before the coverage number. Symmetric misses mean the interval is too
narrow — widen it. Lopsided misses mean it is mis-centred, and widening is the
wrong fix entirely.

**How this got fixed is instructive.** The 6-9 bucket read 73.8% and the misses
were 0.8% below / 24.2% above — a $2,150 bias, not a width problem. Three
hypotheses were wrong: it was not the ledger (netting checked out at -$179), not
the tail interpolation (fixed anyway, on its own merits), and not the priors
(fitting them from comps helped accuracy but not coverage). The actual bug was
in the eval harness: `int(2026 - age_years)` truncates, ageing every truck by
~0.5 years on average, which at 14%/yr depreciation reads as a -7% price bias
and looks exactly like a broken model. Fix the ruler before you rebuild the
thing being measured.

---

## Layout

```
truckval/
  schema.py        Observation[T] with explicit NOT_VISIBLE. Read this first.
  gate.py          Object class, cross-photo consistency, capture quality
  perception.py    VLM call + the JSON schema that makes abstention cheap
  identity.py      vPIC decode, VIN check digit, claim cross-checking
  priors.py        Conditioned samplers for unseen fields
  features.py      Evidence + one draw → model row
  pricing.py       Quantile regression, conformal calibration, comps k-NN
  ledger.py        Condition adjustments, cohort netting
  uncertainty.py   Monte Carlo, quantile inversion, variance attribution
  report.py        Tiered copy, condition report, reasoning from the ledger
  calibration.py   Coverage, reliability curve, bucketed diagnostics
  validate.py      Hardened schema + cross-field invariants for model output
  pipeline.py      Tier routing, sticky abstain, anchor-leak assert, logging
ingest/
  vpic.py          VIN decode cache
  comps.py         Normalize realized sales into the training schema
scripts/           make_synth_comps, train_pricing, demo, eval_calibration
scripts/           + sizing.py (how much data each stage needs)
tests/             35 tests; test_range_widens_when_blind is the one that matters
```

~4,400 lines. If `test_range_widens_when_blind` ever goes green-to-red, the
product is broken even if nothing throws.

---

## Build order from here

1. **Replace the detector.** One afternoon. Off-the-shelf COCO model, threshold
   on frame coverage. Your Tier 0 refusals become real.
2. **Wire live perception.** The schema is written. Test it against fifty real
   seller photo sets and count how often it hallucinates a value where it should
   have said `not_visible`. That number is your perception quality metric.
3. **Collect 300 real comps by hand.** Retrain. Everything downstream already
   works; only the numbers get real.
4. **Re-run `eval_calibration.py` on real data.** Expect coverage to be worse
   than synthetic. Fix the priors, not the thresholds.
5. **Then** the truck-specific labels nobody has: tread depth on duals, box
   length against wheel diameter, liftgate/reefer/crane presence, rust
   perforation vs surface, cab interior wear as an odometer corroborant.
   Bootstrap them from Ritchie Bros listing text as weak labels on the photos in
   the same lot — thousands of free noisy labels before you hand-annotate
   anything.

Step 5 is the moat. Steps 1–4 are the machine that makes step 5 pay off.

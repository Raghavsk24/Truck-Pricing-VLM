"""Scrape TruckPaper.com listing photos for a Class 7/8 truck image dataset.

Positives are Class 7/8 trucks: sleeper, day cab and dump trucks, plus a slice of
other heavy trucks (mixers, garbage, tow, ...). Negatives are Class 2-6 trucks.
The weight class comes from TruckPaper's Gross Vehicle Weight Rating search filter
and only decides which folder a listing lands in; it is not saved.

Output (created before any request is made):

    TruckPaper Scraped Images Dataset/
      truckpaper_listings.json          id, brand, price, currency, image paths per listing
      Positive (Class 7-8)/<category>/sleeper_truck_01/sleeper_truck_01_image_01.jpg ...
      Negative (Class 2-6)/<category>/box_truck_01/box_truck_01_image_01.jpg ...
      .state/                           search cache + progress, used to resume

Listings are named by category, not by TruckPaper's numeric listing ID, so a folder or
file can be traced back to its listing by eye: "sleeper_truck_07" is the 7th sleeper
truck saved, and its 3rd photo is "sleeper_truck_07_image_03.jpg". Auction-style
listings (bid prices, not asking prices) are skipped entirely.

Usage:
    python scraper/truckpaper_scraper.py            # 300-image test run (200 positive, 100 negative)
    python scraper/truckpaper_scraper.py --full     # 10,000 positive + 5,000 negative images
    python scraper/truckpaper_scraper.py plan       # print strata and quotas, download nothing

Re-running resumes: listings that are already saved are skipped, so `--full` builds
on top of the test run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from urllib.parse import quote, urlencode

from curl_cffi import requests
from curl_cffi.requests.exceptions import RequestException

# TruckPaper's Cloudflare fronting distinguishes clients by TLS fingerprint, not just
# User-Agent: plain `requests`/urllib3 gets a JS challenge (HTTP 403) that a script can
# never solve, while curl and real browsers pass. curl_cffi's `impersonate=` reproduces
# a real Chrome TLS handshake, which is why it's used here instead of plain `requests`.
IMPERSONATE = "chrome124"

try:
    from tqdm import tqdm
except ImportError:  # the progress bar is optional
    tqdm = None

BASE_URL = "https://www.truckpaper.com"
CATEGORY_SITEMAP = BASE_URL + "/sitemaps/Truck/com/for-sale-equipment-category-sitemap-1.xml.gz"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

OUT_DIR = Path(__file__).resolve().parent.parent / "TruckPaper Scraped Images Dataset"
POS_DIR = "Positive (Class 7-8)"
NEG_DIR = "Negative (Class 2-6)"
JSON_NAME = "truckpaper_listings.json"

CATEGORIES = {"sleeper-trucks": 16045, "day-cab-trucks": 16013, "dump-trucks": 16014}
CORE_CATEGORY_IDS = set(CATEGORIES.values())
POS_CLASS_SHARES = {7: 0.15, 8: 0.85}  # within sleeper / day cab / dump
OTHER_POS_SHARE = 0.20  # share of positives from other Class 7/8 trucks
OTHER_POS_CLASS_SHARES = {7: 0.25, 8: 0.75}
NEG_SHARES = {6: 0.45, 5: 0.22, 4: 0.18, 3: 0.10, 2: 0.05}
YEAR_BUCKETS = [(1900, 2012), (2013, 2017), (2018, 2021), (2022, date.today().year + 1)]

TEST_TARGETS = (200, 100)
FULL_TARGETS = (10_000, 5_000)

TRUCKS_BASE_CATEGORY = 27  # trailers are 28, cranes 4
PAGE_SIZE = 28
MAX_PAGES = 357  # the site stops paginating at ~10,000 results
MIN_PHOTOS = 5
TERCILE_WEIGHTS = {"low": 0.45, "mid": 0.30, "high": 0.25, "none": 0.10}
BRAND_CAP = 0.30
MAX_PENDING_LISTINGS = 4  # listings whose images may download while the next page is fetched
TOPUP_SLACK = 10  # stop topping up a group once it is this close to its image target
MAX_CONSECUTIVE_FAILURES = 5

APP_JSON_ANCHOR = re.compile(r"React\.createElement\(App,\s*")
IMAGE_EXTS = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp", "image/gif": ".gif"}


class Blocked(RuntimeError):
    """Too many page requests in a row failed; the site is probably blocking us."""


def log(msg: str) -> None:
    if tqdm:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(5):  # Windows refuses the rename while another process has the file open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.2 * (attempt + 1))
    os.replace(tmp, path)


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def extract_app_json(html: str) -> dict | None:
    """Every TruckPaper page embeds its data as ReactDOM.hydrate(React.createElement(App, {...}))."""
    match = APP_JSON_ANCHOR.search(html)
    if not match:
        return None
    try:
        return json.JSONDecoder().raw_decode(html, match.end())[0]
    except json.JSONDecodeError:
        return None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def category_label(category_slug: str) -> str:
    """"sleeper-trucks" -> "sleeper_truck": singularise the last word for use as a name prefix."""
    words = category_slug.replace("-", "_").split("_")
    if words and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]
    return "_".join(words)


def to_int(value, default: int = 0) -> int:
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def to_stub(listing: dict) -> dict:
    """Keep only what sampling and the output need from a search-result listing."""
    try:
        price = float(listing.get("Price"))
    except (TypeError, ValueError):
        price = None
    if price is not None and price <= 0:  # "Call for Price"
        price = None
    return {
        "id": str(listing.get("Id")),
        "url": listing.get("DetailUrl") or "",
        "brand": (listing.get("ManufacturerName") or "").strip() or None,
        "price": price,
        "currency": listing.get("CurrencyCode") or None,
        "year": to_int(listing.get("Year"), 0) or None,
        "images": to_int((listing.get("ListingImageModel") or {}).get("ImageCount")),
        "cat_id": to_int(listing.get("CategoryId")),
        "cat_name": listing.get("DisplayCategoryName") or listing.get("CategoryName") or "",
        "base": to_int(listing.get("BaseCategoryId")),
        "dismantled": bool(listing.get("IsDismantled")),
    }


def page_props(data: dict | None) -> dict:
    return ((data or {}).get("BodyComponent") or {}).get("Props") or {}


def image_urls(props: dict, size: str) -> list[str]:
    urls = []
    for media in (props.get("MediaModel") or {}).get("Media") or []:
        if media.get("MediaTypeName") != "Image" or media.get("Is360") or media.get("IsSpin360"):
            continue
        primary, fallback = ("FullScreenUrl", "MediaUrl") if size == "full" else ("MediaUrl", "FullScreenUrl")
        url = media.get(primary) or media.get(fallback)
        if url and url not in urls:
            urls.append(url)
    return urls


def round_systematic(values: list[float], rng: random.Random) -> list[int]:
    """Round fractional listing counts to whole numbers that keep the total, spreading picks evenly."""
    offset = rng.random()
    out, cumulative = [], 0.0
    for value in values:
        start, cumulative = cumulative, cumulative + value
        out.append(math.ceil(cumulative - offset) - math.ceil(start - offset))
    return out


@dataclass
class Stratum:
    """One search: a GVWR class and year range, within a category or site-wide."""

    key: str
    group: str  # POS_DIR or NEG_DIR
    gvwr_class: int
    years: tuple[int, int]
    weight: float
    category_id: int | None = None  # None = site-wide search
    folder: str | None = None  # fixed category folder for sleeper / day cab / dump
    exclude_core: bool = False  # site-wide positives leave sleeper / day cab / dump to their own strata
    count: int = 0
    total_pages: int = 0
    fetched_pages: list[int] = field(default_factory=list)
    stubs: dict[str, dict] = field(default_factory=dict)
    quota: int = 0  # listings to scrape

    def search_url(self, page: int = 1) -> str:
        params = []
        if self.category_id:
            params.append(("Category", self.category_id))
        params.append(("GrossVehicleWeightRating", f"Class {self.gvwr_class}"))
        params.append(("Year", f"{self.years[0]}*{self.years[1]}"))
        if page > 1:
            params.append(("page", page))
        return f"{BASE_URL}/listings/search?{urlencode(params, quote_via=quote)}"

    def add_page(self, page: int, listings: list[dict]) -> None:
        if page not in self.fetched_pages:
            self.fetched_pages.append(page)
        for listing in listings:
            stub = to_stub(listing)
            self.stubs.setdefault(stub["id"], stub)


def build_strata() -> list[Stratum]:
    strata = []
    n_years = len(YEAR_BUCKETS)
    core_share = (1 - OTHER_POS_SHARE) / len(CATEGORIES)
    for slug, cat_id in CATEGORIES.items():
        for cls, cls_share in POS_CLASS_SHARES.items():
            for years in YEAR_BUCKETS:
                strata.append(Stratum(
                    f"pos:{slug}:class{cls}:{years[0]}-{years[1]}", POS_DIR, cls, years,
                    core_share * cls_share / n_years, category_id=cat_id, folder=slug,
                ))
    for cls, cls_share in OTHER_POS_CLASS_SHARES.items():
        for years in YEAR_BUCKETS:
            strata.append(Stratum(
                f"pos:other:class{cls}:{years[0]}-{years[1]}", POS_DIR, cls, years,
                OTHER_POS_SHARE * cls_share / n_years, exclude_core=True,
            ))
    for cls, share in NEG_SHARES.items():
        for years in YEAR_BUCKETS:
            strata.append(Stratum(
                f"neg:class{cls}:{years[0]}-{years[1]}", NEG_DIR, cls, years, share / n_years,
            ))
    return strata


@dataclass
class Pending:
    stratum: Stratum
    stub: dict
    folder: str
    code: str
    brand: str | None
    futures: list[Future]


@dataclass
class Stats:
    pages: int = 0
    page_seconds: float = 0.0
    listings: int = 0
    images: int = 0
    images_failed: int = 0
    bytes: int = 0


class Scraper:
    def __init__(self, out_dir: Path, image_size: str, delay: float, workers: int, seed: int, refresh: bool):
        self.out_dir = out_dir
        self.state_dir = out_dir / ".state"
        self.image_size = image_size
        self.delay = delay
        self.seed = seed
        self.refresh = refresh

        # impersonate= reproduces a real Chrome TLS handshake so Cloudflare doesn't
        # challenge us; use_thread_local_curl (curl_cffi's default) makes it safe to
        # share img_session across the download thread pool.
        self.session = requests.Session(impersonate=IMPERSONATE, headers={
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self.img_session = requests.Session(impersonate=IMPERSONATE)
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.lock = threading.Lock()

        self.next_request_at = 0.0
        self.failures = 0
        self.stats = Stats()
        self.bar = None
        self.pending: list[Pending] = []
        self.rngs: dict[str, random.Random] = {}
        self.next_seq: dict[str, int] = {}  # label -> highest code number used so far, globally unique

        self.data: dict[str, list] = read_json(out_dir / JSON_NAME, {})
        self.progress: dict[str, dict] = read_json(self.state_dir / "progress.json", {})
        self.search_cache: dict[str, dict] = {} if refresh else read_json(self.state_dir / "search_cache.json", {})
        self.cat_slugs: dict[int, str] = {}
        self.strata = build_strata()

    # ---------- HTTP ----------

    def throttle(self) -> None:
        wait = self.next_request_at - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self.next_request_at = time.monotonic() + self.delay * random.uniform(1.0, 1.3)

    def fetch_app_json(self, url: str) -> dict | None:
        started = time.monotonic()
        error = ""
        try:
            for attempt in range(3):
                self.throttle()
                try:
                    resp = self.session.get(url, timeout=30)
                except RequestException as exc:
                    error = str(exc)
                else:
                    self.stats.pages += 1
                    if resp.status_code in (404, 410):
                        self.failures = 0
                        return None
                    if resp.ok:
                        data = extract_app_json(resp.content.decode("utf-8", "replace"))
                        if data is not None:
                            self.failures = 0
                            return data
                        error = "page had no embedded listing data"
                    else:
                        error = f"HTTP {resp.status_code}"
                time.sleep(5 * (attempt + 1))
        finally:
            self.stats.page_seconds += time.monotonic() - started
        self.failures += 1
        log(f"  giving up on {url}: {error}")
        if self.failures >= MAX_CONSECUTIVE_FAILURES:
            raise Blocked(f"{self.failures} page requests in a row failed (last: {error})")
        return None

    def download(self, url: str, stem: Path) -> Path | None:
        existing = [p for p in stem.parent.glob(stem.name + ".*") if not p.name.endswith((".part", ".tmp"))]
        if existing:
            return existing[0]
        resp = None
        try:
            for attempt in range(3):  # curl_cffi has no built-in retry adapter
                resp = self.img_session.get(url, timeout=60, stream=True)
                if resp.ok:
                    break
                if resp.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                    resp.raise_for_status()
                time.sleep(2 * (attempt + 1))
            ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if not ctype.startswith("image/"):
                raise ValueError(f"not an image ({ctype or 'no content type'})")
            dest = stem.with_suffix(IMAGE_EXTS.get(ctype, ".jpg"))
            tmp = dest.with_name(dest.name + ".part")
            size = 0
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(64 * 1024):
                    fh.write(chunk)
                    size += len(chunk)
            os.replace(tmp, dest)
        except Exception as exc:  # one bad image shouldn't stop the listing
            with self.lock:
                self.stats.images_failed += 1
            log(f"  image failed ({exc}): {url[:90]}")
            return None
        with self.lock:
            self.stats.bytes += size
        return dest

    # ---------- setup: categories, sizing, quotas ----------

    def load_category_slugs(self) -> None:
        cached = read_json(self.state_dir / "category_slugs.json", {})
        if cached and not self.refresh:
            self.cat_slugs = {int(k): v for k, v in cached.items()}
            return
        self.throttle()
        try:
            body = self.session.get(CATEGORY_SITEMAP, timeout=30).content
            try:
                body = gzip.decompress(body)
            except OSError:
                pass  # already decompressed in transit
            for match in re.finditer(rb"/listings/for-sale/([a-z0-9-]+)/(\d+)</loc>", body):
                self.cat_slugs[int(match.group(2))] = match.group(1).decode().replace("-slash-", "-")
        except RequestException as exc:
            log(f"Couldn't load category names ({exc}); folders will use TruckPaper's display names.")
        if self.cat_slugs:
            write_json_atomic(self.state_dir / "category_slugs.json", self.cat_slugs)

    def size_strata(self) -> None:
        todo = [s for s in self.strata if s.key not in self.search_cache]
        for s in self.strata:
            if s.key in self.search_cache:
                cached = self.search_cache[s.key]
                s.count, s.total_pages = cached["count"], cached["total_pages"]
                s.fetched_pages, s.stubs = cached["fetched_pages"], cached["stubs"]
        if todo:
            log(f"Counting listings in {len(todo)} searches (about {len(todo) * self.delay * 1.3 / 60:.0f} min)...")
        for i, s in enumerate(todo, 1):
            model = page_props(self.fetch_app_json(s.search_url(1))).get("ListPageAjaxModel")
            if model is None:
                log(f"  [{i}/{len(todo)}] {s.key}: no results page, skipping")
                continue
            s.count = to_int(model.get("ListingsCount"))
            s.total_pages = min(to_int(model.get("TotalPages")), MAX_PAGES)
            s.add_page(1, model.get("Listings") or [])
            self.cache_stratum(s)
            log(f"  [{i}/{len(todo)}] {s.key}: {s.count:,} listings")

    def cache_stratum(self, s: Stratum) -> None:
        self.search_cache[s.key] = {
            "count": s.count, "total_pages": s.total_pages,
            "fetched_pages": s.fetched_pages, "stubs": s.stubs,
        }
        write_json_atomic(self.state_dir / "search_cache.json", self.search_cache)

    def is_truck(self, s: Stratum, stub: dict) -> bool:
        """A truck listing with enough photos and a real asking price. Trailers, bodies-only,
        cranes and auctions (bid prices, not asking prices) are out."""
        if not stub["url"].startswith("/listing/for-sale/"):
            return False  # links out to an auction site (auctiontime.com, equipmentfacts.com, ...)
        if stub["base"] != TRUCKS_BASE_CATEGORY or stub["dismantled"] or stub["images"] < MIN_PHOTOS:
            return False
        text = f"{self.cat_slugs.get(stub['cat_id'], '')} {stub['url']}".lower()
        if "trailer" in text or "bodies-only" in text:
            return False
        return not (s.exclude_core and stub["cat_id"] in CORE_CATEGORY_IDS)

    def capacity(self, s: Stratum) -> tuple[float, float]:
        """(listings we could take, average photos per listing), estimated from the pages fetched so far."""
        if not s.stubs or not s.count:
            return 0.0, 0.0
        usable = [c for c in s.stubs.values() if self.is_truck(s, c)]
        if not usable:
            return 0.0, 0.0
        reachable = min(s.count, s.total_pages * PAGE_SIZE)
        avg_images = sum(c["images"] for c in usable) / len(usable)
        return reachable * len(usable) / len(s.stubs), avg_images

    def allocate(self, group: str, target_images: int) -> dict[str, int]:
        """Split an image target across a group's strata by weight, capped by what each search holds."""
        strata = [s for s in self.strata if s.group == group]
        caps = {s.key: self.capacity(s) for s in strata}
        active = [s for s in strata if caps[s.key][0] >= 1]
        quota_images: dict[str, float] = {}
        remaining = float(target_images)
        while active:
            total_weight = sum(s.weight for s in active)
            capped = [s for s in active
                      if remaining * s.weight / total_weight >= caps[s.key][0] * caps[s.key][1]]
            if not capped:
                for s in active:
                    quota_images[s.key] = remaining * s.weight / total_weight
                break
            for s in capped:
                quota_images[s.key] = caps[s.key][0] * caps[s.key][1]
                remaining -= quota_images[s.key]
                active.remove(s)
        fractions = [quota_images.get(s.key, 0.0) / caps[s.key][1] if caps[s.key][1] else 0.0 for s in strata]
        counts = round_systematic(fractions, random.Random(f"{self.seed}:{group}:{target_images}"))
        return {s.key: n for s, n in zip(strata, counts)}

    def prepare(self, pos_target: int, neg_target: int) -> None:
        self.load_category_slugs()
        self.size_strata()
        quotas = {**self.allocate(POS_DIR, pos_target), **self.allocate(NEG_DIR, neg_target)}
        for s in self.strata:
            s.quota = quotas.get(s.key, 0)

    # ---------- progress bookkeeping ----------

    def listing_records(self, status: str = "done"):
        return (r for r in self.progress.values() if r["status"] == status)

    def done_count(self, s: Stratum) -> int:
        done = sum(1 for r in self.listing_records() if r["stratum"] == s.key)
        return done + sum(1 for p in self.pending if p.stratum is s)

    def group_images(self, group: str) -> int:
        done = sum(r["images"] for r in self.listing_records() if r["group"] == group)
        return done + sum(len(p.futures) for p in self.pending if p.stratum.group == group)

    def brand_counts(self, s: Stratum) -> Counter:
        brands = Counter(r["brand"] for r in self.listing_records() if r["stratum"] == s.key)
        brands.update(p.brand for p in self.pending if p.stratum is s)
        return brands

    def is_taken(self, listing_id: str) -> bool:
        return listing_id in self.progress or any(p.stub["id"] == listing_id for p in self.pending)

    def save_progress(self) -> None:
        write_json_atomic(self.state_dir / "progress.json", self.progress)

    # ---------- sampling ----------

    def next_candidate(self, s: Stratum) -> dict | None:
        """Pick the next listing: spread across price terciles (favouring cheap trucks), with a brand cap."""
        rng = self.rngs.setdefault(s.key, random.Random(f"{self.seed}:{s.key}"))
        need = max(1, s.quota - self.done_count(s))
        while True:
            available = [c for c in s.stubs.values() if self.is_truck(s, c) and not self.is_taken(c["id"])]
            unfetched = [p for p in range(1, s.total_pages + 1) if p not in s.fetched_pages]
            if len(available) >= 2 * need or not unfetched:
                break
            page = rng.choice(unfetched)
            model = page_props(self.fetch_app_json(s.search_url(page))).get("ListPageAjaxModel") or {}
            s.add_page(page, model.get("Listings") or [])
            self.cache_stratum(s)
        if not available:
            return None

        prices = sorted(c["price"] for c in s.stubs.values() if c["price"] is not None and self.is_truck(s, c))
        low, high = (prices[len(prices) // 3], prices[2 * len(prices) // 3]) if prices else (0.0, 0.0)

        def tercile(c: dict) -> str:
            if c["price"] is None:
                return "none"
            return "low" if c["price"] < low else "mid" if c["price"] < high else "high"

        cap = max(2, math.ceil(BRAND_CAP * s.quota))
        brands = self.brand_counts(s)
        pool = [c for c in available if brands[c["brand"]] < cap] or available
        by_tercile = defaultdict(list)
        for c in sorted(pool, key=lambda c: c["id"]):
            by_tercile[tercile(c)].append(c)
        names = sorted(by_tercile)
        chosen = rng.choices(names, weights=[TERCILE_WEIGHTS[n] for n in names])[0]
        return rng.choice(by_tercile[chosen])

    def category_slug(self, s: Stratum, stub: dict) -> str:
        return s.folder or self.cat_slugs.get(stub["cat_id"]) or slugify(stub["cat_name"]) or "other-trucks"

    def folder_for(self, s: Stratum, stub: dict) -> str:
        return f"{s.group}/{self.category_slug(s, stub)}"

    def next_code(self, label: str) -> str:
        """"sleeper_truck_01", "sleeper_truck_02", ... - stable and unique across the whole dataset,
        even though the same category label (e.g. "dump_truck") can appear in both classes."""
        if label not in self.next_seq:
            used = [int(r["code"].rsplit("_", 1)[1]) for r in self.progress.values()
                    if r.get("code", "").rsplit("_", 1)[0] == label]
            self.next_seq[label] = max(used, default=0)
        self.next_seq[label] += 1
        return f"{label}_{self.next_seq[label]:02d}"

    # ---------- scraping ----------

    def start_listing(self, s: Stratum, stub: dict) -> bool:
        """Fetch the detail page and queue its images. Returns False if the listing had to be skipped."""
        props = page_props(self.fetch_app_json(BASE_URL + stub["url"]))
        urls = image_urls(props, self.image_size)
        pricing = props.get("PricingInfo") or {}
        if not urls or props.get("IsExpired") or pricing.get("IsAuctionListing") or pricing.get("IsAuctionResult"):
            self.progress[stub["id"]] = {"stratum": s.key, "group": s.group, "brand": stub["brand"],
                                         "images": 0, "status": "skipped"}
            self.save_progress()
            return False
        brand = stub["brand"] or (props.get("Manufacturer") or "").strip() or None
        folder = self.folder_for(s, stub)
        label = category_label(self.category_slug(s, stub))
        code = self.next_code(label)
        listing_dir = self.out_dir / folder / code
        listing_dir.mkdir(parents=True, exist_ok=True)
        futures = [self.pool.submit(self.download, url, listing_dir / f"{code}_image_{i + 1:02d}")
                  for i, url in enumerate(urls)]
        self.pending.append(Pending(s, stub, folder, code, brand, futures))
        while len(self.pending) > MAX_PENDING_LISTINGS:
            self.finish(self.pending.pop(0))
        self.drain()
        return True

    def finish(self, p: Pending) -> None:
        """Record a listing once all its downloads are done."""
        if p in self.pending:
            self.pending.remove(p)
        if any(f.cancelled() for f in p.futures):
            return  # interrupted; a retry gets a fresh code and re-downloads, leaving this one's files orphaned
        paths = [path for path in (f.result() for f in p.futures) if path]
        record = {"stratum": p.stratum.key, "group": p.stratum.group, "brand": p.brand, "folder": p.folder,
                  "code": p.code, "images": len(paths), "status": "done" if paths else "skipped"}
        self.progress[p.stub["id"]] = record
        if paths:
            self.data.setdefault(p.folder, []).append({
                "id": p.code,
                "brand": p.brand,
                "price": p.stub["price"],
                "currency": p.stub["currency"],
                "images": [path.relative_to(self.out_dir).as_posix() for path in paths],
            })
            write_json_atomic(self.out_dir / JSON_NAME, self.data)
            self.stats.listings += 1
            self.stats.images += len(paths)
            if self.bar:
                self.bar.update(len(paths))
        self.save_progress()

    def drain(self, block: bool = False) -> None:
        for p in list(self.pending):
            if block or all(f.done() for f in p.futures):
                self.finish(p)

    def crawl(self, pos_target: int, neg_target: int) -> None:
        targets = {POS_DIR: pos_target, NEG_DIR: neg_target}
        if tqdm:
            done = sum(min(self.group_images(g), t) for g, t in targets.items())
            self.bar = tqdm(total=pos_target + neg_target, initial=done, unit="img", desc="Images")

        def group_full(group: str, slack: int = 0) -> bool:
            return self.group_images(group) >= targets[group] - slack

        # Round-robin across strata, alternating positives and negatives, until each hits its quota.
        pos = [s for s in self.strata if s.group == POS_DIR]
        neg = [s for s in self.strata if s.group == NEG_DIR]
        order = [s for pair in zip(pos, neg) for s in pair] + pos[len(neg):] + neg[len(pos):]
        active = [s for s in order if s.quota > self.done_count(s)]
        while active:
            for s in list(active):
                if group_full(s.group) or self.done_count(s) >= s.quota:
                    active.remove(s)
                    continue
                stub = self.next_candidate(s)
                if stub is None:
                    log(f"  {s.key}: no more usable listings")
                    active.remove(s)
                    continue
                self.start_listing(s, stub)
        self.drain(block=True)

        # Listings vary in photo count, so top up any group that came in short.
        for group in targets:
            strata = sorted((s for s in self.strata if s.group == group and s.quota > 0),
                            key=lambda s: s.weight, reverse=True)
            while not group_full(group, TOPUP_SLACK):
                added = False
                for s in strata:
                    if group_full(group, TOPUP_SLACK):
                        break
                    stub = self.next_candidate(s)
                    if stub is not None:
                        added = self.start_listing(s, stub) or added
                self.drain(block=True)
                if not added:
                    break

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.drain(block=True)
        self.save_progress()
        if self.bar:
            self.bar.close()

    # ---------- reporting ----------

    def print_plan(self, pos_target: int, neg_target: int) -> None:
        print(f"\n{'search':44} {'on site':>8} {'usable':>7} {'photos':>7} {'take':>5} {'~images':>8}")
        totals = defaultdict(lambda: [0, 0.0])
        for s in self.strata:
            listings, avg_images = self.capacity(s)
            usable = f"{listings / min(s.count, MAX_PAGES * PAGE_SIZE):.0%}" if s.count else "-"
            print(f"{s.key:44} {s.count:>8,} {usable:>7} {avg_images:>7.1f} {s.quota:>5} {s.quota * avg_images:>8.0f}")
            totals[s.group][0] += s.quota
            totals[s.group][1] += s.quota * avg_images
        print()
        for group, target in ((POS_DIR, pos_target), (NEG_DIR, neg_target)):
            n, imgs = totals[group]
            print(f"{group}: {n} listings, ~{imgs:,.0f} images (target {target:,})")
        n_listings = sum(n for n, _ in totals.values())
        minutes = n_listings * (self.delay * 1.15 + 0.6) / 60
        print(f"Estimated scrape time: ~{minutes:.0f} min for the detail pages, "
              f"with images downloading alongside.")

    def print_summary(self, elapsed: float, crawl_seconds: float, full_targets: tuple[int, int]) -> None:
        s = self.stats
        print(f"\nFinished in {elapsed / 60:.1f} min: {s.listings} listings, {s.images} images "
              f"({s.bytes / 1e6:.1f} MB), {s.pages} page requests, {s.images_failed} failed images.")
        for group in (POS_DIR, NEG_DIR):
            records = [r for r in self.listing_records() if r["group"] == group]
            print(f"  {group}: {len(records)} listings, {sum(r['images'] for r in records)} images in total")
        print(f"Output: {self.out_dir}")
        if not s.listings:
            return
        # Project the full run from this run's measured speed.
        avg_images = s.images / s.listings
        sec_per_listing = crawl_seconds / s.listings
        sec_per_page = s.page_seconds / max(s.pages, 1)
        have = sum(self.group_images(g) for g in (POS_DIR, NEG_DIR))
        remaining_listings = max(0.0, (sum(full_targets) - have) / avg_images)
        extra_pages = 0
        for group, target in zip((POS_DIR, NEG_DIR), full_targets):
            for key, n in self.allocate(group, target).items():
                st = next(x for x in self.strata if x.key == key)
                listings, _ = self.capacity(st)
                per_page = PAGE_SIZE * listings / max(min(st.count, MAX_PAGES * PAGE_SIZE), 1)
                needed = min(st.total_pages, math.ceil(2 * n / max(per_page, 1)))
                extra_pages += max(0, needed - len(st.fetched_pages))
        minutes = (remaining_listings * sec_per_listing + extra_pages * sec_per_page) / 60
        print(f"Measured: {sec_per_listing:.1f} s per listing, {avg_images:.0f} photos per listing.")
        print(f"Projected full run ({full_targets[0]:,} + {full_targets[1]:,} images): "
              f"~{minutes:.0f} min more for ~{remaining_listings:.0f} listings. "
              f"Start it with: python scraper/truckpaper_scraper.py --full")


def ensure_output_dirs(out_dir: Path) -> None:
    for path in (out_dir, out_dir / POS_DIR, out_dir / NEG_DIR, out_dir / ".state"):
        path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", choices=("crawl", "plan"), default="crawl",
                        help="crawl (default) downloads images; plan only prints strata and quotas")
    parser.add_argument("--full", action="store_true",
                        help=f"full run: {FULL_TARGETS[0]:,} positive + {FULL_TARGETS[1]:,} negative images "
                             f"(default is a {sum(TEST_TARGETS)}-image test run)")
    parser.add_argument("--positive-images", type=int, help="override the positive image target")
    parser.add_argument("--negative-images", type=int, help="override the negative image target")
    parser.add_argument("--image-size", choices=("full", "medium"), default="full",
                        help="full = original resolution, medium = 614x460")
    parser.add_argument("--delay", type=float, default=1.0, help="seconds between page requests (default 1.0)")
    parser.add_argument("--workers", type=int, default=8, help="parallel image downloads (default 8)")
    parser.add_argument("--seed", type=int, default=7, help="random seed for sampling")
    parser.add_argument("--refresh", action="store_true", help="ignore cached search results")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.out)

    pos_target, neg_target = FULL_TARGETS if args.full else TEST_TARGETS
    if args.positive_images is not None:
        pos_target = args.positive_images
    if args.negative_images is not None:
        neg_target = args.negative_images

    started = time.monotonic()
    scraper = Scraper(args.out, args.image_size, args.delay, args.workers, args.seed, args.refresh)
    crawl_started = None
    try:
        scraper.prepare(pos_target, neg_target)
        if args.command == "plan":
            scraper.print_plan(pos_target, neg_target)
            return
        mode = "full run" if args.full else "test run"
        log(f"Starting {mode}: {pos_target:,} positive + {neg_target:,} negative images -> {args.out}")
        crawl_started = time.monotonic()
        scraper.crawl(pos_target, neg_target)
    except Blocked as exc:
        log(f"Stopped: {exc}. TruckPaper may be rate-limiting; wait a while and re-run to resume.")
    except KeyboardInterrupt:
        log("Interrupted. Progress is saved; re-run the same command to resume.")
    finally:
        scraper.close()
    if crawl_started is not None:
        scraper.print_summary(time.monotonic() - started, time.monotonic() - crawl_started, FULL_TARGETS)


if __name__ == "__main__":
    main()

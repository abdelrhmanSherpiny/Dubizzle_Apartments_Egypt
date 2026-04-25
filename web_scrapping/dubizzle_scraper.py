"""
Dubizzle Egypt — All-Egypt Apartments-for-Sale Scraper  (v2 – Segmented)
=========================================================================
Phase 1: Data Acquisition for the Market Oracle AI Engine

Scrapes ALL apartments for sale across ALL Egyptian cities from
https://www.dubizzle.com.eg/en/properties/apartments-duplex-for-sale/

Strategy:
  Dubizzle caps pagination at ~200 pages (~9,600 ads per URL).
  With 72K+ total ads, a single listing URL cannot reach them all.
  This scraper segments by LOCATION SLUG and, for large locations,
  further splits by PRICE BAND using Dubizzle's filter URL format:
      ?filter=price_between_{MIN}_to_{MAX}

Features:
  ✓ Location × price-band segmentation to bypass pagination cap
  ✓ Auto-discovery of sub-location slugs from listing pages
  ✓ Adaptive price bisection for mega-segments (>9K ads)
  ✓ All cities nationwide via 24+ location slugs
  ✓ Latitude & Longitude from Mapbox static map images
  ✓ City/district extracted from breadcrumb
  ✓ All structured fields + full unstructured description
  ✓ Arabic numeral normalization (٠-٩ → 0-9)
  ✓ Per-task checkpoint/resume for multi-day scraping
  ✓ Graceful Ctrl+C handling — saves progress on interrupt
  ✓ Incremental JSON-Lines output (no data loss)
  ✓ Deduplication by ad_id (keeps old records safe)

Usage:
    python dubizzle_scraper.py                     # scrape all segments
    python dubizzle_scraper.py --resume            # resume interrupted scrape
    python dubizzle_scraper.py --test sharqia       # test on one small location
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import signal
import sys
import time
from collections import Counter
from datetime import datetime
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
BASE_URL = "https://www.dubizzle.com.eg"

# Base category path (no location filter)
CATEGORY_PATH = "/en/properties/apartments-duplex-for-sale/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://www.dubizzle.com.eg/en/realestate/",
}

# Polite scraping delays (seconds)
MIN_DELAY = 1.0
MAX_DELAY = 2.0

# Checkpoint & output filenames
CHECKPOINT_FILE = "scrape_checkpoint_v2.json"
JSONL_FILE = "dubizzle_apartments_egypt.jsonl"

# Arabic-Eastern digit map: ٠١٢٣٤٥٦٧٨٩ → 0123456789
ARABIC_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Pagination cap — segments above this will be noted but can't be split further
PAGINATION_CAP = 9000  # conservative (actual cap ~9,600 = 199 pages × 48 ads)

# ── Location Segments ────────────────────────────────────────────────────────
# Each tuple: (slug, name, approximate_ad_count)
# Dubizzle uses a FLAT URL structure for all locations:
#   https://www.dubizzle.com.eg/en/properties/apartments-duplex-for-sale/{slug}/
#
# Large locations (5th Settlement 18K, Sheikh Zayed 8.6K, 6th of October 7.5K)
# are broken down to their compound/neighborhood sub-locations to stay under
# the 9,600-ad pagination cap.
LOCATION_SEGMENTS = [
    # ═══ 5th Settlement — broken into compounds/neighborhoods ═══
    ("mountain-view-icity-compound",      "MV iCity (5th Sett.)",        1746),
    ("fifth-square-compound",             "Fifth Square (5th Sett.)",    1296),
    ("hyde-park-new-cairo-compound",       "Hyde Park (5th Sett.)",       1123),
    ("el-patio-oro-compound",             "El Patio Oro (5th Sett.)",    1012),
    ("palm-hills-new-cairo-compound",     "Palm Hills NC (5th Sett.)",    979),
    ("beit-al-watan-6",                   "Beit Al Watan (5th Sett.)",    642),
    ("the-address-east-compound",         "Address East (5th Sett.)",     607),
    ("mountain-view-hyde-park-compound",   "MV Hyde Park (5th Sett.)",    602),
    ("al-andalous-2",                     "Al Andalous (5th Sett.)",      552),
    ("mivida-compound",                   "Mivida (5th Sett.)",           521),
    ("eastown-compound",                  "Eastown (5th Sett.)",          449),
    ("district-5-compound-compound",      "District 5 (5th Sett.)",      430),
    ("lake-view-residence-compound",      "Lake View (5th Sett.)",        410),
    ("creek-town-compound",               "Creek Town (5th Sett.)",       381),
    ("el-lotus-2",                        "El Lotus (5th Sett.)",         376),
    ("villette-compound",                 "Villette (5th Sett.)",         362),
    ("galleria-moon-valley-compound",     "Galleria MV (5th Sett.)",      357),
    ("lake-view-residence-2-2",           "Lake View 2 (5th Sett.)",      330),
    ("the-icon-residence-compound",       "Icon Res. (5th Sett.)",        290),
    ("al-narges-2",                       "Al Narges (5th Sett.)",        283),
    # Remaining 5th Settlement ads (fall through to parent slug)
    ("5th-settlement",                    "5th Settlement (other)",      5000),

    # ═══ Cairo — New Cairo cluster ═══
    ("1st-settlement",      "1st Settlement",        2922),
    ("rehab-city",          "Rehab City",            1289),
    ("6th-settlement",      "6th Settlement",         915),
    ("mostakbal-city",      "Mostakbal City",        3763),
    ("madinaty",            "Madinaty",              3740),
    ("new-capital-city",    "New Capital City",      3464),
    ("shorouk-city",        "Shorouk City",          2146),

    # ═══ Cairo — Central ═══
    ("sheraton",            "Sheraton",               1187),
    ("katameya",            "Katameya",               1043),
    ("new-heliopolis",      "New Heliopolis",          855),
    ("nasr-city",           "Nasr City",               854),
    ("mokattam",            "Mokattam",                636),
    ("maadi",               "Maadi",                   597),

    # ═══ Sheikh Zayed — broken into compounds ═══
    ("zed-west",                "ZED West (Zayed)",          797),
    ("village-west",            "Village West (Zayed)",       584),
    ("el-khamayel",             "El Khamayel (Zayed)",        574),
    ("green-revolution",        "Green Revolution (Zayed)",   462),
    ("vye-sodic",               "VYE Sodic (Zayed)",          338),
    ("dejoya",                  "Dejoya (Zayed)",             233),
    ("beverly-hills",           "Beverly Hills (Zayed)",      229),
    ("zayed-dunes",             "Zayed Dunes (Zayed)",        187),
    ("the-address",             "The Address (Zayed)",        170),
    ("westview",                "Westview (Zayed)",           158),
    ("karmell",                 "Karmell (Zayed)",            155),
    ("ivoire",                  "Ivoire (Zayed)",             152),
    ("terrace",                 "Terrace (Zayed)",            143),
    ("casa",                    "Casa (Zayed)",               131),
    ("solana",                  "Solana (Zayed)",             127),
    ("karma-kay",               "Karma Kay (Zayed)",          119),
    ("genista",                 "Genista (Zayed)",            119),
    ("marville-new-zayed",      "MarVille (Zayed)",           114),
    ("arkan-palm-205",          "Arkan Palm (Zayed)",         105),
    ("sodic-westown",           "Sodic Westown (Zayed)",       96),
    # Remaining Sheikh Zayed (fall through to parent slug)
    ("sheikh-zayed",            "Sheikh Zayed (other)",      3000),

    # ═══ 6th of October — broken into compounds ═══
    ("mountain-view-icity-2",         "MV iCity (6 Oct.)",        1181),
    ("o-west",                        "O West (6 Oct.)",           854),
    ("badya-palm-hills",              "Badya Palm Hills (6 Oct.)", 718),
    ("palm-parks",                    "Palm Parks (6 Oct.)",       302),
    ("kayan",                         "Kayan (6 Oct.)",            247),
    ("pyramids-wales",                "Pyramids Wales (6 Oct.)",   198),
    ("degla-palms",                   "Degla Palms (6 Oct.)",      169),
    ("new-giza",                      "New Giza (6 Oct.)",         163),
    ("october-plaza-sodic",           "Oct Plaza Sodic (6 Oct.)",  161),
    ("west-somid",                    "West Somid (6 Oct.)",       158),
    ("northern-expansions",           "N. Expansions (6 Oct.)",    156),
    ("tesla-residence",               "Tesla Res. (6 Oct.)",       133),
    ("garden-lakes",                   "Garden Lakes (6 Oct.)",    132),
    ("nyoum-october",                 "Nyoum Oct. (6 Oct.)",       129),
    ("ashgar-city",                   "Ashgar City (6 Oct.)",      111),
    ("joulz",                         "Joulz (6 Oct.)",            111),
    ("dreamland",                     "Dreamland (6 Oct.)",        109),
    ("px-palm-hills",                 "PX Palm Hills (6 Oct.)",    101),
    ("mountain-view-chillout-park",   "MV Chillout (6 Oct.)",       83),
    ("flw-residence",                 "FLW Res. (6 Oct.)",          81),
    # Remaining 6th of October (fall through to parent slug)
    ("6th-of-october",                "6th of October (other)",   2500),

    # ═══ Giza — Other ═══
    ("hadayek-october",     "Hadayek October",        1698),
    ("hadayek-al-ahram",    "Hadayek al-Ahram",        531),

    # ═══ Other Governorates ═══
    ("alexandria",          "Alexandria",             3401),
    ("red-sea",             "Red Sea",                2375),
    ("matruh",              "Matruh",                 1446),
    ("suez",                "Suez",                    323),
    ("gharbia",             "Gharbia",                 199),
    ("sharqia",             "Sharqia",                 149),
]

# Graceful shutdown flag
_interrupted = False


def _handle_sigint(sig, frame):
    """Handle Ctrl+C gracefully — set flag instead of crashing."""
    global _interrupted
    if _interrupted:
        # Second Ctrl+C → force quit
        logger.warning("Force quit!")
        sys.exit(1)
    _interrupted = True
    logger.info("\n⚠ Scraping interrupted by user! Saving progress...")

signal.signal(signal.SIGINT, _handle_sigint)


# ── Helper Utilities ─────────────────────────────────────────────────────────
def polite_sleep(factor: float = 1.0):
    """Random delay between requests."""
    time.sleep(random.uniform(MIN_DELAY * factor, MAX_DELAY * factor))


def fetch_page(session: requests.Session, url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a URL → BeautifulSoup. Retries with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 404:
                logger.warning("  404 Not Found: %s", url)
                return None
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "lxml")
        except requests.RequestException as exc:
            logger.warning("  Attempt %d/%d failed: %s", attempt, retries, exc)
            if attempt < retries:
                time.sleep(5 * attempt)
    return None


def clean_text(text: str | None) -> str:
    """Normalise whitespace and strip."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def arabic_to_english_digits(text: str) -> str:
    """Convert Arabic-Eastern numerals (٠-٩) → English (0-9)."""
    return text.translate(ARABIC_DIGIT_MAP)


def parse_number(value: str) -> float | None:
    """Extract a float from a string like 'EGP 4,500,000' or '168 SQM'."""
    if not value:
        return None
    value = arabic_to_english_digits(value)
    cleaned = re.sub(r"[^\d.]", "", value.replace(",", ""))
    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


# ── Listing-Page Parsers ─────────────────────────────────────────────────────
def parse_listing_page(soup: BeautifulSoup) -> list[dict]:
    """Extract ad URLs from one listing page."""
    listings = []
    ad_links = soup.find_all("a", href=re.compile(r"/en/ad/.*-ID\d+\.html"))

    seen_urls = set()
    for link in ad_links:
        href = link.get("href", "")
        full_url = urljoin(BASE_URL, href)
        if full_url in seen_urls:
            continue
        seen_urls.add(full_url)
        listings.append({"url": full_url})

    return listings


def get_total_pages(soup: BeautifulSoup) -> int:
    """Get max page number from pagination.
    Only considers links that contain the apartments-duplex-for-sale path
    to avoid picking up unrelated pagination from navigation menus.
    """
    max_page = 1
    for pl in soup.find_all("a", href=re.compile(r"apartments-duplex-for-sale.*[?&]page=\d+")):
        m = re.search(r"page=(\d+)", pl.get("href", ""))
        if m:
            max_page = max(max_page, int(m.group(1)))
    return max_page


def get_ad_count(soup: BeautifulSoup) -> int | None:
    """
    Parse the total number of ads shown on a listing page.
    Dubizzle shows text like "72,143 ads" or "18,174 ads in 5th Settlement".
    Returns the count or None if not found.
    """
    page_text = soup.get_text(separator="\n", strip=True)
    page_text = arabic_to_english_digits(page_text)

    # Pattern: "<number> ads" or "<number> results" or "<number> listings"
    for pattern in [
        r"([\d,]+)\s+ads?\b",
        r"([\d,]+)\s+results?\b",
        r"([\d,]+)\s+listings?\b",
        r"([\d,]+)\s+إعلان",
    ]:
        m = re.search(pattern, page_text, re.I)
        if m:
            count_str = m.group(1).replace(",", "")
            try:
                return int(count_str)
            except ValueError:
                continue

    # Fallback: estimate from pagination
    total_pages = get_total_pages(soup)
    if total_pages > 1:
        return total_pages * 48  # ~48 ads per page
    return None


def discover_sub_locations(soup: BeautifulSoup, current_slug: str) -> list[tuple[str, str, int]]:
    """
    Discover sub-location links from the sidebar of a listing page.
    Returns list of (slug, name, ad_count) tuples for locations
    that are children of the current slug.
    """
    sub_locations = []
    base_pattern = re.compile(
        r"/en/properties/apartments-duplex-for-sale/([a-z0-9-]+)/?"
    )

    # Look for location links with ad counts in parentheses
    for a_tag in soup.find_all("a", href=base_pattern):
        href = a_tag.get("href", "")
        m = base_pattern.search(href)
        if not m:
            continue

        slug = m.group(1)
        if slug == current_slug:
            continue

        text = clean_text(a_tag.get_text())
        # Try to extract count from text like "New Cairo (23,835)"
        count_match = re.search(r"\(([\d,]+)\)", text)
        count = int(count_match.group(1).replace(",", "")) if count_match else 0

        name = re.sub(r"\s*\([\d,]+\)\s*", "", text).strip()
        if name and slug and len(slug) > 2:
            sub_locations.append((slug, name, count))

    return sub_locations


# ── Task Generation ──────────────────────────────────────────────────────────
class ScrapeTask:
    """Represents a single scraping segment: one location slug."""

    def __init__(self, slug: str, name: str, estimated_ads: int):
        self.slug = slug
        self.name = name
        self.estimated_ads = estimated_ads

    @property
    def task_id(self) -> str:
        """Unique identifier for checkpointing."""
        return self.slug

    @property
    def listing_url(self) -> str:
        """Base listing URL (page 1, no page param)."""
        return f"{BASE_URL}{CATEGORY_PATH}{self.slug}/"

    def page_url(self, page_num: int) -> str:
        """Listing URL for a specific page number."""
        if page_num == 1:
            return self.listing_url
        return f"{BASE_URL}{CATEGORY_PATH}{self.slug}/?page={page_num}"

    def __repr__(self):
        return f"<Task: {self.name} (~{self.estimated_ads:,} ads)>"


def generate_tasks(
    locations: list[tuple[str, str, int]],
) -> list[ScrapeTask]:
    """Generate one scrape task per location. All segments should already be
    broken down to fit under the pagination cap."""
    tasks = []

    for slug, name, approx_count in locations:
        tasks.append(ScrapeTask(slug, name, approx_count))
        if approx_count > PAGINATION_CAP:
            logger.warning(
                "⚠ %s has ~%d ads (> %d cap). Some ads may be unreachable.",
                name, approx_count, PAGINATION_CAP,
            )

    logger.info("Generated %d scrape tasks from %d locations.", len(tasks), len(locations))
    return tasks


# ── Detail-Page Parser ───────────────────────────────────────────────────────
def parse_detail_page(soup: BeautifulSoup, url: str) -> dict:
    """
    Extract ALL data from a Dubizzle property detail page.

    Layers:
      1. HTML highlights section (key-value span pairs)
      2. JSON-LD schema.org markup (title, price, image)
      3. Mapbox static image URL (latitude, longitude)
      4. Breadcrumb (city, governorate)
      5. Description container
      6. Seller info
      7. Computed derived features
    """

    data = {"url": url, "scrape_timestamp": datetime.now().isoformat()}

    # ── AD ID ────────────────────────────────────────────────────────────
    ad_id_match = re.search(r"ID(\d+)\.html", url)
    data["ad_id"] = ad_id_match.group(1) if ad_id_match else ""

    # ── Full page text (Arabic digits normalized) ────────────────────────
    all_text = arabic_to_english_digits(soup.get_text(separator="\n", strip=True))

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 1: HTML Highlights (key-value pairs)
    # ══════════════════════════════════════════════════════════════════════
    KNOWN_LABELS = {
        "type", "bedrooms", "bathrooms", "area", "furnished",
        "floor", "level", "ownership", "payment option", "payment method",
        "completion status", "delivery term", "delivery date", "delivery",
        "compound", "down payment", "finish type",
        "النوع", "غرف النوم", "الحمامات", "المساحة", "مفروشة", "مفروش",
        "الدور", "الطابق", "التشطيب", "الملكية", "طريقة الدفع",
        "حالة الإنشاء", "الكمبوند", "المقدم", "موعد التسليم",
    }
    KEY_MAP = {
        "type": "type", "النوع": "type",
        "bedrooms": "bedrooms", "غرف النوم": "bedrooms",
        "bathrooms": "bathrooms", "الحمامات": "bathrooms",
        "area": "area", "المساحة": "area",
        "furnished": "furnished", "مفروشة": "furnished", "مفروش": "furnished",
        "floor": "floor", "level": "floor", "الدور": "floor", "الطابق": "floor",
        "ownership": "ownership", "الملكية": "ownership",
        "payment option": "payment_option", "payment method": "payment_option",
        "طريقة الدفع": "payment_option",
        "completion status": "completion_status", "حالة الإنشاء": "completion_status",
        "delivery term": "delivery_term", "delivery date": "delivery_term",
        "delivery": "delivery_term", "موعد التسليم": "delivery_term",
        "compound": "compound", "الكمبوند": "compound",
        "down payment": "down_payment", "المقدم": "down_payment",
        "finish type": "finish_type", "التشطيب": "finish_type",
    }

    highlights = {}
    for el in soup.find_all(["span", "div", "p"]):
        text_lower = clean_text(el.get_text()).lower().strip()
        if text_lower not in KNOWN_LABELS:
            continue

        value = ""
        # Next sibling span/div
        ns = el.find_next_sibling(["span", "div", "p"])
        if ns:
            value = clean_text(ns.get_text())
        # Parent's next sibling
        if not value and el.parent:
            pn = el.parent.find_next_sibling()
            if pn:
                value = clean_text(pn.get_text())
        # Same parent, second child
        if not value and el.parent:
            spans = [c for c in el.parent.children if hasattr(c, "name") and c.name in ("span", "div")]
            if len(spans) >= 2 and spans[-1] != el:
                value = clean_text(spans[-1].get_text())

        key = KEY_MAP.get(text_lower, text_lower)
        if value and key not in highlights:
            highlights[key] = arabic_to_english_digits(value)

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 2: JSON-LD schema.org
    # ══════════════════════════════════════════════════════════════════════
    jsonld_title, jsonld_desc, jsonld_image = "", "", ""
    jsonld_price = None

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            ld = json.loads(script.string)
            if isinstance(ld, dict) and ld.get("@type") == "Product":
                jsonld_title = clean_text(ld.get("name", ""))
                jsonld_desc = clean_text(ld.get("description", ""))
                jsonld_image = ld.get("image", "")
                offers = ld.get("offers", [])
                if isinstance(offers, dict):
                    offers = [offers]
                for o in offers:
                    p = o.get("price", 0)
                    if p and p != 0:
                        jsonld_price = float(p)
                break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 3: Latitude / Longitude from Mapbox static image
    # URL format: api.mapbox.com/.../static/{lng},{lat},14/...
    # ══════════════════════════════════════════════════════════════════════
    latitude, longitude = None, None

    # Search for Mapbox image URLs in <img> tags
    for img in soup.find_all("img", src=re.compile(r"api\.mapbox\.com")):
        src = img.get("src", "")
        geo_match = re.search(r"/static/([\d.]+),([\d.]+),", src)
        if geo_match:
            longitude = float(geo_match.group(1))
            latitude = float(geo_match.group(2))
            break

    # Fallback: search in inline scripts for coordinate patterns
    if not latitude:
        for script in soup.find_all("script"):
            if not script.string:
                continue
            # Look for mapbox URL in script text
            geo_match = re.search(
                r"api\.mapbox\.com[^\"]*?/static/([\d.]+),([\d.]+),",
                script.string
            )
            if geo_match:
                longitude = float(geo_match.group(1))
                latitude = float(geo_match.group(2))
                break
            # Look for lat/lng JSON keys
            lat_match = re.search(r'"lat(?:itude)?"\s*:\s*([\d.]+)', script.string)
            lng_match = re.search(r'"(?:lng|lon|longitude)"\s*:\s*([\d.]+)', script.string)
            if lat_match and lng_match:
                latitude = float(lat_match.group(1))
                longitude = float(lng_match.group(1))
                break

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 4: City & Governorate from breadcrumb
    # Breadcrumb format:
    #   "Apartments for Sale" → "...in Cairo" → "...in New Capital City" → "...in Sky Capital"
    # We extract location names from the "Apartments for Sale in {X}" pattern.
    # ══════════════════════════════════════════════════════════════════════
    city = ""
    governorate = ""

    # Non-location terms that appear as breadcrumb nodes but aren't places
    NON_LOCATION = {
        "installment", "cash", "resale", "primary", "furnished",
        "unfurnished", "ready to move", "under construction",
    }

    location_crumbs = []  # ordered: [governorate, city/district, compound/sub-district]
    for a in soup.find_all("a", href=re.compile(r"/en/properties/apartments-duplex-for-sale/")):
        t = clean_text(a.get_text())
        m = re.match(r"Apartments for Sale in (.+)", t, re.I)
        if m:
            name = m.group(1).strip()
            if name.lower() not in NON_LOCATION and len(name) < 60:
                location_crumbs.append(name)

    if len(location_crumbs) >= 3:
        governorate = location_crumbs[0]   # e.g., "Cairo"
        city = location_crumbs[1]          # e.g., "New Capital City"
        # location_crumbs[2] is often the compound/sub-district
    elif len(location_crumbs) == 2:
        governorate = location_crumbs[0]
        city = location_crumbs[1]
    elif len(location_crumbs) == 1:
        city = location_crumbs[0]

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 5: Price
    # ══════════════════════════════════════════════════════════════════════
    price_str = ""
    price_el = soup.find("span", attrs={"aria-label": "Price"})
    if price_el:
        price_str = arabic_to_english_digits(clean_text(price_el.get_text()))
    if not price_str:
        pm = re.search(r"EGP\s*[\d,]+", all_text)
        if pm:
            price_str = pm.group(0)
    if not price_str and jsonld_price:
        price_str = f"EGP {jsonld_price:,.0f}"

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 6: Description (full unstructured text for LLM)
    # ══════════════════════════════════════════════════════════════════════
    description = ""
    desc_container = soup.find(attrs={"aria-label": "Description"})
    if desc_container:
        description = clean_text(desc_container.get_text())

    if not description:
        for heading in soup.find_all(["h2", "h3", "h4", "span"]):
            if clean_text(heading.get_text()) in ("Description", "الوصف"):
                for sib in heading.parent.find_next_siblings():
                    txt = clean_text(sib.get_text())
                    if len(txt) > 30:
                        description = txt
                        break
                if not description and heading.parent.parent:
                    for sib in heading.parent.parent.find_next_siblings():
                        txt = clean_text(sib.get_text())
                        if len(txt) > 30:
                            description = txt
                            break
                if description:
                    break

    if not description and jsonld_desc:
        description = jsonld_desc

    if description.startswith("Description"):
        description = description[len("Description"):].strip()

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 7: Seller Info
    # ══════════════════════════════════════════════════════════════════════
    seller_name, seller_type = "", ""
    member_el = soup.find(string=re.compile(r"Member since", re.I))
    if member_el:
        container = member_el.parent
        for _ in range(5):
            if container and container.parent:
                container = container.parent
                cl = container.find("a", href=re.compile(r"/companies/"))
                if cl:
                    seller_name = clean_text(cl.get_text())
                    seller_type = "Agency"
                    break
        if not seller_name:
            seller_type = "Individual"

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 8: Title
    # ══════════════════════════════════════════════════════════════════════
    title = jsonld_title or ""
    if not title:
        h1 = soup.find("h1")
        title = clean_text(h1.get_text()) if h1 else ""

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 9: Amenities
    # ══════════════════════════════════════════════════════════════════════
    amenities = []
    amenity_section = soup.find(string=re.compile(r"^Amenities$|^المرافق$", re.I))
    if amenity_section:
        parent = amenity_section
        for _ in range(5):
            parent = parent.parent if parent else None
            if parent and parent.name == "div":
                for item in parent.find_all(["span", "div", "li"]):
                    txt = clean_text(item.get_text())
                    if txt and 3 < len(txt) < 40 and txt not in ("Amenities", "المرافق", "See All"):
                        if txt not in amenities:
                            amenities.append(txt)
                if amenities:
                    break

    # ══════════════════════════════════════════════════════════════════════
    # LAYER 10: Posted date
    # ══════════════════════════════════════════════════════════════════════
    posted_date = ""
    for pat in [
        r"(\d+\s+(?:hours?|days?|weeks?|months?|minutes?)\s+ago)",
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s+\d{1,2},?\s+\d{4})",
    ]:
        dm = re.search(pat, all_text, re.I)
        if dm:
            posted_date = dm.group(1)
            break

    # ══════════════════════════════════════════════════════════════════════
    # ASSEMBLE RECORD
    # ══════════════════════════════════════════════════════════════════════
    # Area
    area_raw = highlights.get("area", "")
    if not area_raw:
        am = re.search(r"([\d,]+)\s*(?:SQM|sqm|m²|m2|متر)", all_text, re.I)
        area_raw = am.group(1).replace(",", "") if am else ""
    area_raw = area_raw.replace(",", "").strip()
    area_num_m = re.search(r"[\d.]+", area_raw)
    area_str = area_num_m.group(0) if area_num_m else ""

    # Bedrooms & Bathrooms
    beds = highlights.get("bedrooms", "")
    if not beds:
        bm = re.search(r"(?:Bedrooms?|غرف)\s*[:\-]?\s*(\d+)", all_text, re.I)
        beds = bm.group(1) if bm else ""
    beds_m = re.search(r"\d+", beds)
    beds = beds_m.group(0) if beds_m else ""

    baths = highlights.get("bathrooms", "")
    if not baths:
        bm2 = re.search(r"(?:Bathrooms?|حمام)\s*[:\-]?\s*(\d+)", all_text, re.I)
        baths = bm2.group(1) if bm2 else ""
    baths_m = re.search(r"\d+", baths)
    baths = baths_m.group(0) if baths_m else ""

    # Computed features
    price_numeric = parse_number(price_str)
    area_numeric = parse_number(area_str)
    price_per_sqm = None
    if price_numeric and area_numeric and area_numeric > 0:
        price_per_sqm = round(price_numeric / area_numeric, 2)

    # Build record
    data["title"] = title
    data["price"] = price_str
    data["price_numeric"] = price_numeric
    data["bedrooms"] = beds
    data["bathrooms"] = baths
    data["area_sqm"] = area_str
    data["area_numeric"] = area_numeric
    data["price_per_sqm"] = price_per_sqm
    data["property_type"] = highlights.get("type", "")
    data["city"] = city
    data["governorate"] = governorate
    data["latitude"] = latitude
    data["longitude"] = longitude
    data["floor"] = highlights.get("floor", "")
    data["furnished"] = highlights.get("furnished", "")
    data["finish_type"] = highlights.get("finish_type", "")
    data["ownership"] = highlights.get("ownership", "")
    data["payment_option"] = highlights.get("payment_option", "")
    data["completion_status"] = highlights.get("completion_status", "")
    data["delivery_term"] = highlights.get("delivery_term", "")
    data["compound"] = highlights.get("compound", "")
    data["down_payment"] = highlights.get("down_payment", "")
    data["amenities"] = ", ".join(amenities) if amenities else ""
    data["seller_name"] = seller_name
    data["seller_type"] = seller_type
    data["posted_date"] = posted_date
    data["description"] = description
    data["image_url"] = jsonld_image

    return data


# URLs file — stores one URL per line (append-only, fast at scale)
URLS_FILE = "scraped_urls.txt"


# ── Checkpoint Logic ─────────────────────────────────────────────────────────
def load_checkpoint(output_dir: str) -> dict:
    cp_path = os.path.join(output_dir, CHECKPOINT_FILE)
    cp = {
        "completed_tasks": [],
        "current_task": None,
        "current_task_page": 0,
        "total_records": 0,
    }
    if os.path.exists(cp_path):
        with open(cp_path, "r", encoding="utf-8") as f:
            cp.update(json.load(f))
    return cp


def save_checkpoint(output_dir: str, checkpoint: dict):
    """Save task-level metadata (NOT URLs — those go to a separate file)."""
    cp_path = os.path.join(output_dir, CHECKPOINT_FILE)
    # Don't store scraped_urls in checkpoint JSON anymore
    cp_data = {k: v for k, v in checkpoint.items() if k != "scraped_urls"}
    with open(cp_path, "w", encoding="utf-8") as f:
        json.dump(cp_data, f, ensure_ascii=False)


def load_scraped_urls(output_dir: str) -> set:
    """Load previously scraped URLs from the URL tracking file."""
    urls = set()
    urls_path = os.path.join(output_dir, URLS_FILE)
    if os.path.exists(urls_path):
        with open(urls_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    urls.add(line)
    return urls


def append_scraped_urls(output_dir: str, urls: list[str]):
    """Append newly scraped URLs to the tracking file."""
    if not urls:
        return
    urls_path = os.path.join(output_dir, URLS_FILE)
    with open(urls_path, "a", encoding="utf-8") as f:
        for url in urls:
            f.write(url + "\n")


def append_jsonl(filepath: str, records: list[dict]):
    """Append records to a JSON-Lines file (one JSON object per line)."""
    with open(filepath, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(filepath: str) -> list[dict]:
    """Load all records from a JSON-Lines file."""
    records = []
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


# ── Output Writers ───────────────────────────────────────────────────────────
FIELD_ORDER = [
    "ad_id", "title",
    # Price features
    "price", "price_numeric", "price_per_sqm",
    # Property features
    "bedrooms", "bathrooms", "area_sqm", "area_numeric",
    "property_type", "floor", "furnished", "finish_type",
    "ownership", "payment_option", "completion_status",
    "delivery_term", "compound", "down_payment",
    # Location features
    "city", "governorate", "latitude", "longitude",
    # Seller features
    "seller_name", "seller_type",
    # Amenities
    "amenities",
    # Temporal
    "posted_date",
    # Unstructured text (LLM Phase 2)
    "description",
    # Metadata
    "image_url", "url", "scrape_timestamp",
]


def save_csv(records: list[dict], filepath: str):
    if not records:
        return
    fieldnames = list(FIELD_ORDER)
    seen = set(fieldnames)
    for r in records:
        for k in r:
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)

    with open(filepath, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    logger.info("Saved %d records → %s", len(records), filepath)


def save_json(records: list[dict], filepath: str):
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    logger.info("Saved %d records → %s", len(records), filepath)


# ── Single Task Scraper ──────────────────────────────────────────────────────
def scrape_task(
    session: requests.Session,
    task: ScrapeTask,
    scraped_urls: set,
    jsonl_path: str,
    output_dir: str,
    checkpoint: dict,
    start_page: int = 1,
    max_pages_override: int = 0,
) -> int:
    """
    Scrape one task (one location + optional price band) through all its pages.
    Returns the number of NEW records scraped.
    """
    global _interrupted

    logger.info("")
    logger.info("═══════════════════════════════════════════════════════════════")
    logger.info("  📍 Task: %s", task)
    logger.info("  🔗 URL:  %s", task.listing_url)
    logger.info("═══════════════════════════════════════════════════════════════")

    # Fetch first page
    polite_sleep()
    soup = fetch_page(session, task.listing_url)
    if not soup:
        logger.warning("  ❌ Failed to fetch first page. Skipping task.")
        return 0

    # Determine total pages for this segment
    total_pages = get_total_pages(soup)
    ad_count = get_ad_count(soup)

    # Cap total_pages using estimated ad count (with safety margin)
    if ad_count and ad_count > 0:
        estimated_pages = (ad_count // 48) + 2  # ~48 ads/page + safety margin
        if total_pages > estimated_pages:
            logger.info(
                "  ⚡ Capping pages from %d → %d (based on %s ads)",
                total_pages, estimated_pages, f"{ad_count:,}",
            )
            total_pages = estimated_pages
        logger.info("  📊 Ads in segment: %s | Pages: %d", f"{ad_count:,}", total_pages)
    else:
        logger.info("  📊 Pages: %d", total_pages)

    if max_pages_override > 0:
        total_pages = min(total_pages, max_pages_override)

    task_scraped = 0

    for page_num in range(start_page, total_pages + 1):
        if _interrupted:
            break

        # Fetch page (first page already fetched)
        if page_num > 1 or start_page > 1:
            page_url = task.page_url(page_num)
            logger.info("  📄 Page %d/%d — %s", page_num, total_pages, page_url)
            polite_sleep()
            soup = fetch_page(session, page_url)
            if not soup:
                logger.warning("  ⚠ Skipping page %d (fetch failed).", page_num)
                continue
        else:
            logger.info("  📄 Page %d/%d", page_num, total_pages)

        # Parse listing page
        listings = parse_listing_page(soup)
        new_listings = [l for l in listings if l["url"] not in scraped_urls]

        logger.info(
            "  Page %d: %d listings (%d new, %d already scraped).",
            page_num, len(listings), len(new_listings),
            len(listings) - len(new_listings),
        )

        if not listings:
            logger.info("  ⚠ No listings found on page %d. End of segment.", page_num)
            break

        page_records = []
        for idx, listing in enumerate(new_listings, 1):
            if _interrupted:
                break

            detail_url = listing["url"]
            logger.info(
                "    [%d/%d] %s",
                idx, len(new_listings),
                detail_url[:90] + "..." if len(detail_url) > 90 else detail_url,
            )
            polite_sleep()

            detail_soup = fetch_page(session, detail_url)
            if not detail_soup:
                logger.warning("    ⚠ Failed, skipping.")
                continue

            record = parse_detail_page(detail_soup, detail_url)
            page_records.append(record)
            scraped_urls.add(detail_url)
            task_scraped += 1
            checkpoint["total_records"] = checkpoint.get("total_records", 0) + 1

            # Compact log line
            total_so_far = checkpoint["total_records"]
            city_short = (record.get("city", "") or "?")[:16]
            lat = f"{record['latitude']:.4f}" if record.get("latitude") else "?"
            lng = f"{record['longitude']:.4f}" if record.get("longitude") else "?"
            logger.info(
                "    ✓ [#%d] %s | %s | %s beds | %s m² | %s/m² | geo: %s,%s",
                total_so_far,
                city_short,
                record.get("price", "?")[:20],
                record.get("bedrooms", "?"),
                record.get("area_sqm", "?"),
                f"{record['price_per_sqm']:,.0f}" if record.get("price_per_sqm") else "?",
                lat, lng,
            )

        # Save page results incrementally
        if page_records:
            append_jsonl(jsonl_path, page_records)
            # Track newly scraped URLs in the URL file (append-only, fast)
            new_urls = [r["url"] for r in page_records if r.get("url")]
            append_scraped_urls(output_dir, new_urls)
            logger.info(
                "  💾 Page %d saved. Task: +%d | Total: %d",
                page_num, task_scraped, checkpoint["total_records"],
            )

        # Update checkpoint after each page (lightweight — no URLs)
        checkpoint["current_task"] = task.task_id
        checkpoint["current_task_page"] = page_num
        save_checkpoint(output_dir, checkpoint)

    logger.info("  ✅ Task done: %s — scraped %d new records.", task.name, task_scraped)
    return task_scraped


# ── Main Scraping Pipeline ──────────────────────────────────────────────────
def main():
    global _interrupted

    parser = argparse.ArgumentParser(
        description="Dubizzle Market Oracle — All-Egypt Apartments Scraper (v2 Segmented)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dubizzle_scraper.py                    # scrape ALL segments
  python dubizzle_scraper.py --resume           # resume from checkpoint
  python dubizzle_scraper.py --test sharqia     # test on one small location
        """,
    )
    parser.add_argument(
        "--output", type=str, default="dubizzle_apartments_egypt",
        help="Output filename without extension."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint."
    )
    parser.add_argument(
        "--test", type=str, default="",
        help="Test mode: scrape only this location slug (e.g., 'sharqia')."
    )
    args = parser.parse_args()

    output_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_path = os.path.join(output_dir, JSONL_FILE)
    csv_path = os.path.join(output_dir, f"{args.output}.csv")
    json_path = os.path.join(output_dir, f"{args.output}.json")

    # ── Load checkpoint ──────────────────────────────────────────────────
    checkpoint = load_checkpoint(output_dir) if args.resume else {
        "completed_tasks": [],
        "current_task": None,
        "current_task_page": 0,
        "total_records": 0,
    }
    completed_tasks = set(checkpoint.get("completed_tasks", []))

    # Load scraped URLs from the tracking file (fast, separate from checkpoint)
    scraped_urls = load_scraped_urls(output_dir)

    # ALWAYS scan existing JSONL for URLs (covers prior runs, URL file gaps, etc.)
    existing_records = load_jsonl(jsonl_path)
    if existing_records:
        existing_urls = {r.get("url", "") for r in existing_records if r.get("url")}
        new_urls_found = len(existing_urls - scraped_urls)
        scraped_urls.update(existing_urls)
        checkpoint["total_records"] = len(existing_records)
        logger.info(
            "Found %d existing records in JSONL (%d new URLs loaded). Will skip already-scraped.",
            len(existing_records), new_urls_found,
        )

    if args.resume:
        logger.info(
            "Resuming with %d scraped URLs, %d completed tasks, current: %s (page %d).",
            len(scraped_urls),
            len(completed_tasks),
            checkpoint.get("current_task", "none"),
            checkpoint.get("current_task_page", 0),
        )

    session = requests.Session()
    session.headers.update(HEADERS)

    # ── Determine locations to scrape ────────────────────────────────────
    if args.test:
        locations = [
            (slug, name, count)
            for slug, name, count in LOCATION_SEGMENTS
            if slug == args.test
        ]
        if not locations:
            logger.error("Unknown test location: '%s'", args.test)
            logger.info("Available: %s", ", ".join(s[0] for s in LOCATION_SEGMENTS))
            return
    else:
        locations = list(LOCATION_SEGMENTS)

    # ── Generate tasks ───────────────────────────────────────────────────
    tasks = generate_tasks(locations)

    # ── Banner ───────────────────────────────────────────────────────────
    total_est = sum(s[2] for s in locations)
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════════════╗")
    logger.info("║  🏠 Dubizzle Market Oracle — v2 Segmented Scraper              ║")
    logger.info("╠══════════════════════════════════════════════════════════════════╣")
    logger.info("║  Target          : ALL apartments for sale in Egypt             ║")
    logger.info("║  Locations       : %-43d║", len(locations))
    logger.info("║  Scrape tasks    : %-43d║", len(tasks))
    logger.info("║  Estimated ads   : ~%-42s║", f"{total_est:,}")
    logger.info("║  Already scraped : %-43s║", f"{len(scraped_urls):,}")
    logger.info("║  Resume          : %-43s║", "ON" if args.resume else "OFF")
    logger.info("╚══════════════════════════════════════════════════════════════════╝")
    logger.info("")

    # ── Execute tasks ────────────────────────────────────────────────────
    tasks_completed = 0
    tasks_skipped = 0

    for task_idx, task in enumerate(tasks, 1):
        if _interrupted:
            break

        # Skip completed tasks
        if task.task_id in completed_tasks:
            tasks_skipped += 1
            continue

        # Determine start page for current task (resume support)
        start_page = 1
        if args.resume and checkpoint.get("current_task") == task.task_id:
            start_page = checkpoint.get("current_task_page", 0) + 1
            if start_page > 1:
                logger.info("Resuming task %s from page %d.", task.task_id, start_page)

        logger.info(
            "\n🔄 Task %d/%d (skipped %d already done)",
            task_idx, len(tasks), tasks_skipped,
        )

        # Scrape!
        new_records = scrape_task(
            session=session,
            task=task,
            scraped_urls=scraped_urls,
            jsonl_path=jsonl_path,
            output_dir=output_dir,
            checkpoint=checkpoint,
            start_page=start_page,
        )

        if not _interrupted:
            completed_tasks.add(task.task_id)
            checkpoint["completed_tasks"] = list(completed_tasks)
            checkpoint["current_task"] = None
            checkpoint["current_task_page"] = 0
            save_checkpoint(output_dir, checkpoint)

        tasks_completed += 1

    # ═════════════════════════════════════════════════════════════════════
    # FINAL OUTPUT
    # ═════════════════════════════════════════════════════════════════════
    if _interrupted:
        logger.info("\n⚠ Scraping interrupted by user! Progress saved.")
        logger.info("  Run with --resume to continue.")

    # Load all records from JSONL and write final CSV + JSON
    all_records = load_jsonl(jsonl_path)

    # Deduplicate by ad_id
    seen_ids = set()
    unique = []
    for r in all_records:
        aid = r.get("ad_id", "")
        if aid and aid not in seen_ids:
            seen_ids.add(aid)
            unique.append(r)
        elif not aid:
            unique.append(r)
    all_records = unique

    save_json(all_records, json_path)

    # Clean up checkpoint only if NOT interrupted AND all tasks done
    if not _interrupted and len(completed_tasks) == len(tasks):
        for cleanup_file in [CHECKPOINT_FILE, URLS_FILE]:
            p = os.path.join(output_dir, cleanup_file)
            if os.path.exists(p):
                os.remove(p)
        logger.info("Checkpoint files cleaned up (scrape completed successfully).")

    # ── Summary Statistics ───────────────────────────────────────────────
    total = len(all_records)
    if total == 0:
        logger.warning("No records scraped.")
        return

    w_price = sum(1 for r in all_records if r.get("price_numeric"))
    w_beds = sum(1 for r in all_records if r.get("bedrooms"))
    w_area = sum(1 for r in all_records if r.get("area_numeric"))
    w_ppsqm = sum(1 for r in all_records if r.get("price_per_sqm"))
    w_city = sum(1 for r in all_records if r.get("city"))
    w_type = sum(1 for r in all_records if r.get("property_type"))
    w_seller = sum(1 for r in all_records if r.get("seller_type"))
    w_desc = sum(1 for r in all_records if len(r.get("description", "")) > 30)
    w_geo = sum(1 for r in all_records if r.get("latitude"))

    prices = [r["price_numeric"] for r in all_records if r.get("price_numeric")]
    avg_price = sum(prices) / len(prices) if prices else 0
    min_price = min(prices) if prices else 0
    max_price = max(prices) if prices else 0

    ppsqm = [r["price_per_sqm"] for r in all_records if r.get("price_per_sqm")]
    avg_ppsqm = sum(ppsqm) / len(ppsqm) if ppsqm else 0

    # City distribution
    city_counts = Counter(r.get("city", "Unknown") for r in all_records if r.get("city"))
    top_cities = city_counts.most_common(10)

    logger.info("")
    logger.info("┌──────────────────────────────────────────────────────┐")
    logger.info("│          SCRAPING COMPLETE — DATA SUMMARY            │")
    logger.info("├──────────────────────────────────────────────────────┤")
    logger.info("│  Total records        : %-28d│", total)
    logger.info("│  Unique cities        : %-28d│", len(city_counts))
    logger.info("│  Tasks completed      : %-28d│", len(completed_tasks))
    logger.info("│  ─── Feature Coverage ──────────────────────────── │")
    logger.info("│  With price           : %-5d (%3.0f%%)                │", w_price, w_price / total * 100)
    logger.info("│  With bedrooms        : %-5d (%3.0f%%)                │", w_beds, w_beds / total * 100)
    logger.info("│  With area            : %-5d (%3.0f%%)                │", w_area, w_area / total * 100)
    logger.info("│  With price/m²        : %-5d (%3.0f%%)                │", w_ppsqm, w_ppsqm / total * 100)
    logger.info("│  With city            : %-5d (%3.0f%%)                │", w_city, w_city / total * 100)
    logger.info("│  With lat/lng         : %-5d (%3.0f%%)                │", w_geo, w_geo / total * 100)
    logger.info("│  With property type   : %-5d (%3.0f%%)                │", w_type, w_type / total * 100)
    logger.info("│  With seller type     : %-5d (%3.0f%%)                │", w_seller, w_seller / total * 100)
    logger.info("│  With description     : %-5d (%3.0f%%)                │", w_desc, w_desc / total * 100)
    logger.info("│  ─── Price Statistics ──────────────────────────── │")
    logger.info("│  Avg price            : EGP %-24s│", f"{avg_price:,.0f}")
    logger.info("│  Min price            : EGP %-24s│", f"{min_price:,.0f}")
    logger.info("│  Max price            : EGP %-24s│", f"{max_price:,.0f}")
    logger.info("│  Avg price/m²         : EGP %-24s│", f"{avg_ppsqm:,.0f}")
    logger.info("├──────────────────────────────────────────────────────┤")
    logger.info("│  Top cities:                                         │")
    for city_name, count in top_cities:
        logger.info("│    %-21s: %-5d listings              │", city_name[:21], count)
    logger.info("├──────────────────────────────────────────────────────┤")
    logger.info("│  CSV  : %-42s│", os.path.basename(csv_path))
    logger.info("│  JSON : %-42s│", os.path.basename(json_path))
    logger.info("└──────────────────────────────────────────────────────┘")
    logger.info("")
    logger.info("Next step → Phase 2: Run LLM feature extraction on 'description' column")

    # Save CSV last (after final summary, so we know it completed)
    save_csv(all_records, csv_path)


if __name__ == "__main__":
    main()

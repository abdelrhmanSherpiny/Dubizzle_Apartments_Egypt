"""
database.py – SQLite database layer for Real Estate Analytics Hub.

Two-table schema:
  • raw_listings   – append-only staging table (stores scraped text as-is)
  • refined_listings – analytics / ML-ready table (parsed, validated, enriched)

Usage:
  from database import (
      init_db, get_connection, seed_mock_data,
      query_refined, search_listings, get_filter_options, get_stats,
  )

  init_db()                     # creates tables + indexes if they don't exist
  seed_mock_data()              # populates both tables with realistic mock rows
  rows = search_listings(query="Maadi apartment", city="Cairo", max_price=5_000_000)
  opts = get_filter_options()   # distinct values for every dropdown
  stats = get_stats()           # KPI counts for the Explore dashboard
"""

import sqlite3
import hashlib
import json
import os
import random
from datetime import datetime, timedelta

# ── Database path (same folder as this script) ─────────────────────────────────
_DB_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DB_DIR, "real_estate.db")


def get_connection() -> sqlite3.Connection:
    """Return a connection to the SQLite database with FK enforcement."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


# ── Schema DDL ──────────────────────────────────────────────────────────────────
_CREATE_RAW = """
CREATE TABLE IF NOT EXISTS raw_listings (
  id              INTEGER   PRIMARY KEY AUTOINCREMENT,
  scrape_id       TEXT      NOT NULL UNIQUE,
  source_url      TEXT      NOT NULL,
  raw_title       TEXT,
  raw_price       TEXT,
  raw_location    TEXT,
  raw_area        TEXT,
  raw_rooms       TEXT,
  raw_bathrooms   TEXT,
  raw_property_type TEXT,
  raw_description TEXT,
  raw_furnished   TEXT,
  raw_floor       TEXT,
  extra_attributes TEXT,
  scrape_status   TEXT      NOT NULL DEFAULT 'raw',
  scraped_at      DATETIME  NOT NULL,
  created_at      DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_REFINED = """
CREATE TABLE IF NOT EXISTS refined_listings (
  id                INTEGER   PRIMARY KEY AUTOINCREMENT,
  raw_id            INTEGER   NOT NULL REFERENCES raw_listings(id),
  scrape_id         TEXT      NOT NULL UNIQUE,
  source_url        TEXT      NOT NULL,
  title             TEXT      NOT NULL,
  short_description TEXT,
  property_type     TEXT      CHECK(property_type IN
                        ('Apartment','Villa','Duplex','Townhouse','Studio','Penthouse','Other')),
  city              TEXT,
  neighborhood      TEXT,
  area_sqm          REAL      CHECK(area_sqm > 0),
  num_rooms         INTEGER   CHECK(num_rooms >= 0),
  num_bathrooms     INTEGER   CHECK(num_bathrooms >= 0),
  furnished_status  TEXT      CHECK(furnished_status IN
                        ('furnished','unfurnished','semi-furnished')),
  floor_number      INTEGER,
  actual_price      REAL      NOT NULL CHECK(actual_price > 0),
  currency          TEXT      NOT NULL DEFAULT 'EGP',
  price_per_sqm     REAL      CHECK(price_per_sqm > 0),
  expected_price    REAL,
  listing_year      INTEGER,
  listing_month     INTEGER   CHECK(listing_month BETWEEN 1 AND 12),
  data_quality_score TEXT     CHECK(data_quality_score IN ('high','medium','low')),
  listed_at         DATETIME,
  refined_at        DATETIME  NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_scrape_id        ON raw_listings(scrape_id);",
    "CREATE INDEX IF NOT EXISTS idx_refined_raw_id              ON refined_listings(raw_id);",
    "CREATE INDEX IF NOT EXISTS idx_refined_city                ON refined_listings(city);",
    "CREATE INDEX IF NOT EXISTS idx_refined_property_type       ON refined_listings(property_type);",
    "CREATE INDEX IF NOT EXISTS idx_refined_actual_price        ON refined_listings(actual_price);",
    "CREATE INDEX IF NOT EXISTS idx_refined_quality             ON refined_listings(data_quality_score);",
    "CREATE INDEX IF NOT EXISTS idx_refined_expected_price      ON refined_listings(expected_price);",
    "CREATE INDEX IF NOT EXISTS idx_refined_city_type_price     ON refined_listings(city, property_type, actual_price);",
    "CREATE INDEX IF NOT EXISTS idx_refined_time                ON refined_listings(listing_year, listing_month);",
]


def init_db() -> None:
    """Create tables and indexes if they don't exist."""
    conn = get_connection()
    try:
        conn.execute(_CREATE_RAW)
        conn.execute(_CREATE_REFINED)
        for idx_sql in _INDEXES:
            conn.execute(idx_sql)
        conn.commit()
    finally:
        conn.close()


# ── Mock-data seeding ───────────────────────────────────────────────────────────
def _make_scrape_id(url: str, date_str: str) -> str:
    """Deterministic hash of URL + date → stable scrape_id."""
    return hashlib.sha256(f"{url}|{date_str}".encode()).hexdigest()[:16]


def seed_mock_data(n: int = 120, *, force: bool = False) -> int:
    """
    Populate both tables with *n* realistic mock listings.

    Returns the number of rows inserted.  Skips seeding if data already
    exists (unless *force* is True, which deletes everything first).
    """
    conn = get_connection()
    try:
        existing = conn.execute("SELECT COUNT(*) FROM refined_listings").fetchone()[0]
        if existing > 0 and not force:
            return 0
        if force:
            conn.execute("DELETE FROM refined_listings")
            conn.execute("DELETE FROM raw_listings")
            conn.commit()

        random.seed(42)  # reproducible

        # ── Realistic Egyptian real-estate look-up tables ─────────────────────
        cities = ["Cairo", "Giza", "Alexandria"]
        neighborhoods = {
            "Cairo": ["Maadi", "Nasr City", "New Cairo", "Zamalek", "Heliopolis",
                       "Tagamoa", "Mokattam", "Dokki"],
            "Giza":  ["6th of October", "Sheikh Zayed", "Haram", "Faisal"],
            "Alexandria": ["Smouha", "Gleem", "Sidi Gaber", "Stanley"],
        }
        property_types = ["Apartment", "Villa", "Duplex", "Townhouse",
                          "Studio", "Penthouse", "Other"]
        furnished_opts = ["furnished", "unfurnished", "semi-furnished"]
        descriptions = [
            "Bright and spacious unit with an open floor plan, modern kitchen, and balcony overlooking lush greenery.",
            "Prime location close to international schools, shopping malls, and major transport links.",
            "Newly renovated with premium finishes, marble flooring, and central air conditioning throughout.",
            "Quiet residential compound with 24/7 security, swimming pool, gym, and children's play area.",
            "Corner unit offering panoramic city views, natural light all day, and ample storage space.",
            "Ground-floor garden apartment perfect for families, with private outdoor area and covered parking.",
            "High-floor penthouse with a private rooftop terrace, jacuzzi, and smart-home automation system.",
            "Fully furnished turnkey property ideal for expats — includes all appliances and designer furniture.",
        ]
        base_rates = {
            "Maadi": 18_000, "Nasr City": 12_000, "New Cairo": 28_000,
            "Zamalek": 35_000, "Heliopolis": 16_000, "Tagamoa": 30_000,
            "Mokattam": 10_000, "Dokki": 20_000,
            "6th of October": 15_000, "Sheikh Zayed": 25_000,
            "Haram": 8_000, "Faisal": 7_500,
            "Smouha": 14_000, "Gleem": 22_000,
            "Sidi Gaber": 11_000, "Stanley": 26_000,
        }

        inserted = 0
        for i in range(n):
            city = random.choice(cities)
            hood = random.choice(neighborhoods[city])
            ptype = random.choice(property_types)
            area = random.randint(55, 400)
            rooms = random.randint(1, 6)
            baths = random.randint(1, 4)
            floor = random.randint(0, 15)
            furn = random.choice(furnished_opts)
            rate = base_rates.get(hood, 15_000) * random.uniform(0.85, 1.15)
            price = round(rate * area + rooms * 200_000)
            expected = round(price * random.uniform(0.82, 1.18))
            ppsqm = round(price / area, 2)
            desc = random.choice(descriptions)
            listed_date = datetime.now() - timedelta(days=random.randint(1, 180))
            scraped_date = listed_date + timedelta(hours=random.randint(1, 48))
            url = f"https://dubizzle.com.eg/listing/{100_000 + i}"
            scrape_id = _make_scrape_id(url, scraped_date.strftime("%Y-%m-%d"))

            title_ptype = ptype if ptype != "Other" else "Property"
            title = f"{rooms}-Room {title_ptype} in {hood}"
            short_desc = desc[:300]

            # Quality score
            nulls = sum([
                area is None, rooms is None, baths is None, price is None,
            ])
            quality = "high" if nulls == 0 else ("medium" if nulls <= 1 else "low")

            # ── Insert into raw_listings ──────────────────────────────────────
            conn.execute(
                """INSERT INTO raw_listings
                   (scrape_id, source_url, raw_title, raw_price, raw_location,
                    raw_area, raw_rooms, raw_bathrooms, raw_property_type,
                    raw_description, raw_furnished, raw_floor,
                    extra_attributes, scrape_status, scraped_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    scrape_id, url, title,
                    f"EGP {price:,}", f"{hood}, {city}",
                    f"{area} m²", f"{rooms} Bedrooms", f"{baths} Bathrooms",
                    ptype, desc, furn.capitalize(), f"Floor {floor}",
                    json.dumps({"compound": random.choice(["Yes", "No"])}),
                    "processed",
                    scraped_date.isoformat(),
                ),
            )
            raw_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

            # ── Insert into refined_listings ──────────────────────────────────
            conn.execute(
                """INSERT INTO refined_listings
                   (raw_id, scrape_id, source_url, title, short_description,
                    property_type, city, neighborhood, area_sqm,
                    num_rooms, num_bathrooms, furnished_status, floor_number,
                    actual_price, currency, price_per_sqm, expected_price,
                    listing_year, listing_month, data_quality_score, listed_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    raw_id, scrape_id, url, title, short_desc,
                    ptype, city, hood, area,
                    rooms, baths, furn, floor,
                    price, "EGP", ppsqm, expected,
                    listed_date.year, listed_date.month, quality,
                    listed_date.isoformat(),
                ),
            )
            inserted += 1

        conn.commit()
        return inserted
    finally:
        conn.close()


# ── Query helpers ───────────────────────────────────────────────────────────────
def query_refined(
    *,
    city: str | None = None,
    neighborhood: str | None = None,
    property_type: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    quality: str | None = None,
    limit: int = 200,
) -> list[dict]:
    """
    Query refined_listings with optional filters.
    Returns a list of dicts (one per row).
    """
    clauses: list[str] = []
    params: list = []

    if city:
        clauses.append("city = ?")
        params.append(city)
    if neighborhood:
        clauses.append("neighborhood = ?")
        params.append(neighborhood)
    if property_type:
        clauses.append("property_type = ?")
        params.append(property_type)
    if min_price is not None:
        clauses.append("actual_price >= ?")
        params.append(min_price)
    if max_price is not None:
        clauses.append("actual_price <= ?")
        params.append(max_price)
    if min_area is not None:
        clauses.append("area_sqm >= ?")
        params.append(min_area)
    if max_area is not None:
        clauses.append("area_sqm <= ?")
        params.append(max_area)
    if quality:
        clauses.append("data_quality_score = ?")
        params.append(quality)

    where = " AND ".join(clauses) if clauses else "1=1"
    sql = f"SELECT * FROM refined_listings WHERE {where} ORDER BY actual_price LIMIT ?"
    params.append(limit)

    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def search_listings(
    *,
    query: str = "",
    city: str | None = None,
    neighborhood: str | None = None,
    property_type: str | None = None,
    furnished_status: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    min_rooms: int | None = None,
    max_rooms: int | None = None,
    sort_by: str = "actual_price",
    sort_dir: str = "ASC",
    limit: int = 100,
) -> list[dict]:
    """
    Full-text + filter search over refined_listings.

    *query* is matched with LIKE against title, neighborhood, city,
    short_description, and property_type — giving a simple but effective
    fuzzy/partial match without needing FTS5.

    All other keyword arguments are exact / range filters that compose with AND.
    Returns a list of dicts ordered by *sort_by*.
    """
    clauses: list[str] = []
    params: list = []

    # ── Free-text search (OR across key text columns) ─────────────────────────
    if query and query.strip():
        terms = [t.strip() for t in query.strip().split() if t.strip()]
        for term in terms:
            pattern = f"%{term}%"
            clauses.append(
                "(title LIKE ? OR neighborhood LIKE ? OR city LIKE ? "
                "OR short_description LIKE ? OR property_type LIKE ?)"
            )
            params.extend([pattern, pattern, pattern, pattern, pattern])

    # ── Exact / range filters ─────────────────────────────────────────────────
    if city:
        clauses.append("city = ?")
        params.append(city)
    if neighborhood:
        clauses.append("neighborhood LIKE ?")
        params.append(f"%{neighborhood}%")
    if property_type:
        clauses.append("property_type = ?")
        params.append(property_type)
    if furnished_status:
        clauses.append("furnished_status = ?")
        params.append(furnished_status)
    if min_price is not None:
        clauses.append("actual_price >= ?")
        params.append(min_price)
    if max_price is not None:
        clauses.append("actual_price <= ?")
        params.append(max_price)
    if min_area is not None:
        clauses.append("area_sqm >= ?")
        params.append(min_area)
    if max_area is not None:
        clauses.append("area_sqm <= ?")
        params.append(max_area)
    if min_rooms is not None:
        clauses.append("num_rooms >= ?")
        params.append(min_rooms)
    if max_rooms is not None:
        clauses.append("num_rooms <= ?")
        params.append(max_rooms)

    # ── Validate sort column against allowlist (prevent SQL injection) ────────
    _SORTABLE = {
        "actual_price", "price_per_sqm", "area_sqm",
        "num_rooms", "listed_at", "expected_price",
    }
    if sort_by not in _SORTABLE:
        sort_by = "actual_price"
    direction = "DESC" if sort_dir.upper() == "DESC" else "ASC"

    where = " AND ".join(clauses) if clauses else "1=1"
    sql = (
        f"SELECT * FROM refined_listings "
        f"WHERE {where} "
        f"ORDER BY {sort_by} {direction} "
        f"LIMIT ?"
    )
    params.append(limit)

    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_filter_options() -> dict:
    """
    Return distinct values for every dropdown filter in the search UI.
    Keeps the frontend in sync with whatever data is actually in the DB.
    """
    conn = get_connection()
    try:
        cities = [
            r[0] for r in
            conn.execute("SELECT DISTINCT city FROM refined_listings WHERE city IS NOT NULL ORDER BY city").fetchall()
        ]
        neighborhoods = [
            r[0] for r in
            conn.execute("SELECT DISTINCT neighborhood FROM refined_listings WHERE neighborhood IS NOT NULL ORDER BY neighborhood").fetchall()
        ]
        property_types = [
            r[0] for r in
            conn.execute("SELECT DISTINCT property_type FROM refined_listings WHERE property_type IS NOT NULL ORDER BY property_type").fetchall()
        ]
        furnished_opts = [
            r[0] for r in
            conn.execute("SELECT DISTINCT furnished_status FROM refined_listings WHERE furnished_status IS NOT NULL ORDER BY furnished_status").fetchall()
        ]
        price_bounds = conn.execute(
            "SELECT MIN(actual_price), MAX(actual_price) FROM refined_listings"
        ).fetchone()
        area_bounds = conn.execute(
            "SELECT MIN(area_sqm), MAX(area_sqm) FROM refined_listings"
        ).fetchone()
        return {
            "cities": cities,
            "neighborhoods": neighborhoods,
            "property_types": property_types,
            "furnished_opts": furnished_opts,
            "price_min": int(price_bounds[0] or 0),
            "price_max": int(price_bounds[1] or 50_000_000),
            "area_min": int(area_bounds[0] or 0),
            "area_max": int(area_bounds[1] or 1000),
        }
    finally:
        conn.close()


def get_stats() -> dict:
    """Return high-level stats for the Explore Market dashboard."""
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM refined_listings").fetchone()[0]
        raw_total = conn.execute("SELECT COUNT(*) FROM raw_listings").fetchone()[0]
        low_quality = conn.execute(
            "SELECT COUNT(*) FROM refined_listings WHERE data_quality_score = 'low'"
        ).fetchone()[0]
        latest = conn.execute(
            "SELECT MAX(refined_at) FROM refined_listings"
        ).fetchone()[0]
        avg_ppsqm_by_neighborhood = conn.execute(
            """SELECT neighborhood, ROUND(AVG(price_per_sqm)) as avg_ppsqm
               FROM refined_listings
               WHERE price_per_sqm IS NOT NULL
               GROUP BY neighborhood
               ORDER BY avg_ppsqm DESC"""
        ).fetchall()
        return {
            "total_refined": total,
            "total_raw": raw_total,
            "low_quality_count": low_quality,
            "outlier_pct": round((low_quality / max(total, 1)) * 100, 1),
            "last_update": latest,
            "neighborhood_avg_ppsqm": {r["neighborhood"]: r["avg_ppsqm"] for r in avg_ppsqm_by_neighborhood},
        }
    finally:
        conn.close()


# ── CLI bootstrap ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    count = seed_mock_data(n=50, force=True)
    print(f"✅ Database initialised at: {DB_PATH}")
    print(f"   Inserted {count} mock listings into raw_listings + refined_listings.")

    # Quick sanity check
    stats = get_stats()
    print(f"   Total refined rows : {stats['total_refined']}")
    print(f"   Total raw rows     : {stats['total_raw']}")
    print(f"   Low-quality rows   : {stats['low_quality_count']}")
    print(f"\n   Avg price/sqm by neighborhood:")
    for hood, avg in stats["neighborhood_avg_ppsqm"].items():
        print(f"     {hood:20s} → {avg:>10,.0f} EGP/sqm")

"""
database.py – SQLite database layer seeded from test_data.csv.

On first import, automatically creates data/test_data.db from data/test_data.csv
if the database file does not yet exist.
"""

import sqlite3
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "test_data.db")
CSV_PATH = os.path.join(DATA_DIR, "test_data.csv")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the apartments table from test_data.csv if it doesn't exist."""
    if os.path.exists(DB_PATH):
        conn = get_connection()
        count = conn.execute("SELECT COUNT(*) FROM apartments").fetchone()[0]
        conn.close()
        if count > 0:
            return

    df = pd.read_csv(CSV_PATH)
    # Add an explicit integer ID column
    df.insert(0, "id", range(1, len(df) + 1))

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("apartments", conn, if_exists="replace", index=False)

    # Create indexes for fast filtering
    conn.execute("CREATE INDEX IF NOT EXISTS idx_city ON apartments(city);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_governorate ON apartments(governorate);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_property_type ON apartments(property_type);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_bedrooms ON apartments(bedrooms);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_price ON apartments(price_numeric);")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_area ON apartments(area_sqm);")
    conn.commit()
    conn.close()
    print(f"[OK] SQLite database created at {DB_PATH} with {len(df)} rows.")


def get_filter_options() -> dict:
    """Return distinct values for every dropdown filter in the search UI."""
    conn = get_connection()
    try:
        def _distinct(col):
            rows = conn.execute(
                f"SELECT DISTINCT [{col}] FROM apartments WHERE [{col}] IS NOT NULL ORDER BY [{col}]"
            ).fetchall()
            return [r[0] for r in rows]

        price_bounds = conn.execute(
            "SELECT MIN(price_numeric), MAX(price_numeric) FROM apartments"
        ).fetchone()
        area_bounds = conn.execute(
            "SELECT MIN(area_sqm), MAX(area_sqm) FROM apartments WHERE area_sqm > 0"
        ).fetchone()
        bed_bounds = conn.execute(
            "SELECT MIN(bedrooms), MAX(bedrooms) FROM apartments"
        ).fetchone()
        bath_bounds = conn.execute(
            "SELECT MIN(bathrooms), MAX(bathrooms) FROM apartments"
        ).fetchone()

        return {
            "cities": _distinct("city"),
            "governorates": _distinct("governorate"),
            "property_types": _distinct("property_type"),
            "finish_types": _distinct("finish_type"),
            "view_types": _distinct("view_type"),
            "price_min": int(price_bounds[0] or 0),
            "price_max": int(price_bounds[1] or 50_000_000),
            "area_min": int(area_bounds[0] or 0),
            "area_max": int(area_bounds[1] or 1000),
            "bed_min": int(bed_bounds[0] or 1),
            "bed_max": int(bed_bounds[1] or 10),
            "bath_min": int(bath_bounds[0] or 1),
            "bath_max": int(bath_bounds[1] or 6),
        }
    finally:
        conn.close()


def search_apartments(
    *,
    city: str | None = None,
    governorate: str | None = None,
    property_type: str | None = None,
    finish_type: str | None = None,
    min_price: float | None = None,
    max_price: float | None = None,
    min_area: float | None = None,
    max_area: float | None = None,
    min_beds: int | None = None,
    max_beds: int | None = None,
    min_baths: int | None = None,
    max_baths: int | None = None,
    furnished: str | None = None,
    ownership: str | None = None,
    payment: str | None = None,
    completion: str | None = None,
    sort_by: str = "price_numeric",
    sort_dir: str = "ASC",
    limit: int = 100,
    offset: int = 0,
) -> list[dict]:
    """Search apartments with filters. Returns list of dicts."""
    clauses: list[str] = []
    params: list = []

    if city:
        clauses.append("city = ?")
        params.append(city)
    if governorate:
        clauses.append("governorate = ?")
        params.append(governorate)
    if property_type:
        clauses.append("property_type = ?")
        params.append(property_type)
    if finish_type:
        clauses.append("finish_type = ?")
        params.append(finish_type)
    if min_price is not None:
        clauses.append("price_numeric >= ?")
        params.append(min_price)
    if max_price is not None:
        clauses.append("price_numeric <= ?")
        params.append(max_price)
    if min_area is not None:
        clauses.append("area_sqm >= ?")
        params.append(min_area)
    if max_area is not None:
        clauses.append("area_sqm <= ?")
        params.append(max_area)
    if min_beds is not None:
        clauses.append("bedrooms >= ?")
        params.append(min_beds)
    if max_beds is not None:
        clauses.append("bedrooms <= ?")
        params.append(max_beds)
    if min_baths is not None:
        clauses.append("bathrooms >= ?")
        params.append(min_baths)
    if max_baths is not None:
        clauses.append("bathrooms <= ?")
        params.append(max_baths)

    # OHE boolean filters
    if furnished == "yes":
        clauses.append("furnished_yes = 1")
    elif furnished == "no":
        clauses.append("furnished_no = 1")
    if ownership == "primary":
        clauses.append("ownership_primary = 1")
    elif ownership == "resale":
        clauses.append("ownership_resale = 1")
    if payment == "cash":
        clauses.append("payment_option_cash = 1")
    elif payment == "installment":
        clauses.append("payment_option_installment = 1")
    elif payment == "cash or installment":
        clauses.append("[payment_option_cash or installment] = 1")
    if completion == "ready":
        clauses.append("completion_status_ready = 1")
    elif completion == "off-plan":
        clauses.append("[completion_status_off-plan] = 1")

    _SORTABLE = {"price_numeric", "area_sqm", "bedrooms", "bathrooms"}
    if sort_by not in _SORTABLE:
        sort_by = "price_numeric"
    direction = "DESC" if sort_dir.upper() == "DESC" else "ASC"

    where = " AND ".join(clauses) if clauses else "1=1"
    sql = f"SELECT * FROM apartments WHERE {where} ORDER BY [{sort_by}] {direction} LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    conn = get_connection()
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_apartments_by_ids(ids: list[int]) -> list[dict]:
    """Return full details for a list of apartment IDs."""
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    sql = f"SELECT * FROM apartments WHERE id IN ({placeholders})"
    conn = get_connection()
    try:
        rows = conn.execute(sql, ids).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_cities_for_governorate(governorate: str) -> list[str]:
    """Return distinct cities for a given governorate (cascading filter)."""
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT city FROM apartments WHERE governorate = ? ORDER BY city",
            (governorate,),
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


# Auto-init on import
init_db()

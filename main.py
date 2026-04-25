from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from utils.config import APP_NAME, VERSION, preprocessor, target_encoder, lgb_model
from utils.ApartmentData import ApartmentData
from utils.inference import predict_new
from utils.database import (
    search_apartments,
    get_filter_options,
    get_apartments_by_ids,
    get_cities_for_governorate,
)
import pandas as pd
import os

app = FastAPI(title=APP_NAME, version=VERSION)

# Allow Streamlit to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ── Cache the processed CSV in memory ────────────────────────────────────────
_explore_df = None


def _get_explore_df() -> pd.DataFrame:
    global _explore_df
    if _explore_df is None:
        _explore_df = pd.read_csv(os.path.join(DATA_DIR, "processed_dubizzle_data.csv"))
    return _explore_df


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/", tags=["General"])
async def home():
    return {"msg": f"Welcome to the {APP_NAME} API v{VERSION}"}


# ═══════════════════════════════════════════════════════════════════════════════
# EXPLORE MARKET
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/explore/stats", tags=["Explore"])
async def explore_stats():
    """Return aggregated statistics from the full processed dataset."""
    df = _get_explore_df()
    return {
        "total_listings": len(df),
        "avg_price": float(df["price_numeric"].mean()),
        "median_price": float(df["price_numeric"].median()),
        "avg_area": float(df["area_sqm"].median()),
        "governorates": sorted(df["governorate"].unique().tolist()),
        "num_governorates": int(df["governorate"].nunique()),
        "num_cities": int(df["city"].nunique()),
        "property_types": df["property_type"].value_counts().to_dict(),
        "avg_price_by_governorate": df.groupby("governorate")["price_numeric"]
        .mean()
        .round(0)
        .to_dict(),
        "avg_price_per_sqm_by_city": df[df["area_sqm"] > 0]
        .assign(ppsqm=lambda x: x["price_numeric"] / x["area_sqm"])
        .groupby("city")["ppsqm"]
        .median()
        .round(0)
        .sort_values(ascending=False)
        .head(25)
        .to_dict(),
    }


@app.get("/api/explore/data", tags=["Explore"])
async def explore_data(governorate: str | None = None, limit: int = 5000):
    """Return processed dataset rows as JSON (with optional governorate filter)."""
    df = _get_explore_df()
    if governorate:
        df = df[df["governorate"] == governorate]
    df = df.head(limit)
    return df.to_dict(orient="records")


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH (SQLite backed)
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/search/filters", tags=["Search"])
async def search_filters():
    """Return distinct values for every filter dropdown."""
    return get_filter_options()


@app.get("/api/search/cities", tags=["Search"])
async def cities_for_gov(governorate: str):
    """Return cities for a given governorate (cascading filter)."""
    return get_cities_for_governorate(governorate)


@app.get("/api/search", tags=["Search"])
async def search(
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
):
    """Search apartments in the test dataset with filters."""
    return search_apartments(
        city=city,
        governorate=governorate,
        property_type=property_type,
        finish_type=finish_type,
        min_price=min_price,
        max_price=max_price,
        min_area=min_area,
        max_area=max_area,
        min_beds=min_beds,
        max_beds=max_beds,
        min_baths=min_baths,
        max_baths=max_baths,
        furnished=furnished,
        ownership=ownership,
        payment=payment,
        completion=completion,
        sort_by=sort_by,
        sort_dir=sort_dir,
        limit=limit,
        offset=offset,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# COMPARE
# ═══════════════════════════════════════════════════════════════════════════════
@app.get("/api/compare", tags=["Compare"])
async def compare(ids: str = Query(..., description="Comma-separated apartment IDs")):
    """Return full details for selected apartment IDs."""
    try:
        id_list = [int(x.strip()) for x in ids.split(",") if x.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="IDs must be comma-separated integers")
    if not id_list:
        raise HTTPException(status_code=400, detail="At least one ID is required")
    return get_apartments_by_ids(id_list)


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION (no auth)
# ═══════════════════════════════════════════════════════════════════════════════
@app.post("/models/lgb_baseline", tags=["LightGBM"])
async def lgb_predict(data: ApartmentData) -> dict:
    if lgb_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded properly.")
    try:
        result = predict_new(
            data=data,
            preprocessor=preprocessor,
            target_encoder=target_encoder,
            model=lgb_model,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

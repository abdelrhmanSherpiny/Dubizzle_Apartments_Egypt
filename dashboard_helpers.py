"""Helper functions for the Streamlit dashboard."""
import requests, pandas as pd, numpy as np

API = "http://127.0.0.1:8000"

def api_get(path, params=None):
    try: return requests.get(f"{API}{path}", params=params, timeout=30).json()
    except: return None

def api_post(path, json_data):
    try: return requests.post(f"{API}{path}", json=json_data, timeout=30).json()
    except: return None

def get_explore_stats(): return api_get("/api/explore/stats")
def get_explore_data(gov=None): return api_get("/api/explore/data", {"governorate": gov, "limit": 5000} if gov else {"limit": 5000})
def get_filters(): return api_get("/api/search/filters")
def get_cities(gov): return api_get("/api/search/cities", {"governorate": gov})
def search(params): return api_get("/api/search", params)
def compare(ids): return api_get("/api/compare", {"ids": ",".join(str(i) for i in ids)})
def predict(apt_dict): return api_post("/models/lgb_baseline", apt_dict)

def price_verdict(predicted, actual):
    if actual == 0 or predicted == 0: return "---", "#888", 0
    delta = (actual - predicted) / predicted * 100
    if abs(delta) < 10: return "Fair Price", "#2ecc71", delta
    if delta < -10: return "Great Deal", "#3498db", delta
    if delta < 30: return "Overpriced", "#f39c12", delta
    return "Possibly a Scam", "#e74c3c", delta

def build_predict_payload(apt):
    return {
        "bedrooms": apt["bedrooms"], "bathrooms": apt["bathrooms"],
        "area_numeric": apt["area_numeric"], "latitude": apt["latitude"],
        "longitude": apt["longitude"], "property_type": apt["property_type"],
        "city": apt["city"], "governorate": apt["governorate"],
        "finish_type": apt["finish_type"], "seller_name": apt.get("seller_name","unknown"),
        "view_type": apt.get("view_type","unknown"), "compound_name": apt.get("compound_name","unknown"),
        "delivery_date": apt.get("delivery_date", 0.0),
        "seller_type": "agency" if apt.get("seller_type_agency") else "individual",
        "furnished": "yes" if apt.get("furnished_yes") else ("no" if apt.get("furnished_no") else "unknown"),
        "ownership": "primary" if apt.get("ownership_primary") else "resale",
        "payment_option": "cash" if apt.get("payment_option_cash") else ("installment" if apt.get("payment_option_installment") else "cash or installment"),
        "completion_status": "ready" if apt.get("completion_status_ready") else "off-plan",
        "Electricity Meter": bool(apt.get("Electricity Meter",0)),
        "Water Meter": bool(apt.get("Water Meter",0)),
        "Natural Gas": bool(apt.get("Natural Gas",0)),
        "Security": bool(apt.get("Security",0)),
        "Covered Parking": bool(apt.get("Covered Parking",0)),
        "Pets Allowed": bool(apt.get("Pets Allowed",0)),
        "Landline": bool(apt.get("Landline",0)),
        "Balcony": bool(apt.get("Balcony",0)),
        "Private Garden": bool(apt.get("Private Garden",0)),
        "Pool": bool(apt.get("Pool",0)),
        "Built in Kitchen Appliances": bool(apt.get("Built in Kitchen Appliances",0)),
        "Elevator": bool(apt.get("Elevator",0)),
        "Central A/C & heating": bool(apt.get("Central A/C & heating",0)),
        "Maids Room": bool(apt.get("Maids Room",0)),
        "roof": bool(apt.get("roof",0)),
    }

_CITY_COORDS = None
def get_city_coords():
    global _CITY_COORDS
    if _CITY_COORDS is None:
        data = get_explore_data()
        if data:
            df = pd.DataFrame(data)
            _CITY_COORDS = df.groupby("city")[["latitude","longitude"]].median().to_dict("index")
        else: _CITY_COORDS = {}
    return _CITY_COORDS

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 1.2rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }
[data-testid="stSidebar"] * { color: #ccc !important; }
h1 { background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.2rem !important; }
h2, h3 { color: #e0e0e0 !important; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; background: rgba(30,30,50,0.6); border-radius: 14px; padding: 5px 8px;
    backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); }
.stTabs [data-baseweb="tab"] { border-radius: 10px; padding: 10px 24px; font-weight: 600; font-size: 0.95rem;
    transition: all 0.3s ease; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #667eea, #764ba2) !important;
    box-shadow: 0 4px 15px rgba(102,126,234,0.4); }
.kpi-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 18px;
    padding: 1.4rem 1rem; text-align: center; border: 1px solid rgba(102,126,234,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3); transition: transform 0.3s, box-shadow 0.3s; }
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(102,126,234,0.2); }
.kpi-card h2 { background: linear-gradient(135deg, #f1c40f, #f39c12); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent; font-size: 1.8rem; margin: 0; font-weight: 800; }
.kpi-card p { color: #8892b0; font-size: 0.82rem; margin: 6px 0 0; font-weight: 500; letter-spacing: 0.5px; text-transform: uppercase; }
.apt-card { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 18px;
    padding: 1.4rem 1.6rem; color: #e0e0e0; box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    margin-bottom: 1rem; border: 1px solid rgba(255,255,255,0.04);
    transition: transform 0.3s ease, box-shadow 0.3s ease; }
.apt-card:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(102,126,234,0.15); border-color: rgba(102,126,234,0.2); }
.apt-price { font-size: 1.6rem; font-weight: 800; background: linear-gradient(135deg, #f1c40f, #f39c12);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0.3rem 0; }
.apt-specs { display: flex; gap: 1.2rem; flex-wrap: wrap; padding: 0.6rem 0;
    border-top: 1px solid rgba(255,255,255,0.06); margin-top: 0.5rem; font-size: 0.85rem; color: #8892b0; }
.verdict-badge { display: inline-block; padding: 5px 14px; border-radius: 20px;
    font-weight: 700; font-size: 0.82rem; letter-spacing: 0.3px; }
.section-divider { height: 2px; background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
    margin: 1.5rem 0; border: none; }
</style>
"""

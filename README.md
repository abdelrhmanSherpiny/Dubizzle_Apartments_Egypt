# 🏠 Dubizzle Apartments Egypt — Big Data Analytics & ML Pipeline

A full-stack big data project that scrapes, processes, analyzes, and predicts apartment prices across Egypt using data from Dubizzle.

## 📋 Project Overview

| Component | Technology | Description |
|-----------|-----------|-------------|
| **Web Scraping** | Python, Selenium | Scraped 70,000+ apartment listings from Dubizzle Egypt |
| **Feature Extraction** | Regex + Gemini LLM | Extracted structured features from raw listing text |
| **Data Processing** | Pandas, NumPy | Cleaned, validated, and engineered 49 features |
| **ML Model** | LightGBM | Trained on 60K+ listings, tested on 10K holdout set |
| **Backend API** | FastAPI | REST API serving predictions and data queries |
| **Database** | SQLite | Auto-seeded from test data for fast search queries |
| **Dashboard** | Streamlit, Plotly, Folium | Interactive 4-tab analytics dashboard |

## 🏗️ Architecture

```
Dubizzle_Apartments_Egypt/
├── app.py                    # Streamlit dashboard (4 tabs)
├── dashboard_helpers.py      # API helpers, styling, utilities
├── main.py                   # FastAPI backend
├── requirements.txt          # Python dependencies
├── data/
│   ├── processed_dubizzle_data.csv   # Full dataset (70K rows)
│   └── test_data.csv                 # Test split (10K rows)
├── models/
│   ├── lgb_baseline.txt              # LightGBM model
│   ├── target_encoder.pkl            # Category encoder
│   └── standard_scaler.pkl           # Feature scaler
├── notebooks/
│   └── Dubizzle_Apartments_Enhanced_SOTA.ipynb
├── utils/
│   ├── ApartmentData.py      # Pydantic input schema
│   ├── config.py             # Model loading & config
│   ├── database.py           # SQLite layer (auto-seeds)
│   └── inference.py          # Feature engineering + prediction
└── web_scrapping/            # Scraping scripts
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install streamlit plotly streamlit-folium folium
```

### 2. Start the Backend API
```bash
uvicorn main:app --reload --port 8000
```

### 3. Start the Dashboard
```bash
streamlit run app.py --server.port 8501
```

### 4. Open in Browser
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

## 📊 Dashboard Tabs

| Tab | Description |
|-----|-------------|
| 🗺️ **Explore Market** | Interactive heatmap, price insights, market overview with filterable charts |
| 🔍 **Search** | Filter 10K+ apartments, get ML price predictions, identify deals vs scams |
| ⚖️ **Compare** | Side-by-side comparison with radar charts and deal ranking |
| 💰 **Price Predictor** | Enter property specs, get instant LightGBM price estimate |

## 🤖 ML Pipeline

- **Model**: LightGBM (`lgb_baseline.txt`)
- **Features**: 48 engineered features including interaction terms, amenity scores, delivery buckets
- **Preprocessing**: Target encoding for categorical features, StandardScaler for numerics
- **Target**: `log1p(price)` — predictions are inverse-transformed via `expm1()`

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/explore/stats` | Market statistics |
| `GET` | `/api/explore/data` | Full dataset (with optional governorate filter) |
| `GET` | `/api/search` | Search apartments with filters |
| `GET` | `/api/search/filters` | Available filter values |
| `GET` | `/api/search/cities` | Cities for a governorate |
| `GET` | `/api/compare` | Compare apartments by IDs |
| `POST` | `/models/lgb_baseline` | Predict apartment price |

## 👥 Team

Big Data Course Project — Senior 1, Second Term

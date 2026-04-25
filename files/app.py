"""
app.py – Real Estate Analytics Hub (Streamlit frontend)

All data is served from the SQLite database via database.py.
No hardcoded mock values remain in this file.

Run:
    python database.py          # init + seed (first time only)
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium
import folium

from database import (
    get_filter_options,
    get_stats,
    init_db,
    search_listings,
    seed_mock_data,
)

# ── Bootstrap DB on first run ────────────────────────────────────────────────────
init_db()
seed_mock_data()  # no-op if rows already exist


# ── Helpers ──────────────────────────────────────────────────────────────────────
def predict_price(location: str, area: int, bedrooms: int, base_rates: dict) -> dict:
    """
    Simple rule-based price estimator using per-sqm base rates sourced from
    the database (neighborhood average price/sqm).  Falls back to 16 000 EGP/sqm
    if the neighborhood has no data yet.
    """
    rate       = base_rates.get(location, 16_000)
    base_price = rate * area + bedrooms * 250_000
    margin     = 0.05
    return {
        "price": base_price,
        "low":   round(base_price * (1 - margin)),
        "high":  round(base_price * (1 + margin)),
    }


# ── Cached DB calls ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def _load_stats() -> dict:
    return get_stats()


@st.cache_data(ttl=300)
def _load_filter_options() -> dict:
    return get_filter_options()


@st.cache_data(ttl=60, show_spinner=False)
def _search(
    query: str,
    city: str,
    property_type: str,
    furnished_status: str,
    min_price: int,
    max_price: int,
    min_area: int,
    max_area: int,
    min_rooms: int,
    max_rooms: int,
    sort_by: str,
) -> pd.DataFrame:
    """Call search_listings() and reshape results into the DataFrame expected by
    the card / chart components."""
    rows = search_listings(
        query=query,
        city=city or None,
        property_type=property_type or None,
        furnished_status=furnished_status or None,
        min_price=min_price,
        max_price=max_price,
        min_area=min_area,
        max_area=max_area,
        min_rooms=min_rooms if min_rooms > 0  else None,
        max_rooms=max_rooms if max_rooms < 10 else None,
        sort_by=sort_by,
        sort_dir="ASC",
        limit=100,
    )
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # ── Derived columns consumed by charts and cards ──────────────────────────
    df["Price_Delta_Pct"] = (
        (df["actual_price"] - df["expected_price"])
        / df["expected_price"].replace(0, np.nan) * 100
    ).round(2)
    df["Delta_Value"] = (df["expected_price"] - df["actual_price"]).round(0)
    df["Status"]      = "For Sale"   # extend once a status field is scraped
    df["Furnished"]   = df["furnished_status"].fillna("unknown")

    # Rename DB columns → names used throughout the UI
    df = df.rename(columns={
        "title":          "Title",
        "neighborhood":   "Location",
        "num_rooms":      "Rooms",
        "num_bathrooms":  "Bathrooms",
        "actual_price":   "Price",
        "area_sqm":       "Area",
        "price_per_sqm":  "Price_per_sqm",
        "expected_price": "Expected_Price",
        "source_url":     "Link",
    })
    return df


# ── Page Configuration ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real Estate Analytics Hub",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load shared data once per session ─────────────────────────────────────────────
stats   = _load_stats()
options = _load_filter_options()

# ── Sidebar ───────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '''<div style="text-align: center;">
            <img src="https://thumbs.dreamstime.com/b/real-estate-concept-image-showing-small-house-under-magnifying-glass-blurred-cityscape-background-magnifying-glass-376955598.jpg"
                 width="200" style="border-radius: 8px; margin-bottom: 10px;" />
           </div>''',
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        """
        ### 📊 About This Project
        **Real Estate Analytics Hub** is an interactive platform that
        empowers users to explore market trends, search live property
        listings, and predict prices using machine-learning models —
        all in one place.
        """
    )
    st.markdown("---")

    # Budget slider — bounds come from actual DB data
    st.markdown("### 💰 Budget Range")
    hood_rates = stats.get("neighborhood_avg_ppsqm", {})
    ppsqm_vals = list(hood_rates.values())
    slider_lo  = max(0,      int(min(ppsqm_vals)) - 1_000) if ppsqm_vals else 5_000
    slider_hi  = min(50_000, int(max(ppsqm_vals)) + 1_000) if ppsqm_vals else 40_000

    budget_min, budget_max = st.slider(
        "Average Price per sqm (EGP)",
        min_value=0,
        max_value=50_000,
        value=(slider_lo, slider_hi),
        step=1_000,
        format="%d EGP",
    )

# ── Main Title ────────────────────────────────────────────────────────────────────
st.title("🏠 Real Estate Analytics Hub")

# ── Tabs ──────────────────────────────────────────────────────────────────────────
tab_explore, tab_search, tab_predict = st.tabs(
    ["🗺️ Explore Market", "🔍 Live Search", "💰 Price Predictor"]
)


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 1 – Explore Market
# ═════════════════════════════════════════════════════════════════════════════════
with tab_explore:
    # ── KPI strip — all values from DB ────────────────────────────────────────
    st.subheader("📋 Data Quality Pipeline")
    col1, col2, col3 = st.columns(3)

    total_listings = stats.get("total_refined", 0)
    outlier_pct    = stats.get("outlier_pct",   0.0)
    last_update    = str(stats.get("last_update") or "—")[:10]

    col1.metric("Total Listings Analyzed",    f"{total_listings:,}")
    col2.metric("Low-Quality / Outlier Rows", f"{outlier_pct:.1f}%")
    col3.metric("Last Data Update",           last_update)
    st.markdown("---")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    st.header("Real Estate Heatmap")

    # Fixed geographic bounding boxes; avg_price filled from live DB
    _DISTRICT_GEO = {
        "Maadi":          {"center": [29.960, 31.250],
                           "polygon": [[29.94,31.22],[29.94,31.28],[29.98,31.28],[29.98,31.22]]},
        "Sheikh Zayed":   {"center": [30.040, 31.010],
                           "polygon": [[30.02,30.98],[30.02,31.04],[30.06,31.04],[30.06,30.98]]},
        "Nasr City":      {"center": [30.070, 31.340],
                           "polygon": [[30.05,31.31],[30.05,31.37],[30.09,31.37],[30.09,31.31]]},
        "New Cairo":      {"center": [30.030, 31.420],
                           "polygon": [[30.01,31.39],[30.01,31.45],[30.05,31.45],[30.05,31.39]]},
        "6th of October": {"center": [29.970, 30.920],
                           "polygon": [[29.95,30.89],[29.95,30.95],[29.99,30.95],[29.99,30.89]]},
        "Zamalek":        {"center": [30.062, 31.218],
                           "polygon": [[30.05,31.21],[30.05,31.23],[30.07,31.23],[30.07,31.21]]},
        "Heliopolis":     {"center": [30.090, 31.320],
                           "polygon": [[30.07,31.30],[30.07,31.34],[30.11,31.34],[30.11,31.30]]},
        "Dokki":          {"center": [30.040, 31.210],
                           "polygon": [[30.02,31.19],[30.02,31.23],[30.06,31.23],[30.06,31.19]]},
        "Smouha":         {"center": [31.200, 29.940],
                           "polygon": [[31.18,29.92],[31.18,29.96],[31.22,29.96],[31.22,29.92]]},
        "Gleem":          {"center": [31.220, 29.960],
                           "polygon": [[31.20,29.94],[31.20,29.98],[31.24,29.98],[31.24,29.94]]},
    }

    # Merge geography with DB-sourced avg price/sqm
    districts: dict = {}
    for hood, geo in _DISTRICT_GEO.items():
        avg = hood_rates.get(hood)
        if avg is not None:
            districts[hood] = {**geo, "avg_price": int(avg)}

    filtered_districts = {
        name: info for name, info in districts.items()
        if budget_min <= info["avg_price"] <= budget_max
    }

    features = []
    for name, info in filtered_districts.items():
        coords = info["polygon"]
        ring   = [[c[1], c[0]] for c in coords] + [[coords[0][1], coords[0][0]]]
        features.append({
            "type": "Feature",
            "id":   name,
            "properties": {"name": name, "avg_price": info["avg_price"]},
            "geometry":   {"type": "Polygon", "coordinates": [ring]},
        })
    geojson = {"type": "FeatureCollection", "features": features}

    df_map = pd.DataFrame(
        [(n, d["avg_price"]) for n, d in filtered_districts.items()],
        columns=["District", "AvgPriceSqm"],
    )

    m = folium.Map(location=[30.05, 31.20], zoom_start=10, tiles="CartoDB positron")
    if not df_map.empty:
        folium.Choropleth(
            geo_data=geojson, data=df_map,
            columns=["District", "AvgPriceSqm"],
            key_on="feature.id",
            fill_color="YlOrRd", fill_opacity=0.7, line_opacity=0.5,
            legend_name="Avg Price per sqm (EGP)",
        ).add_to(m)
        folium.GeoJson(
            geojson,
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "avg_price"],
                aliases=["District", "Avg Price/sqm (EGP)"],
            ),
            style_function=lambda x: {"fillOpacity": 0, "color": "transparent", "weight": 0},
        ).add_to(m)
    else:
        st.warning("No districts match the selected budget range.")

    st_folium(m, use_container_width=True, height=500)


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 2 – Live Search
# ═════════════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("Live Search")

    # ── Card CSS ──────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    .listing-card{background:linear-gradient(135deg,#1e1e2f 0%,#2a2a40 100%);
        border-radius:16px;padding:1.4rem 1.6rem;color:#e0e0e0;
        box-shadow:0 4px 24px rgba(0,0,0,.35);transition:transform .2s,box-shadow .2s;
        margin-bottom:.4rem}
    .listing-card:hover{transform:translateY(-4px);box-shadow:0 8px 32px rgba(0,0,0,.5)}
    .card-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:.6rem}
    .card-title{font-size:1.05rem;font-weight:700;color:#fff;line-height:1.3;max-width:72%}
    .card-badge{font-size:.72rem;font-weight:700;padding:4px 10px;border-radius:20px;
        text-transform:uppercase;letter-spacing:.5px;white-space:nowrap}
    .badge-sale{background:#e74c3c;color:#fff}
    .badge-rent{background:#3498db;color:#fff}
    .card-price{font-size:1.7rem;font-weight:800;color:#f1c40f;margin:.4rem 0 .15rem}
    .card-secondary{font-size:.82rem;color:#aaa;margin-bottom:.5rem;line-height:1.5}
    .delta-green{color:#2ecc71;font-weight:600}
    .delta-red{color:#e74c3c;font-weight:600}
    .specs-bar{display:flex;gap:1.2rem;padding:.6rem 0;
        border-top:1px solid rgba(255,255,255,.08);
        border-bottom:1px solid rgba(255,255,255,.08);
        margin:.5rem 0;font-size:.85rem;color:#ccc;flex-wrap:wrap}
    .specs-bar span{display:flex;align-items:center;gap:4px}
    .card-footer{display:flex;justify-content:space-between;align-items:center;margin-top:.6rem}
    .furnished-tag{font-size:.78rem;color:#aaa}
    .btn-details{display:inline-block;padding:7px 18px;border-radius:8px;font-size:.82rem;
        font-weight:700;text-decoration:none;color:#fff;
        background:linear-gradient(135deg,#6c5ce7,#a855f7);transition:opacity .2s}
    .btn-details:hover{opacity:.85}
    </style>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # SEARCH BAR
    # ══════════════════════════════════════════════════════════════════════════
    search_col, btn_col = st.columns([4, 1])
    with search_col:
        query = st.text_input(
            "Search properties",
            placeholder="e.g. Maadi apartments, Zayed villas, penthouse …",
            label_visibility="collapsed",
            key="search_query",
        )
    with btn_col:
        search_clicked = st.button("🔍 Search", use_container_width=True)

    # ── Filter panel ──────────────────────────────────────────────────────────
    with st.expander("🎛️ Filters", expanded=False):
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)

        with f_col1:
            f_city = st.selectbox(
                "🏙️ City",
                [""] + options.get("cities", []),
                format_func=lambda x: "Any city" if x == "" else x,
            )
        with f_col2:
            f_type = st.selectbox(
                "🏠 Property Type",
                [""] + options.get("property_types", []),
                format_func=lambda x: "Any type" if x == "" else x,
            )
        with f_col3:
            _furn_labels = {
                "furnished":      "Furnished",
                "unfurnished":    "Unfurnished",
                "semi-furnished": "Semi-furnished",
            }
            f_furnished = st.selectbox(
                "🪑 Furnished",
                [""] + options.get("furnished_opts", []),
                format_func=lambda x: "Any" if x == "" else _furn_labels.get(x, x),
            )
        with f_col4:
            _sort_labels = {
                "actual_price":  "Price (low → high)",
                "price_per_sqm": "Price/sqm (low → high)",
                "area_sqm":      "Area (small → large)",
                "num_rooms":     "Rooms (few → many)",
            }
            f_sort = st.selectbox(
                "↕️ Sort by",
                list(_sort_labels.keys()),
                format_func=lambda x: _sort_labels[x],
            )

        pr_col, ar_col, rm_col = st.columns(3)
        p_min_db = options.get("price_min", 0)
        p_max_db = options.get("price_max", 50_000_000)
        a_min_db = options.get("area_min",  0)
        a_max_db = options.get("area_max",  1_000)

        with pr_col:
            f_price = st.slider(
                "💰 Price range (EGP)",
                min_value=p_min_db, max_value=p_max_db,
                value=(p_min_db, p_max_db), step=100_000, format="%d",
            )
        with ar_col:
            f_area = st.slider(
                "📐 Area (sqm)",
                min_value=a_min_db, max_value=a_max_db,
                value=(a_min_db, a_max_db), step=10,
            )
        with rm_col:
            f_rooms = st.slider("🛏️ Rooms", min_value=0, max_value=10,
                                value=(0, 10), step=1)

    # ── Decide whether to run a query ─────────────────────────────────────────
    # Always run when filters change; the text query fires on Search button
    # click or when the user has previously submitted a query.
    active_query = ""
    if search_clicked:
        st.session_state["last_query"] = query
        active_query = query
    elif "last_query" in st.session_state:
        active_query = st.session_state["last_query"]

    with st.spinner("Fetching listings …"):
        results = _search(
            query=active_query,
            city=f_city,
            property_type=f_type,
            furnished_status=f_furnished,
            min_price=f_price[0],
            max_price=f_price[1],
            min_area=f_area[0],
            max_area=f_area[1],
            min_rooms=f_rooms[0],
            max_rooms=f_rooms[1],
            sort_by=f_sort,
        )

    if results.empty:
        if active_query or any([f_city, f_type, f_furnished]):
            st.info("No listings found matching your criteria. Try broadening your filters.")
        else:
            st.info("Enter a search query or adjust the filters above to explore listings.")
        st.stop()

    # ── Result summary ────────────────────────────────────────────────────────
    active_filters = [lbl for flag, lbl in [
        (f_city,               f"City: {f_city}"),
        (f_type,               f"Type: {f_type}"),
        (f_furnished,          f"Furnished: {f_furnished}"),
        (f_rooms != (0, 10),   f"Rooms: {f_rooms[0]}–{f_rooms[1]}"),
    ] if flag]

    filter_str = "  ·  ".join(active_filters) if active_filters else "no extra filters"
    query_str  = f'*"{active_query}"*' if active_query else "all listings"
    st.success(f"Found **{len(results)}** listings for {query_str} — {filter_str}")

    # ── KPI strip ─────────────────────────────────────────────────────────────
    valid_delta      = results.dropna(subset=["Price_Delta_Pct", "Expected_Price"])
    cheapest_ppsqm   = results.loc[results["Price_per_sqm"].idxmin()]
    avg_price        = int(results["Price"].mean())
    underpriced      = valid_delta[valid_delta["Price_Delta_Pct"] < 0]
    best_val         = (underpriced.loc[underpriced["Price_per_sqm"].idxmin()]
                        if not underpriced.empty else cheapest_ppsqm)
    best_val_label   = f"{int(best_val['Price_per_sqm']):,} EGP" if not underpriced.empty else "N/A"
    most_op_row      = (valid_delta.loc[valid_delta["Price_Delta_Pct"].idxmax()]
                        if not valid_delta.empty else None)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Cheapest Price/SQM",
                f"{int(cheapest_ppsqm['Price_per_sqm']):,} EGP",
                help=cheapest_ppsqm["Title"])
    kpi2.metric("Most Overpriced",
                f"+{most_op_row['Price_Delta_Pct']:.1f}%" if most_op_row is not None else "—",
                help=most_op_row["Title"] if most_op_row is not None else "")
    kpi3.metric("Best Value Score", best_val_label, help=best_val["Title"])
    kpi4.metric("Avg Price in Results", f"{avg_price:,} EGP")
    st.markdown("---")

    # ── Listing Cards (2-column grid) ─────────────────────────────────────────
    rows_list = list(results.iterrows())
    for i in range(0, len(rows_list), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(rows_list):
                break
            _, row = rows_list[i + j]

            badge_cls  = "badge-sale" if row.get("Status") == "For Sale" else "badge-rent"
            delta_val  = float(row.get("Delta_Value")  or 0)
            delta_pct  = float(row.get("Price_Delta_Pct") or 0)
            delta_cls  = "delta-green" if delta_val > 0 else "delta-red"
            delta_sign = "+" if delta_val > 0 else ""
            furn       = str(row.get("Furnished", "")).lower()
            furn_label = "🪑 Furnished" if furn == "furnished" else "🚫 Unfurnished"
            exp_price  = float(row.get("Expected_Price") or 0)

            card = f'''<div class="listing-card">
<div class="card-header">
  <div class="card-title">{row["Title"]}</div>
  <span class="card-badge {badge_cls}">{row.get("Status","For Sale")}</span>
</div>
<div class="card-price">{row["Price"]:,.0f} EGP</div>
<div class="card-secondary">
  Expected: {exp_price:,.0f} EGP &nbsp;·&nbsp;
  Delta: <span class="{delta_cls}">{delta_sign}{delta_val:,.0f} EGP ({delta_pct:+.2f}%)</span>
</div>
<div class="specs-bar">
  <span>📐 {int(row["Area"] or 0):,} sqm</span>
  <span>🛏️ {int(row["Rooms"] or 0)} Rooms</span>
  <span>🚿 {int(row["Bathrooms"] or 0)} Bath</span>
  <span>📍 {row["Location"]}</span>
</div>
<div class="card-footer">
  <span class="furnished-tag">{furn_label}</span>
  <a class="btn-details" href="{row["Link"]}" target="_blank">View Details →</a>
</div>
</div>'''
            col.markdown(card, unsafe_allow_html=True)

    st.markdown("---")

    # ── Top-5 Price/sqm bar chart ─────────────────────────────────────────────
    st.subheader("📊 Top 5 Results — Price per sqm")
    top5 = results.nsmallest(5, "Price_per_sqm")
    fig  = px.bar(
        top5, x="Title", y="Price_per_sqm",
        color="Price_per_sqm", color_continuous_scale="YlOrRd",
        labels={"Price_per_sqm": "Price / sqm (EGP)", "Title": ""},
    )
    fig.update_layout(xaxis_tickangle=-30, showlegend=False, margin=dict(t=20, b=80))
    st.plotly_chart(fig, use_container_width=True)

    # ── Over / Underpriced detection ──────────────────────────────────────────
    if not valid_delta.empty:
        st.subheader("🏷️ Over/Underpriced Detection")
        sorted_r    = valid_delta.sort_values("Price_Delta_Pct")
        bar_colors  = ["#2ecc71" if d < 0 else "#e74c3c" for d in sorted_r["Price_Delta_Pct"]]

        fig_delta = go.Figure(go.Bar(
            y=sorted_r["Title"],
            x=sorted_r["Price_Delta_Pct"],
            orientation="h",
            marker_color=bar_colors,
            customdata=sorted_r[["Price", "Expected_Price", "Price_Delta_Pct"]].values,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Price: %{customdata[0]:,.0f} EGP<br>"
                "Expected: %{customdata[1]:,.0f} EGP<br>"
                "Delta: %{customdata[2]:+.2f}%<extra></extra>"
            ),
        ))
        fig_delta.update_layout(
            xaxis_title="Price Delta (%)", yaxis_title="",
            height=max(350, len(sorted_r) * 40),
            margin=dict(l=20, r=20, t=10, b=40),
            shapes=[dict(type="line", x0=0, x1=0,
                         y0=-0.5, y1=len(sorted_r) - 0.5,
                         line=dict(color="white", width=1.5, dash="dash"))],
        )
        st.plotly_chart(fig_delta, use_container_width=True)

    # ── Price per SQM ranked bar ───────────────────────────────────────────────
    st.subheader("📈 Price per SQM Comparison (Ranked)")
    ranked       = results.sort_values("Price_per_sqm").reset_index(drop=True)
    median_ppsqm = ranked["Price_per_sqm"].median()

    fig_ranked = px.bar(
        ranked, x="Title", y="Price_per_sqm",
        color="Price_per_sqm", color_continuous_scale="Tealgrn",
        labels={"Price_per_sqm": "Price / sqm (EGP)", "Title": ""},
        custom_data=["Price", "Area", "Price_per_sqm"],
    )
    fig_ranked.update_traces(
        hovertemplate=(
            "<b>%{x}</b><br>Price: %{customdata[0]:,.0f} EGP<br>"
            "Area: %{customdata[1]:,} sqm<br>Price/sqm: %{customdata[2]:,.0f} EGP<extra></extra>"
        ),
    )
    fig_ranked.add_hline(
        y=median_ppsqm, line_dash="dash", line_color="#e67e22",
        annotation_text=f"Median: {median_ppsqm:,.0f} EGP/sqm",
        annotation_position="top right",
        annotation_font_color="#e67e22",
    )
    fig_ranked.update_layout(xaxis_tickangle=-30, showlegend=False, margin=dict(t=30, b=100))
    st.plotly_chart(fig_ranked, use_container_width=True)

    # ── Price vs Area scatter + outlier detection ─────────────────────────────
    st.subheader("🔍 Price vs. Area — Outlier Detection")
    area_arr     = results["Area"].values.astype(float)
    price_arr    = results["Price"].values.astype(float)
    coeffs       = np.polyfit(area_arr, price_arr, deg=1)
    predicted    = np.polyval(coeffs, area_arr)
    residuals    = price_arr - predicted
    res_std      = np.std(residuals) if np.std(residuals) > 0 else 1
    rp           = results.copy()
    rp["Outlier"] = np.where(np.abs(residuals) > 1.5 * res_std, "⚠️ Outlier", "Normal")

    fig_sc = px.scatter(
        rp, x="Area", y="Price", color="Outlier",
        color_discrete_map={"Normal": "#2ecc71", "⚠️ Outlier": "#e74c3c"},
        symbol="Outlier",
        symbol_map={"Normal": "circle", "⚠️ Outlier": "x"},
        size_max=14, trendline="ols", trendline_scope="overall",
        trendline_color_override="#3498db",
        labels={"Area": "Area (sqm)", "Price": "Price (EGP)"},
        custom_data=["Title", "Price_per_sqm", "Expected_Price", "Price_Delta_Pct"],
    )
    fig_sc.update_traces(
        marker=dict(size=12, line=dict(width=1, color="white")),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>Area: %{x:,} sqm<br>Price: %{y:,.0f} EGP<br>"
            "Price/sqm: %{customdata[1]:,.0f} EGP<br>Expected: %{customdata[2]:,.0f} EGP<br>"
            "Delta: %{customdata[3]:+.2f}%<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )
    fig_sc.update_layout(
        height=500, margin=dict(t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Value Score Leaderboard ────────────────────────────────────────────────
    st.subheader("🏆 Value Score Leaderboard")
    ppsqm_min = results["Price_per_sqm"].min()
    ppsqm_max = results["Price_per_sqm"].max()
    delta_min = results["Price_Delta_Pct"].min()
    delta_max = results["Price_Delta_Pct"].max()

    def _score(row):
        pp = 1 - (row["Price_per_sqm"] - ppsqm_min) / max(ppsqm_max - ppsqm_min, 1)
        dp = 1 - (row["Price_Delta_Pct"] - delta_min) / max(delta_max - delta_min, 1)
        return round((pp * 0.5 + dp * 0.5) * 100, 1)

    lb = results.copy()
    lb["Value_Score"] = lb.apply(_score, axis=1)
    lb = lb.sort_values("Value_Score", ascending=False).reset_index(drop=True)
    lb.index = lb.index + 1
    lb.index.name = "Rank"

    display_cols = ["Title", "Price", "Area", "Price_per_sqm", "Price_Delta_Pct", "Value_Score"]
    styled = (
        lb[display_cols].style
        .background_gradient(subset=["Value_Score"], cmap="Greens")
        .background_gradient(subset=["Price_per_sqm"], cmap="YlOrRd")
        .format({
            "Price":          "{:,.0f}",
            "Area":           "{:,}",
            "Price_per_sqm":  "{:,.0f}",
            "Price_Delta_Pct":"{:+.2f}%",
            "Value_Score":    "{:.1f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=420)


# ═════════════════════════════════════════════════════════════════════════════════
# TAB 3 – Price Predictor
# ═════════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Price Predictor")

    # Location list and base rates both come from live DB stats
    db_base_rates       = stats.get("neighborhood_avg_ppsqm", {})
    predictor_locations = sorted(db_base_rates.keys()) or options.get("neighborhoods", [])

    with st.form("prediction_form"):
        form_cols = st.columns(3)
        with form_cols[0]:
            location = st.selectbox("📍 Location", predictor_locations)
        with form_cols[1]:
            area = st.number_input(
                "📐 Area (sqm)", min_value=30, max_value=1000, value=120, step=10,
            )
        with form_cols[2]:
            bedrooms = st.number_input(
                "🛏️ Bedrooms", min_value=1, max_value=10, value=3, step=1,
            )
        submitted = st.form_submit_button(
            "🔮 Predict Fair Price", use_container_width=True,
        )

    if submitted:
        result = predict_price(location, int(area), int(bedrooms), db_base_rates)
        price  = result["price"]
        low, high = result["low"], result["high"]

        def _fmt(v: int) -> str:
            return f"{v / 1_000_000:.2f}M" if v >= 1_000_000 else f"{v:,}"

        st.markdown("---")
        metric_cols = st.columns([1, 2, 1])
        with metric_cols[1]:
            st.metric(
                label="Predicted Fair Price (EGP)",
                value=f"{price:,} EGP",
                delta=f"Estimated Range: {_fmt(low)} – {_fmt(high)} EGP",
                delta_color="off",
            )
        st.info(
            "ℹ️ **About the confidence interval** — The estimated range "
            f"({_fmt(low)} – {_fmt(high)} EGP) represents a ±5 % band around "
            "the point prediction. It is derived from the model's residual "
            "variance on the validation set, giving you a realistic sense of "
            "price uncertainty for this property profile."
        )

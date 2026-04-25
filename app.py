"""Dubizzle Apartments Analytics Dashboard"""
import numpy as np, pandas as pd, plotly.express as px, plotly.graph_objects as go
import streamlit as st, folium
from streamlit_folium import st_folium
from dashboard_helpers import *

st.set_page_config(page_title="Dubizzle Analytics Hub", page_icon="🏠", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🏠 Dubizzle Analytics")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("**Explore** the Egyptian apartment market, **search** real listings, **compare** properties, and **predict** prices with ML.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown("##### ⚡ Powered by LightGBM")
    st.markdown(f"##### 📊 70,000+ listings analyzed")

st.title("🏠 Dubizzle Apartments Analytics Hub")
tab_explore, tab_search, tab_compare, tab_predict = st.tabs(["🗺️ Explore Market", "🔍 Search", "⚖️ Compare", "💰 Price Predictor"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — EXPLORE
# ══════════════════════════════════════════════════════════════════════════
with tab_explore:
    stats = get_explore_stats()
    if not stats:
        st.error("Cannot connect to backend. Run: uvicorn main:app --reload --port 8000"); st.stop()
    k1,k2,k3,k4 = st.columns(4)
    for col,val,lbl in [(k1,f"{stats['total_listings']:,}","Total Listings"),(k2,f"{stats['median_price']:,.0f}","Median Price (EGP)"),
                         (k3,f"{stats['avg_area']:,.0f}","Median Area (sqm)"),(k4,f"{stats['num_governorates']}","Governorates")]:
        col.markdown(f'<div class="kpi-card"><h2>{val}</h2><p>{lbl}</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    gov_filter = st.selectbox("🌍 Filter by Governorate", ["All"] + stats["governorates"], key="explore_gov")
    raw = get_explore_data(None if gov_filter=="All" else gov_filter)
    if not raw: st.warning("No data."); st.stop()
    df = pd.DataFrame(raw)
    df_valid = df[df["area_sqm"]>0].copy()
    df_valid["ppsqm"] = df_valid["price_numeric"] / df_valid["area_sqm"]

    # ── HEATMAP — Where are apartments concentrated? ──
    st.subheader("🗺️ Where Are Apartments Priced Highest?")
    st.caption("Zoom in to explore. Red = expensive areas, Blue = affordable areas.")
    map_df = df[(df["latitude"].notna()) & (df["longitude"].notna()) & (df["area_sqm"]>0)].copy()
    map_df = map_df.sample(min(2000, len(map_df)), random_state=42)
    center_lat, center_lng = map_df["latitude"].median(), map_df["longitude"].median()
    m = folium.Map(location=[center_lat, center_lng], zoom_start=7, tiles="CartoDB dark_matter")
    from folium.plugins import HeatMap
    heat_data = [[r["latitude"], r["longitude"], r["price_numeric"]/1e6] for _,r in map_df.iterrows()]
    HeatMap(heat_data, radius=12, blur=15, max_zoom=13, gradient={0.2:'blue',0.4:'cyan',0.6:'lime',0.8:'yellow',1:'red'}).add_to(m)
    st_folium(m, use_container_width=True, height=480)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 1. Most Affordable Cities (user question: "Where can I get the best deal?") ──
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("💡 Most Affordable Cities")
        st.caption("Cities with the lowest median price per sqm — best value for your money")
        affordable = df_valid.groupby("city")["ppsqm"].median().sort_values().head(15).reset_index()
        affordable.columns = ["City","Price/sqm"]
        fig = px.bar(affordable, x="Price/sqm", y="City", orientation="h",
                     color="Price/sqm", color_continuous_scale="Greens_r",
                     labels={"Price/sqm":"Median EGP/sqm","City":""})
        fig.update_layout(showlegend=False, margin=dict(t=10,l=10), height=480,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 2. Most Expensive Cities (user question: "Where is the premium market?") ──
    with c2:
        st.subheader("🏆 Most Premium Cities")
        st.caption("Cities with the highest median price per sqm — luxury market")
        premium = df_valid.groupby("city")["ppsqm"].median().sort_values(ascending=True).tail(15).reset_index()
        premium.columns = ["City","Price/sqm"]
        fig = px.bar(premium, x="Price/sqm", y="City", orientation="h",
                     color="Price/sqm", color_continuous_scale="YlOrRd",
                     labels={"Price/sqm":"Median EGP/sqm","City":""})
        fig.update_layout(showlegend=False, margin=dict(t=10,l=10), height=480,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 3. What Can You Get For Your Budget? (interactive) ──
    st.subheader("🎯 What Can You Get For Your Budget?")
    st.caption("Select a budget to see the typical apartment you can afford")
    budget = st.slider("Set your budget (EGP)", 1_500_000, 30_000_000, 7_000_000, step=500_000, key="budget_slider", format="%d EGP")
    margin = 0.15
    budget_df = df_valid[(df_valid["price_numeric"] >= budget*(1-margin)) & (df_valid["price_numeric"] <= budget*(1+margin))]
    if len(budget_df) > 5:
        bc1,bc2,bc3,bc4 = st.columns(4)
        bc1.metric("Avg Area", f"{budget_df['area_sqm'].median():,.0f} sqm")
        bc2.metric("Avg Bedrooms", f"{budget_df['bedrooms'].median():.0f}")
        bc3.metric("Avg Bathrooms", f"{budget_df['bathrooms'].median():.0f}")
        bc4.metric("Listings Available", f"{len(budget_df):,}")
        top_cities = budget_df["city"].value_counts().head(5)
        fig = px.bar(top_cities.reset_index(), x="count", y="city", orientation="h",
                     color="count", color_continuous_scale="Purples",
                     labels={"count":"Listings","city":""})
        fig.update_layout(title="Top cities in your budget", showlegend=False, margin=dict(t=40,l=10), height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Few listings match this exact budget. Try adjusting the slider.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 4. Price by Bedrooms (user question: "How much more for an extra room?") ──
    c3,c4 = st.columns(2)
    with c3:
        st.subheader("🛏️ How Much More For an Extra Room?")
        st.caption("Median price by number of bedrooms — plan your upgrade")
        bed_price = df.groupby("bedrooms")["price_numeric"].median().reset_index()
        bed_price.columns = ["Bedrooms","Median Price"]
        fig = px.bar(bed_price, x="Bedrooms", y="Median Price", color="Median Price",
                     color_continuous_scale="Plasma", labels={"Median Price":"Median Price (EGP)"},
                     text=bed_price["Median Price"].apply(lambda x: f"{x/1e6:.1f}M"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10,b=40), plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 5. Property Types Available ──
    with c4:
        st.subheader("🏗️ What Types of Properties Are Available?")
        st.caption("Find the right property type for your lifestyle")
        pt = df["property_type"].value_counts().reset_index(); pt.columns=["Type","Count"]
        fig = px.pie(pt, values="Count", names="Type", hole=0.5, color_discrete_sequence=px.colors.sequential.Plasma_r)
        fig.update_traces(textposition='inside', textinfo='percent+label', textfont_size=11)
        fig.update_layout(margin=dict(t=10,b=10), paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── 6. Finish Type Impact on Price ──
    c5,c6 = st.columns(2)
    with c5:
        st.subheader("🔨 Does Finish Level Affect Price?")
        st.caption("Compare median prices across finish types")
        finish_price = df_valid.groupby("finish_type")["ppsqm"].median().sort_values().reset_index()
        finish_price.columns = ["Finish Type","Median EGP/sqm"]
        fig = px.bar(finish_price, x="Finish Type", y="Median EGP/sqm", color="Median EGP/sqm",
                     color_continuous_scale="Tealgrn", text=finish_price["Median EGP/sqm"].apply(lambda x: f"{x:,.0f}"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10,b=40), plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa", coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 7. Which Amenities Are Common? ──
    with c6:
        st.subheader("🏢 Which Amenities Can You Expect?")
        st.caption("Percentage of listings offering each amenity")
        acols = ["Electricity Meter","Water Meter","Natural Gas","Security","Covered Parking","Pets Allowed",
                 "Landline","Balcony","Private Garden","Pool","Built in Kitchen Appliances","Elevator",
                 "Central A/C & heating","Maids Room","roof"]
        avail = [c for c in acols if c in df.columns]
        am_pct = (df[avail].mean() * 100).sort_values(ascending=True).reset_index()
        am_pct.columns = ["Amenity","Percentage"]
        fig = px.bar(am_pct, x="Percentage", y="Amenity", orientation="h", color="Percentage",
                     color_continuous_scale="Purples", labels={"Percentage":"% of listings","Amenity":""},
                     text=am_pct["Percentage"].apply(lambda x: f"{x:.0f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(t=10,l=10), height=450,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa",
                          coloraxis_showscale=False, xaxis=dict(ticksuffix="%"))
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — SEARCH
# ══════════════════════════════════════════════════════════════════════════
with tab_search:
    st.header("🔍 Search Apartments")
    if "compare_ids" not in st.session_state: st.session_state["compare_ids"] = []
    filters = get_filters()
    if not filters: st.error("Cannot load filters."); st.stop()

    with st.expander("🎛️ Filters", expanded=True):
        r1 = st.columns(4)
        with r1[0]: f_gov = st.selectbox("Governorate",[""] + filters.get("governorates",[]), format_func=lambda x:"Any" if x=="" else x, key="s_gov")
        with r1[1]:
            copts = get_cities(f_gov) if f_gov else filters.get("cities",[])
            f_city = st.selectbox("City",[""] + (copts or []), format_func=lambda x:"Any" if x=="" else x, key="s_city")
        with r1[2]: f_ptype = st.selectbox("Property Type",[""] + filters.get("property_types",[]), format_func=lambda x:"Any" if x=="" else x, key="s_ptype")
        with r1[3]: f_finish = st.selectbox("Finish Type",[""] + filters.get("finish_types",[]), format_func=lambda x:"Any" if x=="" else x, key="s_finish")
        r2 = st.columns(3)
        with r2[0]: f_price = st.slider("Price (EGP)", filters["price_min"], filters["price_max"], (filters["price_min"],filters["price_max"]), step=500000, format="%d", key="s_price")
        with r2[1]: f_area = st.slider("Area (sqm)", filters["area_min"], min(filters["area_max"],1000), (filters["area_min"],min(filters["area_max"],1000)), step=10, key="s_area")
        with r2[2]: f_beds = st.slider("Bedrooms", filters["bed_min"], filters["bed_max"], (filters["bed_min"],filters["bed_max"]), key="s_beds")
        r3 = st.columns(4)
        with r3[0]: f_furn = st.selectbox("Furnished",["","yes","no"], format_func=lambda x:"Any" if x=="" else x, key="s_furn")
        with r3[1]: f_own = st.selectbox("Ownership",["","primary","resale"], format_func=lambda x:"Any" if x=="" else x, key="s_own")
        with r3[2]: f_pay = st.selectbox("Payment",["","cash","installment","cash or installment"], format_func=lambda x:"Any" if x=="" else x, key="s_pay")
        with r3[3]: f_comp = st.selectbox("Completion",["","ready","off-plan"], format_func=lambda x:"Any" if x=="" else x, key="s_comp")

    params = {"min_price":f_price[0],"max_price":f_price[1],"min_area":f_area[0],"max_area":f_area[1],"min_beds":f_beds[0],"max_beds":f_beds[1],"limit":50}
    if f_gov: params["governorate"]=f_gov
    if f_city: params["city"]=f_city
    if f_ptype: params["property_type"]=f_ptype
    if f_finish: params["finish_type"]=f_finish
    if f_furn: params["furnished"]=f_furn
    if f_own: params["ownership"]=f_own
    if f_pay: params["payment"]=f_pay
    if f_comp: params["completion"]=f_comp

    results = search(params)
    if not results: st.info("No results. Adjust filters."); st.stop()
    st.success(f"Found **{len(results)}** apartments")

    for i in range(0, len(results), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i+j >= len(results): break
            apt = results[i+j]; aid = apt["id"]; price = apt.get("price_numeric",0); area = apt.get("area_sqm",0)
            ppsqm = round(price/area,0) if area>0 else 0
            with col:
                st.markdown(f'''<div class="apt-card">
<div style="display:flex;justify-content:space-between;align-items:center">
<span style="font-weight:700;color:#fff;font-size:1.05rem">{apt.get("property_type","").title()} in {apt.get("city","").title()}</span>
<span style="background:linear-gradient(135deg,#667eea,#764ba2);padding:4px 12px;border-radius:12px;font-size:0.72rem;font-weight:700;color:#fff">{apt.get("finish_type","")}</span>
</div>
<div class="apt-price">{price:,.0f} EGP</div>
<div style="color:#8892b0;font-size:0.82rem">📍 {apt.get("city","").title()}, {apt.get("governorate","").title()} · {ppsqm:,.0f} EGP/sqm</div>
<div class="apt-specs">
<span>📐 {area:,.0f} sqm</span><span>🛏️ {int(apt.get("bedrooms",0))} bed</span><span>🚿 {int(apt.get("bathrooms",0))} bath</span>
<span>👀 {apt.get("view_type","")}</span><span>🏘️ {apt.get("compound_name","")}</span>
</div></div>''', unsafe_allow_html=True)
                bc1,bc2 = st.columns(2)
                with bc1:
                    if st.button("🔮 Predict Price", key=f"pred_{aid}"):
                        res = predict(build_predict_payload(apt))
                        if res and "Prediction" in res:
                            pred_price = res["Prediction"]; verdict, vcolor, delta = price_verdict(pred_price, price)
                            st.markdown(f'Predicted: **{pred_price:,.0f} EGP** · Delta: **{delta:+.1f}%**')
                            st.markdown(f'<span class="verdict-badge" style="background:{vcolor};color:#fff">{verdict}</span>', unsafe_allow_html=True)
                        else: st.error("Prediction failed")
                with bc2:
                    in_compare = aid in st.session_state["compare_ids"]
                    if st.button("✅ Added" if in_compare else "➕ Compare", key=f"cmp_{aid}", disabled=in_compare):
                        if aid not in st.session_state["compare_ids"]:
                            st.session_state["compare_ids"].append(aid); st.rerun()

    if st.session_state["compare_ids"]:
        st.info(f"🛒 **{len(st.session_state['compare_ids'])}** apartments selected for comparison. Go to ⚖️ Compare tab.")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — COMPARE
# ══════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.header("⚖️ Compare Apartments")
    if "compare_ids" not in st.session_state: st.session_state["compare_ids"] = []
    if not st.session_state["compare_ids"]:
        st.info("No apartments selected. Go to 🔍 Search and click ➕ Compare."); st.stop()
    if st.button("🗑️ Clear All Selections"):
        st.session_state["compare_ids"] = []; st.rerun()

    apts = compare(st.session_state["compare_ids"])
    if not apts: st.warning("Could not fetch data."); st.stop()

    # Predict all and add delta
    st.subheader("📊 Deal Ranking (Best Deals First)")
    st.caption("Sorted by price delta — negative delta = better deal")
    ranked = []
    for apt in apts:
        res = predict(build_predict_payload(apt))
        pred = res["Prediction"] if res and "Prediction" in res else 0
        actual = apt.get("price_numeric",0)
        delta_pct = (actual - pred) / pred * 100 if pred else 0
        verdict, vcolor, _ = price_verdict(pred, actual)
        ranked.append({**apt, "_pred": pred, "_delta": delta_pct, "_verdict": verdict, "_vcolor": vcolor})
    ranked.sort(key=lambda x: x["_delta"])

    for r in ranked:
        c1,c2,c3,c4 = st.columns([3,2,2,2])
        c1.markdown(f"**{r.get('property_type','').title()} in {r.get('city','').title()}** · {r.get('area_sqm',0):,.0f} sqm · {int(r.get('bedrooms',0))} bed")
        c2.markdown(f"Actual: **{r.get('price_numeric',0):,.0f}** EGP")
        c3.markdown(f"Predicted: **{r['_pred']:,.0f}** EGP · Δ **{r['_delta']:+.1f}%**")
        c4.markdown(f'<span class="verdict-badge" style="background:{r["_vcolor"]};color:#fff">{r["_verdict"]}</span>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Side-by-side table
    st.subheader("📋 Feature Comparison")
    dkeys = ["property_type","city","governorate","price_numeric","area_sqm","bedrooms","bathrooms","finish_type","view_type","compound_name","amenity_score"]
    labels = {"property_type":"Type","city":"City","governorate":"Gov.","price_numeric":"Price (EGP)","area_sqm":"Area (sqm)",
              "bedrooms":"Beds","bathrooms":"Baths","finish_type":"Finish","view_type":"View","compound_name":"Compound","amenity_score":"Amenities"}
    rows_html = ""
    for key in dkeys:
        vals = [apt.get(key,"---") for apt in apts]
        if key == "price_numeric": vals = [f"{v:,.0f}" if isinstance(v,(int,float)) else v for v in vals]
        cells = "".join(f"<td style='padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.06)'>{v}</td>" for v in vals)
        rows_html += f"<tr><td style='font-weight:600;color:#667eea;padding:8px 12px;border-bottom:1px solid rgba(255,255,255,0.06)'>{labels.get(key,key)}</td>{cells}</tr>"
    hdrs = "".join(f"<th style='background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:10px 12px'>Apt #{a['id']}</th>" for a in apts)
    st.markdown(f'<table style="width:100%;border-collapse:collapse;border-radius:12px;overflow:hidden"><tr><th style="background:linear-gradient(135deg,#667eea,#764ba2);color:white;padding:10px 12px">Feature</th>{hdrs}</tr>{rows_html}</table>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Radar chart
    st.subheader("🕸️ Normalized Comparison")
    rkeys = ["price_numeric","area_sqm","bedrooms","bathrooms","amenity_score"]
    rlabels = ["Price","Area","Bedrooms","Bathrooms","Amenities"]
    fig = go.Figure()
    colors = ["#667eea","#f1c40f","#2ecc71","#e74c3c","#764ba2"]
    for idx,apt in enumerate(apts):
        vraw = [float(apt.get(k,0)) for k in rkeys]
        maxv = [max(float(a.get(k,1)) for a in apts) or 1 for k in rkeys]
        vnorm = [v/mx*100 for v,mx in zip(vraw,maxv)]
        fig.add_trace(go.Scatterpolar(r=vnorm+[vnorm[0]], theta=rlabels+[rlabels[0]],
                                       name=f"#{apt['id']} {apt.get('city','').title()}",
                                       line=dict(color=colors[idx%len(colors)], width=2), fill='toself', fillcolor=f"rgba({','.join(str(int(colors[idx%len(colors)].lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.1)"))
    fig.update_layout(polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True,range=[0,110],color="#666")),
                      showlegend=True, margin=dict(t=40,b=40), paper_bgcolor="rgba(0,0,0,0)", font_color="#aaa", height=450)
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — PRICE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("💰 AI Price Predictor")
    st.markdown("Adjust inputs and click **Predict** for an instant LightGBM-powered estimate.")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    coords = get_city_coords()
    es = get_explore_stats() or {}
    all_govs = es.get("governorates", [])

    with st.form("predictor_form"):
        st.subheader("📍 Location")
        p1,p2,p3 = st.columns(3)
        with p1: p_gov = st.selectbox("Governorate", all_govs if all_govs else ["cairo"], key="p_gov")
        with p2:
            pc = get_cities(p_gov) if p_gov else []
            p_city = st.selectbox("City", pc if pc else ["unknown"], key="p_city")
        with p3: p_type = st.selectbox("Property Type", ["apartment","penthouse","studio","duplex","roof","chalet","stand alone villa","town house"], key="p_type")

        st.subheader("📐 Specifications")
        s1,s2,s3,s4 = st.columns(4)
        with s1: p_area = st.number_input("Area (sqm)", 30, 2000, 150, step=10, key="p_area")
        with s2: p_beds = st.number_input("Bedrooms", 1, 10, 3, key="p_beds")
        with s3: p_baths = st.number_input("Bathrooms", 1, 6, 2, key="p_baths")
        with s4: p_finish = st.selectbox("Finish", ["unknown","fully_finished","finished","lux","super_lux","semi_finished","unfinished"], key="p_finish")

        st.subheader("📋 Details (Optional)")
        d1,d2,d3,d4 = st.columns(4)
        with d1: p_furn = st.selectbox("Furnished", ["unknown","yes","no"], key="p_furn")
        with d2: p_own = st.selectbox("Ownership", ["resale","primary"], key="p_own")
        with d3: p_pay = st.selectbox("Payment", ["cash","installment","cash or installment"], key="p_pay")
        with d4: p_compst = st.selectbox("Completion", ["ready","off-plan"], key="p_comp")

        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True, type="primary")

    if submitted:
        cc = coords.get(p_city, {"latitude":30.05,"longitude":31.25})
        payload = {
            "bedrooms":float(p_beds),"bathrooms":float(p_baths),"area_numeric":float(p_area),
            "latitude":cc.get("latitude",30.05),"longitude":cc.get("longitude",31.25),
            "property_type":p_type,"city":p_city,"governorate":p_gov,
            "finish_type":p_finish,"seller_name":"unknown","view_type":"unknown",
            "compound_name":"unknown","delivery_date":0.0,"seller_type":"agency",
            "furnished":p_furn,"ownership":p_own,"payment_option":p_pay,"completion_status":p_compst,
            "Electricity Meter":False,"Water Meter":False,"Natural Gas":False,"Security":False,
            "Covered Parking":False,"Pets Allowed":False,"Landline":False,"Balcony":False,
            "Private Garden":False,"Pool":False,"Built in Kitchen Appliances":False,
            "Elevator":False,"Central A/C & heating":False,"Maids Room":False,"roof":False,
        }
        res = predict(payload)
        if res and "Prediction" in res:
            pred = res["Prediction"]; ppsqm = pred/p_area if p_area>0 else 0
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            mc1,mc2,mc3 = st.columns([1,2,1])
            with mc2:
                st.markdown(f'<div class="kpi-card" style="padding:2rem"><h2 style="font-size:2.4rem">{pred:,.0f} EGP</h2><p>Predicted Fair Price</p></div>', unsafe_allow_html=True)
                st.markdown("")
                st.markdown(f"**{ppsqm:,.0f} EGP/sqm** · {p_beds} bed · {p_baths} bath · {p_area} sqm · {p_city.title()}, {p_gov.title()}")
                st.info(f"This prediction is generated by a LightGBM model trained on 60,000+ Egyptian apartment listings from Dubizzle.")
        else:
            st.error("Prediction failed. Check that the FastAPI backend is running.")

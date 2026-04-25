#!/usr/bin/env python3
"""
Phase 3 — Data Cleaning & Analysis for Dubizzle Egypt Apartments Dataset.

Produces:
  1. Cleaned dataset (dubizzle_cleaned.csv)
  2. Comprehensive HTML dashboard with interactive charts
  3. Summary statistics

Usage:
  python analyze_data.py
"""

import json
import os
import csv
import math
import logging
from collections import Counter, defaultdict
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, "dubizzle_apartments_enriched.jsonl")
CLEANED_CSV = os.path.join(SCRIPT_DIR, "dubizzle_cleaned.csv")
CLEANED_JSON = os.path.join(SCRIPT_DIR, "dubizzle_cleaned.json")
DASHBOARD_HTML = os.path.join(SCRIPT_DIR, "dashboard.html")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════

def clean_data(records: list[dict]) -> list[dict]:
    """Clean and standardize apartment records."""
    logger.info("Starting data cleaning on %d records...", len(records))
    cleaned = []
    removed = Counter()

    for r in records:
        # ── Remove records missing critical fields ──
        price = r.get("price_numeric")
        if not price or price <= 0:
            removed["no_price"] += 1
            continue

        area = r.get("area_numeric")
        city = r.get("city", "")

        # ── Remove extreme outliers ──
        if price < 100_000:
            removed["price_too_low"] += 1
            continue
        if price > 150_000_000:
            removed["price_too_high"] += 1
            continue

        if area and area < 15:
            removed["area_too_small"] += 1
            continue
        if area and area > 1500:
            removed["area_too_large"] += 1
            continue

        ppsqm = r.get("price_per_sqm")
        if ppsqm and (ppsqm < 500 or ppsqm > 500_000):
            removed["ppsqm_outlier"] += 1
            continue

        # ── Standardize fields ──
        # City normalization
        city_map = {
            "New Cairo City": "New Cairo",
            "El Tagamoa El Khames": "New Cairo",
            "El Tagamoa' El Khames": "New Cairo",
            "5th Settlement": "New Cairo",
            "El Rehab": "Rehab City",
            "El Shorouk": "Shorouk City",
            "Al Shorouk City": "Shorouk City",
            "October": "6th of October",
            "6 October": "6th of October",
            "Sheikh Zayed City": "Sheikh Zayed",
            "Zayed": "Sheikh Zayed",
            "Hadayek Al Ahram": "Hadayek al-Ahram",
            "Garden City": "Garden City",
        }
        if city in city_map:
            r["city"] = city_map[city]

        # Region classification
        r["region"] = classify_region(r.get("city", ""))

        # Bedrooms normalization
        beds = r.get("bedrooms")
        if isinstance(beds, str):
            try:
                r["bedrooms"] = int(beds.replace("+", "").strip())
            except (ValueError, AttributeError):
                r["bedrooms"] = None

        # Floor number cap
        floor = r.get("floor_number")
        if floor is not None and (floor < 0 or floor > 40):
            r["floor_number"] = None

        # Delivery date standardization
        delivery = r.get("delivery_date")
        if delivery and delivery not in ("immediate",):
            try:
                yr = int(str(delivery)[:4])
                if yr < 2020 or yr > 2035:
                    r["delivery_date"] = None
            except (ValueError, TypeError):
                pass

        # Price tier
        r["price_tier"] = classify_price_tier(price)

        # Area tier
        if area:
            r["area_tier"] = classify_area_tier(area)

        cleaned.append(r)

    logger.info("Cleaning complete:")
    logger.info("  Before: %d records", len(records))
    logger.info("  After:  %d records", len(cleaned))
    logger.info("  Removed breakdown:")
    for reason, count in removed.most_common():
        logger.info("    %-25s: %d", reason, count)

    return cleaned


def classify_region(city: str) -> str:
    """Classify city into broad region."""
    cairo_east = {"New Cairo", "Madinaty", "Shorouk City", "Rehab City", "Mostakbal City",
                  "Heliopolis", "Nasr City", "Sheraton", "New Heliopolis", "Katameya",
                  "Mokattam", "Ain Sukhna"}
    cairo_west = {"Sheikh Zayed", "6th of October", "Hadayek October", "Hadayek al-Ahram",
                  "Giza", "Dokki", "Mohandessin", "Agouza"}
    cairo_central = {"Maadi", "Garden City", "Zamalek", "Downtown", "Helwan"}
    new_capital = {"New Capital City"}
    north_coast = {"North Coast"}
    red_sea = {"Hurghada", "El Gouna", "Marsa Alam"}
    alex = {"Alexandria", "Borg El Arab"}

    if city in cairo_east:
        return "Cairo East"
    elif city in cairo_west:
        return "Cairo West (Giza)"
    elif city in cairo_central:
        return "Cairo Central"
    elif city in new_capital:
        return "New Capital"
    elif city in north_coast:
        return "North Coast"
    elif city in red_sea:
        return "Red Sea"
    elif city in alex:
        return "Alexandria"
    else:
        return "Other"


def classify_price_tier(price: float) -> str:
    if price < 1_000_000:
        return "Under 1M"
    elif price < 3_000_000:
        return "1M - 3M"
    elif price < 5_000_000:
        return "3M - 5M"
    elif price < 8_000_000:
        return "5M - 8M"
    elif price < 12_000_000:
        return "8M - 12M"
    elif price < 20_000_000:
        return "12M - 20M"
    else:
        return "20M+"


def classify_area_tier(area: float) -> str:
    if area < 80:
        return "Studio/Small (<80m²)"
    elif area < 120:
        return "Medium (80-120m²)"
    elif area < 170:
        return "Large (120-170m²)"
    elif area < 250:
        return "XLarge (170-250m²)"
    else:
        return "Luxury (250m²+)"


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: STATISTICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stats(records: list[dict]) -> dict:
    """Compute comprehensive statistics."""
    stats = {}

    prices = [r["price_numeric"] for r in records if r.get("price_numeric")]
    areas = [r["area_numeric"] for r in records if r.get("area_numeric")]
    ppsqm = [r["price_per_sqm"] for r in records if r.get("price_per_sqm")]

    stats["total"] = len(records)
    stats["price_mean"] = sum(prices) / len(prices) if prices else 0
    stats["price_median"] = sorted(prices)[len(prices)//2] if prices else 0
    stats["price_min"] = min(prices) if prices else 0
    stats["price_max"] = max(prices) if prices else 0
    stats["area_mean"] = sum(areas) / len(areas) if areas else 0
    stats["area_median"] = sorted(areas)[len(areas)//2] if areas else 0
    stats["ppsqm_mean"] = sum(ppsqm) / len(ppsqm) if ppsqm else 0
    stats["ppsqm_median"] = sorted(ppsqm)[len(ppsqm)//2] if ppsqm else 0

    # By city
    city_data = defaultdict(list)
    for r in records:
        c = r.get("city", "Unknown")
        if r.get("price_per_sqm"):
            city_data[c].append(r["price_per_sqm"])

    stats["city_ppsqm"] = {}
    for city, vals in city_data.items():
        if len(vals) >= 20:
            stats["city_ppsqm"][city] = {
                "median": sorted(vals)[len(vals)//2],
                "mean": sum(vals) / len(vals),
                "count": len(vals),
            }

    # By region
    region_data = defaultdict(list)
    for r in records:
        reg = r.get("region", "Other")
        if r.get("price_per_sqm"):
            region_data[reg].append(r["price_per_sqm"])

    stats["region_ppsqm"] = {}
    for reg, vals in region_data.items():
        stats["region_ppsqm"][reg] = {
            "median": sorted(vals)[len(vals)//2],
            "mean": sum(vals) / len(vals),
            "count": len(vals),
        }

    # Distributions
    stats["city_counts"] = dict(Counter(r.get("city", "?") for r in records).most_common(25))
    stats["region_counts"] = dict(Counter(r.get("region", "?") for r in records).most_common())
    stats["bedroom_counts"] = dict(Counter(str(r.get("bedrooms", "?")) for r in records).most_common())
    stats["finish_counts"] = dict(Counter(r.get("finish_type", "Unknown") for r in records).most_common())
    stats["price_tier_counts"] = dict(Counter(r.get("price_tier", "?") for r in records).most_common())
    stats["area_tier_counts"] = dict(Counter(r.get("area_tier", "?") for r in records if r.get("area_tier")).most_common())
    stats["view_counts"] = dict(Counter(r.get("view_type", "None") for r in records).most_common())
    stats["delivery_counts"] = dict(Counter(r.get("delivery_date") or "Unknown" for r in records).most_common(10))
    stats["furnished_counts"] = dict(Counter(r.get("furnished", "Unknown") for r in records).most_common())
    stats["compound_counts"] = dict(Counter(r.get("compound_name") or "Unknown" for r in records).most_common(20))

    # Price by bedrooms
    beds_price = defaultdict(list)
    for r in records:
        b = r.get("bedrooms")
        p = r.get("price_numeric")
        if b and p and 0 < b <= 6:
            beds_price[str(b)].append(p)
    stats["beds_price_median"] = {k: sorted(v)[len(v)//2] for k, v in beds_price.items()}

    # Boolean features summary
    bool_feats = ["has_garden", "has_roof", "has_elevator", "has_parking", "has_pool", "has_security"]
    stats["amenities"] = {}
    for feat in bool_feats:
        count = sum(1 for r in records if r.get(feat))
        stats["amenities"][feat.replace("has_", "")] = count

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dashboard(stats: dict, output_path: str):
    """Generate an interactive HTML dashboard using Chart.js."""

    def fmt(n):
        if n >= 1_000_000:
            return f"{n/1_000_000:.1f}M"
        elif n >= 1_000:
            return f"{n/1_000:.0f}K"
        return str(int(n))

    # Prepare chart data
    city_labels = list(stats["city_counts"].keys())[:15]
    city_values = [stats["city_counts"][c] for c in city_labels]

    region_labels = list(stats["region_counts"].keys())
    region_values = [stats["region_counts"][r] for r in region_labels]

    # Price/sqm by city (top 15 by count)
    ppsqm_cities = sorted(stats["city_ppsqm"].items(), key=lambda x: x[1]["count"], reverse=True)[:15]
    ppsqm_cities_sorted = sorted(ppsqm_cities, key=lambda x: x[1]["median"], reverse=True)
    ppsqm_labels = [c[0] for c in ppsqm_cities_sorted]
    ppsqm_values = [c[1]["median"] for c in ppsqm_cities_sorted]

    # Region price/sqm
    reg_ppsqm = sorted(stats["region_ppsqm"].items(), key=lambda x: x[1]["median"], reverse=True)
    reg_ppsqm_labels = [r[0] for r in reg_ppsqm]
    reg_ppsqm_values = [r[1]["median"] for r in reg_ppsqm]

    # Bedrooms
    bed_labels = sorted([k for k in stats["bedroom_counts"].keys() if k.isdigit()], key=int)
    bed_values = [stats["bedroom_counts"].get(b, 0) for b in bed_labels]

    # Price tier
    tier_order = ["Under 1M", "1M - 3M", "3M - 5M", "5M - 8M", "8M - 12M", "12M - 20M", "20M+"]
    tier_values = [stats["price_tier_counts"].get(t, 0) for t in tier_order]

    # Finish type (exclude Unknown/None)
    finish_labels = [k for k in stats["finish_counts"].keys() if k and k not in ("Unknown", "None", "?")]
    finish_values = [stats["finish_counts"][k] for k in finish_labels]

    # Area tier
    area_order = ["Studio/Small (<80m²)", "Medium (80-120m²)", "Large (120-170m²)", "XLarge (170-250m²)", "Luxury (250m²+)"]
    area_values = [stats["area_tier_counts"].get(t, 0) for t in area_order]

    # View types (exclude None)
    view_labels = [k for k in stats["view_counts"].keys() if k and k != "None"]
    view_values = [stats["view_counts"][k] for k in view_labels]

    # Amenities
    amen_labels = list(stats["amenities"].keys())
    amen_values = list(stats["amenities"].values())

    # Beds vs price
    beds_price_labels = sorted(stats["beds_price_median"].keys(), key=int)
    beds_price_values = [stats["beds_price_median"][b] for b in beds_price_labels]

    # Top compounds
    compound_labels = [k for k in list(stats["compound_counts"].keys())[:15] if k != "Unknown"]
    compound_values = [stats["compound_counts"][k] for k in compound_labels]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dubizzle Egypt Apartments — Market Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: #0f0f1a;
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a3e 0%, #2d1b69 50%, #1a1a3e 100%);
            padding: 40px 20px;
            text-align: center;
            border-bottom: 2px solid rgba(139, 92, 246, 0.3);
        }}
        .header h1 {{
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }}
        .header p {{
            color: #94a3b8;
            font-size: 1rem;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            padding: 30px 40px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .kpi {{
            background: linear-gradient(145deg, #1e1e3a, #252550);
            border: 1px solid rgba(139, 92, 246, 0.2);
            border-radius: 16px;
            padding: 24px;
            text-align: center;
        }}
        .kpi .value {{
            font-size: 2rem;
            font-weight: 700;
            color: #a78bfa;
        }}
        .kpi .label {{
            font-size: 0.85rem;
            color: #94a3b8;
            margin-top: 4px;
        }}
        .charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(550px, 1fr));
            gap: 24px;
            padding: 20px 40px 60px;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .chart-card {{
            background: linear-gradient(145deg, #1e1e3a, #252550);
            border: 1px solid rgba(139, 92, 246, 0.15);
            border-radius: 16px;
            padding: 24px;
        }}
        .chart-card h3 {{
            font-size: 1rem;
            font-weight: 600;
            color: #c4b5fd;
            margin-bottom: 16px;
        }}
        .chart-card canvas {{
            max-height: 350px;
        }}
        .full-width {{
            grid-column: 1 / -1;
        }}
        .footer {{
            text-align: center;
            padding: 20px;
            color: #64748b;
            font-size: 0.8rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏠 Dubizzle Egypt — Apartments Market Dashboard</h1>
        <p>73K+ listings analyzed • {datetime.now().strftime("%B %Y")} • Powered by Web Scraping + NLP</p>
    </div>

    <div class="kpi-row">
        <div class="kpi"><div class="value">{stats['total']:,}</div><div class="label">Total Listings</div></div>
        <div class="kpi"><div class="value">EGP {fmt(stats['price_median'])}</div><div class="label">Median Price</div></div>
        <div class="kpi"><div class="value">{stats['area_median']:.0f} m²</div><div class="label">Median Area</div></div>
        <div class="kpi"><div class="value">EGP {fmt(stats['ppsqm_median'])}</div><div class="label">Median Price/m²</div></div>
        <div class="kpi"><div class="value">133</div><div class="label">Cities Covered</div></div>
    </div>

    <div class="charts">
        <!-- Chart 1: Listings by City -->
        <div class="chart-card">
            <h3>📊 Listings by City (Top 15)</h3>
            <canvas id="cityChart"></canvas>
        </div>

        <!-- Chart 2: Price/m² by City -->
        <div class="chart-card">
            <h3>💰 Median Price per m² by City</h3>
            <canvas id="ppsqmChart"></canvas>
        </div>

        <!-- Chart 3: Region Distribution -->
        <div class="chart-card">
            <h3>🗺️ Listings by Region</h3>
            <canvas id="regionChart"></canvas>
        </div>

        <!-- Chart 4: Price/m² by Region -->
        <div class="chart-card">
            <h3>💎 Median Price per m² by Region</h3>
            <canvas id="regPpsqmChart"></canvas>
        </div>

        <!-- Chart 5: Bedrooms -->
        <div class="chart-card">
            <h3>🛏️ Bedroom Distribution</h3>
            <canvas id="bedChart"></canvas>
        </div>

        <!-- Chart 6: Price Tiers -->
        <div class="chart-card">
            <h3>📈 Price Tier Distribution</h3>
            <canvas id="tierChart"></canvas>
        </div>

        <!-- Chart 7: Finish Type -->
        <div class="chart-card">
            <h3>🏗️ Finish Type</h3>
            <canvas id="finishChart"></canvas>
        </div>

        <!-- Chart 8: Area Tiers -->
        <div class="chart-card">
            <h3>📐 Area Distribution</h3>
            <canvas id="areaChart"></canvas>
        </div>

        <!-- Chart 9: Price by Bedrooms -->
        <div class="chart-card">
            <h3>💵 Median Price by Bedrooms</h3>
            <canvas id="bedsPriceChart"></canvas>
        </div>

        <!-- Chart 10: View Types -->
        <div class="chart-card">
            <h3>🌅 View Types (NLP extracted)</h3>
            <canvas id="viewChart"></canvas>
        </div>

        <!-- Chart 11: Amenities -->
        <div class="chart-card">
            <h3>🏊 Amenities (NLP extracted)</h3>
            <canvas id="amenChart"></canvas>
        </div>

        <!-- Chart 12: Top Compounds -->
        <div class="chart-card">
            <h3>🏘️ Top Compounds (NLP extracted)</h3>
            <canvas id="compoundChart"></canvas>
        </div>
    </div>

    <div class="footer">
        Built with Python • Data scraped from Dubizzle Egypt • Features extracted via Regex + Gemini AI
    </div>

    <script>
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.borderColor = 'rgba(139, 92, 246, 0.1)';
    Chart.defaults.font.family = 'Inter';

    const purple = ['#8b5cf6','#a78bfa','#c4b5fd','#7c3aed','#6d28d9','#5b21b6','#4c1d95',
                    '#60a5fa','#38bdf8','#22d3ee','#34d399','#fbbf24','#f87171','#fb923c','#a3e635'];

    function barChart(id, labels, data, label, color='#8b5cf6') {{
        new Chart(document.getElementById(id), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{ label: label, data: data, backgroundColor: color + '99',
                              borderColor: color, borderWidth: 1, borderRadius: 6 }}]
            }},
            options: {{
                indexAxis: labels.length > 8 ? 'y' : 'x',
                responsive: true,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{ x: {{ grid: {{ display: false }} }}, y: {{ grid: {{ color: 'rgba(139,92,246,0.08)' }} }} }}
            }}
        }});
    }}

    function doughnut(id, labels, data) {{
        new Chart(document.getElementById(id), {{
            type: 'doughnut',
            data: {{
                labels: labels,
                datasets: [{{ data: data, backgroundColor: purple.slice(0, labels.length),
                              borderWidth: 0, hoverOffset: 8 }}]
            }},
            options: {{
                responsive: true,
                plugins: {{ legend: {{ position: 'right', labels: {{ boxWidth: 14, padding: 10 }} }} }}
            }}
        }});
    }}

    // 1. Listings by City
    barChart('cityChart', {json.dumps(city_labels)}, {json.dumps(city_values)}, 'Listings', '#8b5cf6');

    // 2. Price/sqm by City
    barChart('ppsqmChart', {json.dumps(ppsqm_labels)}, {json.dumps(ppsqm_values)}, 'EGP/m²', '#60a5fa');

    // 3. Region
    doughnut('regionChart', {json.dumps(region_labels)}, {json.dumps(region_values)});

    // 4. Region price/sqm
    barChart('regPpsqmChart', {json.dumps(reg_ppsqm_labels)}, {json.dumps(reg_ppsqm_values)}, 'EGP/m²', '#22d3ee');

    // 5. Bedrooms
    doughnut('bedChart', {json.dumps(bed_labels)}, {json.dumps(bed_values)});

    // 6. Price tiers
    barChart('tierChart', {json.dumps(tier_order)}, {json.dumps(tier_values)}, 'Listings', '#a78bfa');

    // 7. Finish type
    doughnut('finishChart', {json.dumps(finish_labels)}, {json.dumps(finish_values)});

    // 8. Area tier
    barChart('areaChart', {json.dumps(area_order)}, {json.dumps(area_values)}, 'Listings', '#34d399');

    // 9. Price by beds
    barChart('bedsPriceChart', {json.dumps(beds_price_labels)}, {json.dumps(beds_price_values)}, 'Median EGP', '#fbbf24');

    // 10. View types
    doughnut('viewChart', {json.dumps(view_labels)}, {json.dumps(view_values)});

    // 11. Amenities
    barChart('amenChart', {json.dumps(amen_labels)}, {json.dumps(amen_values)}, 'Listings', '#f87171');

    // 12. Compounds
    barChart('compoundChart', {json.dumps(compound_labels)}, {json.dumps(compound_values)}, 'Listings', '#fb923c');
    </script>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Dashboard saved → %s", os.path.basename(output_path))


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    logger.info("Loading enriched data...")
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))
    logger.info("Loaded %d records.", len(records))

    # Clean
    cleaned = clean_data(records)

    # Save cleaned data
    # CSV
    if cleaned:
        fieldnames = list(cleaned[0].keys())
        for r in cleaned[:100]:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)

        with open(CLEANED_CSV, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(cleaned)
        logger.info("Saved %d records → %s", len(cleaned), os.path.basename(CLEANED_CSV))

    # JSON
    with open(CLEANED_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False)
    logger.info("Saved %d records → %s", len(cleaned), os.path.basename(CLEANED_JSON))

    # Compute stats
    stats = compute_stats(cleaned)

    # Generate dashboard
    generate_dashboard(stats, DASHBOARD_HTML)

    # Print summary
    logger.info("")
    logger.info("╔════════════════════════════════════════════════════════════╗")
    logger.info("║           DATASET SUMMARY                                ║")
    logger.info("╠════════════════════════════════════════════════════════════╣")
    logger.info("║  Total listings     : %-35s║", f"{stats['total']:,}")
    logger.info("║  Median price       : %-35s║", f"EGP {stats['price_median']:,.0f}")
    logger.info("║  Median area        : %-35s║", f"{stats['area_median']:.0f} m²")
    logger.info("║  Median price/m²    : %-35s║", f"EGP {stats['ppsqm_median']:,.0f}")
    logger.info("║  Price range        : %-35s║",
                f"EGP {stats['price_min']:,.0f} — {stats['price_max']:,.0f}")
    logger.info("╠════════════════════════════════════════════════════════════╣")
    logger.info("║  Output files:                                           ║")
    logger.info("║    dubizzle_cleaned.csv                                  ║")
    logger.info("║    dubizzle_cleaned.json                                 ║")
    logger.info("║    dashboard.html                                        ║")
    logger.info("╚════════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()

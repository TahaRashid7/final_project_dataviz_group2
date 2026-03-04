import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chicago Housing Distress Dashboard",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e1a;
    color: #e2e8f0;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; }

/* Title */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2rem !important;
    letter-spacing: -0.02em !important;
    color: #f8fafc !important;
    margin-bottom: 0.2rem !important;
}

/* Subheaders */
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.01em !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f1629 !important;
    border-right: 1px solid #1e2d4a !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Syne', sans-serif !important;
    color: #cbd5e1 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #f8fafc !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1rem 1.2rem !important;
    transition: border-color 0.2s ease;
}
[data-testid="metric-container"]:hover {
    border-color: #f97316;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #64748b !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1.6rem !important;
    color: #f97316 !important;
}

/* Divider */
hr {
    border-color: #1e2d4a !important;
    margin: 1.5rem 0 !important;
}

/* Section labels */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #f97316;
    margin-bottom: 0.5rem;
}

/* Ward badge */
.ward-badge {
    display: inline-block;
    background: #f97316;
    color: #0a0e1a;
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 0.75rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}

/* Plotly chart backgrounds */
.js-plotly-plot {
    border-radius: 12px;
    overflow: hidden;
}

/* Selectbox and radio */
[data-testid="stSelectbox"] > div,
[data-testid="stRadio"] > div {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── PATHS ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent

# ── CACHED LOADERS ─────────────────────────────────────────────────────────────
@st.cache_data
def load_wards_geo():
    gdf = gpd.read_file(BASE / "dataset" / "cleaned" / "wards_2023_final_dashboard.geojson")
    gdf = gdf.to_crs(epsg=4326)
    gdf["ward"] = gdf["ward"].astype(int)
    return gdf

@st.cache_data
def load_vacant_csv():
    vacant = pd.read_csv(BASE / "vacant_minimal.csv")
    vacant["ward_spatial"] = vacant["ward_spatial"].astype(int)
    return vacant

@st.cache_data
def load_foreclosures_timeseries():
    foreclosures = pd.read_csv(
        "https://raw.githubusercontent.com/aaryal22/final_project_dataviz_group2/main/dataset/raw/foreclosures_chicago_wards_clean.csv"
    )
    foreclosures["ward"] = foreclosures["Geography"].str.replace("Ward ", "", regex=False).astype(int)
    year_cols = [c for c in foreclosures.columns if c.isdigit()]
    long = foreclosures.melt(id_vars=["ward"], value_vars=year_cols, var_name="year", value_name="foreclosures")
    long["year"] = long["year"].astype(int)
    return long

@st.cache_data
def load_debt_by_ward():
    ward_debt = pd.read_csv(BASE / "dataset" / "cleaned" / "ward_debt_summary.csv")
    ward_debt["ward"] = ward_debt["ward"].astype(int)
    agg_cols = [c for c in ward_debt.columns if c != "ward"]
    ward_debt_long = ward_debt.melt(id_vars="ward", value_vars=agg_cols, var_name="category", value_name="amount")
    ward_debt_long["amount_m"] = (ward_debt_long["amount"] / 1e6).round(3)
    ward_totals = ward_debt[["ward"]].copy()
    ward_totals["total_debt_m"] = (ward_debt[agg_cols].sum(axis=1) / 1e6).round(2)
    return ward_debt_long, ward_totals

@st.cache_data
def load_demolitions_by_ward():
    try:
        demo = pd.read_csv(BASE / "dataset" / "raw" / "demolition_clean.csv")
        zip_ward = pd.read_csv(BASE / "dataset" / "cleaned" / "zip_ward_lookup.csv")
        zip_ward["ZIP5"] = zip_ward["ZIP5"].astype(str).str[:5]

        # Try all contact ZIP columns as fallback chain
        zip_cols = [f"CONTACT_{i}_ZIPCODE" for i in range(1, 4) if f"CONTACT_{i}_ZIPCODE" in demo.columns]
        if zip_cols:
            demo["ZIP5"] = demo[zip_cols].bfill(axis=1).iloc[:, 0].astype(str).str[:5]
        else:
            return pd.DataFrame(columns=["ward", "total_demolitions", "city_initiated"])

        demo = demo.merge(zip_ward, on="ZIP5", how="left")
        demo = demo.dropna(subset=["ward"])
        demo["ward"] = demo["ward"].astype(int)

        return demo.groupby("ward").agg(
            total_demolitions=("PERMIT#", "count"),
            city_initiated=("is_city_initiated", "sum")
        ).reset_index()
    except Exception as e:
        st.warning(f"Demolition data unavailable: {e}")
        return pd.DataFrame(columns=["ward", "total_demolitions", "city_initiated"])

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
gdf         = load_wards_geo()
vacant      = load_vacant_csv()
ts          = load_foreclosures_timeseries()
debt_long, ward_debt_totals = load_debt_by_ward()
ward_demo   = load_demolitions_by_ward()

# Merge debt into gdf
gdf = gdf.merge(ward_debt_totals, on="ward", how="left")

# Merge demolitions
if not ward_demo.empty:
    gdf = gdf.merge(ward_demo[["ward", "total_demolitions"]], on="ward", how="left")
else:
    gdf["total_demolitions"] = 0

gdf["total_debt_m"]      = gdf["total_debt_m"].fillna(0)
gdf["total_demolitions"] = gdf["total_demolitions"].fillna(0)

# ── RECOMPUTE HARDSHIP INDEX WITH DEBT ────────────────────────────────────────
def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0

gdf["foreclosure_norm"] = normalize(gdf["foreclosure_rate_2024"])
gdf["vacancy_norm"]     = normalize(gdf["vacant_count"])
gdf["debt_norm"]        = normalize(gdf["total_debt_m"])

gdf["housing_distress_index"] = (
    gdf["foreclosure_norm"] +
    gdf["vacancy_norm"] +
    gdf["debt_norm"]
) / 3

# Recompute risk tiers
gdf["risk_tier"] = pd.qcut(
    gdf["housing_distress_index"], q=3,
    labels=["Low", "Watch", "Critical"]
)

# Rename for display
gdf = gdf.rename(columns={
    "foreclosure_rate_2024":  "Foreclosures (2024)",
    "vacant_count":           "Vacant parcels",
    "housing_distress_index": "Housing Distress Index",
    "total_debt_m":           "Outstanding Debt ($M)",
    "total_demolitions":      "Demolitions"
})

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Harris School of Public Policy · University of Chicago</p>', unsafe_allow_html=True)
st.title("🏙️ Chicago Housing Distress Dashboard")
st.markdown("*Ward-level analysis of foreclosures, vacancy, outstanding debt, and demolitions across Chicago's 50 wards.*")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## Controls")
ward_list = sorted(gdf["ward"].unique())

selected_ward = st.sidebar.selectbox(
    "Select a Ward",
    options=["Citywide"] + ward_list,
    index=0
)

map_metric = st.sidebar.radio(
    "Choropleth metric",
    options=[
        "Foreclosures (2024)",
        "Vacant parcels",
        "Outstanding Debt ($M)",
        "Demolitions",
        "Risk tier",
        "Housing Distress Index",
    ],
    index=0
)

zoom_mode    = st.sidebar.radio("View", options=["Citywide", "Selected ward"], index=0)
show_parcels = st.sidebar.checkbox("Show vacant parcel dots", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-family: DM Mono, monospace; font-size: 0.6rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em;'>
Data Sources<br><br>
• Chicago Dept. of Water Management<br>
• Cook County Assessor's Office<br>
• Chicago Data Portal — Demolitions<br>
• Chicago Ward Foreclosure Rates
</div>
""", unsafe_allow_html=True)

# ── SUMMARY METRICS ────────────────────────────────────────────────────────────
st.markdown("---")
col1, col2, col3, col4, col5 = st.columns(5)

if selected_ward == "Citywide":
    st.markdown('<p class="section-label">📊 Citywide Summary</p>', unsafe_allow_html=True)
    col1.metric("Avg Foreclosure Rate",   f"{gdf['Foreclosures (2024)'].mean():.2f}%")
    col2.metric("Total Vacant Parcels",   f"{int(gdf['Vacant parcels'].sum()):,}")
    col3.metric("Avg Distress Index",     f"{gdf['Housing Distress Index'].mean():.3f}")
    col4.metric("Total Outstanding Debt", f"${gdf['Outstanding Debt ($M)'].sum():.0f}M")
    col5.metric("Critical Wards",         f"{int((gdf['risk_tier'] == 'Critical').sum())} / 50")
else:
    ward_row = gdf[gdf["ward"] == selected_ward].iloc[0]
    st.markdown(f'<div class="ward-badge">Ward {selected_ward}</div>', unsafe_allow_html=True)
    col1.metric("Foreclosures (2024)", f"{ward_row['Foreclosures (2024)']:.2f}%")
    col2.metric("Vacant Parcels",      f"{int(ward_row['Vacant parcels']):,}")
    col3.metric("Distress Index",      f"{ward_row['Housing Distress Index']:.3f}")
    col4.metric("Outstanding Debt",    f"${ward_row['Outstanding Debt ($M)']:.1f}M")
    col5.metric("Risk Tier",           ward_row["risk_tier"])

# ── MAP ────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-label">🗺️ Ward Map</p>', unsafe_allow_html=True)

hover_cols = ["ward", "Foreclosures (2024)", "Vacant parcels",
              "Housing Distress Index", "Outstanding Debt ($M)", "Demolitions", "risk_tier"]

color_scales = {
    "Foreclosures (2024)":    "Reds",
    "Vacant parcels":         "Blues",
    "Housing Distress Index":  "Purples",
    "Outstanding Debt ($M)":  "Oranges",
    "Demolitions":            "YlOrRd"
}

plot_bgcolor  = "#0a0e1a"
paper_bgcolor = "#0a0e1a"

if map_metric == "Risk tier":
    fig = px.choropleth_mapbox(
        gdf, geojson=gdf.__geo_interface__, locations="ward",
        featureidkey="properties.ward", color="risk_tier",
        mapbox_style="carto-darkmatter", opacity=0.85, hover_data=hover_cols,
        color_discrete_map={"Low": "#22c55e", "Watch": "#f59e0b", "Critical": "#ef4444"}
    )
else:
    fig = px.choropleth_mapbox(
        gdf, geojson=gdf.__geo_interface__, locations="ward",
        featureidkey="properties.ward", color=map_metric,
        mapbox_style="carto-darkmatter", opacity=0.8, hover_data=hover_cols,
        color_continuous_scale=color_scales.get(map_metric, "Blues")
    )

fig.update_layout(
    mapbox=dict(center=dict(lat=41.85, lon=-87.68), zoom=9.4),
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor=paper_bgcolor,
    plot_bgcolor=plot_bgcolor,
    font=dict(family="Syne", color="#e2e8f0")
)

if selected_ward != "Citywide":
    if show_parcels:
        filtered = vacant[vacant["ward_spatial"] == selected_ward]
        fig.add_trace(go.Scattermapbox(
            lat=filtered["latitude"], lon=filtered["longitude"],
            mode="markers", marker=dict(size=4, color="#f97316", opacity=0.7),
            name="Vacant parcels"
        ))
    if zoom_mode == "Selected ward":
        center = gdf[gdf["ward"] == selected_ward].geometry.centroid.iloc[0]
        fig.update_layout(mapbox=dict(
            center=dict(lat=center.y, lon=center.x),
            zoom=11.5, style="carto-darkmatter"
        ))

st.plotly_chart(fig, use_container_width=True)

# ── FORECLOSURE TIME SERIES ────────────────────────────────────────────────────
if map_metric == "Foreclosures (2024)" and selected_ward != "Citywide":
    st.markdown("---")
    st.markdown('<p class="section-label">📈 Foreclosures Over Time</p>', unsafe_allow_html=True)
    ts_ward = ts[ts["ward"] == selected_ward].sort_values("year")
    fig_line = px.line(
        ts_ward, x="year", y="foreclosures", markers=True,
        title=f"Ward {selected_ward} — Foreclosures by Year",
        color_discrete_sequence=["#f97316"]
    )
    fig_line.update_traces(line=dict(width=2.5), marker=dict(size=7))
    fig_line.update_layout(
        paper_bgcolor=paper_bgcolor, plot_bgcolor="#111827",
        font=dict(family="Syne", color="#e2e8f0"),
        title_font=dict(size=14, family="Syne"),
        margin={"r": 20, "t": 50, "l": 20, "b": 20},
        xaxis=dict(gridcolor="#1e2d4a", showline=False),
        yaxis=dict(gridcolor="#1e2d4a", showline=False),
    )
    st.plotly_chart(fig_line, use_container_width=True)

# ── DEBT BREAKDOWN ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<p class="section-label">💰 Outstanding Debt Breakdown — Service vs. Penalties</p>', unsafe_allow_html=True)

color_map = {
    "Water (Service)":     "#1d4ed8",
    "Sewer (Service)":     "#3b82f6",
    "Garbage (Service)":   "#60a5fa",
    "Water Tax (Service)": "#93c5fd",
    "Sewer Tax (Service)": "#bfdbfe",
    "Water Penalty":       "#991b1b",
    "Sewer Penalty":       "#dc2626",
    "Garbage Penalty":     "#ef4444",
    "Water Tax Penalty":   "#f87171",
    "Sewer Tax Penalty":   "#fca5a5",
    "Other":               "#475569",
}

if selected_ward == "Citywide":
    top20 = debt_long.groupby("ward")["amount_m"].sum().nlargest(20).index.tolist()
    plot_data = debt_long[debt_long["ward"].isin(top20)]
    title = "Top 20 Wards — Outstanding Debt by Category ($M)"
    x_col = "ward"
else:
    plot_data = debt_long[debt_long["ward"] == selected_ward]
    title = f"Ward {selected_ward} — Outstanding Debt by Category ($M)"
    x_col = "category"

fig_debt = px.bar(
    plot_data, x=x_col, y="amount_m", color="category",
    barmode="stack", color_discrete_map=color_map,
    labels={"amount_m": "Amount ($M)", "ward": "Ward", "category": "Debt Category"},
    title=title
)
fig_debt.update_layout(
    paper_bgcolor=paper_bgcolor, plot_bgcolor="#111827",
    font=dict(family="Syne", color="#e2e8f0"),
    title_font=dict(size=14, family="Syne"),
    legend_title_text="Debt Category",
    margin={"r": 20, "t": 50, "l": 20, "b": 20},
    xaxis=dict(type="category", gridcolor="#1e2d4a"),
    yaxis=dict(gridcolor="#1e2d4a"),
    legend=dict(orientation="v", x=1.01, y=1, font=dict(size=10))
)
st.plotly_chart(fig_debt, use_container_width=True)

# ── DEMOLITIONS BREAKDOWN ──────────────────────────────────────────────────────
if not ward_demo.empty:
    st.markdown("---")
    st.markdown('<p class="section-label">🏗️ Demolition Permits</p>', unsafe_allow_html=True)

    if selected_ward == "Citywide":
        demo_plot = ward_demo.sort_values("total_demolitions", ascending=False).head(20).copy()
        demo_plot["ward"] = demo_plot["ward"].astype(str)
        demo_plot["private"] = demo_plot["total_demolitions"] - demo_plot["city_initiated"]
        demo_long = demo_plot.melt(
            id_vars="ward",
            value_vars=["city_initiated", "private"],
            var_name="type", value_name="count"
        )
        demo_long["type"] = demo_long["type"].map({
            "city_initiated": "City-Initiated",
            "private": "Private"
        })
        title_demo = "Top 20 Wards — Demolitions by Type"
    else:
        ward_row_demo = ward_demo[ward_demo["ward"] == selected_ward]
        if ward_row_demo.empty:
            st.info(f"No demolition data available for Ward {selected_ward}.")
            ward_row_demo = pd.DataFrame()
        if not ward_row_demo.empty:
            r = ward_row_demo.iloc[0]
            private = int(r["total_demolitions"]) - int(r["city_initiated"])
            demo_long = pd.DataFrame({
                "type": ["City-Initiated", "Private"],
                "count": [int(r["city_initiated"]), private]
            })
            demo_long["ward"] = str(selected_ward)
            title_demo = f"Ward {selected_ward} — Demolitions by Type"

    if not ward_demo.empty and (selected_ward == "Citywide" or not ward_row_demo.empty):
        fig_demo = px.bar(
            demo_long,
            x="ward" if selected_ward == "Citywide" else "type",
            y="count",
            color="type",
            barmode="stack",
            color_discrete_map={"City-Initiated": "#f97316", "Private": "#334155"},
            labels={"count": "Demolition Permits", "ward": "Ward", "type": "Type"},
            title=title_demo
        )
        fig_demo.update_layout(
            paper_bgcolor=paper_bgcolor, plot_bgcolor="#111827",
            font=dict(family="Syne", color="#e2e8f0"),
            title_font=dict(size=14, family="Syne"),
            legend_title_text="Demolition Type",
            margin={"r": 20, "t": 50, "l": 20, "b": 20},
            xaxis=dict(type="category", gridcolor="#1e2d4a"),
            yaxis=dict(gridcolor="#1e2d4a"),
        )
        st.plotly_chart(fig_demo, use_container_width=True)
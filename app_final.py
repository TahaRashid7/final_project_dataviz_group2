import streamlit as st
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(layout="wide")
st.title("Chicago Housing Distress Dashboard (2026)")

BASE = Path(__file__).parent

# -----------------------------
# CACHED LOADERS
# -----------------------------
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
    summary_path = BASE / "dataset" / "cleaned" / "ward_demolition_summary.csv"
    detail_path  = BASE / "dataset" / "cleaned" / "demolition_with_ward.csv"

    if summary_path.exists():
        demo = pd.read_csv(summary_path)
    elif detail_path.exists():
        raw = pd.read_csv(detail_path)
        raw["is_city_initiated"] = raw["is_city_initiated"].astype(str).str.upper() == "TRUE"
        demo = raw.groupby("ward").agg(
            total_demolitions=("PERMIT#", "count"),
            city_initiated=("is_city_initiated", "sum")
        ).reset_index()
    else:
        return pd.DataFrame(columns=["ward", "total_demolitions", "city_initiated"])

    demo["ward"]              = demo["ward"].astype(int)
    demo["city_initiated"]    = demo["city_initiated"].astype(int)
    demo["total_demolitions"] = demo["total_demolitions"].astype(int)
    return demo[["ward", "total_demolitions", "city_initiated"]]

# -----------------------------
# LOAD DATA
# -----------------------------
gdf                         = load_wards_geo()
vacant                      = load_vacant_csv()
ts                          = load_foreclosures_timeseries()
debt_long, ward_debt_totals = load_debt_by_ward()
ward_demo                   = load_demolitions_by_ward()

# Water debt calculation
water_categories = ["Water (Service)", "Water Penalty", "Water Tax (Service)", "Water Tax Penalty"]
water_debt = (
    debt_long[debt_long["category"].isin(water_categories)]
    .groupby("ward")["amount_m"]
    .sum()
    .reset_index()
    .rename(columns={"amount_m": "Water Debt ($M)"})
)

# Merge
gdf = gdf.merge(ward_debt_totals, on="ward", how="left")
gdf = gdf.merge(water_debt, on="ward", how="left")
if not ward_demo.empty:
    gdf = gdf.merge(ward_demo[["ward", "total_demolitions"]], on="ward", how="left")
else:
    gdf["total_demolitions"] = 0

gdf["total_debt_m"]      = gdf["total_debt_m"].fillna(0)
gdf["Water Debt ($M)"]   = gdf["Water Debt ($M)"].fillna(0)
gdf["total_demolitions"] = gdf["total_demolitions"].fillna(0)

# Recompute Housing Distress Index (foreclosure + vacancy + debt, normalized)
def normalize(s):
    mn, mx = s.min(), s.max()
    return (s - mn) / (mx - mn) if mx > mn else s * 0

gdf["housing_distress_index"] = (
    normalize(gdf["foreclosure_rate_2024"]) +
    normalize(gdf["vacant_count"]) +
    normalize(gdf["total_debt_m"])
) / 3

gdf["risk_tier"] = pd.qcut(
    gdf["housing_distress_index"], q=3,
    labels=["Low", "Watch", "Critical"]
)

gdf = gdf.rename(columns={
    "foreclosure_rate_2024":  "Foreclosures (2024)",
    "vacant_count":           "Vacant parcels",
    "housing_distress_index": "Housing Distress Index",
    "total_debt_m":           "Outstanding Debt ($M)",
    "total_demolitions":      "Demolitions"
})

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Controls")
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
        "Water Debt ($M)",
        "Demolitions",
        "Risk tier",
        "Housing Distress Index",
    ],
    index=0
)

zoom_mode    = st.sidebar.radio("View", options=["Citywide", "Selected ward"], index=0)
show_parcels = st.sidebar.checkbox("Show vacant parcel dots", value=True)

# -----------------------------
# SUMMARY METRICS
# -----------------------------
col1, col2, col3, col4, col5 = st.columns(5)

if selected_ward == "Citywide":
    col1.metric("Avg Foreclosure Rate",   f"{gdf['Foreclosures (2024)'].mean():.2f}%")
    col2.metric("Total Vacant Parcels",   f"{int(gdf['Vacant parcels'].sum()):,}")
    col3.metric("Avg Distress Index",     f"{gdf['Housing Distress Index'].mean():.3f}")
    col4.metric("Total Outstanding Debt", f"${gdf['Outstanding Debt ($M)'].sum():.0f}M")
    col5.metric("Critical Wards",         f"{int((gdf['risk_tier'] == 'Critical').sum())} of 50")
else:
    ward_row = gdf[gdf["ward"] == selected_ward].iloc[0]
    col1.metric("Foreclosures (2024)", f"{ward_row['Foreclosures (2024)']:.2f}%")
    col2.metric("Vacant Parcels",      f"{int(ward_row['Vacant parcels']):,}")
    col3.metric("Distress Index",      f"{ward_row['Housing Distress Index']:.3f}")
    col4.metric("Outstanding Debt",    f"${ward_row['Outstanding Debt ($M)']:.1f}M")
    col5.metric("Risk Tier",           ward_row["risk_tier"])

# -----------------------------
# MAP
# -----------------------------
hover_cols = [
    "ward", "Foreclosures (2024)", "Vacant parcels",
    "Housing Distress Index", "Outstanding Debt ($M)",
    "Water Debt ($M)", "Demolitions", "risk_tier"
]

color_scales = {
    "Foreclosures (2024)":    "Reds",
    "Vacant parcels":         "Blues",
    "Housing Distress Index": "Purples",
    "Outstanding Debt ($M)":  "Oranges",
    "Water Debt ($M)":        "GnBu",
    "Demolitions":            "YlOrRd"
}

if map_metric == "Risk tier":
    fig = px.choropleth_mapbox(
        gdf, geojson=gdf.__geo_interface__, locations="ward",
        featureidkey="properties.ward", color="risk_tier",
        mapbox_style="carto-positron", opacity=0.8, hover_data=hover_cols,
        color_discrete_map={"Low": "#2ecc71", "Watch": "#f1c40f", "Critical": "#e74c3c"}
    )
else:
    fig = px.choropleth_mapbox(
        gdf, geojson=gdf.__geo_interface__, locations="ward",
        featureidkey="properties.ward", color=map_metric,
        mapbox_style="carto-positron", opacity=0.75, hover_data=hover_cols,
        color_continuous_scale=color_scales.get(map_metric, "Blues")
    )

fig.update_layout(
    mapbox=dict(center=dict(lat=41.85, lon=-87.68), zoom=9.4),
    margin={"r": 0, "t": 0, "l": 0, "b": 0}
)

if selected_ward != "Citywide":
    if show_parcels:
        filtered = vacant[vacant["ward_spatial"] == selected_ward]
        fig.add_trace(go.Scattermapbox(
            lat=filtered["latitude"], lon=filtered["longitude"],
            mode="markers", marker=dict(size=4, color="blue", opacity=0.6),
            name="Vacant parcels"
        ))
    if zoom_mode == "Selected ward":
        center = gdf[gdf["ward"] == selected_ward].geometry.centroid.iloc[0]
        fig.update_layout(
            mapbox=dict(center=dict(lat=center.y, lon=center.x), zoom=11.5)
        )

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# LINE GRAPH
# -----------------------------
if map_metric == "Foreclosures (2024)" and selected_ward != "Citywide":
    st.subheader("Foreclosures over time")
    ts_ward = ts[ts["ward"] == selected_ward].sort_values("year")
    fig_line = px.line(ts_ward, x="year", y="foreclosures", markers=True,
                       title=f"Ward {selected_ward}: Foreclosures by Year")
    fig_line.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    st.plotly_chart(fig_line, use_container_width=True)

# -----------------------------
# DEBT BREAKDOWN
# -----------------------------
st.markdown("---")
st.subheader("Outstanding Debt Breakdown — Service vs. Penalties by Category")

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
    "Other":               "#d1d5db",
}

if selected_ward == "Citywide":
    top20     = debt_long.groupby("ward")["amount_m"].sum().nlargest(20).index.tolist()
    plot_data = debt_long[debt_long["ward"].isin(top20)]
    title     = "Top 20 Wards — Outstanding Debt by Category ($M)"
    x_col     = "ward"
else:
    plot_data = debt_long[debt_long["ward"] == selected_ward]
    title     = f"Ward {selected_ward} — Outstanding Debt by Category ($M)"
    x_col     = "category"

fig_debt = px.bar(
    plot_data, x=x_col, y="amount_m", color="category",
    barmode="stack", color_discrete_map=color_map,
    labels={"amount_m": "Amount ($M)", "ward": "Ward", "category": "Debt Category"},
    title=title
)
fig_debt.update_layout(
    legend_title_text="Debt Category",
    margin={"r": 0, "t": 40, "l": 0, "b": 0},
    xaxis=dict(type="category"),
    legend=dict(orientation="v", x=1.01, y=1)
)
st.plotly_chart(fig_debt, use_container_width=True)

# -----------------------------
# DEMOLITIONS
# -----------------------------
if not ward_demo.empty:
    st.markdown("---")
    st.subheader("Demolition Permits — City-Initiated vs. Private")

    if selected_ward == "Citywide":
        # Stacked bar for citywide top 20
        demo_plot = ward_demo.sort_values("total_demolitions", ascending=False).head(20).copy()
        demo_plot["ward"]    = demo_plot["ward"].astype(str)
        demo_plot["private"] = demo_plot["total_demolitions"] - demo_plot["city_initiated"]
        demo_long = demo_plot.melt(
            id_vars="ward", value_vars=["city_initiated", "private"],
            var_name="type", value_name="count"
        )
        demo_long["type"] = demo_long["type"].map({
            "city_initiated": "City-Initiated", "private": "Private"
        })
        fig_demo = px.bar(
            demo_long, x="ward", y="count", color="type", barmode="stack",
            color_discrete_map={"City-Initiated": "#3b82f6", "Private": "#94a3b8"},
            labels={"count": "Demolition Permits", "ward": "Ward", "type": "Type"},
            title="Top 20 Wards — Demolition Permits by Type"
        )
        fig_demo.update_layout(
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            xaxis=dict(type="category"),
        )
        st.plotly_chart(fig_demo, use_container_width=True)

    else:
        # Donut chart for single ward
        wr = ward_demo[ward_demo["ward"] == selected_ward]
        if not wr.empty:
            r           = wr.iloc[0]
            city_n      = int(r["city_initiated"])
            private_n   = int(r["total_demolitions"]) - city_n
            total_n     = int(r["total_demolitions"])

            fig_demo = go.Figure(data=[go.Pie(
                labels=["City-Initiated", "Private"],
                values=[city_n, private_n],
                hole=0.5,
                marker_colors=["#3b82f6", "#94a3b8"],
                textinfo="label+percent",
                textfont=dict(size=13),
                hovertemplate="%{label}: %{value} permits (%{percent})<extra></extra>"
            )])
            fig_demo.update_layout(
                title=f"Ward {selected_ward} — Demolitions by Type ({total_n:,} total)",
                margin={"r": 20, "t": 50, "l": 20, "b": 20},
                showlegend=True,
                legend=dict(orientation="h", x=0.3, y=-0.1),
                annotations=[dict(
                    text=f"<b>{total_n:,}</b><br>total",
                    x=0.5, y=0.5,
                    font_size=16,
                    showarrow=False
                )]
            )
            st.plotly_chart(fig_demo, use_container_width=True)
        else:
            st.info(f"No demolition data available for Ward {selected_ward}.")
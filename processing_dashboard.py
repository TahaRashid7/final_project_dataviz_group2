import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# --------------------------------------------------
# 1️⃣ LOAD 2023 WARD BOUNDARIES
# --------------------------------------------------

wards = gpd.read_file(
    "~/Downloads/Boundaries - Wards (2023-)_20260301/geo_export_dd898801-215f-493e-988d-d356065abd03.shp"
)

wards = wards.to_crs(epsg=4326)
wards["ward"] = wards["ward"].astype(int)

# --------------------------------------------------
# 2️⃣ LOAD VACANT PARCEL DATA
# --------------------------------------------------

vacant = pd.read_csv(
    "~/Downloads/Chicago_Vacant_Land_Parcels(in).csv"
)

vacant["latitude"] = vacant["latitude"].astype(float)
vacant["longitude"] = vacant["longitude"].astype(float)

geometry = [Point(xy) for xy in zip(vacant["longitude"], vacant["latitude"])]

vacant_gdf = gpd.GeoDataFrame(
    vacant,
    geometry=geometry,
    crs="EPSG:4326"
)

# --------------------------------------------------
# 3️⃣ SPATIAL JOIN VACANT → WARD (2023)
# --------------------------------------------------

# Drop any leftover join columns
for col in ["index_right", "index_left"]:
    if col in vacant_gdf.columns:
        vacant_gdf = vacant_gdf.drop(columns=[col])

vacant_joined = gpd.sjoin(
    vacant_gdf,
    wards[["ward", "geometry"]],
    how="left",
    predicate="within"
)

vacant_joined = vacant_joined.rename(columns={"ward": "ward_spatial"})
vacant_joined = vacant_joined.drop(columns=["index_right"])

# --------------------------------------------------
# 4️⃣ AGGREGATE VACANT COUNT BY WARD
# --------------------------------------------------

vacant_by_ward = (
    vacant_joined
    .groupby("ward_spatial")
    .size()
    .reset_index(name="vacant_count")
)

# --------------------------------------------------
# 5️⃣ LOAD FORECLOSURE RATE (2024)
# --------------------------------------------------

foreclosures = pd.read_csv(
    "https://raw.githubusercontent.com/aaryal22/final_project_dataviz_group2/main/dataset/raw/foreclosures_chicago_wards_clean.csv"
)


foreclosures["ward"] = (
    foreclosures["Geography"]
    .str.replace("Ward ", "", regex=False)
    .astype(int)
)

foreclosures_2024 = foreclosures[["ward", "2024"]]
foreclosures_2024 = foreclosures_2024.rename(
    columns={"2024": "foreclosure_rate_2024"}
)

# --------------------------------------------------
# 6️⃣ MERGE EVERYTHING INTO WARDS
# --------------------------------------------------

wards = wards.merge(
    vacant_by_ward,
    left_on="ward",
    right_on="ward_spatial",
    how="left"
)

wards = wards.merge(
    foreclosures_2024,
    on="ward",
    how="left"
)

wards["vacant_count"] = wards["vacant_count"].fillna(0)
wards["foreclosure_rate_2024"] = wards["foreclosure_rate_2024"].fillna(0)

# --------------------------------------------------
# 7️⃣ BUILD HOUSING DISTRESS INDEX
# --------------------------------------------------

# Normalize foreclosure rate
wards["foreclosure_norm"] = (
    (wards["foreclosure_rate_2024"] - wards["foreclosure_rate_2024"].min()) /
    (wards["foreclosure_rate_2024"].max() - wards["foreclosure_rate_2024"].min())
)

# Normalize vacancy
wards["vacancy_norm"] = (
    (wards["vacant_count"] - wards["vacant_count"].min()) /
    (wards["vacant_count"].max() - wards["vacant_count"].min())
)

# Composite index
wards["housing_distress_index"] = (
    wards["foreclosure_norm"] +
    wards["vacancy_norm"]
) / 2

# --------------------------------------------------
# 8️⃣ CREATE RISK TIERS (3 LEVELS)
# --------------------------------------------------

wards["risk_tier"] = pd.qcut(
    wards["housing_distress_index"],
    q=3,
    labels=["Low", "Watch", "Critical"]
)

# --------------------------------------------------
# 9️⃣ SAVE FINAL DASHBOARD FILE
# --------------------------------------------------

wards.to_file(
    "wards_2023_final_dashboard.geojson",
    driver="GeoJSON"
)

vacant_joined.to_file(
    "vacant_with_ward_2023.geojson",
    driver="GeoJSON"
)

print("Processing complete. Files exported.")



import geopandas as gpd

# Load full parcel geojson
vacant_full = gpd.read_file("vacant_with_ward_2023.geojson")

# Keep ONLY what the map needs
vacant_minimal = vacant_full[[
    "ward_spatial",
    "geometry"
]].copy()

# Convert geometry → lat/lon columns
vacant_minimal["longitude"] = vacant_minimal.geometry.x
vacant_minimal["latitude"] = vacant_minimal.geometry.y

# Drop geometry entirely
vacant_minimal = vacant_minimal.drop(columns="geometry")

# Save as CSV
vacant_minimal.to_csv("vacant_minimal.csv", index=False)
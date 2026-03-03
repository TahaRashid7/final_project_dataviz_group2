import pandas as pd
import requests
from bs4 import BeautifulSoup
import geopandas as gpd
from pathlib import Path

#Making reproducible paths
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "dataset" / "raw"
CLEANED_DIR = BASE_DIR / "dataset" / "cleaned"
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

#Loading data
df_vacant = pd.read_excel(RAW_DIR / "Chicago_Vacant_Land_Parcels.xlsx")
df_foreclosure = pd.read_csv(RAW_DIR / "foreclosures_chicago_wards_clean.csv")
ward_gdf = gpd.read_file(RAW_DIR / "wards_shapefile.csv")

#Exploring the vacant land data
df_vacant["class"].nunique() 
#190 and 100 are vacant lands: https://prodassets.cookcountyassessoril.gov/s3fs-public/form_documents/Class_codes_definitions_12.16.24_0.pdf?VersionId=O6TFbl9Nop_06LDdiBKus1AXNVzeiRoa
df_vacant.shape
#32,614 rows and 130 cols? The assessor parcel data had 50.7M rows and 124 cols.
assert df_vacant["ward_num"].nunique() == 50
assert df_vacant["ward_num"].isna().sum() == 0
assert df_vacant["class"].isna().sum() == 0


#Converting the vacant land data to ward level
df_vacant_wardlvl = df_vacant.groupby("ward_num").size().reset_index(name="vacant_land")
assert df_vacant_wardlvl.isna().sum().sum() == 0


#Exploring the foreclosure data
df_foreclosure.shape
#50 rows and 21 cols
df_foreclosure.head()
assert df_foreclosure["Geography"].nunique() == 50
assert df_foreclosure.isna().sum().sum() == 0


#Changing the ward names in the foreclosure data
df_foreclosure = df_foreclosure.rename(columns={"Geography" : "ward_num"})
df_foreclosure["ward_num"] = df_foreclosure["ward_num"].str.replace("Ward", "").astype(int)


#Merging the foreclosure data with vacant land data
combined_data = df_foreclosure.merge(df_vacant_wardlvl, on="ward_num", how ="left")
combined_data.columns


#Cleaning the combined dataset
year_cols = ['2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
for year in year_cols:
    combined_data = combined_data.rename(columns={year: f"foreclosure_{year}"})


#Merging the combined dataset with ward shapefile
assert ward_gdf["Ward"].nunique() == 50
assert ward_gdf["Ward"].isna().sum() == 0

ward_gdf = ward_gdf.rename(columns={"Ward" : "ward_num"})
ward_gdf["ward_num"] = ward_gdf["ward_num"].astype(int)
combined_gdf= ward_gdf.merge(combined_data, on="ward_num", how="left")
cols_to_drop = ["objectid", "edit_date", "ward_id", "globalid"]
combined_gdf = combined_gdf.drop(columns=[c for c in cols_to_drop if c in combined_gdf.columns])

combined_gdf.to_csv(CLEANED_DIR / "Merged_data.csv", index=False)






import pandas as pd

# Load data
print("Loading data...")
df = pd.read_parquet("cache/profiles.parquet")

# Filter to Indian Ocean (default dashboard filter)
df_io = df[
    (df["longitude"] >= 20.0) & (df["longitude"] <= 145.0) & 
    (df["latitude"] >= -70.1) & (df["latitude"] <= 30.0)
]

print("\n--- Metric 1: New Floats Deployed per Year (Current Dashboard Logic) ---")
float_years = df_io.dropna(subset=["year"]).groupby("wmo_id")["year"].min().reset_index()
yearly_new = float_years.groupby("year")["wmo_id"].nunique().reset_index()
print(yearly_new[yearly_new["year"].isin([1999, 2001, 2003, 2011, 2014, 2016, 2020, 2025, 2026])].to_string(index=False))

print("\n--- Metric 2: Active Floats per Year (Unique WMOs reporting in that year) ---")
yearly_active = df_io.dropna(subset=["year"]).groupby("year")["wmo_id"].nunique().reset_index()
print(yearly_active[yearly_active["year"].isin([1999, 2000, 2001, 2003, 2004, 2008, 2016, 2020, 2026])].to_string(index=False))

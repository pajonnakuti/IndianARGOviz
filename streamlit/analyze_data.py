"""Quick analysis: how many floats survive each filtering stage?"""
import pandas as pd
from global_land_mask import globe
import numpy as np

print("=== Loading raw data ===")
df = pd.read_csv("ar_index_global_prof.txt", comment="#")
df.columns = df.columns.str.strip()
df["wmo_id"] = df["file"].str.extract(r"/(\d+)/")

print(f"1. Raw rows: {len(df):,}")
print(f"   Raw unique floats: {df['wmo_id'].nunique():,}")

# After dropping NaN lat/lon
df2 = df.dropna(subset=["latitude", "longitude"])
print(f"\n2. After dropping NaN coords: {len(df2):,} rows, {df2['wmo_id'].nunique():,} floats")

# After valid range filter
df3 = df2[
    (df2["latitude"] >= -90) & (df2["latitude"] <= 90)
    & (df2["longitude"] >= -180) & (df2["longitude"] <= 180)
]
print(f"3. After valid range: {len(df3):,} rows, {df3['wmo_id'].nunique():,} floats")

# After land masking
is_land = globe.is_land(df3["latitude"].values, df3["longitude"].values)
land_count = int(is_land.sum())
df4 = df3[~is_land]
print(f"4. Land-masked removed: {land_count:,} rows")
print(f"   After land mask: {len(df4):,} rows, {df4['wmo_id'].nunique():,} floats")

# After Indian Ocean bounding box (default filters)
df5 = df4[
    (df4["longitude"] >= 20.0) & (df4["longitude"] <= 145.0)
    & (df4["latitude"] >= -70.1) & (df4["latitude"] <= 30.0)
]
print(f"\n5. Indian Ocean box (20-145E, 70.1S-30N):")
print(f"   Rows: {len(df5):,}, Unique floats: {df5['wmo_id'].nunique():,}")

# Latest position per float (what map shows)
df5_sorted = df5.copy()
df5_sorted["date"] = pd.to_datetime(df5_sorted["date"], format="%Y%m%d%H%M%S", errors="coerce")
map_df = (
    df5_sorted.dropna(subset=["latitude", "longitude"])
    .sort_values("date")
    .groupby("wmo_id")
    .tail(1)
)
print(f"   Map markers (latest pos per float): {len(map_df):,}")

# Check institutions in Indian Ocean
print(f"\n6. Institutions in Indian Ocean box:")
inst_counts = df5.groupby("institution")["wmo_id"].nunique().sort_values(ascending=False)
for inst, count in inst_counts.items():
    print(f"   {inst}: {count:,} floats")
print(f"   TOTAL: {inst_counts.sum():,}")

# Also check: how many floats does the GLOBAL dataset have whose LATEST position is in Indian Ocean?
print("\n7. Floats whose LATEST position falls in Indian Ocean:")
df4["date"] = pd.to_datetime(df4["date"], format="%Y%m%d%H%M%S", errors="coerce")
latest_pos = df4.dropna(subset=["date"]).sort_values("date").groupby("wmo_id").tail(1)
io_latest = latest_pos[
    (latest_pos["longitude"] >= 20.0) & (latest_pos["longitude"] <= 145.0)
    & (latest_pos["latitude"] >= -70.1) & (latest_pos["latitude"] <= 30.0)
]
print(f"   Floats with latest pos in IO: {len(io_latest):,}")

# Check: floats that EVER reported from Indian Ocean
print("\n8. Floats that EVER reported from Indian Ocean box:")
io_ever = df4[
    (df4["longitude"] >= 20.0) & (df4["longitude"] <= 145.0)
    & (df4["latitude"] >= -70.1) & (df4["latitude"] <= 30.0)
]
print(f"   Unique floats ever in IO: {io_ever['wmo_id'].nunique():,}")
print(f"   Total profiles in IO: {len(io_ever):,}")

# Argo reference numbers
print("\n=== For reference ===")
print(f"Global active floats (latest profile in last 30 days):")
latest_date = df4["date"].max()
thirty_days = pd.Timestamp(latest_date - pd.Timedelta(days=30))
active_global = latest_pos[latest_pos["date"] >= thirty_days]
print(f"   Global active: {len(active_global):,}")
active_io = io_latest[io_latest["date"] >= thirty_days]
print(f"   Indian Ocean active: {len(active_io):,}")

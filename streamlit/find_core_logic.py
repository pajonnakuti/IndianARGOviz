import pandas as pd
import numpy as np

target_values = {
    2014: 973,
    2016: 1094,
    2020: 1056,
}

df = pd.read_parquet("cache/profiles.parquet")
df_io = df[
    (df["longitude"] >= 20.0) & (df["longitude"] <= 145.0) & 
    (df["latitude"] >= -70.1) & (df["latitude"] <= 30.0)
].copy()
df_io["date"] = pd.to_datetime(df_io["date"])
df_io["year"] = df_io["date"].dt.year

df_bio = pd.read_parquet("cache/bgc_profiles.parquet")
bgc_wmos = set(df_bio["wmo_id"].unique())
DEEP_PROFILER_TYPES = {862, 864, 876, 882, 869, 863, 873, 874, 886, 877, 875, 884, 872, 879, 865, 860, 878, 861, 871, 870, 881, 853}

df_io["is_bgc"] = df_io["wmo_id"].isin(bgc_wmos)
if "profiler_type" in df_io.columns:
    df_io["is_deep"] = df_io["profiler_type"].isin(DEEP_PROFILER_TYPES)
else:
    df_io["is_deep"] = False

print("Total unique active floats in IO:")
active_yearly = df_io.dropna(subset=["year"]).groupby("year")["wmo_id"].nunique().to_dict()
for y in target_values: print(f"  {y}: Target={target_values[y]}, Calc={active_yearly.get(y, 0)}")

print("Core floats only (Not BGC, Not Deep):")
core_df = df_io[~df_io["is_bgc"] & ~df_io["is_deep"]]
active_core = core_df.dropna(subset=["year"]).groupby("year")["wmo_id"].nunique().to_dict()
for y in target_values: print(f"  {y}: Target={target_values[y]}, Calc={active_core.get(y, 0)}")

print("BGC floats only:")
bgc_df = df_io[df_io["is_bgc"]]
active_bgc = bgc_df.dropna(subset=["year"]).groupby("year")["wmo_id"].nunique().to_dict()
for y in target_values: print(f"  {y}: Target={target_values[y]}, Calc={active_bgc.get(y, 0)}")

# Let's try matching exactly by seeing if there's a specific month filter or something.
# Or maybe the data from the image is NOT Indian Ocean, but a specific subset of Global?

import pandas as pd
import numpy as np

# Target values from the image
target_values = {
    2000: 10,
    2001: 33,
    2003: 222,
    2004: 361,
    2006: 598,
    2007: 669,
    2009: 762,
    2010: 848,
    2011: 946,
    2013: 905,
    2014: 973,
    2016: 1094,
    2017: 1029,
    2018: 1011,
    2020: 1056,
    2022: 866,
    2023: 848,
    2025: 966
}

print("Loading data...")
df = pd.read_parquet("cache/profiles.parquet")
df_io = df[
    (df["longitude"] >= 20.0) & (df["longitude"] <= 145.0) & 
    (df["latitude"] >= -70.1) & (df["latitude"] <= 30.0)
].copy()
df_io["date"] = pd.to_datetime(df_io["date"])
df_io["year"] = df_io["date"].dt.year

print("\n--- Testing different metrics to match the target values ---")

def compare_to_target(calculated_dict, name):
    print(f"\n{name}")
    diffs = []
    for y, target in target_values.items():
        calc = calculated_dict.get(y, 0)
        diff = calc - target
        diffs.append(abs(diff))
        print(f"  {y}: Target={target}, Calc={calc}, Diff={diff}")
    print(f"  Avg absolute difference: {np.mean(diffs):.1f}")

# 1. Total unique floats per year (active in that year)
active_yearly = df_io.dropna(subset=["year"]).groupby("year")["wmo_id"].nunique().to_dict()
compare_to_target(active_yearly, "Metric 1: Active floats per year in Indian Ocean")

# 2. Total unique BGC floats per year? No, the title says "No. of Floats", let's try it anyway.
df_bio = pd.read_parquet("cache/bgc_profiles.parquet")
# The BGC profiles file doesn't have lat/lon in it in our cache, so we use wmo_id
bgc_wmos = set(df_bio["wmo_id"].unique())
active_bgc_yearly = df_io[df_io["wmo_id"].isin(bgc_wmos)].groupby("year")["wmo_id"].nunique().to_dict()
compare_to_target(active_bgc_yearly, "Metric 2: Active BGC floats per year in Indian Ocean")

# 3. Active floats per year globally (no lat/lon filter)
global_active = df.dropna(subset=["year"]).groupby("year")["wmo_id"].nunique().to_dict()
compare_to_target(global_active, "Metric 3: Active floats per year GLOBALLY")

# 4. Floats that were "Live" at the end of each year?
# A float is live if it reported a profile in the 90 days before Dec 31 of that year.
def active_at_end_of_year(df, days=90):
    res = {}
    years = sorted(df["year"].dropna().unique())
    for y in years:
        end_of_year = pd.Timestamp(f"{int(y)}-12-31")
        start_period = end_of_year - pd.Timedelta(days=days)
        # Floats that reported in the last 90 days of the year
        active = df[(df["date"] >= start_period) & (df["date"] <= end_of_year)]["wmo_id"].nunique()
        res[y] = active
    return res

compare_to_target(active_at_end_of_year(df_io, 90), "Metric 4: Active in last 90 days of year (IO)")
compare_to_target(active_at_end_of_year(df_io, 365), "Metric 5: Active in last 365 days of year (IO)")

# 6. Core floats only? (Not deep, not BGC)
core_wmos = set(df_io[~df_io["is_bgc"] & ~df_io["is_deep"]]["wmo_id"])
active_core = df_io[df_io["wmo_id"].isin(core_wmos)].groupby("year")["wmo_id"].nunique().to_dict()
compare_to_target(active_core, "Metric 6: Active CORE floats per year (IO)")


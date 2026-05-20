import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"c:\Users\harsh\incois\dashboard")
PROF_FILE = BASE_DIR / "ar_index_global_prof.txt"

def check_land_points():
    print(f"Reading {PROF_FILE}...")
    df = pd.read_csv(PROF_FILE, comment="#")
    df.columns = df.columns.str.strip()
    
    # India land area approx: 8-36N, 68-95E
    india_land = df[
        (df["latitude"] > 8) & (df["latitude"] < 36) & 
        (df["longitude"] > 68) & (df["longitude"] < 95)
    ].copy()
    
    india_land["wmo_id"] = india_land["file"].str.extract(r"/(\d+)/")
    latest = india_land.sort_values("date").groupby("wmo_id").tail(1)
    
    # Specific search for JA floats in this box
    ja_in_india = latest[latest["institution"] == "JA"]
    print(f"JA floats in India box: {len(ja_in_india)}")
    if len(ja_in_india) > 0:
        print(ja_in_india[["wmo_id", "latitude", "longitude", "institution", "date"]].to_string())

    # All floats in India land box
    print(f"\nAll unique floats in India box: {len(latest)}")
    # Print top 20 suspicious ones (high latitude, inland)
    suspicious = latest[latest["latitude"] > 10]
    print(suspicious[["wmo_id", "latitude", "longitude", "institution", "date"]].head(20))

if __name__ == "__main__":
    check_land_points()

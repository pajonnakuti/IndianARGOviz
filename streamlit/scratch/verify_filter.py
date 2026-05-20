import pandas as pd
from global_land_mask import globe
from pathlib import Path

PROF_FILE = Path(r"c:\Users\harsh\incois\dashboard\ar_index_global_prof.txt")

def verify():
    print("Verifying land mask filtering...")
    df = pd.read_csv(PROF_FILE, comment="#")
    df.columns = df.columns.str.strip()
    
    initial_count = len(df)
    print(f"Initial row count: {initial_count:,}")
    
    # Filter logic from dashboard.py
    df = df.dropna(subset=["latitude", "longitude"])
    
    # Ensure coordinates are within valid ranges for global-land-mask
    df = df[(df["latitude"] >= -90) & (df["latitude"] <= 90) & 
            (df["longitude"] >= -180) & (df["longitude"] <= 180)]
            
    is_on_land = globe.is_land(df["latitude"].values, df["longitude"].values)
    df_filtered = df[~is_on_land]
    
    final_count = len(df_filtered)
    removed_count = initial_count - final_count
    print(f"Final row count: {final_count:,}")
    print(f"Rows removed (on land): {removed_count:,}")
    
    # Check if any profiles in the filtered set are still on land
    still_on_land = globe.is_land(df_filtered["latitude"].values, df_filtered["longitude"].values)
    land_count = still_on_land.sum()
    print(f"Profiles remaining in sea-only set that are still on land: {land_count}")
    
    if land_count == 0:
        print("\nSUCCESS: No profiles remain on land.")
    else:
        print(f"\nFAILURE: {land_count} profiles are still on land.")

if __name__ == "__main__":
    verify()

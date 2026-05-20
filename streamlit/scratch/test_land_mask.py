import pandas as pd
from global_land_mask import globe
import warnings

warnings.filterwarnings("ignore")

PROF_FILE = "ar_index_global_prof.txt"

df = pd.read_csv(PROF_FILE, comment="#")
df.columns = df.columns.str.strip()
df = df.dropna(subset=["latitude", "longitude"])
df = df[
    (df["latitude"] >= -90) & (df["latitude"] <= 90) &
    (df["longitude"] >= -180) & (df["longitude"] <= 180)
]

total_points = len(df)
is_land = globe.is_land(df["latitude"].values, df["longitude"].values)
total_land_points = is_land.sum()

print(f"Total points: {total_points}")
print(f"Total points on land globally: {total_land_points}")

# India bounding box
in_india_bbox = (df["latitude"] >= 6.0) & (df["latitude"] <= 36.0) & (df["longitude"] >= 68.0) & (df["longitude"] <= 98.0)
india_land_points = (in_india_bbox & is_land).sum()

print(f"Total points on land in India BBox: {india_land_points}")

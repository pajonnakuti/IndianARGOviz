import pandas as pd

META_FILE = r"c:\Users\harsh\incois\dashboard\ar_index_global_meta.txt"
PROF_FILE = r"c:\Users\harsh\incois\dashboard\ar_index_global_prof.txt"

df_meta = pd.read_csv(META_FILE, comment="#")
df_meta.columns = df_meta.columns.str.strip()
df_meta['wmo_id'] = df_meta['file'].str.extract(r'/(\d+)/')

df_prof = pd.read_csv(PROF_FILE, comment="#", usecols=['file', 'date', 'institution'])
df_prof.columns = df_prof.columns.str.strip()
df_prof['wmo_id'] = df_prof['file'].str.extract(r'/(\d+)/')
df_prof['date'] = pd.to_datetime(df_prof['date'], format='%Y%m%d%H%M%S', errors='coerce')

# Get earliest profile date
incois_prof = df_prof[df_prof["institution"] == "IN"]
deployments = incois_prof.groupby("wmo_id")["date"].min().reset_index()

# Get all 615 INCOIS meta floats
meta_in = df_meta[df_meta["institution"] == "IN"].copy()
print(f"Meta IN WMOs: {len(meta_in)}")

# Merge
merged = pd.merge(meta_in, deployments, on="wmo_id", how="left")

# Fill missing dates with date_update
merged["date_update"] = pd.to_datetime(merged["date_update"], format="%Y%m%d%H%M%S", errors='coerce')
merged["date_final"] = merged["date"].fillna(merged["date_update"])

print(f"Missing dates before fallback: {merged['date'].isna().sum()}")
print(f"Missing dates after fallback: {merged['date_final'].isna().sum()}")

merged["Year"] = merged["date_final"].dt.year
merged["Month"] = merged["date_final"].dt.month

print(f"Missing Year: {merged['Year'].isna().sum()}")
print(f"Missing Month: {merged['Month'].isna().sum()}")

pivot = merged.pivot_table(index="Month", columns="Year", values="wmo_id", aggfunc="count", fill_value=0)
print(pivot.sum().sum())

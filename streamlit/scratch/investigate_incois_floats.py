import pandas as pd

PROF_FILE = r"c:\Users\harsh\incois\dashboard\ar_index_global_prof.txt"
META_FILE = r"c:\Users\harsh\incois\dashboard\ar_index_global_meta.txt"

df_meta = pd.read_csv(META_FILE, comment="#")
df_meta.columns = df_meta.columns.str.strip()
df_meta['wmo_id'] = df_meta['file'].str.extract(r'/(\d+)/')

# Filter INCOIS
meta_in = df_meta[df_meta['institution'] == 'IN']
meta_wmos = set(meta_in['wmo_id'].unique())
print(f"Total INCOIS WMOs in meta: {len(meta_wmos)}")

# Now check prof file before any filtering
df_prof_raw = pd.read_csv(PROF_FILE, comment="#", usecols=['file', 'date', 'latitude', 'longitude'])
df_prof_raw.columns = df_prof_raw.columns.str.strip()
df_prof_raw['wmo_id'] = df_prof_raw['file'].str.extract(r'/(\d+)/')

prof_in_raw = df_prof_raw[df_prof_raw['wmo_id'].isin(meta_wmos)]
prof_raw_wmos = set(prof_in_raw['wmo_id'].unique())

print(f"INCOIS WMOs in raw prof file: {len(prof_raw_wmos)}")
print(f"Missing from raw prof file: {len(meta_wmos - prof_raw_wmos)}")
print(meta_wmos - prof_raw_wmos)

# Now check after coordinate filtering
df_prof_coords = prof_in_raw.dropna(subset=['latitude', 'longitude'])
df_prof_coords = df_prof_coords[
    (df_prof_coords["latitude"] >= -90) & (df_prof_coords["latitude"] <= 90) &
    (df_prof_coords["longitude"] >= -180) & (df_prof_coords["longitude"] <= 180)
]
prof_coord_wmos = set(df_prof_coords['wmo_id'].unique())
print(f"INCOIS WMOs after coordinate filter: {len(prof_coord_wmos)}")

# Now check after date formatting and filtering
df_prof_coords['date'] = pd.to_datetime(df_prof_coords['date'], format='%Y%m%d%H%M%S', errors='coerce')
df_prof_dates = df_prof_coords.dropna(subset=['date'])
prof_date_wmos = set(df_prof_dates['wmo_id'].unique())
print(f"INCOIS WMOs after valid date filter: {len(prof_date_wmos)}")

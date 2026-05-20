"""Quick analysis of ar_index_global_meta.txt to understand the data."""
import pandas as pd

META_FILE = r"c:\Users\harsh\incois\dashboard\ar_index_global_meta.txt"
df = pd.read_csv(META_FILE, comment="#")
df.columns = df.columns.str.strip()

print(f"Total rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"\nUnique profiler_type codes: {df['profiler_type'].nunique()}")
print(f"\nProfiler type distribution (top 20):")
print(df['profiler_type'].value_counts().head(20).to_string())

print(f"\n\nUnique institutions: {df['institution'].nunique()}")
print(f"\nInstitution distribution:")
print(df['institution'].value_counts().to_string())

# Extract DAC and WMO from file path
df['dac'] = df['file'].str.extract(r'^([^/]+)/')
df['wmo_id'] = df['file'].str.extract(r'/(\d+)/')

print(f"\n\nUnique DACs: {df['dac'].nunique()}")
print(f"\nDAC distribution:")
print(df['dac'].value_counts().to_string())

print(f"\n\nTotal unique floats (WMOs): {df['wmo_id'].nunique():,}")

# Check overlap with prof file
PROF_FILE = r"c:\Users\harsh\incois\dashboard\ar_index_global_prof.txt"
df_prof = pd.read_csv(PROF_FILE, comment="#", usecols=['file'])
df_prof.columns = df_prof.columns.str.strip()
df_prof['wmo_id'] = df_prof['file'].str.extract(r'/(\d+)/')
prof_wmos = set(df_prof['wmo_id'].dropna().unique())
meta_wmos = set(df['wmo_id'].dropna().unique())

print(f"\nWMOs in meta but NOT in prof: {len(meta_wmos - prof_wmos):,}")
print(f"WMOs in prof but NOT in meta: {len(prof_wmos - meta_wmos):,}")
print(f"WMOs in BOTH: {len(meta_wmos & prof_wmos):,}")

# Check which profiler_type codes are NOT in the prof file
df_prof_full = pd.read_csv(PROF_FILE, comment="#", usecols=['file', 'profiler_type'])
df_prof_full.columns = df_prof_full.columns.str.strip()
prof_ptypes = set(df_prof_full['profiler_type'].dropna().unique())
meta_ptypes = set(df['profiler_type'].dropna().unique())
print(f"\nProfiler types in meta only: {meta_ptypes - prof_ptypes}")
print(f"Profiler types in prof only: {prof_ptypes - meta_ptypes}")

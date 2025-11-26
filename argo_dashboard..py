# argo_dashboard.py
# Full dashboard (Static map + Bar + Donut + Interactive map + DAC table)
# - Static & Interactive map use: C:\Users\medagam INDU\Desktop\total dataset.csv
# - Bar, Donut, Table use:    C:\Users\medagam INDU\Desktop\bio_dataset.csv

import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime

# ------------------------------
# UPDATE THESE PATHS if needed
# ------------------------------
TOTAL_CSV = r"C:\Users\medagam INDU\Desktop\total dataset.csv"
BIO_CSV   = r"C:\Users\medagam INDU\Desktop\bio_dataset.csv"

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Argo Float Dashboard", layout="wide")
# Simple headings only (no extra notes)
st.title("Argo Float Dashboard")

# ------------------------------
# Helper: detect lat/lon column names
# ------------------------------
def detect_latlon(df):
    lat = None
    lon = None
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("lat", "latitude") or cl.startswith("lat"):
            lat = c
        if cl in ("lon", "long", "longitude") or cl.startswith("lon") or cl.startswith("long"):
            lon = c
    return lat, lon

# ------------------------------
# Load datasets (with caching)
# ------------------------------
@st.cache_data
def load_total():
    df = pd.read_csv(TOTAL_CSV, low_memory=False, encoding='latin1')
    return df

@st.cache_data
def load_bio():
    df = pd.read_csv(BIO_CSV, low_memory=False, encoding='latin1')
    return df

total_df = load_total()
bio_df = load_bio()

# ------------------------------
# Prepare total dataset for static map: try find core/bio type or split
# ------------------------------
def prepare_total_map(df, N_split=10):
    d = df.copy()
    # detect lat/lon
    lat_col, lon_col = detect_latlon(d)
    # find a type column if exists
    type_col = None
    for cand in ("Type", "FLOAT_TYPE", "Float_Type", "float_type", "TYPE"):
        if cand in d.columns:
            type_col = cand
            break
    if type_col is not None:
        core = d[d[type_col].astype(str).str.lower().str.contains("core", na=False)].copy()
        bio  = d[d[type_col].astype(str).str.lower().str.contains("bio|bgc", na=False)].copy()
        if core.empty and bio.empty:
            core = d.copy()
            bio = d.iloc[0:0].copy()
    else:
        # fallback to Excel-like split (first N columns core, rest bio)
        try:
            core = d.iloc[:, :N_split].copy()
            bio  = d.iloc[:, N_split:].copy()
        except Exception:
            core = d.copy()
            bio = d.iloc[0:0].copy()
    return core, bio, lat_col, lon_col

core_map_df, bio_map_df, total_lat_col, total_lon_col = prepare_total_map(total_df, N_split=10)

# Ensure we have lat/lon for total dataset (fallback common names)
if total_lat_col is None or total_lon_col is None:
    for a,b in [("LATITUDE","LONGITUDE"), ("LAT","LONG"), ("Latitude","Longitude")]:
        if a in total_df.columns and b in total_df.columns:
            total_lat_col, total_lon_col = a,b
            break

# ------------------------------
# Prepare bio dataset (dates, per-float aggregated)
# ------------------------------
def prepare_bio(df):
    b = df.copy()
    # find birth and death columns
    birth_col = None
    death_col = None
    for c in ("DATE_UPDATE","DATE"):
        if c in b.columns:
            birth_col = c
            break
    for c in ("DATE_UPDATE.1","DATE.1","DATE_UPDATE1","DATE_UPDATE_1"):
        if c in b.columns:
            death_col = c
            break
    # fallback search any 'date' columns
    if birth_col is None:
        for c in b.columns:
            if 'date' in c.lower():
                birth_col = c
                break
    # parse dates
    if birth_col is not None:
        b[birth_col] = pd.to_datetime(b[birth_col], errors='coerce', dayfirst=False)
    if death_col is not None:
        b[death_col] = pd.to_datetime(b[death_col], errors='coerce', dayfirst=False)
    else:
        b['DATE_UPDATE.1'] = pd.NaT
        death_col = 'DATE_UPDATE.1'
    # Year for bar chart
    if birth_col is not None:
        b['Year'] = b[birth_col].dt.year
    else:
        b['Year'] = pd.NA
    # id column
    id_col = 'WMOID' if 'WMOID' in b.columns else b.columns[0]
    # per-float aggregation: birth=min(birth_col), death=max(death_col)
    grouped = b.groupby(id_col).agg({
        birth_col: 'min' if birth_col in b.columns else (lambda x: pd.NaT),
        death_col: 'max' if death_col in b.columns else (lambda x: pd.NaT),
        'DAC': 'first' if 'DAC' in b.columns else (lambda x: None)
    }).reset_index().rename(columns={birth_col: 'birth', death_col: 'death'})
    today = pd.Timestamp.today()
    grouped['end_date'] = grouped['death'].fillna(today)
    grouped['age_days'] = (grouped['end_date'] - grouped['birth']).dt.days
    # classify by 90 days
    grouped['status_90'] = grouped['age_days'].apply(lambda x: 'Live' if pd.notnull(x) and x >= 90 else 'Dead')
    return b, id_col, birth_col, death_col, grouped

bio_prepared, bio_id_col, bio_birth_col, bio_death_col, bio_floats_grouped = prepare_bio(bio_df)

# ------------------------------
# Sidebar: filters and search
# ------------------------------
st.sidebar.header("Controls")
# Year filter (bio)
years_available = sorted([int(y) for y in bio_prepared['Year'].dropna().unique() if pd.notnull(y)])
selected_years = st.sidebar.multiselect("Year(s) for bar chart", years_available, default=years_available)
# DAC filter
dac_options = sorted(bio_prepared['DAC'].dropna().unique().astype(str)) if 'DAC' in bio_prepared.columns else []
selected_dacs = st.sidebar.multiselect("DAC(s)", dac_options, default=dac_options)
# Search box
search_text = st.sidebar.text_input("Search WMOID or DAC")

# Apply filters to bio_prepared for bar/donut
bio_filtered = bio_prepared.copy()
if selected_years:
    bio_filtered = bio_filtered[bio_filtered['Year'].isin(selected_years)]
if selected_dacs:
    bio_filtered = bio_filtered[bio_filtered['DAC'].astype(str).isin(selected_dacs)]
if search_text and search_text.strip():
    bio_filtered = bio_filtered[
        bio_filtered['WMOID'].astype(str).str.contains(search_text, case=False, na=False) |
        bio_filtered['DAC'].astype(str).str.contains(search_text, case=False, na=False)
    ]

# Also filter grouped_for_table (per-float) by DAC/search
grouped_for_table = bio_floats_grouped.copy()
if selected_dacs:
    grouped_for_table = grouped_for_table[grouped_for_table['DAC'].astype(str).isin(selected_dacs)]
if search_text and search_text.strip():
    grouped_for_table = grouped_for_table[
        grouped_for_table[bio_id_col].astype(str).str.contains(search_text, case=False, na=False) |
        grouped_for_table['DAC'].astype(str).str.contains(search_text, case=False, na=False)
    ]

# ------------------------------
# Static Map (top) — core pink, bio blue (use total dataset)
# ------------------------------
st.subheader("Static Map")

plt.figure(figsize=(14,7))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# detect lat/lon in core_map_df and bio_map_df returned earlier from total split
core_lat, core_lon = detect_latlon(core_map_df)
bio_lat, bio_lon = detect_latlon(bio_map_df)

# fallback common names
if core_lat is None or core_lon is None:
    for a,b in (("LATITUDE","LONGITUDE"), ("LAT","LONG"), ("Latitude","Longitude")):
        if a in core_map_df.columns and b in core_map_df.columns:
            core_lat, core_lon = a,b
            break
if bio_lat is None or bio_lon is None:
    for a,b in (("LATITUDE","LONGITUDE"), ("LAT","LONG"), ("Latitude","Longitude")):
        if a in bio_map_df.columns and b in bio_map_df.columns:
            bio_lat, bio_lon = a,b
            break

# plot if available
if core_lat and core_lon and (core_lat in core_map_df.columns) and (core_lon in core_map_df.columns):
    ax.scatter(core_map_df[core_lon], core_map_df[core_lat], color='pink', s=15, alpha=0.7, label='Core')
if bio_lat and bio_lon and (bio_lat in bio_map_df.columns) and (bio_lon in bio_map_df.columns):
    ax.scatter(bio_map_df[bio_lon], bio_map_df[bio_lat], color='blue', s=15, alpha=0.7, label='Bio')

plt.title("Core (pink) vs Bio (blue) — Map")
plt.legend(loc='upper right')
st.pyplot(plt.gcf())

# ------------------------------
# Middle row: Bar chart (left) and Donut chart (right) — both from bio dataset
# ------------------------------
col1, col2 = st.columns(2)

# Bar chart: number of unique floats per Year and DAC
with col1:
    st.subheader("Bar Chart")
    id_col = bio_id_col
    df_grouped = bio_filtered.groupby(['Year','DAC'])[id_col].nunique().reset_index(name='Float_Count')
    totals = df_grouped.groupby('Year')['Float_Count'].sum().reset_index(name='Total_Floats')

    fig_bar = px.bar(
        df_grouped,
        x='Year',
        y='Float_Count',
        color='DAC',
        text='Float_Count',
        title='Number of Floats per DAC'
    )
    for _, r in totals.iterrows():
        fig_bar.add_annotation(x=r['Year'], y=r['Total_Floats'], text=str(int(r['Total_Floats'])), showarrow=False, yshift=10)
    fig_bar.update_layout(barmode='stack', xaxis=dict(dtick=1), height=520)
    st.plotly_chart(fig_bar, use_container_width=True)

# Donut chart: age distribution of alive floats (bio dataset)
with col2:
    st.subheader("Donut Chart")
    g = bio_floats_grouped.copy()
    # alive = death is NaT
    alive = g[g['death'].isna()].copy()
    if not alive.empty:
        alive['age_years'] = ((pd.Timestamp.today() - alive['birth']).dt.days // 365).astype('Int64')
        alive = alive[alive['age_years'].notna() & (alive['age_years'] >= 0)]
        age_counts = alive['age_years'].value_counts().sort_index().reset_index()
        age_counts.columns = ['Age_Years','Count']
        if not age_counts.empty:
            fig_donut = px.pie(age_counts, names='Age_Years', values='Count', hole=0.55, title='Age (years) distribution of alive floats')
            fig_donut.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.write("No alive float age groups available for selected filters.")
    else:
        st.write("No alive floats found for selected filters.")

# ------------------------------
# Interactive Plotly Map (uses total dataset)
# ------------------------------
# ------------------------------
# Interactive Plotly Map (ALL DACs always visible)
# ------------------------------
st.subheader("Interactive Plotly Map (All DACs)")

# copy full total dataset (no DAC filtering)
map_df = total_df.copy()

# detect lat/lon
lat_t, lon_t = detect_latlon(map_df)
if lat_t is None and "LATITUDE" in map_df.columns:
    lat_t = "LATITUDE"
if lon_t is None and "LONGITUDE" in map_df.columns:
    lon_t = "LONGITUDE"
if lat_t is None and "LAT" in map_df.columns:
    lat_t = "LAT"
if lon_t is None and "LONG" in map_df.columns:
    lon_t = "LONG"

if lat_t is None or lon_t is None:
    st.warning("Latitude/Longitude not found in total dataset for interactive map.")
else:
    # Convert dates
    if 'DATE_UPDATE' in map_df.columns:
        map_df['DATE_UPDATE'] = pd.to_datetime(map_df['DATE_UPDATE'], errors='coerce')
    if 'DATE_UPDATE.1' in map_df.columns:
        map_df['DATE_UPDATE.1'] = pd.to_datetime(map_df['DATE_UPDATE.1'], errors='coerce')

    # Compute status (90-day rule)
    if 'DATE_UPDATE' in map_df.columns:
        map_df['end_date'] = map_df['DATE_UPDATE.1'].fillna(pd.Timestamp.today()) \
                             if 'DATE_UPDATE.1' in map_df.columns else pd.Timestamp.today()
        map_df['age_days'] = (map_df['end_date'] - map_df['DATE_UPDATE']).dt.days
        map_df['Status'] = map_df['age_days'].apply(
            lambda x: 'Live' if pd.notnull(x) and x >= 90 else 'Dead'
        )
    else:
        map_df['Status'] = 'Unknown'

    # Hover information
    hover_data = {
        lat_t: True,
        lon_t: True,
        "Status": True
    }
    if "WMOID" in map_df.columns:
        hover_data["WMOID"] = True
    if "DAC" in map_df.columns:
        hover_data["DAC"] = True
    if "DATE_UPDATE" in map_df.columns:
        hover_data["DATE_UPDATE"] = True
    if "DATE_UPDATE.1" in map_df.columns:
        hover_data["DATE_UPDATE.1"] = True

    # Plot ALL DACs (no filters)
    fig_map = px.scatter_geo(
        map_df,
        lat=lat_t,
        lon=lon_t,
        color="DAC" if "DAC" in map_df.columns else None,
        hover_name="WMOID" if "WMOID" in map_df.columns else None,
        hover_data=hover_data,
        title="Interactive Map — All DACs",
        projection="natural earth"
    )

    fig_map.update_layout(height=650)
    st.plotly_chart(fig_map, use_container_width=True)
# ------------------------------
# DAC summary table (bio dataset) - bottom
# Live = age_days >= 90, Dead = age_days < 90, Total = unique floats
# ------------------------------
st.subheader("DAC Table")

g = grouped_for_table.copy()
# ensure age_days already present (end_date computed earlier)
g['age_days'] = (g['end_date'] - g['birth']).dt.days
g['Live90'] = g['age_days'].apply(lambda x: 1 if pd.notnull(x) and x >= 90 else 0)
g['Dead90'] = g['age_days'].apply(lambda x: 1 if pd.notnull(x) and x < 90 else 0)

summary_table = g.groupby('DAC').agg(
    Live=('Live90','sum'),
    Dead=('Dead90','sum'),
    Total=(bio_id_col, 'nunique')
).reset_index()

# present in requested order: DAC | Live | Dead | Total
summary_table = summary_table[['DAC','Live','Dead','Total']]
st.dataframe(summary_table, use_container_width=True)

# Compact text lines
st.markdown("**Compact summary (DAC — Live / Dead / Total)**")
for _, row in summary_table.iterrows():
    st.write(f"{row['DAC']} — {int(row['Live'])} / {int(row['Dead'])} / {int(row['Total'])}")

import streamlit as st
import pandas as pd
import plotly.express as px
import geopandas as gpd
from shapely.geometry import Point

st.set_page_config(layout="wide")

# ---------------- DATA FILES ----------------
bio_data = "argo_bio-profile_index.txt"
core_data = "ar_index_global_prof.txt"

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df_bio = pd.read_csv(bio_data, comment="#")
    df_core = pd.read_csv(core_data, comment="#")
    return df_bio, df_core

df_bio, df_core = load_data()

st.title("Indian ARGO CTD_BGC")
st.caption("Global profiling float data visualization | Data Source: IFREMER")

# ---------------- DATE CONVERSION ----------------
df_bio['date'] = pd.to_datetime(df_bio['date'], format='%Y%m%d%H%M%S', errors='coerce')
df_core['date'] = pd.to_datetime(df_core['date'], format='%Y%m%d%H%M%S', errors='coerce')

# ---------------- YEAR COLUMN ----------------
df_bio['year'] = df_bio['date'].dt.year
df_core['year'] = df_core['date'].dt.year

# ---------------- LATEST LOCATION PER FLOAT ----------------
df_bio_latest = df_bio.sort_values('date').groupby('file').tail(1).copy()
df_bio_latest['type'] = "BGC"

df_core_latest = df_core.sort_values('date').groupby('file').tail(1).copy()
df_core_latest['type'] = "CTD"

# ---------------- COMBINE BOTH ----------------
df_all = pd.concat([
    df_bio_latest[['latitude','longitude','institution','type','file']],
    df_core_latest[['latitude','longitude','institution','type','file']]
], ignore_index=True)

# ---------------- CLEAN DATA ----------------
df_all = df_all.dropna(subset=['latitude','longitude'])

# ---------------- GEO FILTER (OCEAN ONLY) ----------------
@st.cache_data
def filter_ocean(df):
    geometry = [Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    world = gpd.read_file(
        "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
    )

    land = world[world['CONTINENT'] != 'Antarctica']

    gdf_ocean = gdf[~gdf.within(land.geometry.union_all())]

    gdf_ocean = gdf_ocean[
        (gdf_ocean['latitude'] >= -60) & (gdf_ocean['latitude'] <= 30)
    ]

    return pd.DataFrame(gdf_ocean.drop(columns='geometry'))

df_map_full = filter_ocean(df_all)

# ---------------- SIDEBAR FILTER ----------------
option = st.sidebar.radio(
    "Select Network",
    ["ALL", "BGC", "CTD"]
)

if option == "ALL":
    df_map = df_map_full
elif option == "BGC":
    df_map = df_map_full[df_map_full['type'] == "BGC"]
else:
    df_map = df_map_full[df_map_full['type'] == "CTD"]

# ---------------- REDUCE DATA SIZE ----------------
if len(df_map) > 8000:
    df_map = df_map.sample(8000, random_state=42)

# ---------------- KPI COUNTS ----------------
doxy = df_bio[df_bio['parameters'].str.contains("DOXY", na=False)].shape[0]
chla = df_bio[df_bio['parameters'].str.contains("CHLA", na=False)].shape[0]
nitrate = df_bio[df_bio['parameters'].str.contains("NITRATE", na=False)].shape[0]
ph = df_bio[df_bio['parameters'].str.contains("PH", na=False)].shape[0]

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([3, 1])

# ---------------- MAP ----------------
with col1:
    st.subheader("Geographic Distribution (Indian Ocean → Antarctica)")

    fig_map = px.scatter_mapbox(
        df_map,
        lat="latitude",
        lon="longitude",
        color="type",
        hover_data=["file", "institution", "type"],
        zoom=3,
        height=650
    )

    fig_map.update_traces(marker=dict(size=5))

    fig_map.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=-10, lon=80), zoom=3)
    )

    st.plotly_chart(fig_map, width="stretch")

# ---------------- KPI ----------------
with col2:
    st.metric("DOXY Profiles", f"{doxy:,}")
    st.metric("Chla Profiles", f"{chla:,}")
    st.metric("Nitrate Profiles", f"{nitrate:,}")
    st.metric("pH Profiles", f"{ph:,}")

# ---------------- BAR CHART ----------------
st.markdown("---")

st.subheader("Number of Floats Deployed Over Years")

# combine years
df_years = pd.concat([
    df_bio[['file','year']],
    df_core[['file','year']]
])

# remove duplicate floats
df_years = df_years.drop_duplicates('file')

# count per year
year_counts = df_years.groupby('year').size().reset_index(name='count')

# plot
fig_bar = px.bar(
    year_counts,
    x="year",
    y="count"
)

st.plotly_chart(fig_bar, width="stretch")
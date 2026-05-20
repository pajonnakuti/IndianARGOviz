"""
Indian ARGO CTD / BGC Float Dashboard
======================================
Streamlit re-implementation per INCOIS PRD.
Data Source: Argo GDAC (IFREMER)

Components
----------
1. Geospatial float-position map (colour-coded by institution/region)
2. Annual float-count bar chart (1999–present)
3. BGC profile KPI tiles (DOXY, Chla, Nitrate, pH)
4. Active floats/profiles last-7-days treemap
5. Float-age donut chart
6. DAC/Institution summary table
"""

# ==================== IMPORTS ====================
import streamlit as st
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import xarray as xr

warnings.filterwarnings("ignore")

# ==================== PATHS & CONSTANTS ====================
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
PROF_FILE = BASE_DIR / "ar_index_global_prof.txt"
BIO_FILE = BASE_DIR / "argo_bio-profile_index.txt"
META_FILE = BASE_DIR / "ar_index_global_meta.txt"

# Institution → colour (PRD §7.1.2 / Table 6)
REGION_COLORS = {
    "IN": "#8BC34A",   # Indian Ocean – olive green
    "AO": "#00BCD4",   # Arabian / Atlantic Ocean – cyan
    "BO": "#FF5722",   # Bay of Bengal – deep orange
    "CS": "#FFC107",   # Coral Sea – amber
    "HZ": "#9E9E9E",   # Marginal seas – grey
    "IF": "#4CAF50",   # Intermediate / Far seas – green
    "JA": "#2196F3",   # JMA – blue
    "KO": "#E91E63",   # KIOST – pink
    "KM": "#9C27B0",   # KMA – purple
    "ME": "#795548",   # MEDS – brown
    "NM": "#607D8B",   # NMDIS – blue-grey
}



# KPI tile colours (PRD §7.3.2 / Table 7)
KPI_COLORS = {
    "DOXY": "#00BCD4",
    "Chla": "#8BC34A",
    "Nitrate": "#FF8F00",
    "pH": "#4CAF50",
}

# Age-group colours (PRD §7.5.2)
AGE_COLORS = {
    "00-02": "#673AB7",
    "03-05": "#2196F3",
    "06-08": "#FF9800",
    "09-11": "#F44336",
    "12+": "#795548",
}

# Profiler Type (WMO R08 Table) → Human-readable instrument name
PROFILER_TYPE_NAMES = {
    831: "P-ALACE", 834: "Provor-II", 835: "Provor-III", 836: "Provor-MT",
    837: "Arvor-C", 838: "Arvor-D", 839: "Provor-IV", 840: "Provor (no CT)",
    841: "Provor-SBE", 842: "Arvor-CM", 843: "Provor-V", 844: "Arvor",
    845: "Webb-PALACE", 846: "APEX", 847: "APEX-EM", 848: "APEX-EM-SBE",
    849: "APEX-Deep", 850: "SOLO (no CT)", 851: "SOLO-SBE", 852: "SOLO-FSI",
    853: "SOLO2", 854: "S2A", 855: "Ninja (no CT)", 856: "Ninja-D",
    857: "Ninja-BGC", 858: "Ninja-Deep", 859: "Ninja-SBE", 860: "Ninja",
    861: "ALTO", 862: "Navis-EBR", 863: "Navis-A", 864: "Navis-Deep",
    865: "Nova", 869: "Deep ARVOR", 870: "APEX-APF11", 871: "APEX-Deep-APF11",
    872: "APEX-BGC", 873: "Arvor-Deep", 874: "APEX-Deep-SBE",
    875: "Provor-BGC", 876: "Deep SOLO", 877: "Deep SOLO-MRV",
    878: "Deep NINJA", 879: "HM2000", 880: "HM4000", 881: "Deep Arvor-O",
    882: "Deep S2A", 883: "Provor-BGC-II", 884: "Arvor-I", 885: "TWR",
    886: "SOLO-BGC", 887: "Arvor-RBR", 888: "ALTO-RBR", 889: "Arvor-Deep-RBR",
    890: "APEX-RBR", 891: "Navis-RBR",
}

# Colors for top profiler type families
PROFILER_COLORS = {
    "APEX": "#4FC3F7", "Arvor": "#FF7043", "SOLO-SBE": "#26A69A",
    "SOLO2": "#BA68C8", "Deep ARVOR": "#FFB74D", "Provor-SBE": "#00BCD4",
    "S2A": "#F06292", "Navis-A": "#9CCC65", "Provor-MT": "#9575CD",
    "SOLO-FSI": "#FFD54F", "Nova": "#90A4AE", "Ninja": "#EF5350",
    "Arvor-D": "#42A5F5", "Provor-II": "#66BB6A", "Navis-EBR": "#AB47BC",
    "Arvor-CM": "#FFA726", "APEX-Deep-SBE": "#78909C", "APEX-APF11": "#29B6F6",
    "Deep NINJA": "#EC407A", "Deep SOLO-MRV": "#5C6BC0",
    "Other": "#607D8B",
}

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Indian ARGO CTD_BGC Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Indian ARGO CTD/BGC Float Dashboard · INCOIS · Data: IFREMER GDAC"
    },
)


# ==================== CUSTOM CSS ====================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', 'Inter', sans-serif;
    }

    /* ── Seamless App Background ── */
    .stApp {
        background: radial-gradient(circle at 10% 20%, rgba(5, 12, 33, 1) 0%, rgba(1, 4, 15, 1) 90%);
        background-attachment: fixed;
    }

    /* ── Glass Containers ── */
    [data-testid="stMetric"], .kpi-tile, [data-testid="stExpander"], .dac-table, .treemap-info {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        padding: 20px;
    }

    /* ── Sidebar Glass UI ── */
    section[data-testid="stSidebar"] {
        background: rgba(6, 11, 25, 0.82) !important;
        backdrop-filter: blur(15px);
        border-right: 1px solid rgba(0, 188, 212, 0.15);
    }
    section[data-testid="stSidebar"] * { color: #d1d9e6 !important; }
    section[data-testid="stSidebar"] h2 { color: #00BCD4 !important; font-weight: 700; letter-spacing: 0.5px; }

    /* ── KPI Tiles - Revamped ── */
    .kpi-tile {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 140px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
    }
    .kpi-tile:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(0, 188, 212, 0.4) !important;
        box-shadow: 0 12px 40px rgba(0, 188, 212, 0.15);
    }
    .kpi-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: rgba(255, 255, 255, 0.7);
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        white-space: nowrap; /* Prevent wrapping */
        line-height: 1.1;
    }

    /* ── Header ── */
    .header-bar {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(0, 188, 212, 0.2);
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
        position: relative;
    }
    .header-bar h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #ffffff, #00BCD4, #4CAF50);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Modern Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: rgba(0, 188, 212, 0.4); border-radius: 10px; }

    /* ── Fix Streamlit gaps ── */
    .stPlotlyChart {
         background: rgba(255, 255, 255, 0.02) !important;
         border-radius: 18px;
         padding: 10px;
         border: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* ── Legend Chips ── */
    .legend-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 10px;
        margin: 3px 4px;
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        font-size: 0.75rem;
        color: #c8d6e5;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    /* ── Footer ── */
    .footer-bar {
        text-align: center;
        padding: 20px 30px;
        margin-top: 30px;
        font-size: 0.8rem;
        color: rgba(255,255,255,0.45);
        background: rgba(255,255,255,0.02);
        border-top: 1px solid rgba(0,188,212,0.12);
        border-radius: 16px;
        letter-spacing: 0.3px;
    }

    /* ── Treemap Summary Stats ── */
    .stat {
        font-size: 1.6rem;
        font-weight: 800;
        color: #ffffff;
        text-align: center;
    }
    .stat-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: rgba(255,255,255,0.5);
        text-align: center;
        margin-top: 2px;
    }

    /* ── DAC Table ── */
    .dac-table table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        font-size: 0.85rem;
        color: #c8d6e5;
    }
    .dac-table th {
        padding: 12px 14px;
        text-align: center;
        font-weight: 700;
        color: #00BCD4;
        border-bottom: 2px solid rgba(0,188,212,0.2);
        font-size: 0.8rem;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }
    .dac-table td {
        padding: 10px 14px;
        text-align: center;
        border-bottom: 1px solid rgba(255,255,255,0.04);
    }
    .dac-table tbody tr:nth-child(odd) {
        background: rgba(255,255,255,0.02);
    }
    .dac-table tbody tr:hover {
        background: rgba(0,188,212,0.06);
    }

    /* ── Refresh timestamp ── */
    .refresh-ts {
        font-size: 0.75rem;
        color: rgba(255,255,255,0.4);
        margin-top: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ==================== HELPER: dark plotly layout ====================
def _dark_layout(**overrides):
    """Return a dark-themed plotly layout dict."""
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#c8d6e5", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
    )
    base.update(overrides)
    return base


# ==================== DATA LOADING ====================
# Global constants for classification
DEEP_PROFILER_TYPES = {862, 864, 876, 882, 869, 863, 873, 874, 886, 877, 875, 884, 872, 879, 865, 860, 878, 861, 871, 870, 881, 853}

@st.cache_data(show_spinner="Loading core-profile index …")
def load_profile_data():
    """Load ar_index_global_prof.txt with Parquet cache (24-h TTL)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "profiles.parquet"

    if cache_path.exists():
        age_h = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            df = pd.read_parquet(cache_path)
            # Ensure is_deep exists (handles stale caches from before this column was added)
            if "is_deep" not in df.columns:
                df["is_deep"] = df["profiler_type"].isin(DEEP_PROFILER_TYPES) if "profiler_type" in df.columns else False
            if "dac" not in df.columns:
                df["dac"] = df["file"].str.extract(r"^([^/]+)/")
            return df

    df = pd.read_csv(PROF_FILE, comment="#")
    # Strip whitespace from column names (GDAC files sometimes have spaces)
    df.columns = df.columns.str.strip()

    # --- Land-mask filtering removed: caused discrepancies ---
    df = df.dropna(subset=["latitude", "longitude"])
    
    # Ensure coordinates are within valid ranges [-90, 90] and [-180, 180]
    df = df[
        (df["latitude"] >= -90) & (df["latitude"] <= 90) &
        (df["longitude"] >= -180) & (df["longitude"] <= 180)
    ]

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")
    if "date_update" in df.columns:
        df["date_update"] = pd.to_datetime(
            df["date_update"], format="%Y%m%d%H%M%S", errors="coerce"
        )
    df["wmo_id"] = df["file"].str.extract(r"/(\d+)/")
    df["dac"] = df["file"].str.extract(r"^([^/]+)/")
    df["year"] = df["date"].dt.year
    df["is_deep"] = df["profiler_type"].isin(DEEP_PROFILER_TYPES)

    df.to_parquet(cache_path, index=False)
    return df


@st.cache_data(show_spinner="Loading BGC-profile index …")
def load_bio_data():
    """Load argo_bio-profile_index.txt with Parquet cache (24-h TTL)."""
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "bgc_profiles.parquet"

    if cache_path.exists():
        age_h = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            return pd.read_parquet(cache_path)

    df = pd.read_csv(BIO_FILE, comment="#")
    df.columns = df.columns.str.strip()

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d%H%M%S", errors="coerce")
    df["wmo_id"] = df["file"].str.extract(r"/(\d+)/")
    df["year"] = df["date"].dt.year

    params_upper = df["parameters"].fillna("").str.upper()
    df["has_doxy"] = params_upper.str.contains("DOXY")
    df["has_chla"] = params_upper.str.contains("CHLA")
    df["has_nitrate"] = params_upper.str.contains("NITRATE")
    df["has_ph"] = params_upper.str.contains("PH_IN_SITU")

    df.to_parquet(cache_path, index=False)
    return df


@st.cache_data(show_spinner="Loading float metadata index …")
def load_meta_data():
    """Load ar_index_global_meta.txt with Parquet cache (24-h TTL).

    Provides one row per float (WMO) with profiler_type, institution,
    dac, and a human-readable profiler_name from WMO R08.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / "meta.parquet"

    if cache_path.exists():
        age_h = (datetime.now().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_h < 24:
            return pd.read_parquet(cache_path)

    df = pd.read_csv(META_FILE, comment="#")
    df.columns = df.columns.str.strip()
    df["wmo_id"] = df["file"].str.extract(r"/(\d+)/")
    df["dac"] = df["file"].str.extract(r"^([^/]+)/")
    df["date_update"] = pd.to_datetime(
        df["date_update"], format="%Y%m%d%H%M%S", errors="coerce"
    )
    # Map numeric profiler_type code to human-readable name
    df["profiler_name"] = (
        df["profiler_type"]
        .map(PROFILER_TYPE_NAMES)
        .fillna("Unknown")
    )
    df.to_parquet(cache_path, index=False)
    return df


@st.cache_data
def _bgc_wmo_set(_df_bio):
    """Set of WMO IDs that have at least one BGC profile."""
    return set(_df_bio["wmo_id"].dropna().unique())


# ==================== LOAD DATA ====================
with st.spinner("🌊 Initialising ARGO Dashboard …"):
    df_prof = load_profile_data()
    df_bio = load_bio_data()
    df_meta = load_meta_data()
    bgc_wmos = _bgc_wmo_set(df_bio)

# Derived column: is this float a BGC float?
df_prof["is_bgc"] = df_prof["wmo_id"].isin(bgc_wmos)

# Enrich profiles with profiler_name from meta (authoritative per-float source)
_meta_pname = df_meta.set_index("wmo_id")["profiler_name"]
df_prof["profiler_name"] = df_prof["wmo_id"].map(_meta_pname).fillna("Unknown")

@st.dialog("Float Information", width="large")
def show_float_details(wmo):
    meta_path = BASE_DIR / f"more_components/{wmo}_meta.nc"
    prof_path = BASE_DIR / f"more_components/{wmo}_prof.nc"
    
    # Auto-download from IFREMER GDAC if files do not exist
    if not meta_path.exists() or not prof_path.exists():
        import urllib.request
        
        dac_row = df_meta[df_meta["wmo_id"] == wmo]
        if len(dac_row) > 0:
            dac = dac_row.iloc[0]["dac"]
        else:
            dac = "incois"  # fallback
            
        meta_url = f"ftp://ftp.ifremer.fr/ifremer/argo/dac/{dac}/{wmo}/{wmo}_meta.nc"
        prof_url = f"ftp://ftp.ifremer.fr/ifremer/argo/dac/{dac}/{wmo}/{wmo}_prof.nc"
        
        target_dir = BASE_DIR / "more_components"
        target_dir.mkdir(exist_ok=True)
        
        with st.spinner(f"Downloading GDAC NetCDF files for {wmo} ({dac})..."):
            try:
                if not meta_path.exists():
                    urllib.request.urlretrieve(meta_url, meta_path)
                if not prof_path.exists():
                    urllib.request.urlretrieve(prof_url, prof_path)
            except Exception as e:
                st.error(f"Failed to download files from {meta_url}. Error: {e}")
                return
        
    try:
        ds_meta = xr.open_dataset(meta_path)
        ds_prof = xr.open_dataset(prof_path)
        
        def d(val):
            if hasattr(val, "item") and callable(val.item):
                try:
                    val = val.item()
                except:
                    pass
            if isinstance(val, bytes):
                return val.decode('utf-8', errors='ignore').strip()
            elif isinstance(val, np.ndarray) and val.dtype.kind == 'S':
                return ", ".join([v.decode('utf-8', errors='ignore').strip() for v in val.flat if v.decode('utf-8', errors='ignore').strip()])
            elif isinstance(val, (list, np.ndarray)):
                return ", ".join([d(v) for v in val])
            return str(val).strip()

        maker = d(ds_meta.PLATFORM_MAKER.values) if 'PLATFORM_MAKER' in ds_meta else 'N/A'
        serial = d(ds_meta.FLOAT_SERIAL_NO.values) if 'FLOAT_SERIAL_NO' in ds_meta else 'N/A'
        ptype = d(ds_meta.PLATFORM_TYPE.values) if 'PLATFORM_TYPE' in ds_meta else 'N/A'
        trans = d(ds_meta.TRANS_SYSTEM.values) if 'TRANS_SYSTEM' in ds_meta else 'N/A'
        owner = d(ds_meta.FLOAT_OWNER.values) if 'FLOAT_OWNER' in ds_meta else 'N/A'
        
        dc_map = {
            "AO": "AOML", "BO": "BODC", "CO": "Coriolis", "CS": "CSIRO", 
            "IN": "INCOIS", "JA": "JMA", "KM": "KMA", "ME": "MEDS", 
            "RU": "RU", "HZ": "CSIO", "NM": "NMDIS"
        }
        if 'DATA_CENTRE' in ds_meta:
            dc_code = d(ds_meta.DATA_CENTRE.values).upper()
            dc = dc_map.get(dc_code, dc_code)
        elif 'OPERATING_INSTITUTION' in ds_meta:
            dc = d(ds_meta.OPERATING_INSTITUTION.values)
        else:
            dc = 'N/A'
        sensors = d(ds_meta.SENSOR.values) if 'SENSOR' in ds_meta else 'N/A'
        ptt = d(ds_meta.PTT.values) if 'PTT' in ds_meta else 'N/A'
        
        launch_date = d(ds_meta.LAUNCH_DATE.values) if 'LAUNCH_DATE' in ds_meta else 'N/A'
        if launch_date != 'N/A' and len(launch_date) == 14:
            try:
                dt = datetime.strptime(launch_date, '%Y%m%d%H%M%S')
                launch_date_fmt = dt.strftime('%d/%m/%Y %H:%M:%S')
                age = f"{(datetime.now() - dt).days / 365.25:.2f} years ago"
            except:
                launch_date_fmt = launch_date
                age = "N/A"
        else:
            launch_date_fmt = launch_date
            age = "N/A"
            
        launch_lat = float(ds_meta.LAUNCH_LATITUDE.values) if 'LAUNCH_LATITUDE' in ds_meta else 'N/A'
        launch_lon = float(ds_meta.LAUNCH_LONGITUDE.values) if 'LAUNCH_LONGITUDE' in ds_meta else 'N/A'
        
        project = d(ds_meta.PROJECT_NAME.values) if 'PROJECT_NAME' in ds_meta else 'N/A'
        pi = d(ds_meta.PI_NAME.values) if 'PI_NAME' in ds_meta else 'N/A'
        
        if 'CYCLE_NUMBER' in ds_prof and len(ds_prof.CYCLE_NUMBER) > 0:
            cycle = int(np.nanmax(ds_prof.CYCLE_NUMBER.values))
            juld = ds_prof.JULD.values
            last_date_np = juld[~np.isnat(juld)]
            if len(last_date_np) > 0:
                dt_last = pd.to_datetime(last_date_np[-1])
                last_date = dt_last.strftime('%d/%m/%Y %H:%M:%S')
                if launch_date != 'N/A' and len(launch_date) == 14:
                    try:
                        dt_launch = datetime.strptime(launch_date, '%Y%m%d%H%M%S')
                        cycle_age_years = (dt_last - dt_launch).days / 365.25
                        cycle_age = f"{cycle_age_years:.2f} years old"
                    except:
                        cycle_age = "N/A"
                else:
                    cycle_age = "N/A"
            else:
                last_date = "N/A"
                cycle_age = "N/A"
                
            try:
                pres_data = ds_prof.PRES.values
                valid_cycles = np.where(~np.isnan(pres_data).all(axis=1))[0]
                if len(valid_cycles) > 0:
                    last_valid_idx = valid_cycles[-1]
                    last_pres = pres_data[last_valid_idx]
                    last_temp = ds_prof.TEMP.values[last_valid_idx] if 'TEMP' in ds_prof else np.full_like(last_pres, np.nan)
                    last_psal = ds_prof.PSAL.values[last_valid_idx] if 'PSAL' in ds_prof else np.full_like(last_pres, np.nan)
                    
                    valid_idx = ~np.isnan(last_pres)
                    pres_v = last_pres[valid_idx]
                    temp_v = last_temp[valid_idx]
                    psal_v = last_psal[valid_idx]
                    
                    if len(pres_v) > 0:
                        surface_idx = np.argmin(pres_v)
                        bottom_idx = np.argmax(pres_v)
                        surf_data = f"{pres_v[surface_idx]:.2f} dbar {temp_v[surface_idx]:.3f}°C {psal_v[surface_idx]:.3f} PSU"
                        bott_data = f"{pres_v[bottom_idx]:.2f} dbar {temp_v[bottom_idx]:.3f}°C {psal_v[bottom_idx]:.3f} PSU"
                    else:
                        surf_data = "N/A"
                        bott_data = "N/A"
                else:
                    surf_data = "N/A"
                    bott_data = "N/A"
            except:
                surf_data = "N/A"
                bott_data = "N/A"
        else:
            cycle = "N/A"
            last_date = "N/A"
            cycle_age = "N/A"
            surf_data = "N/A"
            bott_data = "N/A"
            
        status = "Inactive" 
        if last_date != "N/A":
             try:
                 dt_last = datetime.strptime(last_date, '%d/%m/%Y %H:%M:%S')
                 if (datetime.now() - dt_last).days <= 90:
                     status = "Active"
             except:
                 pass
                 
        status_color = "#EF5350" if status == "Inactive" else "#66BB6A"

        st.markdown("### Main Information")
        st.markdown(f"""
        <div style='display: flex; gap: 20px; flex-wrap: wrap; margin-top: 10px;'>
            <!-- About Float -->
            <div style='flex: 1; min-width: 250px; background: rgba(255,255,255,0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
                <h4 style='color: #4FC3F7; margin-top: 0; font-family: Outfit, sans-serif; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;'>About Float</h4>
                <table style='width: 100%; border: none; font-size: 0.85em; line-height: 1.5;'>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>WMO<br><span style='color: #4FC3F7; font-size: 1.1em;'>{wmo}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Platform maker<br><span style='color: white; font-size: 1.1em;'>{maker}</span></td>
                    </tr>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Float serial number<br><span style='color: white; font-size: 1.1em;'>{serial}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Platform type<br><span style='color: white; font-size: 1.1em;'>{ptype}</span></td>
                    </tr>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Transmission system<br><span style='color: white; font-size: 1.1em;'>{trans}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>PTT<br><span style='color: white; font-size: 1.1em;'>{ptt}</span></td>
                    </tr>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Owner<br><span style='color: white; font-size: 1.1em;'>{owner}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Data Centre<br><span style='color: #4FC3F7; font-size: 1.1em;'>{dc}</span></td>
                    </tr>
                    <tr>
                        <td colspan='2' style='color: rgba(255,255,255,0.5);'>Sensors<br><span style='color: white; font-size: 0.95em;'>{sensors}</span></td>
                    </tr>
                </table>
            </div>
            <!-- Deployment -->
            <div style='flex: 1; min-width: 250px; background: rgba(255,255,255,0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
                <h4 style='color: #4FC3F7; margin-top: 0; font-family: Outfit, sans-serif; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;'>Deployment</h4>
                <table style='width: 100%; border: none; font-size: 0.85em; line-height: 1.5;'>
                    <tr>
                        <td colspan='2' style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Launched &nbsp; <span style='color: rgba(255,255,255,0.4);'>{age}</span><br><span style='color: white; font-size: 1.1em;'>{launch_date_fmt}</span></td>
                    </tr>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Deployment Latitude<br><span style='color: white; font-size: 1.1em;'>{launch_lat}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Deployment Longitude<br><span style='color: white; font-size: 1.1em;'>{launch_lon}</span></td>
                    </tr>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Ship<br><span style='color: white; font-size: 1.1em;'>frv sagar sampada</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Cruise<br><span style='color: white; font-size: 1.1em;'></span></td>
                    </tr>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Project<br><span style='color: white; font-size: 1.1em;'>{project}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Principal Investigator<br><span style='color: white; font-size: 1.1em;'>{pi}</span></td>
                    </tr>
                </table>
            </div>
            <!-- Cycle activity -->
            <div style='flex: 1; min-width: 250px; background: rgba(255,255,255,0.02); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
                <h4 style='color: #4FC3F7; margin-top: 0; font-family: Outfit, sans-serif; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px;'>Cycle activity</h4>
                <table style='width: 100%; border: none; font-size: 0.85em; line-height: 1.5;'>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Status<br><span style='color: {status_color}; font-size: 1.1em; font-weight: bold;'>{status}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Age<br><span style='color: white; font-size: 1.1em;'>{cycle_age}</span></td>
                    </tr>
                    <tr>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Last profile date<br><span style='color: white; font-size: 1.1em;'>{last_date}</span></td>
                        <td style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Cycle<br><span style='color: white; font-size: 1.1em;'>{cycle}</span></td>
                    </tr>
                    <tr>
                        <td colspan='2' style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Last Surface Data<br><span style='color: white; font-size: 1.05em;'>{surf_data}</span></td>
                    </tr>
                    <tr>
                        <td colspan='2' style='color: rgba(255,255,255,0.5); padding-bottom: 10px;'>Last Bottom Data<br><span style='color: white; font-size: 1.05em;'>{bott_data}</span></td>
                    </tr>
                </table>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("#### Argo parameters section charts and overlaid profiles")
        try:
            import plot_utils
            cycles, dates, pres, temp, psal, rho = plot_utils.get_valid_data(ds_prof)
            
            if len(pres) > 0:
                c1, c2, c3 = st.columns(3)
                with c1:
                    fig = plot_utils.create_ts_diagram(cycles, temp, psal, wmo)
                    st.pyplot(fig, clear_figure=True)
                with c2:
                    fig = plot_utils.create_section_chart(dates, pres, temp, "Temperature (°C)", "Section chart TEMP", wmo)
                    st.pyplot(fig, clear_figure=True)
                with c3:
                    fig = plot_utils.create_section_chart(dates, pres, psal, "Salinity (PSU)", "Section chart PSAL", wmo)
                    st.pyplot(fig, clear_figure=True)
                    
                c4, c5, c6 = st.columns(3)
                with c4:
                    fig = plot_utils.create_section_chart(dates, pres, rho, "Potential Density (kg/m³)", "Section chart RHO", wmo)
                    st.pyplot(fig, clear_figure=True)
                with c5:
                    fig = plot_utils.create_overlaid_profiles(temp, pres, cycles, "Temperature (°C)", "Overlaid profiles TEMP", wmo)
                    st.pyplot(fig, clear_figure=True)
                with c6:
                    fig = plot_utils.create_overlaid_profiles(psal, pres, cycles, "Salinity (PSU)", "Overlaid profiles PSAL", wmo)
                    st.pyplot(fig, clear_figure=True)
                    
                c7, c8, c9 = st.columns(3)
                with c7:
                    fig = plot_utils.create_overlaid_profiles(rho, pres, cycles, "Potential Density (kg/m³)", "Overlaid profiles RHO", wmo)
                    st.pyplot(fig, clear_figure=True)
            else:
                st.info("No valid profile data available for technical plots.")
        except Exception as e:
            st.error(f"Error rendering technical plots: {e}")
                

    except Exception as e:
        st.error(f"Error loading float details: {e}")


# ==================== HEADER ====================
_cache_path = CACHE_DIR / "profiles.parquet"
_last_refresh = (
    datetime.fromtimestamp(_cache_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    if _cache_path.exists() else "N/A"
)
st.markdown(
    f"""
<div class="header-bar">
    <h1>🌊 Indian ARGO CTD / BGC Dashboard</h1>
    <p>Real-time visibility into the Indian Ocean ARGO float network · Data Source: IFREMER GDAC</p>
    <div class="refresh-ts">Last data refresh: {_last_refresh} UTC</div>
</div>
""",
    unsafe_allow_html=True,
)

# ==================== SIDEBAR FILTERS (PRD §6) ====================
# Read URL query params for shareable filter state
qp = st.query_params

with st.sidebar:
    st.markdown("## 🔍 Filters")

    # ── Refresh ──
    if st.button("🔄 Refresh Data", use_container_width=True, type="primary"):
        for f in CACHE_DIR.glob("*.parquet"):
            f.unlink()
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")

    # ── WMO search ──
    search_wmo = st.text_input(
        "🔎 Search WMO Float ID",
        value=qp.get("wmo", ""),
        placeholder="e.g. 2902115, 2902116",
        help="Comma-separated WMO numbers",
    )

    # ── QC Mode ──
    _qc_options = ["All", "Delayed", "Real time"]
    _qc_default = _qc_options.index(qp.get("qc", "All")) if qp.get("qc", "All") in _qc_options else 0
    qc_mode = st.selectbox(
        "QC Mode",
        _qc_options,
        index=_qc_default,
        help="All = all data; Delayed = quality-checked; Real time = latest",
    )

    # ── Community ──
    st.markdown("### Community")
    comm_all = st.checkbox("ALL", value=qp.get("comm_all", "1") == "1", key="comm_all")
    comm_null = st.checkbox("NULL", value=qp.get("comm_null", "0") == "1", key="comm_null")
    comm_argos = st.checkbox("ARGOS", value=qp.get("comm_argos", "0") == "1", key="comm_argos")
    comm_beidou = st.checkbox("BEIDOU", value=qp.get("comm_beidou", "0") == "1", key="comm_beidou")
    comm_iridium = st.checkbox("IRIDIUM", value=qp.get("comm_iridium", "0") == "1", key="comm_iridium")

    # ── Network ──
    st.markdown("### Network")
    net_all = st.checkbox("All (Inclusive)", value=qp.get("net_all", "1") == "1", key="net_all")
    net_bgc = st.checkbox("BGC (Bio-Argo)", value=qp.get("net_bgc", "0") == "1", key="net_bgc")
    net_ctd = st.checkbox("CTD (Core Argo)", value=qp.get("net_ctd", "0") == "1", key="net_ctd")
    net_dep = st.checkbox("DEP (Deep Argo)", value=qp.get("net_dep", "0") == "1", key="net_dep")

    # ── Float Model / Profiler Type ──
    st.markdown("### Float Model")
    _available_models = sorted(df_meta["profiler_name"].dropna().unique().tolist())
    selected_profiler_types = st.multiselect(
        "Select Float Model(s)",
        options=_available_models,
        default=[],
        placeholder="All models (no filter)",
        help="Filter by instrument model from metadata registry (WMO R08)",
    )

    # ── Map Options ──
    st.markdown("### Map Options")
    show_live_only = st.toggle("Live Floats Only (90d)", value=qp.get("live_only", "0") == "1", help="Hide historical dead floats to reduce map clutter")

    # ── Date range ──
    st.markdown("### Date Range")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        _min_d = datetime(1960, 1, 1)
        _max_d = datetime.now()
        
        # Determine default start date (earliest profile or 1960)
        _default_start = _min_d
        if "df_prof" in locals() and len(df_prof) > 0 and pd.notna(df_prof["date"].min()):
            _default_start = df_prof["date"].min().to_pydatetime()

        _sd = datetime.strptime(qp.get("sd", ""), "%Y-%m-%d") if "sd" in qp and qp.get("sd", "") else _default_start
        start_date = st.date_input("Start", value=_sd, min_value=_min_d, max_value=_max_d)
    with d_col2:
        _ed = datetime.strptime(qp.get("ed", ""), "%Y-%m-%d") if "ed" in qp and qp.get("ed", "") else _max_d
        end_date = st.date_input("End", value=_ed, min_value=_min_d, max_value=_max_d)

    # ── Longitude ──
    st.markdown("### Longitude")
    _lon_lo = float(qp.get("lon_lo", "20.0"))
    _lon_hi = float(qp.get("lon_hi", "145.0"))
    lon_range = st.slider(
        "Longitude range",
        min_value=-180.0,
        max_value=180.0,
        value=(_lon_lo, _lon_hi),
        step=0.5,
        label_visibility="collapsed",
    )

    # ── Latitude ──
    st.markdown("### Latitude")
    _lat_lo = float(qp.get("lat_lo", "-70.1"))
    _lat_hi = float(qp.get("lat_hi", "30.0"))
    lat_range = st.slider(
        "Latitude range",
        min_value=-90.0,
        max_value=90.0,
        value=(_lat_lo, _lat_hi),
        step=0.5,
        label_visibility="collapsed",
    )


# ── Sync current filter state to URL query params ──
st.query_params.update({
    "wmo": search_wmo,
    "qc": qc_mode,
    "comm_all": "1" if comm_all else "0",
    "comm_null": "1" if comm_null else "0",
    "comm_argos": "1" if comm_argos else "0",
    "comm_beidou": "1" if comm_beidou else "0",
    "comm_iridium": "1" if comm_iridium else "0",
    "net_all": "1" if net_all else "0",
    "net_bgc": "1" if net_bgc else "0",
    "net_ctd": "1" if net_ctd else "0",
    "net_dep": "1" if net_dep else "0",
    "sd": str(start_date),
    "ed": str(end_date),
    "lon_lo": str(lon_range[0]),
    "lon_hi": str(lon_range[1]),
    "lat_lo": str(lat_range[0]),
    "lat_hi": str(lat_range[1]),
    "live_only": "1" if show_live_only else "0",
})

# ==================== FILTER LOGIC (PRD §6.1) ====================
def apply_filters(df, *, is_bio=False):
    """Apply every sidebar filter to *df* and return the filtered copy."""
    out = df.copy()

    # Date
    if "date" in out.columns:
        out = out[
            (out["date"] >= pd.Timestamp(start_date))
            & (out["date"] <= pd.Timestamp(end_date))
        ]

    # Lon / Lat
    if "longitude" in out.columns:
        out = out[
            (out["longitude"] >= lon_range[0]) & (out["longitude"] <= lon_range[1])
        ]
    if "latitude" in out.columns:
        out = out[
            (out["latitude"] >= lat_range[0]) & (out["latitude"] <= lat_range[1])
        ]

    # Network Logic — only apply to core profiles (bio df lacks is_bgc/is_deep)
    if not net_all and not is_bio and "is_bgc" in out.columns and "is_deep" in out.columns:
        masks = []
        if net_bgc:
            masks.append(out["is_bgc"])
        if net_ctd:
            # Core = NOT BGC and NOT Deep
            masks.append(~out["is_bgc"] & ~out["is_deep"])
        if net_dep:
            masks.append(out["is_deep"])
        
        if masks:
            combined_mask = masks[0]
            for m in masks[1:]:
                combined_mask |= m
            out = out[combined_mask]
        elif not (net_bgc or net_ctd or net_dep):
             # If nothing selected and All is off, show nothing
             out = out.iloc[0:0]

    # WMO search
    if search_wmo.strip():
        wmo_list = [w.strip() for w in search_wmo.split(",") if w.strip()]
        out = out[out["wmo_id"].isin(wmo_list)]

    # Community Logic
    if not comm_all and "positioning_system" in out.columns:
        masks = []
        if comm_null:
            masks.append(out["positioning_system"].isna() | (out["positioning_system"] == ""))
        if comm_argos:
            masks.append(out["positioning_system"].fillna("").str.upper().str.contains("ARGOS"))
        if comm_beidou:
            masks.append(out["positioning_system"].fillna("").str.upper().str.contains("BEIDOU"))
        if comm_iridium:
            masks.append(out["positioning_system"].fillna("").str.upper().str.contains("IRIDIUM"))
        
        if masks:
            combined_mask = masks[0]
            for m in masks[1:]:
                combined_mask |= m
            out = out[combined_mask]
        elif not (comm_null or comm_argos or comm_beidou or comm_iridium):
            out = out.iloc[0:0]

    # QC mode (bio only)
    if is_bio and "parameter_data_mode" in out.columns:
        if qc_mode == "Delayed":
            out = out[out["parameter_data_mode"].fillna("").str.contains("D")]
        elif qc_mode == "Real time":
            out = out[~out["parameter_data_mode"].fillna("").str.contains("D")]

    # Profiler Type / Float Model filter (from meta enrichment)
    if selected_profiler_types and "profiler_name" in out.columns:
        out = out[out["profiler_name"].isin(selected_profiler_types)]

    return out


filt_prof = apply_filters(df_prof)
filt_bio = apply_filters(df_bio, is_bio=True)

# ================================================================
#  FLEET OVERVIEW KPI ROW (from meta registry)
# ================================================================
_total_registered = len(df_meta)
_total_profiled = df_meta["wmo_id"].isin(df_prof["wmo_id"].unique()).sum()
_never_profiled = _total_registered - _total_profiled
_unique_models = df_meta["profiler_name"].nunique()

st.markdown("### 🛰️ Fleet Overview (from Metadata Registry)")
fo1, fo2, fo3, fo4 = st.columns(4)
for col, label, value, color, icon in [
    (fo1, "Registered Floats", _total_registered, "#00BCD4", "📋"),
    (fo2, "Profiled Floats", _total_profiled, "#8BC34A", "✅"),
    (fo3, "Never Profiled", _never_profiled, "#FF5722", "⚠️"),
    (fo4, "Float Models", _unique_models, "#9C27B0", "🔧"),
]:
    with col:
        st.markdown(
            f"""
            <div class="kpi-tile" style="border-bottom: 4px solid {color} !important;">
                <div class="kpi-label">{icon} {label}</div>
                <div class="kpi-value">{value:,}</div>
                <div style="font-size: 10px; color: rgba(255,255,255,0.4); margin-top: 4px;">META REGISTRY</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ================================================================
#  ROW 1 — MAP (left ~55 %) + BAR CHART & KPIs (right ~45 %)
# ================================================================
col_left, col_right = st.columns([55, 45], gap="medium")

with col_left:
    # ── Component 1: Geospatial Float Position Map (PRD §7.1) ──
    st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
    st.markdown("### 📍 Geographic Float Positions")

    if len(filt_prof) > 0:
        # --- Check map selection from session state ---
        selected_wmo_from_map = None
        if "main_map" in st.session_state:
            sel = st.session_state.main_map
            if sel and "selection" in sel and "points" in sel["selection"] and len(sel["selection"]["points"]) > 0:
                pt = sel["selection"]["points"][0]
                if "customdata" in pt and len(pt["customdata"]) > 0:
                    selected_wmo_from_map = str(pt["customdata"][0])
        
        is_sidebar_search = bool(search_wmo.strip())
        is_wmo_searched = is_sidebar_search or bool(selected_wmo_from_map)

        # Apply Live-Only filter if toggled and not searching specific WMOs
        map_source = filt_prof.copy()
        
        # If user clicked a float on the map, filter source to just that float
        if selected_wmo_from_map:
            map_source = map_source[map_source["wmo_id"] == selected_wmo_from_map]

        if show_live_only and not is_wmo_searched:
            latest_d = map_source["date"].max()
            ninety_days_ago = latest_d - timedelta(days=90)
            # Find WMOs that have a profile in the last 90 days
            live_wmos = map_source[map_source["date"] >= ninety_days_ago]["wmo_id"].unique()
            map_source = map_source[map_source["wmo_id"].isin(live_wmos)]

        if is_wmo_searched:
            # Check if this float is newly selected from map to show dialog
            if selected_wmo_from_map:
                if st.session_state.get("last_viewed_wmo") != selected_wmo_from_map:
                    st.session_state["last_viewed_wmo"] = selected_wmo_from_map
                    show_float_details(selected_wmo_from_map)
                
            # Show full trajectory for specific floats
            map_df = (
                map_source.dropna(subset=["latitude", "longitude"])
                .sort_values(["wmo_id", "date"])
                .copy()
            )
            # Add a profile sequence number for each float
            map_df["profile_seq"] = map_df.groupby("wmo_id").cumcount() + 1
            
            fig_map = go.Figure()
            for wmo, group in map_df.groupby("wmo_id"):
                inst = group["institution"].iloc[0]
                color = REGION_COLORS.get(inst, "#ff0000")
                fig_map.add_trace(go.Scattermapbox(
                    lat=group["latitude"].tolist(),
                    lon=group["longitude"].tolist(),
                    mode="lines+markers+text",
                    text=group["profile_seq"].astype(str).tolist(),
                    customdata=[[wmo]] * len(group),
                    textposition="top right",
                    textfont=dict(size=11, color="white"),
                    marker=dict(size=7, color=color, opacity=0.9),
                    line=dict(width=2, color=color),
                    name=str(wmo),
                    hoverinfo="text",
                    hovertext=group.apply(lambda r: f"WMO: {wmo}<br>Date: {r['date']}<br>Lat: {r['latitude']:.2f}, Lon: {r['longitude']:.2f}<br>Profile: {r['profile_seq']}", axis=1).tolist()
                ))
            
            center_lat = float(map_df["latitude"].mean()) if len(map_df) > 0 else 0.0
            center_lon = float(map_df["longitude"].mean()) if len(map_df) > 0 else 0.0
            
            fig_map.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                mapbox=dict(
                    style="carto-darkmatter",
                    center=dict(lat=center_lat, lon=center_lon), 
                    zoom=4
                ),
                legend=dict(
                    title="WMO ID",
                    bgcolor="rgba(10,14,39,0.85)",
                    bordercolor="rgba(0,188,212,0.18)",
                    borderwidth=1,
                    font=dict(size=11, color="#c8d6e5"),
                    yanchor="bottom",
                    y=0.01,
                    xanchor="left",
                    x=0.01,
                )
            )
        else:
            # Latest position per float (one marker per WMO)
            map_df = (
                map_source.dropna(subset=["latitude", "longitude"])
                .sort_values("date")
                .groupby("wmo_id")
                .tail(1)
                .copy()
            )

            # Cap at 12 000 for browser performance
            if len(map_df) > 12_000:
                map_df = map_df.sample(12_000, random_state=42)

            fig_map = px.scatter_mapbox(
                map_df,
                lat="latitude",
                lon="longitude",
                color="institution",
                color_discrete_map=REGION_COLORS,
                hover_name="wmo_id",
                custom_data=["wmo_id"],
                hover_data={
                    "institution": True,
                    "date": True,
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                },
                zoom=2,
                center={"lat": -10, "lon": 80},
                category_orders={"institution": list(REGION_COLORS.keys())},
            )
            fig_map.update_traces(marker=dict(size=8, opacity=0.9))
            fig_map.update_layout(
                mapbox_style="carto-darkmatter",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    title="Region",
                    bgcolor="rgba(10,14,39,0.85)",
                    bordercolor="rgba(0,188,212,0.18)",
                    borderwidth=1,
                    font=dict(size=11, color="#c8d6e5"),
                    yanchor="bottom",
                    y=0.01,
                    xanchor="left",
                    x=0.01,
                    orientation="h",
                ),
            )
            
        fig_map.update_layout(height=620)
        st.plotly_chart(fig_map, use_container_width=True, key="main_map", on_select="rerun", config={"toImageButtonOptions": {"format": "png", "scale": 2, "filename": "argo_float_map"}})
        
        if selected_wmo_from_map:
            if st.button(f"📄 View Info for Float {selected_wmo_from_map}"):
                show_float_details(selected_wmo_from_map)
        elif is_sidebar_search and len([w for w in search_wmo.split(",") if w.strip()]) == 1:
            searched_id = search_wmo.strip()
            if st.button(f"📄 View Info for Float {searched_id}"):
                show_float_details(searched_id)

        if is_wmo_searched:
            st.caption(f"📌 {len(map_df['wmo_id'].unique()):,} floats displayed with full trajectory ({len(map_df):,} total profiles)")
        else:
            st.caption(f"📌 {len(map_df):,} unique floats displayed")
    else:
        st.info("No float data for current filters.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Component 2 + 3: Bar chart + KPI tiles ──
with col_right:
    st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
    # ── Bar chart (PRD §7.2) ──
    st.markdown("### 📈 Number of Floats per DAC")

    if len(filt_prof) > 0 and "dac" in filt_prof.columns:
        # Active floats in the last 90 days of each year
        latest_ds_date = filt_prof["date"].max()
        res = []
        years = sorted(filt_prof["year"].dropna().unique())
        for y in years:
            if y == latest_ds_date.year:
                end_of_year = latest_ds_date
            else:
                end_of_year = pd.Timestamp(f"{int(y)}-12-31")
            
            start_period = end_of_year - pd.Timedelta(days=90)
            active_df = filt_prof[(filt_prof["date"] >= start_period) & (filt_prof["date"] <= end_of_year)]
            active_floats = active_df.drop_duplicates(subset=["wmo_id"])
            
            for dac, count in active_floats["dac"].value_counts().items():
                res.append({"Year": int(y), "DAC": dac, "Count": count})
        yearly = pd.DataFrame(res)
        if len(yearly) > 0:
            yearly["Year"] = yearly["Year"].astype(int)
            yearly = yearly.sort_values(["Year", "Count"], ascending=[True, False])
            totals = yearly.groupby("Year")["Count"].sum().reset_index()

            # Professional DAC Color Mapping
            DAC_COLORS = {
                "aoml": "#4FC3F7", "coriolis": "#FF7043", "kiost": "#26A69A",
                "meds": "#BA68C8", "csiro": "#FFB74D", "jma": "#00BCD4",
                "incois": "#F06292", "csio": "#9CCC65", "bodc": "#9575CD",
                "kma": "#FFD54F", "nmdis": "#90A4AE",
            }

            fig_bar = px.bar(
                yearly,
                x="Year",
                y="Count",
                color="DAC",
                color_discrete_map=DAC_COLORS,
                category_orders={"Year": sorted(yearly["Year"].unique())}
            )
            
            fig_bar.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{x}</b><br>DAC: %{fullData.name}<br>Floats: %{y:,}<extra></extra>"
            )

            fig_bar.add_trace(go.Scatter(
                x=totals["Year"],
                y=totals["Count"],
                mode="text",
                text=totals["Count"],
                textposition="top center",
                textfont=dict(size=10, color="#ffffff", family="Outfit"),
                showlegend=False,
                hoverinfo="skip"
            ))

            fig_bar.update_layout(
                **_dark_layout(
                    height=420,
                    barmode="stack",
                    xaxis=dict(title="", type="category", tickangle=-45, gridcolor="rgba(255,255,255,0.03)"),
                    yaxis=dict(title="Active Floats", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None, font=dict(size=10)),
                    bargap=0.3,
                    margin=dict(l=50, r=20, t=80, b=40),
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True, key="bar_chart", config={"toImageButtonOptions": {"format": "png", "scale": 2, "filename": "argo_annual_floats"}})
        else:
            st.info("No active float data for bar chart.")
    else:
        st.info("No data for bar chart.")

    # ── KPI tiles (PRD §7.3) ──
    st.markdown("### 🧪 BGC Profile Counts")

    doxy_n = int(filt_bio["has_doxy"].sum()) if len(filt_bio) > 0 else 0
    chla_n = int(filt_bio["has_chla"].sum()) if len(filt_bio) > 0 else 0
    nit_n = int(filt_bio["has_nitrate"].sum()) if len(filt_bio) > 0 else 0
    ph_n = int(filt_bio["has_ph"].sum()) if len(filt_bio) > 0 else 0

    k1, k2, k3, k4 = st.columns(4)
    for col, label, value, color in [
        (k1, "DOXY", doxy_n, KPI_COLORS["DOXY"]),
        (k2, "Chla", chla_n, KPI_COLORS["Chla"]),
        (k3, "Nitrate", nit_n, KPI_COLORS["Nitrate"]),
        (k4, "pH", ph_n, KPI_COLORS["pH"]),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="kpi-tile" style="border-bottom: 4px solid {color} !important;" role="status" aria-label="{label}: {value:,} profiles">
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value:,}</div>
                    <div style="font-size: 10px; color: rgba(255,255,255,0.4); margin-top: 4px;">PROFILES</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
#  ROW 2 — TREEMAP (left) + DONUT (right)
# ================================================================
st.markdown("---")
col_tree, col_donut = st.columns(2, gap="medium")
# ── Component 4: Active Floats & Profiles last 1 day ──
with col_tree:
    st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
    st.markdown("### 📊 Active Floats & Profiles — Last 1 Day")

    if len(filt_prof) > 0:
        latest_date = filt_prof["date"].max()
        one_day_ago = pd.Timestamp(latest_date - timedelta(days=1))
        last1 = filt_prof[filt_prof["date"] >= one_day_ago].copy()

        if len(last1) > 0:
            tree_data = (
                last1.groupby("institution")
                .agg(floats=("wmo_id", "nunique"), profiles=("file", "count"))
                .reset_index()
            )

            total_f1 = int(tree_data["floats"].sum())
            total_p1 = int(tree_data["profiles"].sum())

            # Summary card
            st.markdown(
                f"""
                <div class="treemap-info">
                    <h3>All Communities</h3>
                    <div style="display:flex;justify-content:space-around;">
                        <div><div class="stat">{total_f1:,}</div>
                             <div class="stat-label">Active Floats</div></div>
                        <div><div class="stat">{total_p1:,}</div>
                             <div class="stat-label">Profiles</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Treemap
            fig_tree = px.treemap(
                tree_data,
                path=["institution"],
                values="profiles",
                color="profiles",
                color_continuous_scale=[
                    [0, "#1a2744"],
                    [0.5, "#1e3a5f"],
                    [1.0, "#2C5F8A"],
                ],
                hover_data=["floats", "profiles"],
                height=340,
            )
            fig_tree.update_traces(
                textinfo="label+value",
                textfont=dict(size=14, color="white"),
                marker=dict(line=dict(width=2, color="#0a0e27"), cornerradius=5),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Profiles: %{value:,}<br>"
                    "Floats: %{customdata[0]:,}<extra></extra>"
                ),
            )
            fig_tree.update_layout(
                **_dark_layout(margin=dict(l=0, r=0, t=10, b=0)),
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_tree, use_container_width=True, key="treemap", config={"toImageButtonOptions": {"format": "png", "scale": 2, "filename": "argo_last1day_treemap"}})
        else:
            st.info("No active floats in the last 1 day for current filters.")
    else:
        st.info("No data available.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Component 5: Float Age Donut (PRD §7.5) ──
with col_donut:
    st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
    st.markdown("### 🕐 Float Age Distribution")

    if len(filt_prof) > 0:
        # Only calculate age for active floats (reported in the last 90 days)
        latest_ds_date = filt_prof["date"].max()
        ninety_days_ago = pd.Timestamp(latest_ds_date - timedelta(days=90))
        
        float_last = filt_prof.dropna(subset=["date"]).groupby("wmo_id")["date"].max().reset_index()
        active_wmos = float_last[float_last["date"] >= ninety_days_ago]["wmo_id"]
        
        active_prof = filt_prof[filt_prof["wmo_id"].isin(active_wmos)]
        
        # Use earliest profile date per active float as proxy for launch date
        float_first = (
            active_prof.dropna(subset=["date"])
            .groupby("wmo_id")["date"]
            .min()
            .reset_index()
        )
        float_first["age_years"] = (
            (pd.Timestamp.now() - float_first["date"]).dt.days / 365.25
        )

        bins = [0, 3, 6, 9, 12, 999]
        labels = ["00-02", "03-05", "06-08", "09-11", "12+"]
        float_first["age_group"] = pd.cut(
            float_first["age_years"], bins=bins, labels=labels, right=False
        )

        age_counts = float_first["age_group"].value_counts().reset_index()
        age_counts.columns = ["Age Group", "Count"]
        age_counts["Age Group"] = pd.Categorical(
            age_counts["Age Group"], categories=labels, ordered=True
        )
        age_counts = age_counts.sort_values("Age Group")
        age_counts = age_counts[age_counts["Count"] > 0]

        if len(age_counts) > 0:
            fig_donut = px.pie(
                age_counts,
                values="Count",
                names="Age Group",
                hole=0.45,
                color="Age Group",
                color_discrete_map=AGE_COLORS,
                height=420,
            )
            fig_donut.update_traces(
                textinfo="label+percent",
                textposition="outside",
                textfont=dict(size=12, color="#c8d6e5"),
                pull=[0.02] * len(age_counts),
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Count: %{value:,}<br>"
                    "Percent: %{percent}<extra></extra>"
                ),
                marker=dict(line=dict(color="#0a0e27", width=2)),
            )
            fig_donut.update_layout(
                **_dark_layout(margin=dict(l=20, r=80, t=10, b=20)),
                legend=dict(
                    title="Age Group",
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.05,
                    font=dict(size=12, color="#c8d6e5"),
                    bgcolor="rgba(0,0,0,0)",
                ),
            )
            st.plotly_chart(fig_donut, use_container_width=True, key="donut", config={"toImageButtonOptions": {"format": "png", "scale": 2, "filename": "argo_age_distribution"}})
        else:
            st.info("No age data available.")
    else:
        st.info("No data available.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
#  ROW 2.5 — PROFILER TYPE DONUT (left) + FLEET COMPOSITION (right)
#  Data source: ar_index_global_meta.txt
# ================================================================
st.markdown("---")
col_profiler, col_fleet = st.columns(2, gap="medium")

# ── Profiler Type / Instrument Breakdown Donut ──
with col_profiler:
    st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
    st.markdown("### 🔧 Float Instrument Types (Meta Registry)")

    if len(df_meta) > 0:
        ptype_counts = df_meta["profiler_name"].value_counts().reset_index()
        ptype_counts.columns = ["Model", "Count"]

        # Group small categories into "Other" for readability
        top_n = 10
        if len(ptype_counts) > top_n:
            top = ptype_counts.head(top_n)
            other_count = ptype_counts.iloc[top_n:]["Count"].sum()
            other_row = pd.DataFrame([{"Model": "Other", "Count": other_count}])
            ptype_counts = pd.concat([top, other_row], ignore_index=True)

        fig_ptype = px.pie(
            ptype_counts,
            values="Count",
            names="Model",
            hole=0.45,
            color="Model",
            color_discrete_map=PROFILER_COLORS,
            height=420,
        )
        fig_ptype.update_traces(
            textinfo="label+percent",
            textposition="outside",
            textfont=dict(size=11, color="#c8d6e5"),
            pull=[0.02] * len(ptype_counts),
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Floats: %{value:,}<br>"
                "Share: %{percent}<extra></extra>"
            ),
            marker=dict(line=dict(color="#0a0e27", width=2)),
        )
        fig_ptype.update_layout(
            **_dark_layout(margin=dict(l=20, r=80, t=10, b=20)),
            legend=dict(
                title="Instrument",
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.05,
                font=dict(size=11, color="#c8d6e5"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig_ptype, use_container_width=True, key="profiler_donut", config={"toImageButtonOptions": {"format": "png", "scale": 2, "filename": "argo_profiler_types"}})
        st.caption(f"📋 {len(df_meta):,} floats across {df_meta['profiler_name'].nunique()} instrument models (source: ar_index_global_meta.txt)")
    else:
        st.info("No metadata available.")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Fleet Composition Stacked Area Chart ──
with col_fleet:
    st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
    st.markdown("### 📊 Fleet Composition Over Time")

    if len(filt_prof) > 0 and "profiler_name" in filt_prof.columns:
        # Get the deployment year per float (earliest profile date)
        float_deploy = (
            filt_prof.dropna(subset=["date"])
            .groupby("wmo_id")
            .agg(deploy_year=("year", "min"), profiler_name=("profiler_name", "first"))
            .reset_index()
        )

        if len(float_deploy) > 0:
            # Count deployments by year and profiler type
            comp = float_deploy.groupby(["deploy_year", "profiler_name"]).size().reset_index(name="Count")

            # Keep only top N models, group rest as "Other"
            top_models = float_deploy["profiler_name"].value_counts().head(8).index.tolist()
            comp["Model"] = comp["profiler_name"].where(comp["profiler_name"].isin(top_models), "Other")
            comp = comp.groupby(["deploy_year", "Model"])["Count"].sum().reset_index()
            comp = comp.sort_values("deploy_year")

            fig_fleet = px.area(
                comp,
                x="deploy_year",
                y="Count",
                color="Model",
                color_discrete_map=PROFILER_COLORS,
                height=420,
            )
            fig_fleet.update_traces(
                line=dict(width=0.5),
                hovertemplate="<b>%{fullData.name}</b><br>Year: %{x}<br>Floats: %{y:,}<extra></extra>",
            )
            fig_fleet.update_layout(
                **_dark_layout(
                    xaxis=dict(title="Deployment Year", gridcolor="rgba(255,255,255,0.03)"),
                    yaxis=dict(title="Floats Deployed", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=None, font=dict(size=10)),
                    margin=dict(l=50, r=20, t=60, b=40),
                ),
            )
            st.plotly_chart(fig_fleet, use_container_width=True, key="fleet_composition", config={"toImageButtonOptions": {"format": "png", "scale": 2, "filename": "argo_fleet_composition"}})
            st.caption("Shows how the fleet instrument mix has evolved per deployment year")
        else:
            st.info("No deployment data available.")
    else:
        st.info("No data available.")
    st.markdown('</div>', unsafe_allow_html=True)

# ================================================================
#  ROW 3 — DAC / Institution Summary Tables (PRD §7.6)
# ================================================================
st.markdown("---")
col_dac1, col_dac2 = st.columns(2, gap="medium")

if len(filt_prof) > 0:
    dac_profs = (
        filt_prof.groupby("institution")
        .agg(Profiles=("file", "count"))
        .reset_index()
    )
    dac_floats = df_meta.groupby("institution").agg(Floats=("wmo_id", "nunique")).reset_index()
    dac = pd.merge(dac_floats, dac_profs, on="institution", how="left").fillna(0)
    dac = dac.sort_values("Profiles", ascending=False)
    
    dacs = dac["institution"].tolist()
    header = "".join(f"<th>{d}</th>" for d in dacs)
    floats_cells = "".join(f"<td>{int(r):,}</td>" for r in dac["Floats"])
    profs_cells = "".join(f"<td>{int(r):,}</td>" for r in dac["Profiles"])

    with col_dac1:
        st.markdown("### 🏢 DAC / Institution Summary")
        st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="overflow-x:auto;">
            <table class="dac-table">
                <thead><tr><th>Metric</th>{header}</tr></thead>
                <tbody>
                    <tr><td>Floats</td>{floats_cells}</tr>
                    <tr><td>Profiles</td>{profs_cells}</tr>
                </tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_dac2:
        st.markdown("### 📡 Float Status Summary")
        latest_date = filt_prof["date"].max()
        ninety_days_ago = pd.Timestamp(latest_date - timedelta(days=90))
        float_latest = filt_prof.dropna(subset=["date"]).groupby(["institution", "wmo_id"])["date"].max().reset_index()
        float_latest["is_live"] = float_latest["date"] >= ninety_days_ago
        
        live_df = float_latest.groupby("institution").agg(
            live_floats=("is_live", "sum")
        ).reset_index()
        
        status_df = pd.merge(dac_floats.rename(columns={"Floats": "total_count"}), live_df, on="institution", how="left").fillna(0)
        status_df["dead_floats"] = status_df["total_count"] - status_df["live_floats"]
        status_df = status_df.set_index("institution").reindex(dacs).reset_index().fillna(0)
        
        # Dominant instrument per institution from meta registry
        _inst_top_model = (
            df_meta.groupby("institution")["profiler_name"]
            .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "—")
        )
        
        header2 = "".join(f"<th>{d}</th>" for d in status_df["institution"])
        total_cells = "".join(f"<td>{int(r):,}</td>" for r in status_df["total_count"])
        live_cells = "".join(f"<td>{int(r):,}</td>" for r in status_df["live_floats"])
        dead_cells = "".join(f"<td>{int(r):,}</td>" for r in status_df["dead_floats"])
        model_cells = "".join(
            f"<td style='font-size:0.75rem;color:#BA68C8;'>{_inst_top_model.get(d, '—')}</td>"
            for d in status_df["institution"]
        )
        
        st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="overflow-x:auto;">
            <table class="dac-table">
                <thead><tr><th>Status</th>{header2}</tr></thead>
                <tbody>
                    <tr><td>Total Count</td>{total_cells}</tr>
                    <tr><td>Live Floats</td>{live_cells}</tr>
                    <tr><td>Dead Floats</td>{dead_cells}</tr>
                    <tr><td>Top Model</td>{model_cells}</tr>
                </tbody>
            </table>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("No data available for summary tables.")

# ================================================================
#  ROW 4 — INCOIS Deployment Matrix (Month vs Year)
# ================================================================
st.markdown("---")
st.markdown("### 🗓️ INCOIS Float Deployments (Month vs Year)")

if len(df_prof) > 0:
    incois_df = df_prof[df_prof["institution"] == "IN"]
    if len(incois_df) > 0:
        # Find the deployment date (earliest profile date per float)
        deployments = incois_df.groupby("wmo_id")["date"].min().reset_index()
        
        # Merge with all registered INCOIS floats to capture ones that haven't profiled
        meta_in = df_meta[df_meta["institution"] == "IN"].copy()
        merged = pd.merge(meta_in, deployments, on="wmo_id", how="left")
        
        # If float hasn't profiled, fallback to metadata registration date (date_update)
        merged["date_update"] = pd.to_datetime(merged["date_update"], format="%Y%m%d%H%M%S", errors='coerce')
        merged["date_final"] = merged["date"].fillna(merged["date_update"])
        
        merged["Year"] = merged["date_final"].dt.year
        merged["Month"] = merged["date_final"].dt.month
        
        # Create a pivot table: Months as rows, Years as columns
        pivot = merged.pivot_table(
            index="Month", 
            columns="Year", 
            values="wmo_id", 
            aggfunc="count", 
            fill_value=0
        )
        
        # Ensure all 12 months are displayed
        all_months = range(1, 13)
        pivot = pivot.reindex(all_months, fill_value=0)
        
        month_names = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
        }
        pivot.index = pivot.index.map(month_names)
        
        # Calculate Row and Column Totals
        pivot["Total"] = pivot.sum(axis=1)
        pivot.loc["Total"] = pivot.sum(axis=0)
        
        total_in_floats = int(pivot.loc["Total", "Total"])
        
        st.markdown(f"<p style='color: #c8d6e5; font-size: 1rem;'>Total INCOIS Floats Registered: <strong style='color: #00BCD4; font-size: 1.2rem;'>{total_in_floats:,}</strong></p>", unsafe_allow_html=True)
        
        st.markdown('<div class="stPlotlyChart">', unsafe_allow_html=True)
        
        # Build HTML for the table
        header_html = "<th>Month</th>" + "".join([f"<th>{y if isinstance(y, str) else int(y)}</th>" for y in pivot.columns])
        
        body_html = ""
        for month in pivot.index:
            is_total_row = (month == "Total")
            row_bg = "background: rgba(0,188,212,0.06);" if is_total_row else ""
            row_html = f"<td style='font-weight:bold; color:#00BCD4;'>{month}</td>"
            for col in pivot.columns:
                val = pivot.loc[month, col]
                val_str = f"{int(val):,}" if val > 0 else "<span style='color:rgba(255,255,255,0.2)'>-</span>"
                
                is_total_col = (col == "Total")
                style = ""
                if is_total_row or is_total_col:
                    style = "font-weight:bold; color:#FFB74D;"
                # Highlight Pending column slightly
                if col == "Pending" and val > 0:
                    style += " color:#EF5350;"
                    
                row_html += f"<td style='{style}'>{val_str}</td>"
            body_html += f"<tr style='{row_bg}'>{row_html}</tr>"
            
        st.markdown(
            f'''
            <div style="overflow-x:auto;">
            <table class="dac-table" style="width:100%; text-align:center;">
                <thead><tr>{header_html}</tr></thead>
                <tbody>
                    {body_html}
                </tbody>
            </table>
            </div>
            ''',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No INCOIS deployment data found.")
else:
    st.info("No data available for deployment matrix.")


# ================================================================
#  RAW DATA VIEWER (bonus — not in PRD but useful for ops)
# ================================================================
st.markdown("---")
with st.expander("📋 View Raw Data", expanded=False):
    tab1, tab2, tab3 = st.tabs(["Core Profiles", "BGC Profiles", "Float Metadata"])
    with tab1:
        st.dataframe(
            filt_prof.head(200), use_container_width=True, hide_index=True
        )
        st.caption(
            f"Showing {min(200, len(filt_prof)):,} of {len(filt_prof):,} records"
        )
        st.download_button(
            "⬇️ Download Filtered Core Profiles (CSV)",
            data=filt_prof.to_csv(index=False),
            file_name="argo_core_profiles_filtered.csv",
            mime="text/csv",
            key="dl_core",
        )
    with tab2:
        st.dataframe(
            filt_bio.head(200), use_container_width=True, hide_index=True
        )
        st.caption(
            f"Showing {min(200, len(filt_bio)):,} of {len(filt_bio):,} records"
        )
        st.download_button(
            "⬇️ Download Filtered BGC Profiles (CSV)",
            data=filt_bio.to_csv(index=False),
            file_name="argo_bgc_profiles_filtered.csv",
            mime="text/csv",
            key="dl_bgc",
        )
    with tab3:
        st.dataframe(
            df_meta.head(500), use_container_width=True, hide_index=True
        )
        st.caption(
            f"Showing {min(500, len(df_meta)):,} of {len(df_meta):,} float metadata records (source: ar_index_global_meta.txt)"
        )
        st.download_button(
            "⬇️ Download Float Metadata (CSV)",
            data=df_meta.to_csv(index=False),
            file_name="argo_float_metadata.csv",
            mime="text/csv",
            key="dl_meta",
        )

# ==================== FOOTER ====================
st.markdown(
    f"""
<div class="footer-bar">
    Indian ARGO CTD/BGC Dashboard · INCOIS · Data: IFREMER GDAC<br>
    {datetime.now().strftime("%Y-%m-%d %H:%M")} ·
    {len(df_prof):,} profiles · {df_prof['wmo_id'].nunique():,} floats ·
    {len(df_bio):,} BGC profiles · {len(df_meta):,} registered floats (meta)
</div>
""",
    unsafe_allow_html=True,
)
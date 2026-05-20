import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import xarray as xr
import gsw

def get_valid_data(ds_prof):
    # Flatten the data for scatter plots
    cycles_2d = np.repeat(ds_prof.CYCLE_NUMBER.values[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)
    dates_2d = np.repeat(ds_prof.JULD.values[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)
    
    # We might have missing values in some profiles
    lon = ds_prof.LONGITUDE.values
    lat = ds_prof.LATITUDE.values
    lon_2d = np.repeat(lon[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)
    lat_2d = np.repeat(lat[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)

    pres = ds_prof.PRES.values.flatten()
    temp = ds_prof.TEMP.values.flatten() if 'TEMP' in ds_prof else np.full_like(pres, np.nan)
    psal = ds_prof.PSAL.values.flatten() if 'PSAL' in ds_prof else np.full_like(pres, np.nan)
    cycles = cycles_2d.flatten()
    dates = dates_2d.flatten()
    lon_flat = lon_2d.flatten()
    lat_flat = lat_2d.flatten()
    
    valid = ~np.isnan(pres) & ~np.isnan(temp) & ~np.isnan(psal) & ~np.isnat(dates)
    
    pres = pres[valid]
    temp = temp[valid]
    psal = psal[valid]
    cycles = cycles[valid]
    dates = dates[valid]
    lon_flat = lon_flat[valid]
    lat_flat = lat_flat[valid]
    
    # Compute Density (sigma0)
    SA = gsw.SA_from_SP(psal, pres, lon_flat, lat_flat)
    CT = gsw.CT_from_t(SA, temp, pres)
    rho = gsw.sigma0(SA, CT)
    
    return cycles, dates, pres, temp, psal, rho

def create_ts_diagram(cycles, temp, psal, wmo):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(psal, temp, c=cycles, cmap='jet', s=5, alpha=0.8)
    ax.set_xlabel("Practical Salinity (PSU)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title("T/S Diagram")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Profile number")
    fig.tight_layout()
    return fig

def create_section_chart(dates, pres, z_var, z_label, title, wmo, cmap='jet'):
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(dates, pres, c=z_var, cmap=cmap, s=15, marker='s', edgecolors='none')
    ax.invert_yaxis()
    ax.set_ylabel("Pressure (dbar)")
    ax.set_title(title)
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(z_label)
    fig.tight_layout()
    return fig

def create_overlaid_profiles(x_var, pres, cycles, x_label, title, wmo, cmap='jet'):
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Instead of lines which might look messy if flattened, scatter is fine, 
    # but to draw lines we group by cycle. For performance and exact match to 
    # the screenshot (which uses lines/scatter with color mapped to cycle), 
    # a scatter plot with small points looks identical to dense overlaid lines.
    sc = ax.scatter(x_var, pres, c=cycles, cmap=cmap, s=2, alpha=0.8)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pressure (dbar)")
    ax.set_title(title)
    
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Profile number")
    fig.tight_layout()
    return fig

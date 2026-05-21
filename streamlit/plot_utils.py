import numpy as np
import xarray as xr
import gsw
import plotly.graph_objects as go
import pandas as pd

def get_valid_data(ds_prof):
    def get_var(name):
        adj_name = f"{name}_ADJUSTED"
        if adj_name in ds_prof:
            val = ds_prof[adj_name].values.flatten()
            if not np.isnan(val).all():
                return val
        if name in ds_prof:
            return ds_prof[name].values.flatten()
        
        # Fallback if variable doesn't exist
        if 'PRES' in ds_prof:
            return np.full_like(ds_prof.PRES.values.flatten(), np.nan)
        return np.array([])

    if 'CYCLE_NUMBER' not in ds_prof or 'PRES' not in ds_prof:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    cycles_2d = np.repeat(ds_prof.CYCLE_NUMBER.values[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)
    dates_2d = np.repeat(ds_prof.JULD.values[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)
    
    lon = ds_prof.LONGITUDE.values
    lat = ds_prof.LATITUDE.values
    lon_2d = np.repeat(lon[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)
    lat_2d = np.repeat(lat[:, np.newaxis], ds_prof.PRES.shape[1], axis=1)

    pres = get_var('PRES')
    temp = get_var('TEMP')
    psal = get_var('PSAL')
    
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
    
    SA = gsw.SA_from_SP(psal, pres, lon_flat, lat_flat)
    CT = gsw.CT_from_t(SA, temp, pres)
    rho = gsw.sigma0(SA, CT)
    
    return cycles, dates, pres, temp, psal, rho

def _dark_layout(title, xlabel, ylabel, invert_y=False):
    layout = dict(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#c8d6e5", size=12),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.1)", zerolinecolor="rgba(255,255,255,0.1)")
    )
    if xlabel == "Date":
        layout["xaxis"]["tickformat"] = "%d/%m/%Y"
    if invert_y:
        layout["yaxis"]["autorange"] = "reversed"
    return layout

def create_ts_diagram(cycles, temp, psal, wmo, title="T/S Diagram"):
    fig = go.Figure()
    
    fig.add_trace(go.Scattergl(
        x=psal, y=temp,
        mode='markers',
        marker=dict(
            size=4,
            color=cycles,
            colorscale='Jet',
            showscale=True,
            colorbar=dict(title="Profile<br>number")
        ),
        customdata=np.stack((cycles,), axis=-1),
        hovertemplate="<b>Cycle:</b> %{customdata[0]}<br><b>Sal:</b> %{x:.3f} PSU<br><b>Temp:</b> %{y:.3f}°C<extra></extra>"
    ))
    fig.update_layout(**_dark_layout(title, "Practical Salinity (PSU)", "Temperature (°C)"))
    return fig

def create_section_chart(dates, pres, z_var, z_label, title, wmo, cmap='Jet'):
    fig = go.Figure()
    
    fig.add_trace(go.Scattergl(
        x=dates, y=pres,
        mode='markers',
        marker=dict(
            size=5,
            symbol='square',
            color=z_var,
            colorscale=cmap,
            showscale=True,
            colorbar=dict(title=z_label.replace(' ', '<br>', 1))
        ),
        customdata=np.stack((z_var,), axis=-1),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d %H:%M}<br><b>Press:</b> %{y:.1f} dbar<br><b>" + z_label + ":</b> %{customdata[0]:.3f}<extra></extra>"
    ))
    fig.update_layout(**_dark_layout(title, "Date", "Pressure (dbar)", invert_y=True))
    return fig

def create_overlaid_profiles(x_var, pres, cycles, x_label, title, wmo, cmap='Jet'):
    fig = go.Figure()
    
    fig.add_trace(go.Scattergl(
        x=x_var, y=pres,
        mode='markers',
        marker=dict(
            size=3,
            color=cycles,
            colorscale=cmap,
            showscale=True,
            colorbar=dict(title="Profile<br>number")
        ),
        customdata=np.stack((cycles,), axis=-1),
        hovertemplate="<b>Cycle:</b> %{customdata[0]}<br><b>" + x_label + ":</b> %{x:.3f}<br><b>Press:</b> %{y:.1f} dbar<extra></extra>"
    ))
    fig.update_layout(**_dark_layout(title, x_label, "Pressure (dbar)", invert_y=True))
    return fig

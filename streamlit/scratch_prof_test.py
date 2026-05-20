import xarray as xr
import numpy as np

ds_prof = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902174_prof.nc")

print("Checking last few cycles for valid PRES data...")
pres_data = ds_prof.PRES.values
valid_cycles = np.where(~np.isnan(pres_data).all(axis=1))[0]

print(f"Total cycles: {pres_data.shape[0]}")
print(f"Total valid cycles: {len(valid_cycles)}")

if len(valid_cycles) > 0:
    last_valid_idx = valid_cycles[-1]
    print(f"Last valid cycle index: {last_valid_idx}, Cycle number: {ds_prof.CYCLE_NUMBER.values[last_valid_idx]}")
    
    last_pres = pres_data[last_valid_idx]
    last_temp = ds_prof.TEMP.values[last_valid_idx] if 'TEMP' in ds_prof else np.full_like(last_pres, np.nan)
    last_psal = ds_prof.PSAL.values[last_valid_idx] if 'PSAL' in ds_prof else np.full_like(last_pres, np.nan)
    
    valid_idx = ~np.isnan(last_pres)
    pres_v = last_pres[valid_idx]
    temp_v = last_temp[valid_idx]
    psal_v = last_psal[valid_idx]
    
    print(f"Found {len(pres_v)} valid pressure points in this cycle.")
    
    if len(pres_v) > 0:
        surface_idx = np.argmin(pres_v)
        bottom_idx = np.argmax(pres_v)
        print(f"Surface: {pres_v[surface_idx]:.2f} dbar {temp_v[surface_idx]:.3f}°C {psal_v[surface_idx]:.3f} PSU")
        print(f"Bottom: {pres_v[bottom_idx]:.2f} dbar {temp_v[bottom_idx]:.3f}°C {psal_v[bottom_idx]:.3f} PSU")

ds_meta = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902174_meta.nc")
def d(val):
    if hasattr(val, "item") and callable(val.item) and val.ndim == 0:
        val = val.item()
    if isinstance(val, bytes):
        return val.decode('utf-8', errors='ignore').strip()
    elif isinstance(val, np.ndarray) and val.dtype.kind == 'S':
        return ", ".join([v.decode('utf-8', errors='ignore').strip() for v in val.flat if v.decode('utf-8', errors='ignore').strip()])
    return str(val).strip()

print("\nData Centre fields:")
print(f"DATA_CENTRE: {d(ds_meta.DATA_CENTRE.values) if 'DATA_CENTRE' in ds_meta else 'N/A'}")
print(f"OPERATING_INSTITUTION: {d(ds_meta.OPERATING_INSTITUTION.values) if 'OPERATING_INSTITUTION' in ds_meta else 'N/A'}")


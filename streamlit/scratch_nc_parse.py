import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

ds_meta = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902174_meta.nc")
ds_prof = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902174_prof.nc")

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

print("Maker:", d(ds_meta.PLATFORM_MAKER.values))
print("Serial:", d(ds_meta.FLOAT_SERIAL_NO.values))
print("Type:", d(ds_meta.PLATFORM_TYPE.values))
print("Trans:", d(ds_meta.TRANS_SYSTEM.values))
print("DC:", d(ds_meta.DATA_CENTRE.values))
print("Sensors:", d(ds_meta.SENSOR.values))
print("PTT:", d(ds_meta.PTT.values) if 'PTT' in ds_meta else "N/A")
print("Launch date:", d(ds_meta.LAUNCH_DATE.values))
print("Project:", d(ds_meta.PROJECT_NAME.values))
print("PI:", d(ds_meta.PI_NAME.values))
print("Lat:", float(ds_meta.LAUNCH_LATITUDE.values))

cycle = int(np.nanmax(ds_prof.CYCLE_NUMBER.values))
juld = ds_prof.JULD.values
last_date_np = juld[~np.isnat(juld)]
last_date = pd.to_datetime(last_date_np[-1]).strftime('%d/%m/%Y %H:%M:%S')

last_pres = ds_prof.PRES.values[-1]
last_temp = ds_prof.TEMP.values[-1]
last_psal = ds_prof.PSAL.values[-1]

valid_idx = ~np.isnan(last_pres)
pres_v = last_pres[valid_idx]
temp_v = last_temp[valid_idx]
psal_v = last_psal[valid_idx]

surface_idx = np.argmin(pres_v)
bottom_idx = np.argmax(pres_v)
print(f"Cycle: {cycle}, Last Date: {last_date}")
print(f"Surface: {pres_v[surface_idx]} dbar {temp_v[surface_idx]} C {psal_v[surface_idx]} PSU")
print(f"Bottom: {pres_v[bottom_idx]} dbar {temp_v[bottom_idx]} C {psal_v[bottom_idx]} PSU")

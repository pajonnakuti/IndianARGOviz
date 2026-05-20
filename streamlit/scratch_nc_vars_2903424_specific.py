import xarray as xr
import numpy as np

wmo = "2903424"
meta_path = f"more_components/{wmo}_meta.nc"

ds_meta = xr.open_dataset(meta_path)
for v in ds_meta.variables:
    if any(kw in v for kw in ["OWNER", "CENTRE", "INST", "DAC", "PI", "PROJECT"]):
        try:
            val = ds_meta[v].values
            if hasattr(val, "item") and callable(val.item) and val.ndim == 0:
                val = val.item()
            if isinstance(val, bytes):
                val = val.decode('utf-8', errors='ignore').strip()
            elif isinstance(val, np.ndarray) and val.dtype.kind == 'S':
                val = ", ".join([x.decode('utf-8', errors='ignore').strip() for x in val.flat if x.decode('utf-8', errors='ignore').strip()])
            else:
                val = str(val).strip()
            print(f"{v}: {val}")
        except Exception as e:
            pass

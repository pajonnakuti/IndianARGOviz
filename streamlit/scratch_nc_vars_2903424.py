import xarray as xr
import urllib.request
from pathlib import Path
import numpy as np

wmo = "2903424"
dac = "aoml"
meta_url = f"ftp://ftp.ifremer.fr/ifremer/argo/dac/{dac}/{wmo}/{wmo}_meta.nc"
meta_path = f"more_components/{wmo}_meta.nc"

if not Path(meta_path).exists():
    urllib.request.urlretrieve(meta_url, meta_path)

ds_meta = xr.open_dataset(meta_path)
for v in ds_meta.variables:
    if ds_meta[v].dtype.kind in ['S', 'U', 'O']:
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

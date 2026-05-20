import xarray as xr
ds = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902174_meta.nc")
for v in ds.variables:
    if ds[v].dtype.kind in ['S', 'U', 'O']:
        try:
            val = ds[v].values
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

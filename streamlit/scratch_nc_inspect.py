import xarray as xr

try:
    ds_meta = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902174_meta.nc")
    print("Meta vars:", list(ds_meta.variables))
    for v in ['PLATFORM_NUMBER', 'PLATFORM_MAKER', 'FLOAT_SERIAL_NO', 'PLATFORM_TYPE', 'TRANS_SYSTEM', 'PROJECT_NAME', 'PI_NAME', 'LAUNCH_DATE', 'LAUNCH_LATITUDE', 'LAUNCH_LONGITUDE', 'DATA_CENTRE', 'FIRMWARE_VERSION', 'SENSOR']:
        if v in ds_meta:
            val = ds_meta[v].values
            print(f"{v}: {val}")
except Exception as e:
    print("Meta error:", e)

try:
    ds_prof = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902174_prof.nc")
    print("\nProf vars:", list(ds_prof.variables))
    for v in ['CYCLE_NUMBER', 'JULD', 'PRES', 'TEMP', 'PSAL']:
        if v in ds_prof:
            print(f"{v} shape: {ds_prof[v].shape}")
except Exception as e:
    print("Prof error:", e)

import xarray as xr
import plot_utils

try:
    ds_prof = xr.open_dataset(r"C:\Users\harsh\incois\dashboard\more_components\2902821_prof.nc")
    cycles, dates, pres, temp, psal, rho = plot_utils.get_valid_data(ds_prof)
    print("Success. Extracted valid data shapes:")
    print("PRES:", pres.shape)
except Exception as e:
    print("Error:", e)

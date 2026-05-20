import urllib.request
import os

wmo = "2902174"
dac = "incois"
meta_url = f"ftp://ftp.ifremer.fr/ifremer/argo/dac/{dac}/{wmo}/{wmo}_meta.nc"
meta_path = f"{wmo}_meta_test.nc"

try:
    print(f"Downloading {meta_url}...")
    urllib.request.urlretrieve(meta_url, meta_path)
    print(f"Success! Size: {os.path.getsize(meta_path)} bytes")
except Exception as e:
    print("Error:", e)

import pandas as pd
from datetime import timedelta

print('Loading data...')
df = pd.read_parquet('cache/profiles.parquet')

print('\n--- Indian Ocean last 7 days ---')
df_io = df[
    (df['longitude'] >= 20.0) & (df['longitude'] <= 145.0) & 
    (df['latitude'] >= -70.1) & (df['latitude'] <= 30.0)
]
latest_io = df_io['date'].max()
last7_io = df_io[df_io['date'] >= latest_io - timedelta(days=7)]
print(f"Floats: {last7_io['wmo_id'].nunique()}")
print(f"Profiles: {len(last7_io)}")

print('\n--- Indian Ocean last 7 days by institution ---')
tree_data = last7_io.groupby('institution').agg(floats=('wmo_id', 'nunique'), profiles=('file', 'count')).reset_index().sort_values('floats', ascending=False)
print(tree_data.to_string(index=False))

print('\n--- Let us also try with ONLY BGC floats? No, the image says All Communities ---')

"""
Create simple csv comparing RF to CNN
"""

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd


# souza
souza_df = gpd.read_file('./data/amazon_dams_clip.shp')
souza_df = souza_df.loc[souza_df['area_2017']>0]
souza_df['souza_count'] = 1
souza_df = souza_df[['area_2017', 'souza_count']].sum()

# CNN
cnn_df = pd.read_csv('./data/cnn_souza_areas.csv')
cnn_df = cnn_df.assign(cnn_nonzero=cnn_df['area']>0)
cnn_df = cnn_df.loc[cnn_df['reg'] != 0]
cnn_df = cnn_df.groupby('reg').sum()
cnn_df['cnn_area'] = cnn_df['area']*0.01
cnn_df.drop(columns=['area'], inplace=True)


print(cnn_df)
print(souza_df)
# Combine and write to csv
full_df = souza_df.join(cnn_df, how='outer')
full_df.to_csv('./data/compare_summary.csv')

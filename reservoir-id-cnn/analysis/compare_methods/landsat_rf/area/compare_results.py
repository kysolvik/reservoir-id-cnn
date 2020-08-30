"""
Create simple csv comparing RF to CNN
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# Random Forest
rf_df = pd.read_csv('./data/rf_2015_area.csv')
rf_df = rf_df.assign(rf_nonzero=rf_df['area']>0)
rf_df = rf_df.loc[rf_df['reg'] != 0]
rf_df = rf_df.groupby('reg').sum()
rf_df['rf_area'] = rf_df['area']*0.01
rf_df.drop(columns=['area'], inplace=True)

# Random Forest Clip
rf_clip_df = pd.read_csv('./data/rf_2014_reproj_area.csv')
rf_clip_df = rf_clip_df.assign(rf_clip_nonzero=rf_clip_df['area']>0)
rf_clip_df = rf_clip_df.loc[rf_clip_df['reg'] != 0]
rf_clip_df = rf_clip_df.groupby('reg').sum()
rf_clip_df['rf_clip_area'] = rf_clip_df['area']*0.01
rf_clip_df.drop(columns=['area'], inplace=True)

# CNN
cnn_df = pd.read_csv('./data/cnn_2017_area.csv')
cnn_df = cnn_df.assign(cnn_nonzero=cnn_df['area']>0)
cnn_df = cnn_df.loc[cnn_df['reg'] != 0]
cnn_df = cnn_df.groupby('reg').sum()
cnn_df['cnn_area'] = cnn_df['area']*0.01
cnn_df.drop(columns=['area'], inplace=True)


# Combine and write to csv
muni_df = rf_df.join(rf_clip_df, how='inner').join(cnn_df, how='inner')
muni_df.to_csv('./data/muni_compare_summary.csv')

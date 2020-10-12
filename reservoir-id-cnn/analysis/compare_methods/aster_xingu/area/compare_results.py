"""
Create simple csv comparing RF to CNN
"""

import pandas as pd
import matplotlib.pyplot as plt


# aster
aster_df = pd.read_csv('./data/xingu_2007_areas.csv')
aster_df = aster_df.assign(aster_nonzero=aster_df['area']>0)
aster_df = aster_df.loc[aster_df['reg'] != 0]
aster_df = aster_df.groupby('reg').sum()
aster_df['aster_area'] = aster_df['area']*0.01
aster_df.drop(columns=['area'], inplace=True)

# CNN
cnn_df = pd.read_csv('./data/cnn_2017_areas.csv')
cnn_df = cnn_df.assign(cnn_nonzero=cnn_df['area']>0)
cnn_df = cnn_df.loc[cnn_df['reg'] != 0]
cnn_df = cnn_df.groupby('reg').sum()
cnn_df['cnn_area'] = cnn_df['area']*0.01
cnn_df.drop(columns=['area'], inplace=True)


# Combine and write to csv
muni_df = aster_df.join(cnn_df, how='inner')
muni_df.to_csv('./data/muni_compare_summary.csv')

"""
Create simple chloropleth maps by ecoregion and state
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# States
state_df = pd.read_csv('./data/state_sizes.csv')
state_df = state_df.assign(nonzero=state_df['area']>0)
state_df = state_df.groupby('reg').sum()
state_df['area'] = state_df['area']*0.01

state_shape = gpd.read_file('../accuracy/data/shapefiles/brazil_states.shp')

state_shape['GEOCODIGO'] = state_shape['GEOCODIGO'].astype(int)
state_shape = state_shape.merge(state_df, left_on='GEOCODIGO', right_on='reg')

# Calculate density based on state area
state_shape['area_density'] = state_shape['area']/state_shape.area
state_shape['count_density'] = state_shape['nonzero']/state_shape.area

# Make plots
fig, ax = plt.subplots(1, 1)
state_shape.plot(column='area_density', ax=ax, legend=True)
state_shape.boundary.plot(ax=ax)
plt.savefig('./figures/state_res_area_density.png')

fig, ax = plt.subplots(1, 1)
state_shape.plot(column='count_density', ax=ax, legend=True)
state_shape.boundary.plot(ax=ax)
plt.savefig('./figures/state_res_count_density.png')

# Ecoregions
eco_df = pd.read_csv('./data/eco_sizes.csv')
eco_df = eco_df.assign(nonzero=eco_df['area']>0)
eco_df = eco_df.groupby('reg').sum()
eco_df['area'] = eco_df['area']*0.01

eco_shape = gpd.read_file('../accuracy//data/shapefiles/ecoregions.shp')

eco_shape['ECO_NUM'] = eco_shape['ECO_NUM'].astype(int)
eco_shape = eco_shape.merge(eco_df, left_on='ECO_NUM', right_on='reg')

# Calculate density
eco_shape['area_density'] = eco_shape['area']/eco_shape.area
eco_shape['count_density'] = eco_shape['nonzero']/eco_shape.area

# Make plots
fig, ax = plt.subplots(1, 1)
eco_shape.plot(column='area_density', ax=ax, legend=True)
eco_shape.boundary.plot(ax=ax)
plt.savefig('./figures/eco_res_area_density.png')


fig, ax = plt.subplots(1, 1)
eco_shape.plot(column='count_density', ax=ax, legend=True)
eco_shape.boundary.plot(ax=ax)
plt.savefig('./figures/eco_res_count_density.png')

# Watersheds
water_df = pd.read_csv('./data/watersheds_sizes.csv')
water_df = water_df.assign(nonzero=water_df['area']>0)
water_df = water_df.groupby('reg').sum()
water_df['area'] = water_df['area']*0.01

water_shape = gpd.read_file('../accuracy/data/shapefiles/watersheds_4digit.shp')

water_shape['NUNIVOTTO4'] = water_shape['NUNIVOTTO4'].astype(int)
water_shape = water_shape.merge(water_df, left_on='NUNIVOTTO4', right_on='reg')

# Calculate density
water_shape['area_density'] = water_shape['area']/water_shape.area
water_shape['count_density'] = water_shape['nonzero']/water_shape.area

fig, ax = plt.subplots(1, 1)
water_shape.plot(column='area_density', ax=ax, legend=True)
water_shape.boundary.plot(ax=ax, linewidth=0.1)
plt.savefig('./figures/water_res_area_density.png')

fig, ax = plt.subplots(1, 1)
water_shape.plot(column='count_density', ax=ax, legend=True)
water_shape.boundary.plot(ax=ax, linewidth=0.1)
plt.savefig('./figures/water_res_count_density.png')

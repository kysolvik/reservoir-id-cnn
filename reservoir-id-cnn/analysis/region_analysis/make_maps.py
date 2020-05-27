"""
Create simple chloropleth maps by ecoregion and state
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


# States
state_df = pd.read_csv('./data/summary_state.csv')
state_shape = gpd.read_file('./data/shapefiles/brazil_states.shp')
state_shape = state_shape.rename(columns={'NOME_UF':'state'})

state_shape = state_shape.merge(state_df, on='state')

fig, ax = plt.subplots(1, 1)
state_shape.plot(column='f1', ax=ax, legend=True)
state_shape.boundary.plot(ax=ax)
plt.savefig('./figures/state_f1.png')

fig, ax = plt.subplots(1, 1)
state_shape = state_shape.assign(
        total_count=state_shape['train_count'] + state_shape['test_count'])
state_shape.plot(column='total_count', ax=ax, legend=True)
state_shape.boundary.plot(ax=ax)
plt.savefig('./figures/state_count.png')

# Ecoregions
eco_df = pd.read_csv('./data/summary_ecoregion.csv')
eco_shape = gpd.read_file('./data/shapefiles/ecoregions.shp')
eco_shape = eco_shape.rename(columns={'ECO_NAME':'ecoregion'})

eco_shape = eco_shape.merge(eco_df, on='ecoregion')

fig, ax = plt.subplots(1, 1)
eco_shape.plot(column='f1', ax=ax, legend=True)
eco_shape.boundary.plot(ax=ax)
plt.savefig('./figures/eco_f1.png')


fig, ax = plt.subplots(1, 1)
eco_shape = eco_shape.assign(
        total_count=eco_shape['train_count'] + eco_shape['test_count'])
eco_shape.plot(column='total_count', ax=ax, legend=True)
eco_shape.boundary.plot(ax=ax)
plt.savefig('./figures/eco_count.png')

#/usr/bin/env python
"""
Assign training/val/test images to state and ecoregion

NOTE: For some reason, geopandas keeps segfaulting when I read a shapefiles
Running on local machine instead of google cloud
"""

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


def gpd_read_csv(csv_path, x_col, y_col):
    df = pd.read_csv(csv_path)

    geometry = [Point(xy) for xy in zip(df[x_col], df[y_col])]
    crs = {'init': 'epsg:4326'} #http://www.spatialreference.org/ref/epsg/2263/
    geo_df = gpd.GeoDataFrame(df, crs=crs, geometry=geometry)

    return geo_df


def spat_join(points_df, poly_df):
    join = gpd.sjoin(points_df, poly_df, how="inner", op="within")

    return join


def main():

    # Read in points
    points_df = gpd_read_csv('./data/centers.csv', 'center_lon', 'center_lat')
    # Read in states polys
    states_df = gpd.read_file('./data/shapefiles/brazil_states.shp')
    states_df =  states_df[['NOME_UF', 'geometry']].rename(
            columns={'NOME_UF':'state'})
    states_join = spat_join(points_df, states_df)
    states_join.drop(columns=['index_right'], inplace=True)

    # Read in ecoregions polys
    eco_df = gpd.read_file('./data/shapefiles/ecoregions.shp')
    eco_df = eco_df[['ECO_NAME', 'BIOME', 'geometry']].rename(
            columns={'ECO_NAME':'ecoregion', 'BIOME':'biome'})
    all_join = spat_join(states_join, eco_df)

    all_join.drop(columns=['geometry']).to_csv(
            './data/centers_ecoregions_states.csv', index=False)

if __name__=='__main__':
    main()

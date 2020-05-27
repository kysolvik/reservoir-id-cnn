#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""Calculate raster stats and save to csv

Usage:
    python3 zonal_stats.py [shapefile] [shapefile-name-column] [raster] [output-csv]

    shapefile-name-column is the name of the column with the region name
    For brazil_states.shp, it's probably "NOME_UF" that you want.
    For ecoregions.shp (from WWF), it's "ECO_NAME"

"""

import rasterstats
import sys
import pandas as pd

def main():
    if len(sys.argv) != 5:
        sys.exit('Script takes exactly four args: shapefile shapefile-name-column raster output-csv')


    shpfile = sys.argv[1]
    shpfile_col = sys.argv[2]
    raster = sys.argv[3]
    output_csv = sys.argv[4]


    zstats = rasterstats.zonal_stats(shpfile, raster, stats=['sum'],
                                     geojson_out=True, prefix = 'zs_')

    zstats_df = pd.DataFrame.from_dict([f['properties'] for f in zstats])\
        [[shpfile_col, 'zs_sum']]

    zstats_df.to_csv(output_csv, index=False)




if __name__ == '__main__':
    main()


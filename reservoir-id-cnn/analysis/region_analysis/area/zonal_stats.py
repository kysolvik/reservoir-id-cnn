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
import rasterio as rio
import numpy as np
import sys
import pandas as pd

box_size = 1000
 
def main():
    if len(sys.argv) != 5:
        sys.exit('Script takes exactly four args: shapefile shapefile-name-column raster output-csv')



    shpfile = sys.argv[1]
    shpfile_col = sys.argv[2]
    raster = sys.argv[3]
    output_csv = sys.argv[4]

    out_df = pd.DataFrame(columns=[shpfile_col, 'zs_sum'])


    # Doing it in blocks
    src = rio.open(raster)
    affine = src.transform

    total_rows, total_cols = src.height, src.width
    current_row = 0
    current_col = 0
    row_starts = np.arange(0, total_rows, box_size)
    col_starts = np.arange(0, total_cols, box_size)

    # Create Nx2 array with row/col start indices
    start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

    for i in range(5):#start_ind.shape[0]):
        # For the indices near edge we need to use a smaller box size
        box_size_rows = min(total_rows - start_ind[i,0], box_size)
        box_size_cols = min(total_cols - start_ind[i,1], box_size)
        ar = src.read(1, 
                window=((int(start_ind[i, 1]), int(start_ind[i,0])),
                    (int(start_ind[i,1] + box_size_cols),int(start_ind[i,0] + box_size_rows)))
                )
        print(ar.shape)

        zstats = rasterstats.zonal_stats(shpfile, ar, affine=affine, stats=['sum'],
                geojson_out=False, prefix = 'zs_', nodata=0)
        zstats_df = pd.DataFrame.from_dict([f['properties'] for f in zstats])[[shpfile_col, 'zs_sum']]

        out_df = out_df.join(zstats_df, on=shpfile_col, how='outer')


    out_df.to_csv(output_csv, index=False)




if __name__ == '__main__':
    main()


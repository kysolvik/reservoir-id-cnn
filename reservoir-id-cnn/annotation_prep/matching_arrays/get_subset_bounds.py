#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Get lat/lon bounds of subset arrays

To get matching arrays of new rasters, need to get the current
training subset bounding coordinates.

Example:
    $ python3 grid_indices_all.csv gs://res-id/ grid_indices_lalo.csv

Output:
    grid indices csv with lat/long corner coords
"""


import os.path
import argparse
import numpy as np
import pandas as pd
from skimage import io
import gdal
import subprocess as sp
from affine import Affine

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Get bounding lat/lon for training arrays',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('grid_indices_in',
        help = 'grid indices csv output by extract_subarrays.py',
        type = str)
    p.add_argument('gcs_raster_dir',
        help = 'Google Cloud Storage path storing input rasters',
        type = str)
    p.add_argument('grid_indices_out',
        help = 'Output grid indices csv with lat/lon added',
        type = str)

    return(p)


def download_im(im_source, gcs_im_dir):
    gcs_image_path = '{}{}'.format(gcs_im_dir, im_source)
    sp.call(['gsutil', '-m', 'cp', gcs_image_path, './temp/'])
    local_im_path = './temp/{}'.format(im_source)
    return local_im_path


def run_single_im(im_source, gcs_im_dir, grid_df):
    # Initialize
    subset_df = grid_df.loc[grid_df['source'] == im_source]
    local_im_path = download_im(im_source, gcs_im_dir)

    # Get geotransformation info
    fh = gdal.Open(local_im_path)
    geo_affine = Affine.from_gdal(*fh.GetGeoTransform())

    # Calculate corners
    (lon_max, lat_min) = (subset_df['ymax'], subset_df['xmax']) * geo_affine
    (lon_min, lat_max) = (subset_df['ymin'], subset_df['xmin']) * geo_affine
    subset_df.loc[:, 'lon_min'] = lon_min
    subset_df.loc[:, 'lat_min'] = lat_min
    subset_df.loc[:, 'lon_max'] = lon_max
    subset_df.loc[:, 'lat_max'] = lat_max

    # Delete tif
    sp.call(['rm', local_im_path])

    return subset_df


def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Read in input dataframe
    grid_df = pd.read_csv(args.grid_indices_in)

    out_list = [run_single_im(im_source, args.gcs_raster_dir, grid_df) for
                im_source in grid_df['source'].unique()]

    out_df = pd.concat(out_list)

    out_df.to_csv(args.grid_indices_out, header=True, index=False)


if __name__ == '__main__':
    main()



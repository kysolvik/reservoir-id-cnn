#!/usr/bin/env python3

"""Extract arrays from new rasters that match previously extracted training
arrays

"""

import argparse
import pandas as pd
import subprocess as sp
import glob

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Extract matching arrays from new rasters',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('grid_indices_latlon',
        help = 'grid indices csv output by extract_subarrays.py',
        type = str)
    p.add_argument('target_tif',
        help = 'Path to tif',
        type = str)
    p.add_argument('output_dir',
        help = 'Output directory.',
        type = str)
    p.add_argument('output_suffix',
        help = 'Output suffix, e.g. sent 2_20m',
        type = str)

    return(p)

def subset_target(target_tif, target_res, subarray_dim, output_file, subset_df_row):
    center_to_edge = target_res*int(subarray_dim)/2
    xmin = str(subset_df_row['center_lon'] - center_to_edge)
    xmax = str(subset_df_row['center_lon'] + center_to_edge)
    ymin = str(subset_df_row['center_lat'] - center_to_edge)
    ymax = str(subset_df_row['center_lat'] + center_to_edge)
    sp.call(['gdalwarp', '-tr', str(target_res), str(target_res), # '-tap',
             '-te', xmin, ymin, xmax, ymax, '-overwrite', '-co', 'COMPRESS=LZW',
             target_tif, output_file])

    return target_tif


def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Size of subarrays
    subarray_dim=500

    # Make output dir
    sp.call(['mkdir', '-p', args.output_dir])

    # Read in input dataframe
    grid_df = pd.read_csv(args.grid_indices_latlon)

    target_tif = args.target_tif

    # Get resolution of target
    target_res = 8.9831528412e-05

    # Create matching arrays
    for row_i in range(grid_df.shape[0]):
        cur_row = grid_df.loc[row_i]
        output_file = '{}/{}'.format(args.output_dir, cur_row['filename'].replace(
            's2_20m_og', args.output_suffix))
        subset_target(target_tif, target_res, subarray_dim, output_file, cur_row)

    return


if __name__ == '__main__':
    main()

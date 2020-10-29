#!/usr/bin/env python3

"""Extract arrays from new rasters that match previously extracted training
arrays

"""

import argparse
import pandas as pd
import subprocess as sp
import gdal
import glob

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Extract matching arrays from new rasters',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('grid_indices_latlon',
        help = 'grid indices csv output by extract_subarrays.py',
        type = str)
    p.add_argument('gcs_raster_dir',
        help = 'Google Cloud Storage path storing target rasters',
        type = str)
    p.add_argument('output_dir',
        help = 'Output directory.',
        type = str)
    p.add_argument('output_suffix',
        help = 'Output suffix, e.g. sent 2_20m',
        type = str)

    return(p)

def subset_target(target_vrt, target_res, output_file, subset_df_row):
    xmin = str(subset_df_row['lon_min'])
    xmax = str(subset_df_row['lon_max'])
    ymin = str(subset_df_row['lat_min'])
    ymax = str(subset_df_row['lat_max'])
    sp.call(['gdalwarp', '-tr', str(target_res), str(target_res), # '-tap',
             '-te', xmin, ymin, xmax, ymax, '-overwrite', '-co', 'COMPRESS=LZW',
             target_vrt, output_file])

    return target_vrt


def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Make output dir
    sp.call(['mkdir', '-p', args.output_dir])

    # Read in input dataframe
    grid_df = pd.read_csv(args.grid_indices_latlon)

    # Mount GS Bucket
    bucket_name = args.gcs_raster_dir.split('/')[2]
    local_raster_dir = 'gcs_mount{}'.format(
        args.gcs_raster_dir.split(bucket_name, maxsplit=1)[1])
    sp.call(['gcsfuse', bucket_name, 'gcs_mount/'])

    # Create target vrt
    target_vrt = 'temp/target.vrt'
    sp.call(['gdalbuildvrt', target_vrt] +
            glob.glob('{}/*'.format(local_raster_dir)))

    # Get resolution of target
    # fh = gdal.Open(target_vrt)
    # target_res = fh.GetGeoTransform()[1]
    target_res = 8.9831528412e-05

    # Create matching arrays
    for row_i in range(grid_df.shape[0]):
        cur_row = grid_df.loc[row_i]
        output_file = '{}/{}.tif'.format(args.output_dir, cur_row['name'].replace(
            'ndwi', args.output_suffix))
        subset_target(target_vrt, target_res, output_file, cur_row)

    sp.call(['sudo', 'umount', 'gcs_mount'])
    sp.call(['rm', target_vrt])

    return


if __name__ == '__main__':
    main()

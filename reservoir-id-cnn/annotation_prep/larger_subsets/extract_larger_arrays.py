#!/usr/bin/env python3

"""Extract arrays around pre-selected points for all-null training arrays
"""

import argparse
import pandas as pd
import subprocess as sp
import gdal
import glob
import math
import os

OG_SIZE = 500
NEW_SIZE = 640
OUT_PAD = (NEW_SIZE - OG_SIZE)/2
only_annotated= True # Set to true to skip files that have not been annotated

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Extract subset arrays around pre-selected points',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('latlon_csv',
        help = 'csv containing lat/lon for previous boundaries (from matching arrays code)',
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

def subset_target(target_vrt, target_res, output_file, latlon_df_row):
    prev_xmin = latlon_df_row['lon_min']
    prev_xmax = latlon_df_row['lon_max']
    prev_ymin = latlon_df_row['lat_min']
    prev_ymax = latlon_df_row['lat_max']

    xmin, xmax = prev_xmin - OUT_PAD*target_res, prev_xmax + OUT_PAD*target_res
    ymin, ymax = prev_ymin - OUT_PAD*target_res, prev_ymax + OUT_PAD*target_res



    sp.call(['gdalwarp', '-tr', str(target_res), str(target_res), # '-tap',
             '-te', str(xmin), str(ymin), str(xmax), str(ymax),
             '-overwrite', '-co', 'COMPRESS=LZW',
             target_vrt, output_file])

    return target_vrt


def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Make output dir
    sp.call(['mkdir', '-p', args.output_dir])

    # Read in input dataframe
    latlon_df = pd.read_csv(args.latlon_csv)

    # Mount GS Bucket
    sp.call(['mkdir', '-p', 'gcs_mount'])
    bucket_name = args.gcs_raster_dir.split('/')[2]
    local_raster_dir = 'gcs_mount{}'.format(
        args.gcs_raster_dir.split(bucket_name, maxsplit=1)[1])
    sp.call(['gcsfuse', bucket_name, 'gcs_mount/'])

    # Create target vrt
    sp.call(['mkdir', '-p', './temp'])
    target_vrt = 'temp/target.vrt'
    sp.call(['gdalbuildvrt', target_vrt] +
            glob.glob('{}/*'.format(local_raster_dir)))

    # Get resolution of target
    # fh = gdal.Open(target_vrt)
    # target_res = fh.GetGeoTransform()[1]
    target_res = 8.9831528412e-05

    # If only annotated flag set, find the ims that have been
    if only_annotated:
        annotated_list = glob.glob('../../train/data/*mask.png')
        annotated_list = [os.path.splitext(os.path.basename(fn))[0]
                          for fn in annotated_list]



    # Create matching arrays
    for row_i in range(latlon_df.shape[0]):
        cur_row = latlon_df.loc[row_i]
        if only_annotated:
            if not cur_row['name'].replace('ndwi','mask') in annotated_list:
                print('Skipping {}'.format(row_i))
                continue
        output_file = '{}/{}.tif'.format(
            args.output_dir,
            cur_row['name'].replace('ndwi',args.output_suffix))
        subset_target(target_vrt, target_res, output_file, cur_row)

    sp.call(['sudo', 'umount', 'gcs_mount'])
    sp.call(['rm', target_vrt])

    return


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

"""Extract arrays around pre-selected points for all-null training arrays
"""

import argparse
import pandas as pd
import subprocess as sp
import gdal
import glob
import math

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Extract subset arrays around pre-selected points',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('centers_csv',
        help = 'csv containing lat/lon for centers',
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

def subset_target(target_vrt, vrt_x_min, vrt_y_max, target_res, output_file, centers_df_row):
    x_center = centers_df_row['X']
    y_center = centers_df_row['Y']
    x_center_round = vrt_x_min + math.floor((x_center - vrt_x_min)
                                            / target_res) * target_res
    y_center_round = vrt_y_max + math.floor((y_center - vrt_y_max)
                                            / target_res) * target_res

    xmin, xmax = x_center_round - 250*target_res, x_center_round + 250*target_res
    ymin, ymax = y_center_round - 250*target_res, y_center_round + 250*target_res



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
    centers_df = pd.read_csv(args.centers_csv)

    # Mount GS Bucket
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

    # get xmin and ymax
    fh = gdal.Open(target_vrt)
    geo_trans = fh.GetGeoTransform()
    xmin = geo_trans[0]
    ymax = geo_trans[3]

    # Create matching arrays
    for row_i in range(centers_df.shape[0]):
        cur_row = centers_df.loc[row_i]
        output_file = '{}/{}.tif'.format(
            args.output_dir,
            'im_floodplains_' + str(row_i) + '_' +  args.output_suffix)
        subset_target(target_vrt, xmin, ymax, target_res, output_file, cur_row)

    sp.call(['sudo', 'umount', 'gcs_mount'])
    sp.call(['rm', target_vrt])

    return


if __name__ == '__main__':
    main()

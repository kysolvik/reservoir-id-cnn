#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract subset images for annotation

This script extracts subset images from a large geoTiff. These images can then 
be annotated to create training/test data for the CNN.

Example:
    Create 5 10x10 sub-images of raster 'eg.tif': 
    $ python3 extract_subarrays.py eg.tif 5 10 10 out/ --out_prefix='eg_sub_'

Notes:
    In order to work with Labelbox, the images must be exported as png or jpg.
"""


import os.path
import argparse
import numpy as np
import pandas as pd
from skimage import io


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Extract subest images from larger raster/image.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source_path',
        help = 'Path to raw input image',
        type = str)
    p.add_argument('num_subsets',
        help = 'Number of subsets to create',
        type = int)
    p.add_argument('subset_dim_x',
        help = 'Subset image X dimension in # pixels',
        type = int)
    p.add_argument('subset_dim_y',
        help = 'Subset image Y dimension in # pixels',
        type = int)
    p.add_argument('out_dir',
        help = 'Output directory for subset images',
        type = str)
    p.add_argument('--out_prefix',
        help = 'Prefix for output tiffs',
        default = 'image_',
        type = str)

    return(p)


def write_append_csv(df,csv_path):
    """Check if csv already exists. Append if it does, write w/ header if not"""

    if not os.path.isfile(csv_path):
        df.to_csv(csv_path, header = True, index=False)
    else:
        df.to_csv(csv_path, mode = 'a', header=False, index=False)

    return()


def scale_image_tobyte(ar):
    """Scale larger data type array to byte"""

    minVals = np.amin(np.amin(ar,1),0)
    maxVals = np.amax(np.amax(ar,1),0)
    byte_ar = np.round(255.0 * (ar - minVals) / (maxVals - minVals - 1.0)) \
        .astype(np.uint8)
    byte_ar[ar == 0] = 0

    return(byte_ar)


def subset_image(vis_im, og_im, num_subsets, dim_x, dim_y, out_dir,
    source_path, out_prefix):
    """Create num_subsets images of (dim_x, dim_y) size from vis_im."""

    # Randomly select locations for sub-arrays
    sub_xmins = np.random.random_integers(0, vis_im.shape[0] - (dim_x + 1),
                    num_subsets)
    sub_ymins = np.random.random_integers(0, vis_im.shape[1] - (dim_y + 1),
                    num_subsets)

    # Create and save csv containing grid coordinates for images
    grid_indices_df = pd.DataFrame({
        'name': ['{}{}_vis'.format(out_prefix,snum) 
                    for snum in range(0,num_subsets)],
        'source': os.path.basename(source_path),
        'xmin': sub_xmins, 
        'xmax': sub_xmins + dim_x,
        'ymin': sub_ymins,
        'ymax': sub_ymins + dim_y
        })
    write_append_csv(grid_indices_df,'{}/grid_indices.csv'.format(out_dir))

    # Save sub-arrays
    for snum in range(0, num_subsets):
        # Vis image, for annotating
        subset_vis_path = '{}/{}{}_vis.png'.format(out_dir,out_prefix,snum)
        sub_vis_im = vis_im[sub_xmins[snum]:sub_xmins[snum] + dim_x,
                      sub_ymins[snum]:sub_ymins[snum] + dim_y,
                      :]
        sub_vis_im_byte = scale_image_tobyte(sub_vis_im)
        io.imsave(subset_vis_path, sub_vis_im_byte)

        # Original image, for training
        subset_og_path = '{}/{}{}_og.tif'.format(out_dir,out_prefix,snum)
        sub_og_im = og_im[sub_xmins[snum]:sub_xmins[snum] + dim_x,
                      sub_ymins[snum]:sub_ymins[snum] + dim_y,
                      :]
        io.imsave(subset_og_path, sub_og_im)

    return()


def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Read image
    base_image = io.imread(args.source_path) 
    base_image_bandselect = base_image[:,:,[2,1,0]]

    # Get subsets
    subset_image(base_image_bandselect, base_image, args.num_subsets,
        args.subset_dim_x,args.subset_dim_y, 
        args.out_dir, args.source_path, args.out_prefix)

    return()  


if __name__ == '__main__':
    main()

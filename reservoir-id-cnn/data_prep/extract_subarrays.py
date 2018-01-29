#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract subset images for annotation

This script extracts subset images from a large geoTiff. These images can then 
be annotated to create training/test data for the CNN.

Example:
    Create 5 10x10 sub-images of raster 'eg.tif': 
    $ python3 extract_subarrays.py eg.tif 5 10 10 out/ --out_prefix='eg_sub_'

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


def subset_image(arr, num_subsets, dim_x, dim_y, out_dir,
    source_path, outfile_prefix):
    """Create num_subsets arrays of (dim_x, dim_y) size from arr."""

    # Randomly select locations for sub-arrays
    sub_xmins = np.random.random_integers(0, arr.shape[0] - (dim_x + 1),
                    num_subsets)
    sub_ymins = np.random.random_integers(0, arr.shape[1] - (dim_y + 1),
                    num_subsets)

    # Create and save csv containing grid coordinates for images
    grid_coords_df = pd.DataFrame({
        'name': ['{}{}'.format(outfile_prefix,snum) 
                    for snum in range(0,num_subsets)],
        'source': os.path.basename(source_path),
        'xmin': sub_xmins, 
        'xmax': sub_xmins + dim_x,
        'ymin': sub_ymins,
        'ymax': sub_ymins + dim_y
        })
    grid_coords_df.to_csv('{}/grid_coords.csv'.format(out_dir), index = False)

    # Save sub-arrays
    for snum in range(0, num_subsets):
        subset_path = '{}/{}{}.tif'.format(out_dir,outfile_prefix,snum)
        sub_arr = arr[sub_xmins[snum]:sub_xmins[snum] + dim_x,
                      sub_ymins[snum]:sub_ymins[snum] + dim_y]
        io.imsave(subset_path, sub_arr, plugin = 'tifffile')

    return()


def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Read image
    base_image = io.imread(args.source_path, plugin = 'tifffile') 

    # Get subsets
    subset_image(base_image, args.num_subsets, args.subset_dim_x,
        args.subset_dim_y, args.out_dir, args.source_path, args.outfile_prefix)

    return()  
    


if __name__ == '__main__':
    main()

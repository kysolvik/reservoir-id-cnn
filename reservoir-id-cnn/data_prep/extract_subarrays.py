#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract subset images for annotation

This script extracts subset images from a large geoTiff. These images can then 
be annotated to create training/test data for the CNN.

Example:
    
Notes: 
"""


import argparse
import numpy as np
from skimage import io


def argparse_init():
    """Prepare ArgumentParser for inputs"""
    p = argparse.ArgumentParser(
            description='Extract subest images from larger raster/image.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('image_path',
        help = 'Path to raw input image',
        type = str)
    p.add_argument('num_subsets',
        help = 'Number of subsets to create',
        type = int)
    p.add_argument('subset_dimX',
        help = 'Subset image X dimension in # pixels',
        type = int)
    p.add_argument('subset_dimY',
        help = 'Subset image Y dimension in # pixels',
        type = int)
    p.add_argument('out_dir',
        help = 'Output directory for subset images',
        type = str)

    return(p)


def subset_image(arr, num_subsets, dimX, dimY, out_dir):
    """Create num_subsets arrays of (dimX, dimY) size from arr."""

    subarr_minxs = np.random.random_integers(0, arr.shape[0] - (dimX + 1),
                       num_subsets)
    subarr_minys = np.random.random_integers(0, arr.shape[1] - (dimY + 1),
                       num_subsets)

    for snum in range(0, num_subsets):
        subset_path = '{}/image_{}.tif'.format(out_dir,snum)
        sub_arr = arr[subarr_minxs[snum]:subarr_minxs[snum] + dimX,
                      subarr_minxs[snum]:subarr_minxs[snum] + dimY]
        io.imsave(subset_path, sub_arr, plugin = 'tifffile')

    return()


def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Read image
    base_image = io.imread(args.image_path, plugin = 'tifffile') 

    # Get subsets
    subset_image(base_image, args.num_subsets, args.subset_dimX,
        args.subset_dimY, args.out_dir)

    return()  
    


if __name__ == '__main__':
    main()



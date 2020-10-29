#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Save empty masks for floodplains images"""


import glob
from skimage import io
import argparse
import numpy as np


def argparse_init():
    """Prepare ArgumentParser for inputs."""

    p = argparse.ArgumentParser(
            description='Create empty masks for floodplains images',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('im_dir',
                   help='Path to directory containing floodplains images',
                   type=str)
    return p


def save_empty_mask(mask_path, dim_x, dim_y):
    """Writes an all-zeros mask file (png) given dim_x and dim_y"""

    mask_array = np.zeros([dim_x,dim_y], dtype=np.uint8)
    io.imsave(mask_path, mask_array)

    return None

def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    for f in glob.glob('{}/*s1_v2_og.tif'.format(args.im_dir)):
        new_mask_path = f.replace('s1_v2_og.tif', 'mask.png')
        print(new_mask_path)
        save_empty_mask(new_mask_path, 500, 500)

if __name__=='__main__':
    main()

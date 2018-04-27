#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict reservoirs on Sentinel-2 4-band raster

This script subsets a large image and predicts on the subsets
then optional merges them back into a full image.

Example:
    $ python3 predict_map.py ./sentinel.tif ./tiledir --mosaic='./fullrast.tif'

"""


import pathlib
import os.path
import argparse
import numpy as np
import pandas as pd
from skimage import io
import gdal
import rasterio
import resnet


DIM_X = 500
DIM_Y = 500
# Dimensions of input images.

RESIZE_ROWS = 512
RESIZE_COLS = 512
# Inputs unet size

NBANDS = 6
# Number of bands for unet

OVERLAP = 100
# Overlap size, in pixels.


def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Predict reservoirs from Sentinel tif.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source_path',
                   help = 'Path to raw input image to predict on.',
                   type = str)
    p.add_argument('model_weights',
                   help = 'Path to hdf5 file containing model weights.',
                   type = str)
    p.add_argument('out_dir',
                   help = 'Output directory for predicted subsets',
                   type = str)
    p.add_argument('--mosaic',
                   help = ('Path to mosaiced output file. If not defined, will not create.'),
                   default = None,
                   type = str)

    return p

def preprocess_batch(img_batch):

    return prepped_im

def predict_batch(img_batch, model, outfile):
    img_batch = preprocess_batch(img_batch)
    preds = model.predict(img_batch, batch_size=img_batch.shape[0], verbose=1)

    for pred_mask in preds:
        new_dataset = rasterio.open(outfile, 'w', driver='GTiff',
                                    height=Z.shape[0], width=Z.shape[1],
                                    count=1, dtype=Z.dtype,
                                    crs='+proj=latlong', transform=transform)
        new_dataset.write(Z, 1)

    return

def mosaic_tiles(tile_dir, mosaic_path):

    return


def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Load model
    unet_model = resnet.get_unet(RESIZE_ROWS, RESIZE_COLS, NUM_BANDS)
    unet_model.load_weights(args.model_weights)

    # Open image
    src = rasterio.open(args.source_path)
    total_rows, total_cols = src.height, src.width

    current_row = 0
    current_col = 0
    row_starts = range(0, total_rows - DIM_X, DIM_X - OVERLAP)
    col_starts = range(0, total_cols - DIM_X, DIM_X - OVERLAP)

    # Add final indices to row_starts and col_starts
    row_starts = list(row_starts).append(total_rows - DIM_X)
    col_starts = list(col_starts).append(total_cols - DIM_Y)

    for current_row in row_starts:
        for current_col in col_starts:
            target_im = src.read(window=((current_row, current_row + DIM_X),
                                         (current_col, current_col + DIMY)))
            target_outfile = '{}/{}_{}-{}.tif'.format(
                args.out_dir, pathlib.Path(args.source_image).stem,
                current_row, current_col)
            predict_im(target_im, unet, target_outfile)

    # Mosaic


    return()


if __name__ == '__main__':
    main()

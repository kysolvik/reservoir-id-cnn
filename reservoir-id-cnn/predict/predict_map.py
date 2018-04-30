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

BATCH_SIZE = 8
# NN prediction batch size


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


def normalized_diff(ar1, ar2):
    """Returns normalized difference of two arrays."""

    # Convert arrays to float32
    ar1 = ar1.astype('float32')
    ar2 = ar2.astype('float32')

    return((ar1 - ar2) / (ar1 + ar2))


class ResPredictBatch(object):
    """Batch of image tiles for reservoir CNN prediction

    Attributes:
        img_src (rasterio DatasetReader): Rasterio object for reading.
        start_indices (array): Nx2 array with row/column indices for starting
            each tile.
        batch_size (int): Number of images for simultaneous prediction.
        dims (tuple): Dimensions of image (dim_x, dim_y).
        nbands (int): Number of bands in image
        resize_dims (tuple): Dimensions for resizing before CNN prediction.
        out_dir (str): Path to output directory.
        model (keras model): CNN model with loaded weights.

    """
    def __init__(self, img_src, start_indices, batch_size, dims, nbands,
                 resize_dims, out_dir='./predict/', model):
        self.img_src = img_src
        self.start_indices = start_indices
        self.batch_size = batch_size
        self.dims = dims
        self.nbands = nbands
        self.resize_dims = resize_dims
        self.out_dir = out_dir
        self.model = model


    @property
    def crs(self):
        self.src_img.crs


    def get_geotransform(self, indice_pair):
        """Calculate geotransform of a tile.

        Args:
            indice_pair (tuple): Row, Col indices of upper left corner of tile.

        """
        geo = self.src_img.transform
        new_upperleft = geo * indice_pair
        geo[2] = new_upperleft[0]
        geo[5] = new_upperleft[1]


    def load_images(self):
        self.imgs = np.empty((self.batch_size, self.dims[0], self.dims[1],
                              self.nbands))
        for i in range(self.batch_size):
            row, col = self.start_indices[i,0], self.start_indices[i,1]
            self.imgs[i] = self.img_src.read(window=((row, row + self.dim_x),
                                                     (col, col + self.dim_y)))


    def add_nd(self, band1, band2):
        """Add band containing NDWI."""

        nd = normalized_diff(self.imgs[:,:,:,band1], self.imgs[:,:,:,band2])

        # Convert to uint16
        nd_min = nd.min()
        nd_max = nd.max()
        nd = 65535 * (nd - nd_min) / (nd_max - nd_min)
        nd = nd.astype(np.uint16)

        # Reshape
        nd = np.reshape(nd, np.append(np.asarray(nd.shape), 1))

        # Append nd to self.imgs
        self.imgs = np.append(self.imgs, nd, axis=3)


    def preprocess(self):
        # Add NDWI band
        self.add_nd(self.imgs, 1, 3)

        # Add NDVI band
        self.add_nd(self.imgs, 3, 2)

        # Apply scaling
        mean = np.mean(self.imgs)
        std = np.std(self.imgs)
        self.imgs -= mean
        self.imgs /= std


    def predict(self):
        self.preds = self.model.predict(self.imgs, self.batch_size)


    def write_images(self):
        for i in range(self.batch_size):
            outfile = '{}/pred_{}-{}.tif'.format(
                self.out_dir, self.start_indices[i, 0], self.start_indices[i, 1])
            new_dataset = rasterio.open(
                outfile, 'w', driver='GTiff',
                height=self.dim_y, width=self.dim_x,
                count=1, dtype='uint8',
                crs=self.crs,
                transform=self.get_geotransform(tuple(self.start_indices[i,:]))
            )
            new_dataset.write(self.preds[i], 1)


    def predict_write_batch(self):
        """Master method for loading, predicting, and writing full batch."""
        self.load_images()
        self.preprocess()
        self.predict()
        self.write_images()


def get_unet(img_rows, img_cols, nbands):
    """Load U-Net structure."""
    inputs = Input((img_rows, img_cols, nbands))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2),
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2),
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2),
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer='adam')

    return model

def mosaic_tiles(tile_dir, mosaic_path):

    return


def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    # Load model
    unet_model = get_unet(RESIZE_ROWS, RESIZE_COLS, NBANDS)
    unet_model.load_weights(args.model_weights)

    # Open image
    src = rasterio.open(args.source_path)
    total_rows, total_cols = src.height, src.width

    current_row = 0
    current_col = 0
    row_starts = np.arange(0, total_rows - DIM_X, DIM_X - OVERLAP)
    col_starts = np.arange(0, total_cols - DIM_Y, DIM_Y - OVERLAP)

    # Add final indices to row_starts and col_starts
    row_starts = np.append(row_starts, total_rows - DIM_X)
    col_starts = np.append(col_starts, total_cols - DIM_Y)

    # Create Nx2 array with row/col start indices
    start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

    # Create batches of BATCH_SIZE
    total_batches = np.ceil(start_ind.shape[0]/BATCH_SIZE)
    start_ind_batches = np.array_split(start_ind, total_batches)

    # Run prediction
    for batch_ind in start_ind_batches:
        res_batch = ResPredictBatch(
            img_src=src, start_indices=batch_ind, batch_size=BATCH_SIZE,
            dims=(DIM_X, DIM_Y), nbands=NBANDS,
            resize_dims=(RESIZE_ROWS, RESIZE_COLS), out_dir=args.out_dir,
            model=unet_model)
        ResPredictBatch.predict_write_batch()


    # Mosaic


    return()


if __name__ == '__main__':
    main()

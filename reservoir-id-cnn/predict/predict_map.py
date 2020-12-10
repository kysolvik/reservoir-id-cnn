#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict reservoirs on Sentinel-2 4-band raster

This script subsets a large image and predicts on the subsets
then optional merges them back into a full image.

Example:
    $ python3 predict_map.py sentinel.tif weights.h5 structure.txt ./tiledir/ --mosaic='fullrast.tif'

"""


import pathlib
import os
import argparse
import numpy as np
from skimage import io, transform
import rasterio
import affine
from keras import models
import tempfile
import subprocess as sp
import glob
import re
import keras.backend as K
import gc
import pandas as pd

OG_ROWS = 640
OG_COLS = 640
# Dimensions of input images.

RESIZE_FLAG = False
RESIZE_ROWS = 640
RESIZE_COLS = 640
# If needed, can resize before feeding to CNN

OUT_ROWS = 500
OUT_COLS = 500
# Size of output from CNN

NBANDS = 12
# Number of bands in original image

OVERLAP = 140
# Overlap size, in pixels.

BATCH_SIZE = 240 #512
# Batch size for process/prediction

BAND_SELECTION = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

def argparse_init():
    """Prepare ArgumentParser for inputs"""

    p = argparse.ArgumentParser(
            description='Predict reservoirs from Sentinel tif.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('source_path',
                   help = 'Path to raw input image to predict on.',
                   type = str)
    p.add_argument('model_structure',
                   help = 'Text file containing model structure saved as json.',
                   type = str)
    p.add_argument('model_weights',
                   help = 'hdf5 file containing saved model weights.',
                   type = str)
    p.add_argument('mean_std_file',
                   help = '.npy file containing train mean and std for scaling',
                   type = str)

    p.add_argument('out_dir',
                   help = 'Output directory for predicted subsets',
                   type = str)
    p.add_argument('--mosaic',
                   help = ('Path to mosaiced output file. If not defined, will not create.'),
                   default = None,
                   type = str)

    return p


def scale_image_tobyte(ar):
    """Scale larger data type array to byte"""
    min_val = np.min(ar)
    max_val = np.max(ar)
    byte_ar = (np.round(255.0 * (ar - min_val) / (max_val - min_val))
               .astype(np.uint8))
    byte_ar[ar == 0] = 0

    return(byte_ar)


def normalized_diff(ar1, ar2):
    """Returns normalized difference of two arrays."""

    # Convert arrays to float32
    ar1 = ar1.astype('float32')
    ar2 = ar2.astype('float32')

    return np.nan_to_num(((ar1 - ar2) / (ar1 + ar2)),0)

class ResPredictBatch(object):
    """Batch of image tiles for reservoir CNN prediction

    Attributes:
        img_srcs (List of rasterio DatasetReader): List of rio sources for predict images
        start_indices (array): Nx2 array with row/column indices for starting
            each tile.
        batch_size (int): Number of images for simultaneous prediction.
        dims (tuple): Dimensions of image (dim_x, dim_y).
        nbands (int): Number of bands in image
        resize_flag (bool): True to run resizing on inputs, False to not
        resize_dims (tuple): Dimensions for resizing before CNN prediction.
        out_dims (tuple): Dimensions of output from CNN, post undo resizing if done
        out_dir (str): Path to output directory.
        model (keras model): CNN model with loaded weights.

    """
    def __init__(self, img_srcs, start_indices, batch_size, batch_start_point, dims,
                 nbands, resize_flag, resize_dims, out_dims, model, mean_std_file,
                 out_dir='./predict/'):
        self.img_srcs = img_srcs
        self.start_indices = start_indices
        self.batch_size = batch_size
        self.batch_start_point = batch_start_point
        self.dims = dims
        self.nbands = nbands
        self.resize_flag = resize_flag
        self.resize_dims = resize_dims
        self.out_dims = out_dims
        self.crop_pad = int((self.resize_dims[0]-self.out_dims[0])/2)
        self.out_dir = out_dir
        self.mean_std_file = mean_std_file
        self.model = model

    @property
    def crs(self):
        self.img_srcs[0].crs


    def get_geotransform(self, indice_pair):
        """Calculate geotransform of a tile.

        Notes:
            Using .affine instead of .transform because it should work with all
            rasterio > 0.9. See https://github.com/mapbox/rasterio/issues/86.

        Args:
            indice_pair (tuple): Row, Col indices of upper left corner of tile.

        """
        geo = self.img_srcs[0].transform
        if self.crop_pad!=0:
            indice_pair = (indice_pair[0]+self.crop_pad,
                           indice_pair[1]+self.crop_pad)
        new_ul = [geo[2] + indice_pair[0]*geo[0] + indice_pair[1]*geo[1],
                  geo[5] + indice_pair[0]*geo[3] + indice_pair[1]*geo[4]]

        new_affine = affine.Affine(geo[0], geo[1], new_ul[0],
                                   geo[3], geo[4], new_ul[1])

        return new_affine


    def load_images(self):
        self.imgs = np.empty((self.batch_size, self.nbands,
                              self.dims[0], self.dims[1]))
        img_count = 0
        i = self.batch_start_point
        invalid_list = []
        valid_list = []
        while img_count < self.batch_size and i < self.start_indices.shape[0]:
            row, col = self.start_indices[i,0], self.start_indices[i,1]
            og_img_list = []
            # First check if valid against for image
            img_src = self.img_srcs[0]
            base_img = img_src.read(window=((row, row + self.dims[0]),
                                            (col, col + self.dims[1])))
            if np.min(base_img) > 0:
                og_img_list += [base_img]
                for img_src in self.img_srcs[1:]:
                    og_img_list += [img_src.read(window=((row, row + self.dims[0]),
                                                        (col, col + self.dims[1])))]
                self.imgs[img_count] = np.vstack(og_img_list)
                img_count += 1
                valid_list += [self.start_indices[i]]
            else:
                invalid_list += [self.start_indices[i]]
            i += 1

        self.batch_end_point = i - 1

        # Save valid indices
        self.batch_indices = np.asarray(valid_list)

        # Append invalid list to file
        with open('invalid_indices.txt', 'a') as f:
            for ind in invalid_list:
                f.write("{},{}\n".format(ind[0], ind[1]))
        print('Bypassed {} invalid images'.format(len(invalid_list)))
        print('Predicting batch of {}'.format(img_count))



    def add_nd(self, band1, band2):
        """Add band containing NDWI."""

        nd = normalized_diff(self.imgs[:,:,:,band1], self.imgs[:,:,:,band2])

        # Convert to uint16
        nd_minmax = np.load('../train/nd_b{}_b{}_minmax.npy'.format(band1, band2))
        nd_min, nd_max = nd_minmax[0], nd_minmax[1]
        nd = 65535 * (nd - nd_min) / (nd_max - nd_min)
        nd = nd.astype(np.int32)

        # Reshape
        nd = np.reshape(nd, np.append(np.asarray(nd.shape), 1))

        # Append nd to self.imgs
        self.imgs = np.append(self.imgs, nd, axis=3)


    def preprocess(self):
        self.imgs = np.moveaxis(self.imgs, [0, 1, 2, 3], [0, 3, 1, 2])
        self.imgs = self.imgs.astype(np.int32)

        # Add Gao NDWI
        self.add_nd(3, 11)

        # Add MNDWI
        self.add_nd(1, 11)

        # Add McFeeters NDWI band
        self.add_nd(1, 3)

        # Add NDVI band
        self.add_nd(3, 2)

        # Select bands
        self.imgs = self.imgs[:, :, :, BAND_SELECTION]

        # Apply scaling
        self.imgs = self.imgs.astype(np.float32)
        mean_std_array = np.load(self.mean_std_file)
        mean = np.array(mean_std_array[0])
        std = np.array(mean_std_array[1])
        self.imgs -= mean
        self.imgs /= std

        if self.resize_flag:
        # Resize and reshape
            new_imgs = np.zeros((self.imgs.shape[0], self.resize_dims[0],
                                self.resize_dims[1], self.imgs.shape[3]))

            for i in range(self.imgs.shape[0]):
                new_imgs[i] = transform.resize(self.imgs[i],
                                                (self.resize_dims[0], self.resize_dims[1],
                                                self.imgs.shape[3]),
                                                preserve_range=True)
            self.imgs = new_imgs


    def predict(self):
        self.preds = self.model.predict(self.imgs, 12, verbose=1)


    def write_images(self):
        self.preds[self.preds >= 0.5] = 255
        self.preds[self.preds < 0.5] = 0
        for i in range(self.batch_indices.shape[0]):
            outfile = '{}/pred_{}-{}.tif'.format(
                self.out_dir, self.batch_indices[i, 0], self.batch_indices[i, 1])

            new_dataset = rasterio.open(
                outfile, 'w', driver='GTiff',
                height=self.out_dims[1], width=self.out_dims[0],
                count=1, dtype='uint8', compress='lzw',
                crs=self.crs, nodata=0,
                transform=self.get_geotransform((self.batch_indices[i,1],
                                                 self.batch_indices[i,0]))
            )
            if self.resize_flag:
                pred = transform.resize(self.preds[i, :, :, 0],
                                        (self.out_dims[0], self.out_dims[1]),
                                        preserve_range=True)
            else:
                pred = self.preds[i,: ,: ,0]
            new_dataset.write(pred.astype('uint8'), 1)

#             # Save NDWI, predicted mask, and actual masks side by side
#             compare_filename = '{}/pred_{}-{}_results.png'.format(
#                 self.out_dir, self.batch_indices[i, 0], self.batch_indices[i, 1])
#             compare_im = 255 * np.ones((OUT_ROWS, OUT_COLS * 2 + 10), dtype=np.uint8)
#             ndwi_img = self.imgs[i,70:570, 70:570, len(BAND_SELECTION)-2]
#             if self.resize_flag:
#                 ndwi_img = transform.resize(ndwi_img,
#                                         (OG_ROWS, OG_COLS),
#                                         preserve_range = True)
#             ndwi_img = scale_image_tobyte(ndwi_img)
#             ndwi_img = ndwi_img.astype('uint8')
#             compare_im[0:OUT_ROWS, 0:OUT_COLS] = ndwi_img
#             compare_im[0:OUT_ROWS, (OUT_COLS + 10):] = pred
#             io.imsave(compare_filename, compare_im)


    def predict_write_batch(self):
        """Master method for loading, predicting, and writing full batch."""
        self.load_images()
        if self.imgs.shape[0] > 0:
            self.preprocess()
            self.predict()
            self.write_images()
        return self.batch_end_point


def get_done_list(out_dir):
    file_list = glob.glob(os.path.join(out_dir, 'pred_*.tif'))
    ind_strs = [re.findall(r'[0-9]+', f) for f in file_list]
    done_indices = np.asarray(ind_strs).astype(int)

    # Check for invalid ones (not in bounds)
    if os.path.isfile('invalid_indices.txt'):

        invalid_df = pd.read_csv('./invalid_indices.txt',
                                 header=None,names=['x','y'])
        invalid_list = np.array([invalid_df['x'].values,
                                 invalid_df['y'].values]
                                ).T
        if done_indices.shape[0] > 0:
            done_indices = np.vstack([done_indices, invalid_list])
        else:
            done_indices = invalid_list


    return done_indices


def prep_batches(source_path, model_structure, model_weights, done_ind):

    # Open primary image
    src = rasterio.open(source_path)
    total_rows, total_cols = src.height, src.width

    row_starts = np.arange(0, total_rows - OG_ROWS, OG_ROWS - OVERLAP)
    col_starts = np.arange(0, total_cols - OG_COLS, OG_COLS - OVERLAP)
#     row_starts = np.arange(97000, total_rows - OG_ROWS, OG_ROWS - OVERLAP)
#     col_starts = np.arange(87500, total_cols - OG_COLS, OG_COLS - OVERLAP)

    # Add final indices to row_starts and col_starts
    row_starts = np.append(row_starts, total_rows - OG_ROWS)
    col_starts = np.append(col_starts, total_cols - OG_COLS)

    # Create Nx2 array with row/col start indices
    start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

    # Eliminate already predicted indices
    if done_ind.shape[0]>0:
        todo_list = np.invert(np.in1d(
            start_ind.view('int, int'), done_ind.view('int, int')
        ))
        print('To do:', todo_list.shape)
        start_ind = start_ind[todo_list]


    # Open other images and create a list of srcs
    s1_10m_path = source_path.replace('s2_10m', 's1_10m')
    s2_20m_path = source_path.replace('s2_10m', 's2_20m')
    src_list = [
        src,
        rasterio.open(s1_10m_path),
        rasterio.open(s2_20m_path)]

    # Load model
    with open(model_structure, 'r') as struct_file:
        structure_json = struct_file.read()
    unet_model = models.model_from_json(structure_json)
    unet_model.load_weights(model_weights)

    return start_ind, unet_model, src_list


def predict_fullmap(source_path, model_structure, model_weights, mean_std_file,
                    out_dir):

    # Create output dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    done_indices = get_done_list(out_dir)

    start_ind, unet_model, img_srcs = prep_batches(
        source_path, model_structure, model_weights, done_indices)

    batch_start_point = 0
    while batch_start_point < start_ind.shape[0]:
        res_batch = ResPredictBatch(
            img_srcs=img_srcs, start_indices=start_ind,
            batch_size=BATCH_SIZE, batch_start_point=batch_start_point,
            dims=(OG_ROWS, OG_COLS), nbands=NBANDS,
            resize_flag=RESIZE_FLAG, resize_dims=(RESIZE_ROWS, RESIZE_COLS),
            out_dims=(OUT_ROWS, OUT_COLS), out_dir=out_dir,
            model=unet_model, mean_std_file=mean_std_file)
        batch_start_point = res_batch.predict_write_batch() + 1
        gc.collect()
#         K.clear_session()

        print('Done with batch, starting new from {}'.format(batch_start_point))

    return


def mosaic_predictions(tile_list, mosaiced_tif):
    """Mosaic output tiles into full raster."""

    tmpdir = tempfile.mkdtemp()
    tmpvrt = '{}/temp.vrt'.format(tmpdir)

    # Make vrt mosaic
    sp.call(['gdalbuildvrt', tmpvrt] + tile_list)

    # Translate it to a tif
    sp.call(['gdal_translate', '-co', 'COMPRESS=LZW', tmpvrt, mosaiced_tif])

    # Cleanup
    os.remove(tmpvrt)
    os.rmdir(tmpdir)

    return


def main():
    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()

    predict_fullmap(args.source_path, args.model_structure, args.model_weights,
                    args.mean_std_file, args.out_dir)

    if args.mosaic is not None:
        tile_list = glob.glob('{}/*.tif'.format(args.out_dir))
        mosaic_predictions(tile_list, args.mosaic)

    return


if __name__ == '__main__':
    main()

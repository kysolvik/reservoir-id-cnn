#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare images and masks for training and testing

These functions download and prepare the training/test images and their
labels/masks. Prepares them for ingestion into model.

Notes:
    Some functions/code taken from:
    https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/storage/cloud-client/snippets.py

"""


from google.cloud import storage
import pandas as pd
import os
import urllib.request
import numpy as np
from PIL import Image
from skimage import io
import random


def find_ims_masks(labelbox_json):
    """Finds locations of images and masks from labelbox-generated json."""

    label_df = pd.read_json(labelbox_json, orient='records')

    # URLs for original images
    og_urls = label_df['Labeled Data'].replace('ndwi.png', 'og.tif', regex=True)

    # URLs a for image masks
    mask_urls = [m.get('Impoundment') for m in label_df['Masks']]

    og_mask_tuples = zip(og_urls, mask_urls)

    return og_mask_tuples


def save_empty_mask(mask_path, dim_x, dim_y):
    """Writes an all-zeros mask file (png) given dim_x and dim_y"""

    mask_array = np.zeros([dim_x,dim_y], dtype='byte')
    im = Image.fromarray(mask_array)
    im.save(mask_path)

    return None


def download_im_mask_pair(og_url, mask_url, destination_dir,
                          dim_x=500, dim_y=500):
    """Downloads original image and mask, renaming mask to match image."""

    og_dest_file = '{}/{}'.format(destination_dir, os.path.basename(og_url))
    mask_dest_file = og_dest_file.replace('og.tif', 'mask.png')

    urllib.request.urlretrieve(og_url, filename=og_dest_file)

    if mask_url is None:
        save_empty_mask(mask_dest_file, dim_x, dim_y)
    else:
        urllib.request.urlretrieve(mask_url, filename=mask_dest_file)

    return None


def create_train_test_data(dim_x, dim_y, data_path='./data/', test_frac=0.2):
    """Save training and test data into easy .npy file"""
    images = os.listdir(data_path)
    total = int(len(images) / 2) # ksolvik: Needed to convert to int

    imgs = np.ndarray((total, dim_x, dim_y), dtype=np.uint16)
    imgs_mask = np.ndarray((total, dim_x, dim_y), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Loading images')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = imread(os.path.join(data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {}/{} images'.format(i, total))
        i += 1
    print('Loading done.')

    # Split into training, test.
    total_ims = imgs.shape[0]
    train_count = round(total_ims * (1 - test_frac))
    train_indices = random.sample(range(total_ims), train_count)

    imgs_train = imgs[train_indices]
    imgs_train_mask = imgs_mask[train_indices]
    imgs_test = np.delete(imgs, train_indices)
    imgs_test_mask = np.delete(imgs_mask, train_indices)

    np.save('imgs_train.npy', imgs_train)
    np.save('imgs_train_mask.npy', imgs_train_mask)
    np.save('imgs_test.npy', imgs_test)
    np.save('imgs_test_mask.npy', imgs_test_mask)

    print('Saving to .npy files done.')

    return


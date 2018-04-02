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

    mask_array = np.zeros([dim_x,dim_y], dtype=np.uint8)
    io.imsave(mask_path, mask_array)

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


def create_train_test_data(dim_x=500, dim_y=500, nbands=4, data_path='./data/',
                           test_frac=0.2):
    """Save training and test data into easy .npy file"""
    images = os.listdir(data_path)
    total = int(len(images) / 2) # ksolvik: Needed to convert to int

    imgs = np.ndarray((total, dim_x, dim_y, nbands), dtype=np.uint16)
    imgs_mask = np.ndarray((total, dim_x, dim_y), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Loading images')
    print('-'*30)
    og_img_names = []
    for image_name in images:
        if 'mask' in image_name:
            continue
        print(image_name)
        image_mask_name = image_name.replace('og.tif', 'mask.png')
        img = io.imread(os.path.join(data_path, image_name), as_grey=False)
        img_mask = io.imread(os.path.join(data_path, image_mask_name),
                             as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {}/{} images'.format(i, total))
        i += 1

        og_img_names += [image_name]

    print('Loading done.')

    # Split into training, test.
    total_ims = imgs.shape[0]
    train_count = round(total_ims * (1 - test_frac))
    train_indices = random.sample(range(total_ims), train_count)
    test_indices = np.delete(np.array(range(total_ims)), train_indices)

    imgs_train = imgs[train_indices]
    imgs_mask_train = imgs_mask[train_indices]
    train_img_names = [og_img_names[i] for i in train_indices]
    imgs_test = imgs[test_indices]
    imgs_mask_test = imgs_mask[test_indices]
    test_img_names = [og_img_names[i] for i in test_indices]

    prepped_path = '{}/prepped/'.format(data_path)
    if not os.path.isdir(prepped_path):
           os.makedirs(prepped_path)
    np.save('{}imgs_train.npy'.format(prepped_path), imgs_train)
    np.save('{}imgs_mask_train.npy'.format(prepped_path), imgs_mask_train)
    np.save('{}imgs_test.npy'.format(prepped_path), imgs_test)
    np.save('{}imgs_mask_test.npy'.format(prepped_path), imgs_mask_test)

    # Write image names
    with open('./train_names.csv', 'w') as wf:
        for img_name in train_img_names:
            wf.write('{}\n'.format(img_name))
    with open('./test_names.csv', 'w') as wf:
        for img_name in test_img_names:
            wf.write('{}\n'.format(img_name))

    print('Saving to .npy files done.')

    return


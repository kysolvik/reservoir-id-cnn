#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare images and masks for training and testing

These functions download and prepare the training/test images and their
labels/masks. Prepares them for ingestion into model.

"""


from google.cloud import storage
import pandas as pd
import subprocess as sp
import os
import urllib.request
import numpy as np
from PIL import Image
from skimage import io
import random
import argparse


def argparse_init():
    """Prepare ArgumentParser for inputs."""

    p = argparse.ArgumentParser(
            description='Prepare images and masks for training and testing.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('labelbox_json',
                   help='Path to LabelBox exported JSON.',
                   type=str)
    p.add_argument('--no_download',
                   help='Skip download step, just do ingestion.',
                   default=False,
                   action='store_true')
    return p


def find_ims_masks(labelbox_json):
    """Finds locations of images and masks from labelbox-generated json."""

    label_df = pd.read_json(labelbox_json, orient='records')
    label_df = label_df.loc[label_df['Masks'].notna()]

    # URLs for original images
    og_urls = label_df['Labeled Data'].replace('ndwi.png', 'og.tif', regex=True)

    # URLs a for image masks
    mask_urls = [m.get('Impoundment') for m in label_df['Masks']]

    og_mask_tuples = zip(og_urls, mask_urls)

    # Find the bucket name
    sample_og_url = og_urls[0]
    og_gs_path = sample_og_url.replace('https://storage.googleapis.com/', '')
    gs_bucket_name = og_gs_path.split('/')[0]

    return og_mask_tuples, gs_bucket_name


def save_empty_mask(mask_path, dim_x, dim_y):
    """Writes an all-zeros mask file (png) given dim_x and dim_y"""

    mask_array = np.zeros([dim_x,dim_y], dtype=np.uint8)
    io.imsave(mask_path, mask_array)

    return None


def download_im_mask_pair(og_url, mask_url, gs_bucket,
                          destination_dir='./data/', dim_x=500, dim_y=500):
    """Downloads original image and mask, renaming mask to match image."""

    og_dest_file = '{}/{}'.format(destination_dir, os.path.basename(og_url))
    mask_dest_file = og_dest_file.replace('og.tif', 'mask.png')

    # Download og file from google cloud storage using gsutil
    og_gs_path = og_url.replace('https://storage.googleapis.com/{}/'
                                .format(gs_bucket.name), '')
    blob = gs_bucket.blob(og_gs_path)
    blob.download_to_filename(og_dest_file)

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
    with open('{}train_names.csv'.format(prepped_path), 'w') as wf:
        for img_name in train_img_names:
            wf.write('{}\n'.format(img_name))
    with open('{}test_names.csv'.format(prepped_path), 'w') as wf:
        for img_name in test_img_names:
            wf.write('{}\n'.format(img_name))

    print('Saving to .npy files done.')

    return


def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()


    if not args.no_download:
        og_mask_tuples, gs_bucket_name = find_ims_masks(args.labelbox_json)

        # Initiate bucket, first stripping bucket name from URL.
        storage_client = storage.Client()
        gs_bucket = storage_client.get_bucket(gs_bucket_name)
        for og_mask_pair in og_mask_tuples:
            download_im_mask_pair(og_mask_pair[0], og_mask_pair[1], gs_bucket)

    create_train_test_data()
    return


if __name__=='__main__':
    main()

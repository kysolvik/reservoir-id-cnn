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
from skimage import morphology
import random
import argparse
import augment_data as augment
import glob

# Set random seed for
random.seed(5783)

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

    # URLS for
    s1_urls = label_df['Labeled Data'].replace(
        'ndwi.png', 's1_og.tif', regex=True)
    s2_20m_urls = label_df['Labeled Data'].replace(
        'ndwi.png', 's2_20m_og.tif', regex=True)

    # URLs a for image masks
    mask_urls = [m.get('Impoundment') for m in label_df['Masks']]

    og_mask_tuples = zip(og_urls, s1_urls, s2_20m_urls, mask_urls)

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


def normalized_diff(ar1, ar2):
    """Returns normalized difference of two arrays."""

    # Convert arrays to float32
    ar1 = ar1.astype('float32')
    ar2 = ar2.astype('float32')

    return((ar1 - ar2) / (ar1 + ar2))


def add_nd(imgs, band1, band2):
    """Add band containing NDWI."""

    nd = normalized_diff(imgs[:,:,:,band1], imgs[:,:,:,band2])

    # Convert to uint16
    nd_min = nd.min()
    nd_max = nd.max()
    nd = 65535 * (nd - nd_min) / (nd_max - nd_min)
    nd = nd.astype(np.uint16)

    # Reshape
    nd = np.reshape(nd, np.append(np.asarray(nd.shape), 1))

    # Append nd to imgs
    imgs_wnd = np.append(imgs, nd, axis=3)

    return imgs_wnd


def download_ims_mask_pair(og_urls, mask_url, gs_bucket,
                          destination_dir='./data/', dim_x=500, dim_y=500):
    """Downloads original image and mask, renaming mask to match image."""

    name_mask = True
    for im_url in og_urls:
        # Download og file from google cloud storage using gsutil
        og_dest_file = '{}/{}'.format(destination_dir, os.path.basename(im_url))
        og_gs_path = im_url.replace('https://storage.googleapis.com/{}/'
                                 .format(gs_bucket.name), '')
        blob = gs_bucket.blob(og_gs_path)
        blob.download_to_filename(og_dest_file)

        # Using first og url (sentinel 2 10m bands) name local mask file
        if name_mask:
            mask_dest_file = og_dest_file.replace('og.tif', 'mask.png')
            name_mask = False

    if mask_url is None:
        save_empty_mask(mask_dest_file, dim_x, dim_y)
    else:
        urllib.request.urlretrieve(mask_url, filename=mask_dest_file)

    return None


def pad_mask(img_mask):
    """To increase area of predicted res, dilate training masks."""

    img_mask_padded = morphology.binary_dilation(img_mask)

    return img_mask_padded


def create_train_test_data(dim_x=500, dim_y=500, nbands=12, data_path='./data/',
                           test_frac=0.2, val_frac=0.20):
    """Save training and test data into easy .npy file"""

    # Get mask image names and base image patterns
    mask_images = glob.glob('{}*mask.png'.format(data_path))
    image_patterns = [mi.replace('mask.png', '') for mi in mask_images]

    total_ims = len(mask_images)

    imgs = np.ndarray((total_ims, dim_x, dim_y, nbands), dtype=np.uint16)
    imgs_mask = np.ndarray((total_ims, dim_x, dim_y), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Loading images')
    print('-'*30)
    og_img_names = []
    for image_base in image_patterns:

        # Prep mask
        image_mask_name = '{}mask.png'.format(os.path.basename(image_base))
        img_mask = io.imread(os.path.join(data_path, image_mask_name),
                                as_grey=True)
        img_mask = np.array(img_mask)
        imgs_mask[i] = img_mask

        og_img_list = []
        for og_img in sorted(glob.glob('{}*og.tif'.format(image_base))):
            # Using sorted, the order is: s2 10m, s1 10m, s2 20m.
            img = io.imread(og_img, as_grey=False)
            img = np.array(img)
            og_img_list += [img]

        imgs[i] = np.dstack(og_img_list)

        if i % 100 == 0:
            print('Done: {}/{} images'.format(i, total_ims))
        i += 1

        og_img_names += [os.path.basename(og_img)]

    print('Loading done.')

    # Add  Gao NDWI
    imgs = add_nd(imgs, 3, 11)

    # Add  MNDWI
    imgs = add_nd(imgs, 1, 11)

    # Add McFeeters NDWI band
    imgs = add_nd(imgs, 1, 3)

    # Add NDVI band
    imgs = add_nd(imgs, 3, 2)


    # Split into training, validation, and test sets.
    train_count = round(total_ims * (1 - test_frac - val_frac))
    train_indices = random.sample(range(total_ims), train_count)
    test_val_indices = np.delete(np.array(range(total_ims)), train_indices)

    test_count = round(total_ims * test_frac)
    test_indices = random.sample(list(test_val_indices), test_count)
    val_indices = np.delete(np.array(range(total_ims)),
                            np.append(train_indices, test_indices))

    imgs_train = imgs[train_indices]
    imgs_mask_train = imgs_mask[train_indices]
    train_img_names = [og_img_names[i] for i in train_indices]
    imgs_test = imgs[test_indices]
    imgs_mask_test = imgs_mask[test_indices]
    test_img_names = [og_img_names[i] for i in test_indices]
    imgs_val = imgs[val_indices]
    imgs_mask_val = imgs_mask[val_indices]
    val_img_names = [og_img_names[i] for i in val_indices]

    # Augment training data
    new_imgs = np.zeros_like(imgs_train)
    new_masks = np.zeros_like(imgs_mask_train)
    for i in range(imgs_train.shape[0]):
        new_imgs[i], new_masks[i] = augment.random_aug(
            imgs_train[i],
            imgs_mask_train[i])
    new_imgs2 = np.zeros_like(imgs_train)
    new_masks2 = np.zeros_like(imgs_mask_train)
    for i in range(imgs_train.shape[0]):
        new_imgs2[i], new_masks2[i] = augment.random_aug(
            imgs_train[i],
            imgs_mask_train[i])
    imgs_train = np.vstack((imgs_train, new_imgs, new_imgs2))
    imgs_mask_train = np.vstack((imgs_mask_train, new_masks, new_masks2))

    # Write images
    prepped_path = '{}/prepped/'.format(data_path)
    if not os.path.isdir(prepped_path):
           os.makedirs(prepped_path)
    np.save('{}imgs_train.npy'.format(prepped_path), imgs_train)
    np.save('{}imgs_mask_train.npy'.format(prepped_path), imgs_mask_train)
    np.save('{}imgs_test.npy'.format(prepped_path), imgs_test)
    np.save('{}imgs_mask_test.npy'.format(prepped_path), imgs_mask_test)
    np.save('{}imgs_val.npy'.format(prepped_path), imgs_val)
    np.save('{}imgs_mask_val.npy'.format(prepped_path), imgs_mask_val)

    # Write image names
    with open('{}train_names.csv'.format(prepped_path), 'w') as wf:
        for img_name in train_img_names:
            wf.write('{}\n'.format(img_name))
    with open('{}test_names.csv'.format(prepped_path), 'w') as wf:
        for img_name in test_img_names:
            wf.write('{}\n'.format(img_name))
    with open('{}val_names.csv'.format(prepped_path), 'w') as wf:
        for img_name in val_img_names:
            wf.write('{}\n'.format(img_name))

    print('Saving to .npy files done.')

    return


def main():

    # Get command line args
    parser = argparse_init()
    args = parser.parse_args()


    if not args.no_download:
        og_mask_tuples, gs_bucket_name = find_ims_masks(args.labelbox_json)

        # Download imgs using Google Cloud Storage client
        storage_client = storage.Client()
        gs_bucket = storage_client.get_bucket(gs_bucket_name)
        for og_mask_pair in og_mask_tuples:
            download_ims_mask_pair(og_mask_pair[0:3], og_mask_pair[3],
                                   gs_bucket)

    create_train_test_data()
    return


if __name__=='__main__':
    main()

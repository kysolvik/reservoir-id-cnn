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


def find_ims_masks(labelbox_json):
    """Finds locations of images and masks from labelbox-generated json."""

    label_df = pd.read_json(labelbox_json, orient='records')
    
    # URLs for original images
    og_urls = label_df['Labeled Data'].replace('ndwi.png', 'og.tif', regex=True)

    # URLs a for image masks
    mask_urls = [m.get('Impoundment') for m in label_df['Masks']]

    og_mask_tuples = zip(og_urls, mask_urls)

    return og_mask_tuples


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


def save_empty_mask(mask_path, dim_x, dim_y):
    """Writes an all-zeros mask file (png) given dim_x and dim_y"""

    mask_array = np.zeros([dim_x,dim_y], dtype='byte')
    im = Image.fromarray(mask_array)
    im.save(mask_path)

    return None

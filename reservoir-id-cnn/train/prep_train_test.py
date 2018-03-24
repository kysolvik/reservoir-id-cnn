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


def find_ims_masks(labelbox_json):
    """Finds locations of images and masks from labelbox-generated json."""

    label_df = pd.read_json(labelbox_json, orient='records')
    
    # URLs for original images
    og_urls = label_df['Labeled Data'].replace('ndwi.png', 'og.tif', regex=True)

    # URLs a for image masks
    mask_urls = [m.get('Impoundment') for m in label_df['Masks']]

    og_mask_tuples = zip(og_urls, mask_urls)

    return og_mask_tuples


def download_im_mask_pair(og_url, mask_url, destination_dir):
    """Downloads original image and mask, renaming mask to match image."""

    og_dest_file = '{}/{}'.format(destination_dir, os.path.basename(og_url))
    urllib.request.urlretrieve(og_url, filename=og_dest_file)  
    
    if mask_url is not None:
        mask_dest_file = og_dest_file.replace('og.tif', 'mask.png')
        urllib.request.urlretrieve(mask_url, filename=mask_dest_file)

    return None

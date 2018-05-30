#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper for running predict_map on all Google Cloud Storage rasters

Example:
    $ python3 predict_wrapper.py

"""

import subprocess as sp
import predict_map
from google.cloud import storage
import os


def download_rast(gcs_rast):

    local_rast_path = './stage/{}'.format(os.path.basename(gcs_rast.name))
    gcs_rast.download_to_filename(local_rast_path)

    return(local_rast_path)


def predict_wrapper():

    storage_client = storage.Client()
    gs_bucket = storage_client.get_bucket('res-id')
    gcs_list = gs_bucket.list_blobs(prefix='ee_exports/sentinel/')

    for gcs_rast in gcs_list:
        local_rast = download_rast(gcs_rast)
        tile_out_dir = os.path.splitext(os.path.basename(local_rast))[0]

        predict_map.predict_fullmap(local_rast, '../train/unet_structure.txt',
                                    '../train/weights.h5',
                                    './out/{}'.format(tile_out_dir))
        os.remove(local_rast)

    return


def main():
    predict_wrapper()

if __name__ == '__main__':
    main()

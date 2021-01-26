#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper for running predict_map on full local vrts

Example:
    $ python3 predict_wrapper.py

"""

import predict_map


def predict_wrapper():

    predict_map.predict_fullmap('/mnt/disks/pred_data/vrts_20m_avg/s2_10m.tif',
                                './model_data/v3/unet_segmodels_10band.txt',
                                './model_data/v3/weights.h5',
                                './model_data/v3/mean_std.npy',
                                '/mnt/disks/pred_data/out_20m_avg/')
    return


def main():
    predict_wrapper()

if __name__ == '__main__':
    main()

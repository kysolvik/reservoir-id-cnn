#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper for running predict_map on full local vrts

Example:
    $ python3 predict_wrapper.py

"""

import predict_map


def predict_wrapper():

    predict_map.predict_fullmap('/mnt/disks/pred_data/vrts/s1_10m.vrt',
                                './model_data/v2/unet_structure.txt',
                                './model_data/v2/weights.h5',
                                './out/full/')
    return


def main():
    predict_wrapper()

if __name__ == '__main__':
    main()

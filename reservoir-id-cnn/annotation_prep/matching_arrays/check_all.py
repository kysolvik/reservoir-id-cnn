#!/usr/bin/env python3

import numpy as np
import os
import gdal
import glob

def check_diff(f):
    new_f = 'test_out/{}'.format(
        os.path.basename(f).replace('og', 'sent2_10m'))

    old_fh = gdal.Open(f)
    old_ar = old_fh.GetRasterBand(1).ReadAsArray()

    new_fh = gdal.Open(new_f)
    new_ar = new_fh.GetRasterBand(1).ReadAsArray()

    return np.sum(new_ar != old_ar)


f_list = glob.glob('../out_ims/all/og/*.tif')
tot_sum = 0
for f in f_list:
    diffs = check_diff(f)
    print(diffs)
    tot_sum += diffs

print(tot_sum)

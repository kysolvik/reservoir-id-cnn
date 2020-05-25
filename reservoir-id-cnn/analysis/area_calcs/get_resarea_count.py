#/usr/bin/env python

import gdal
import numpy
import skimage.measure
import sys

tif = sys.argv[1]

fh = gdal.Open(tif)
ar = fh.GetRasterBand(1).ReadAsArray()

# Get area
ar[ar>1] = 0
tot_pixels = numpy.sum(ar)
tot_area = tot_pixels * 900 * 0.0001
print('Total Area = {}'.format(tot_area))

# Get count
lab = skimage.measure.label(ar,connectivity=2)
res_num = numpy.max(lab)
print('Total Count = {}'.format(res_num))

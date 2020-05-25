#/usr/bin/env python3

import gdal
import numpy
import skimage.measure
import sys
import pandas as pd

tif = sys.argv[1]

fh = gdal.Open(tif)
ar = fh.GetRasterBand(1).ReadAsArray()
ar[ar>1] = 0

# Get count
lab = skimage.measure.label(ar,connectivity=2)
rp = skimage.measure.regionprops(lab)

for r in rp:
    if r.area == 1:
        print(r.centroid)
        print(r.area)


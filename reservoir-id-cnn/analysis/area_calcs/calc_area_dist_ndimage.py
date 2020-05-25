#!/usr/bin/env python3
""" Quick script to get counts and distribution of reservoir areas'"""

import gdal
import numpy as np
import sys
from scipy import ndimage

tif = sys.argv[1]
out_txt = sys.argv[2]
box_size = 10000

fh = gdal.Open(tif)


def get_count(ar):
    mask = ar == 255
    # Get count
    label_im, nb_labels = ndimage.label(mask,
                                    structure = [[1,1,1],[1,1,1],[1,1,1]])
    print('Count done')

    # Region props
    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))
    print('Reg Props done')
    return sizes

if box_size > 0:
    total_rows, total_cols = fh.RasterYSize, fh.RasterXSize
    current_row = 0
    current_col = 0
    row_starts = np.arange(0, total_rows, box_size)
    col_starts = np.arange(0, total_cols, box_size)

    # Create Nx2 array with row/col start indices
    start_ind = np.array(np.meshgrid(row_starts, col_starts)).T.reshape(-1, 2)

    for i in range(start_ind.shape[0]):
        # For the indices near edge we need to use a smaller box size
        box_size_rows = min(total_rows - start_ind[i,0], box_size)
        box_size_cols = min(total_cols - start_ind[i,1], box_size)
        ar = fh.GetRasterBand(1).ReadAsArray(
            int(start_ind[i, 1]), int(start_ind[i,0]),
            int(box_size_cols),int(box_size_rows))
        sizes = get_count(ar)
        with open(out_txt, 'a') as f:
            for item in sizes:
                f.write("%s\n" % int(item))

else:
    ar = fh.GetRasterBand(1).ReadAsArray()
    print('Array read')
    sizes = get_count(mask)
    with open(out_txt, 'w') as f:
        for item in sizes:
            f.write("%s\n" % int(item))

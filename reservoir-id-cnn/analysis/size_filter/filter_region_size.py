#!/usr/bin/env python3
""" Quick script to get counts and distribution of reservoir areas'"""

import gdal
import numpy as np
import sys
from scipy import ndimage
import affine
import rasterio as rio
import os

tif = sys.argv[1]
out_dir = sys.argv[2]
box_size = 20000
MIN_SIZE = 1250

src = rio.open(tif)


def filter_by_size(ar, min_size=10000):
    mask = ar == 255
    # Label regions
    label_im, nb_labels = ndimage.label(mask,
                                    structure = [[1,1,1],[1,1,1],[1,1,1]])

    # Get sizes of regions
    sizes = np.array(ndimage.sum(mask, label_im, range(1,nb_labels + 1)))

    # Find which regions are outside of our size range
    labels_to_rm = np.arange(sizes.shape[0])[(sizes < min_size)] + 1

    # Copy input array and set regions outside of size range to 0
    out = ar.copy()
    rm_mask = np.isin(label_im, labels_to_rm)
    out[rm_mask] = 0

    return out

def get_geotransform(indice_pair, src):
    """Calculate geotransform of a tile.

    Notes:
        Using .affine instead of .transform because it should work with all
        rasterio > 0.9. See https://github.com/mapbox/rasterio/issues/86.

    Args:
        indice_pair (tuple): Row, Col indices of upper left corner of tile.

    """
    geo = src.transform
    new_ul = [geo[2] + indice_pair[0]*geo[0] + indice_pair[1]*geo[1],
                geo[5] + indice_pair[0]*geo[3] + indice_pair[1]*geo[4]]

    new_affine = affine.Affine(geo[0], geo[1], new_ul[0],
                                geo[3], geo[4], new_ul[1])

    return new_affine

def write_output(out, row, col, src, out_dir):
    outfile = os.path.join(out_dir, 'filter_{}_{}.tif'.format(row, col))

    new_dataset = rio.open(
        outfile, 'w', driver='GTiff',
        height=box_size, width=box_size,
        count=1, dtype='uint8', compress='lzw',
        crs=src.crs, nodata=0,
        transform=get_geotransform((col,row), src)
    )
    new_dataset.write(out.astype('uint8'), 1)


def main():

    if box_size > 0:
        total_rows, total_cols = src.height, src.width
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
            ar = src.read(indexes=1,
                          window = ((start_ind[i,0], start_ind[i,0]+box_size_rows),
                                    (start_ind[i,1], start_ind[i,1]+box_size_cols))
                          )
            print(ar.shape)
            if ar.max()>0:
                out = filter_by_size(ar,min_size=MIN_SIZE)
                write_output(out, start_ind[i,0], start_ind[i,1], src, out_dir)

    else:
        ar = src.read()
        out = filter_by_size(ar, min_size=MIN_SIZE)
        write_output(out,0,0 , src, out_dir)

if __name__ == '__main__':
    main()

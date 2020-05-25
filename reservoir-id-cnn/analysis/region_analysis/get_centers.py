#/usr/bin/env python
"""
Get the center coordinates for all train/test/val images

The *s2_20m_og.tif files must in the target data dir
"""

import rasterio as rio
import glob
import os
import pandas as pd
import numpy as np


DATA_DIR = '../../train/data/'
DATA_TAIL = '*_s2_20m_og.tif'

def get_center(file_path):
    dataset = rio.open(file_path)
    bounds = dataset.bounds
    center_lon = (bounds.left + bounds.right)/2
    center_lat = (bounds.bottom + bounds.top)/2
    return center_lon, center_lat


def main():
    file_list = glob.glob(os.path.join(DATA_DIR, DATA_TAIL))
    file_names = [os.path.basename(f) for f in file_list]
    centers = np.array([get_center(file_path) for file_path in file_list])

    centers_df = pd.DataFrame({'filename': file_names,
                               'center_lon':centers[:,0],
                               'center_lat':centers[:,1]})
    centers_df.to_csv('./centers.csv', index=False)


if __name__ == "__main__":
    main()


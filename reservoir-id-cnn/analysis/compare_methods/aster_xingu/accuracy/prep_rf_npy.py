"""
Prep npy file with all RF images
"""

import numpy as np
import glob
import pandas as pd
from skimage import io
import os

rf_img_dir = './training_images/'
train_names_path = '../../../../train/data/prepped/train_names.csv'
test_names_path = '../../../../train/data/prepped/test_names.csv'

def main():
    img_list = glob.glob('./training_images/*.tif')

    train_names = pd.read_csv(train_names_path, names=['name'])
    test_names = pd.read_csv(test_names_path, names=['name'])

    rf_img_basenames = [os.path.basename(img) for img in img_list]
    rf_train_names = pd.merge(train_names, pd.DataFrame({'name':rf_img_basenames}), on='name', how='inner')
    rf_test_names = pd.merge(test_names, pd.DataFrame({'name':rf_img_basenames}), on='name', how='inner')

    rf_train_names.to_csv('./csvs/rf_train_names.csv')
    rf_test_names.to_csv('./csvs/rf_test_names.csv')

    rf_train_imgs = np.ndarray((rf_train_names.shape[0], 500, 500), dtype=np.uint8)
    rf_test_imgs = np.ndarray((rf_test_names.shape[0], 500, 500), dtype=np.uint8)
    for i in range(rf_train_names.shape[0]): 
        rf_train_imgs[i] = io.imread(os.path.join(rf_img_dir, rf_train_names['name'][i]))

    np.save('./training_images/rf_train_predict.npy',rf_train_imgs)

    for i in range(rf_test_names.shape[0]): 
        rf_test_imgs[i] = io.imread(os.path.join(rf_img_dir,rf_test_names['name'][i]))

    np.save('./training_images/rf_test_predict.npy', rf_test_imgs)



if __name__=='__main__':
    main()



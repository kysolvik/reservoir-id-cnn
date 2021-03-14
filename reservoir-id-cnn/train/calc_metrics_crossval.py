#!/usr/bin/env python3
"""Train U-Net on reservoir images

Example:
    python3 train.py

Notes:
    Must be run from reservoir-id-cnn/train/
    Prepped data should be in the: ./data/prepped/ directory

"""


import os
import numpy as np
import math
from keras import models
from sklearn.model_selection import KFold, train_test_split
from segmentation_models import get_preprocessing
import pandas as pd

model_structure = './unet_segmodels_10band.txt'

seed_value= 587

# Set batch szie
BATCH_SIZE=8

NUM_FLOODPLAINS=103

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

def remove_floodplains(imgs_ar, num_floodplains):
    """Splits floodplains and not floodplains into separate arrays"""
    return imgs_ar[-num_floodplains:], imgs_ar[:-num_floodplains]

def preprocess_mask(masks, mask_crop=0):
    """Preprocess imgs and masks, returning preprocessed copies"""
    masks = np.expand_dims(masks, 3)

    masks = masks.astype('float32')

    masks /= 255.  # scale masks to [0, 1]
    masks[masks >= 0.5] = 1
    masks[masks < 0.5] = 0

    # Crop Mask:
    return masks

def eval(band_selection):
    """Master function for training"""
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    # Prep train
    imgs_mask_train = np.load('./data/prepped/imgs_mask_train.npy')
    names_train = pd.read_csv('./data/prepped/train_names.csv',
                              header=None, names=['name'])
    imgs_mask_train = preprocess_mask(imgs_mask_train,
                                 mask_crop=0)

    # Separate aug and non-aug
    num_train = int(imgs_mask_train.shape[0]/2)
    imgs_mask_train_aug, imgs_mask_train = imgs_mask_train[num_train:], imgs_mask_train[:num_train]

    # Remove floodplains
    imgs_mask_train_fp, imgs_mask_train = remove_floodplains(imgs_mask_train, NUM_FLOODPLAINS)
    imgs_mask_train_aug_fp, imgs_mask_train_aug = remove_floodplains(imgs_mask_train_aug, NUM_FLOODPLAINS)
    print('Should be 0: ', imgs_mask_train_fp.max(), imgs_mask_train_aug_fp.max())

    # Prep K-fold cross-val
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed_value)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    cvscores = {
        'iou':[],
        'f1':[],
        'precision':[],
        'recall':[]
    }
    train_test_splits = [s for s in kfold.split(imgs_mask_train)]
    for i in range(len(train_test_splits)):
        # Split up train and test
        # Train set contains augmented AND floodplains data
        # Val set contains no augmented data, but has FP
        # Test set contains no augmented OR FP
        test = train_test_splits[i][1]

        # Test
        y_test = imgs_mask_train[test]
        names_test = names_train.iloc[test]
        names_test.to_csv('./data/prepped/test_names_{}.csv'.format(i), index=False)
        pred = np.load('./data/predict/predict_{}.npy'.format(i))

        print(pred.max())
        print(y_test.max())
        pred[pred>0.5] = 255
        pred[pred<0.5] = 0
        y_test[y_test>0.5] = 255
        y_test[y_test<0.5] = 0
        print(np.unique(y_test))
        print(np.unique(pred))
        pred_sums = np.sum(pred==255, axis=(1,2,3))
        mask_sums = np.sum(y_test==255, axis=(1,2,3))
        print(np.sum(mask_sums>2000))
        print(names_test.iloc[mask_sums>2000])
#         pred = pred[mask_sums<2000]
#         y_test = y_test[mask_sums<2000]
        print(np.mean(mask_sums))
        if i ==0:
            print(names_test)

        # Calc rates
        true_pos = np.sum((pred==255)*(y_test==255))
        false_pos = np.sum((pred==255)*(y_test==0))
        true_neg = np.sum((pred==0)*(y_test==0))
        false_neg = np.sum((pred==0)*(y_test==255))

        # Calc metrics
        precision = true_pos/(true_pos + false_pos)
        recall = true_pos/(true_pos + false_neg)
        pfa = false_pos/(true_pos + false_pos)
        pmd = false_neg/(true_pos + false_neg)
        agree = true_pos + true_neg
        total = (true_pos + false_pos + false_neg + true_neg)
        pcc = (agree)/(total)
        exp_a =  ((((true_pos+false_pos)*(true_pos+false_neg))+
                ((false_pos+true_neg)*(false_neg+true_neg)))/
                total
                )
        kappa = (agree - exp_a)/(total - exp_a)
        f1 = 2*(precision*recall)/(precision+recall)
        iou = true_pos/(true_pos + false_pos + false_neg)
        print('agree, exp_a, kappa',agree, exp_a, kappa)
        print('pfa, pmd, pcc', pfa, pmd, pcc)
        print('prec, rec, f1, iou, tp, fp, tn, fn',precision, recall, f1, iou, true_pos, false_pos, true_neg, false_neg)


if __name__=='__main__':
    eval([0, 1, 2, 3, 4, 5, 12, 13, 14, 15])


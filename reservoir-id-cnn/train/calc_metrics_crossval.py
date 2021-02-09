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

model_structure = './unet_segmodels_10band.txt'

seed_value= 584

# Set batch szie
BATCH_SIZE=8

NUM_FLOODPLAINS=103

BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)

def preprocess(imgs, masks, band_selection, mask_crop=0):
    """Preprocess imgs and masks, returning preprocessed copies"""
    num_bands = len(band_selection)
    # Select target bands
    imgs = imgs[:, :, :, band_selection]

    masks = np.expand_dims(masks, 3)

    imgs = imgs.astype('float32')
    masks = masks.astype('float32')

    masks /= 255.  # scale masks to [0, 1]
    masks[masks >= 0.5] = 1
    masks[masks < 0.5] = 0

    # Crop Mask:
    if mask_crop!=0:
        masks = masks[:,mask_crop:(-1*mask_crop), mask_crop:(-1*mask_crop)]
    return imgs, masks


def eval(band_selection):
    """Master function for training"""
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    num_bands = len(band_selection)

    # Prep train
    imgs_train = np.load('./data/prepped/imgs_train.npy')
    imgs_mask_train = np.load('./data/prepped/imgs_mask_train.npy')
    imgs_train, imgs_mask_train = preprocess(imgs_train, imgs_mask_train,
                                             band_selection, mask_crop=0)

    # Remove floodplains
    imgs_train_fp, imgs_train = imgs_train[-NUM_FLOODPLAINS:], imgs_train[:-NUM_FLOODPLAINS]
    imgs_mask_train_fp, imgs_mask_train = imgs_mask_train[-NUM_FLOODPLAINS:], imgs_mask_train[:-NUM_FLOODPLAINS]

    # Separate aug and non-aug
    num_train = int(imgs_train.shape[0]/2)
    imgs_mask_train_aug, imgs_mask_train = imgs_mask_train[num_train:], imgs_mask_train[:num_train]
    imgs_train_aug, imgs_train = imgs_train[num_train:], imgs_train[:num_train]

    # Prep K-fold cross-val
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed_value)

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    i=0
    cvscores = {
        'iou':[],
        'f1':[],
        'precision':[],
        'recall':[]
    }
    for train, test in kfold.split(imgs_train, imgs_mask_train):
        x_test = imgs_train[test]
        y_test = imgs_mask_train[test]


        if not os.path.exists('./data/predict/predict_{}.npy'.format(i)):
            # Scale imgs based on train mean and std
            mean_std_array = np.load('./mean_std_{}.npy'.format(i))
            mean = np.array(mean_std_array[0])
            std = np.array(mean_std_array[1])

            # Preprocess, scale val and test
            x_test -= mean
            x_test /= std
            x_test = preprocess_input(x_test)

            # Check on shapes
            print(x_test.shape)
            print(y_test.shape)

            num_bands = len(band_selection)

            # Load model
            with open(model_structure, 'r') as struct_file:
                structure_json = struct_file.read()
            unet_model = models.model_from_json(structure_json)
            unet_model.load_weights('./weights_{}.h5'.format(i))

            # Evaluate
            pred = unet_model.predict(x_test, 8,verbose=1)
            np.save('./data/predict/predict_{}.npy'.format(i), pred)
        else:
            pred = np.load('./data/predict/predict_{}.npy'.format(i))
        print(pred.max())
        print(y_test.max())
        pred[pred>0.5] = 255
        pred[pred<0.5] = 0
        y_test[y_test>0.5] = 255
        y_test[y_test<0.5] = 0

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

        i+=1


if __name__=='__main__':
    eval([0, 1, 2, 3, 4, 5, 12, 13, 14, 15])


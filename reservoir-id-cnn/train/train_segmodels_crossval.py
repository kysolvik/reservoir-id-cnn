#!/usr/bin/env python3
"""Train U-Net on reservoir images

Example:
    python3 train.py

Notes:
    Must be run from reservoir-id-cnn/train/
    Prepped data should be in the: ./data/prepped/ directory

"""


import os
from skimage import transform
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Cropping2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from skimage import io
import loss_functions as lf
import math
from keras.callbacks import LearningRateScheduler
from segmentation_models import Unet
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss, DiceLoss
from segmentation_models.metrics import iou_score, f1_score, precision, recall
from sklearn.model_selection import KFold, train_test_split

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 587

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.set_random_seed(seed_value)
# for later versions:
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
from keras import backend as K
K.set_session(sess)

BACKBONE = 'resnet34'
loss = DiceLoss(class_weights=np.array([1,1]))
preprocess_input = get_preprocessing(BACKBONE)

BATCH_SIZE=8
# Set batch szie
OG_ROWS = 500
OG_COLS = 500
# Original image dimensions
# RESIZE_ROWS = 512
# RESIZE_COLS = 512
TIF_ROWS = 640
TIF_COLS = 640
CROP_SIZE = int((TIF_ROWS - OG_ROWS)/2)
# Dimensions of inputs (non-masks)
PRED_THRESHOLD = 0.5
# Prediction threshold. > PRED_THRESHOLD will be classified as res.
VAL = False
# Includes separate validation set (separate from test)
NUM_FLOODPLAINS=103


def scale_image_tobyte(ar):
    """Scale larger data type array to byte"""
    min_val = np.min(ar)
    max_val = np.max(ar)
    byte_ar = (np.round(255.0 * (ar - min_val) / (max_val - min_val))
               .astype(np.uint8))
    byte_ar[ar == 0] = 0

    return(byte_ar)


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


def remove_floodplains(imgs_ar, num_floodplains):
    """Splits floodplains and not floodplains into separate arrays"""
    return imgs_ar[-num_floodplains:], imgs_ar[:-num_floodplains]

def train(learn_rate, loss_func, band_selection, val, epochs=200):
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

    # Separate aug and non-aug
    num_train = int(imgs_train.shape[0]/2)
    imgs_mask_train_aug, imgs_mask_train = imgs_mask_train[num_train:], imgs_mask_train[:num_train]
    imgs_train_aug, imgs_train = imgs_train[num_train:], imgs_train[:num_train]

    # Remove floodplains
    imgs_train_fp, imgs_train = remove_floodplains(imgs_train, NUM_FLOODPLAINS)
    imgs_mask_train_fp, imgs_mask_train = remove_floodplains(imgs_mask_train, NUM_FLOODPLAINS)
    imgs_train_aug_fp, imgs_train_aug = remove_floodplains(imgs_train_aug, NUM_FLOODPLAINS)
    imgs_mask_train_aug_fp, imgs_mask_train_aug = remove_floodplains(imgs_mask_train_aug, NUM_FLOODPLAINS)
    print('Should be 0: ', imgs_mask_train_fp.max(), imgs_mask_train_aug_fp.max())
    assert imgs_train_aug.shape[0] == imgs_train.shape[0]
    assert imgs_train_aug.shape[0] == imgs_train.shape[0]

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
    train_test_splits = [s for s in kfold.split(imgs_train)]
    fp_train_test_splits = [s for s in kfold.split(imgs_train_fp)]
    for i in range(len(train_test_splits)):
        # Split up train and test
        # Train set contains augmented AND floodplains data
        # Val set contains no augmented data, but has FP
        # Test set contains no augmented OR FP
        test = train_test_splits[i][1]
        val = train_test_splits[i-1][1]
        train = np.delete(train, np.vstack([val,test]),axis=0)
        train = train_test_splits[i][0]
        train_fp = fp_train_test_splits[i][0]
        val_fp = fp_train_test_splits[i][1]
        print(train.shape, val.shape, test.shape)

        # Train
        x_train = np.vstack([imgs_train[train],
                             imgs_train_fp[train_fp],
                             imgs_train_aug[train],
                             imgs_train_aug_fp[train_fp]])
        y_train = np.vstack([imgs_mask_train[train],
                             imgs_mask_train_fp[train_fp],
                             imgs_mask_train_aug[train],
                             imgs_mask_train_aug_fp[train_fp]])

        # Val
        x_val = np.vstack([imgs_train[val],
                             imgs_train_fp[val_fp]])
        y_val = np.vstack([imgs_mask_train[val],
                             imgs_mask_train_fp[val_fp]])

        # Test
        x_test = imgs_train[test]
        y_test = imgs_mask_train[test]


        # Check shapes
        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)
        print(x_test.shape, y_test.shape)

        # Scale imgs based on train mean and std
        mean = np.mean(x_train, axis=(0,1,2), dtype='float64')  # mean for data centering
        std = np.std(x_train, axis=(0,1,2),dtype='float64')  # std for data normalization
        print(mean, std)
        np.save('mean_std_{}.npy'.format(i), np.vstack((mean, std)))
        x_train -= mean
        x_train /= std
        print(x_train[...,0].mean())

        # Preprocess, scale val and test
        x_val -= mean
        x_val /= std
        x_val = preprocess_input(x_val)
        x_test -= mean
        x_test /= std
        x_test = preprocess_input(x_test)

        # Check on shapes (again)
        print(x_train.shape, y_train.shape)
        print(x_val.shape, y_val.shape)
        print(x_test.shape, y_test.shape)

        num_bands = len(band_selection)

        base_model = Unet(backbone_name=BACKBONE, encoder_weights=None, input_shape=(None, None, num_bands))
        output = Cropping2D(cropping=(CROP_SIZE, CROP_SIZE))(base_model.layers[-1].output)
        model = Model(base_model.inputs, output, name=base_model.name)
        optimizer = Adam(lr=learn_rate, decay=2E-3)
        model.compile(optimizer, loss=loss, metrics=[iou_score, f1_score, precision, recall])

        model_checkpoint = ModelCheckpoint('weights_{}.h5'.format(i),
                                           save_best_only=True,
                                           save_weights_only=True)
        early_stopping = EarlyStopping(monitor='val_iou_score', min_delta=0, patience=20,
                                    verbose=0, mode='max')
        model.fit(
                x=x_train,
                y=y_train,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                validation_data=(x_val, y_val),
                verbose=2,
                callbacks=[model_checkpoint, early_stopping],
            shuffle=True
        )

        model.load_weights('weights_{}.h5'.format(i))

        scores = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
        pred = model.predict(x_test, batch_size=BATCH_SIZE)
        np.save('./data/predict/predict_{}.npy'.format(i), pred)
        cvscores['iou'].append(scores[1]*100)
        cvscores['f1'].append(scores[2]*100)
        cvscores['precision'].append(scores[3]*100)
        cvscores['recall'].append(scores[4]*100)
        print('CV So Far:', cvscores)

        i+=1
    print('CV Final:', cvscores)


if __name__=='__main__':
    train(2.5E-4, lf.dice_coef_loss, [0, 1, 2, 3, 4, 5, 12, 13, 14, 15],
          val=VAL, epochs=100)

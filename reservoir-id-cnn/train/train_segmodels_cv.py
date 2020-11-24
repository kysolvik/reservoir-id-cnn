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
import tensorflow as tf
from tensorflow.keras.models import Model
import skimage
from tensorflow.keras.layers import Cropping2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
# from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import KFold
import zca_whiten as zca
import segmentation_models as sm
sm.set_framework('tf.keras')

# Seed value
# Apparently you may use different seed values at each stage
seed_value = 584

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
tf.random.set_seed(seed_value)

# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
    config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

BACKBONE = 'resnet34'
loss = sm.losses.DiceLoss(class_weights=np.array([1,2]))
preprocess_input = sm.get_preprocessing(BACKBONE)

BATCH_SIZE=12
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


def preprocess(imgs, masks, mask_crop=0):
    """Preprocess imgs and masks, returning preprocessed copies"""
    # Select target bands
    masks = np.expand_dims(masks, 3)

    imgs = imgs.astype('float32')
    masks = masks.astype('float32')

    masks /= 255.  # scale masks to [0, 1]
    masks[masks >= 0.5] = 1
    masks[masks < 0.5] = 0

    # Crop Mask:
    if mask_crop!=0:
        masks = masks[:, mask_crop:(-1*mask_crop), mask_crop:(-1*mask_crop)]
    return imgs, masks


def prep_and_split_kfold(band_selection, train, test, withhold_fp=False):

    # Prep train
    imgs_train = np.load('./data/prepped/imgs_train.npy'
                         )[:, :, :, band_selection]
    imgs_mask_train = np.load('./data/prepped/imgs_mask_train.npy')
    imgs_train, imgs_mask_train = preprocess(imgs_train, imgs_mask_train,
                                             mask_crop=0)

    # Separate aug and non-aug
    num_train = int(imgs_train.shape[0]/2)
    imgs_mask_train_aug, imgs_mask_train = imgs_mask_train[num_train:], imgs_mask_train[:num_train]
    imgs_train_aug, imgs_train = imgs_train[num_train:], imgs_train[:num_train]

    # Remove floodplains
    if withhold_fp:
        imgs_train_fp, imgs_train = imgs_train[-51:], imgs_train[:-51]
        imgs_mask_train_fp, imgs_mask_train = imgs_mask_train[-51:], imgs_mask_train[:-51]
        imgs_train_aug_fp, imgs_train_aug = imgs_train_aug[-51:], imgs_train_aug[:-51]
        imgs_mask_train_aug_fp, imgs_mask_train_aug = imgs_mask_train_aug[-51:], imgs_mask_train_aug[:-51]
        print(imgs_train_fp.shape)
        print(imgs_train_aug_fp.shape)

    # Stack into train sets
    x_train = np.vstack([imgs_train[train], imgs_train_aug[train]])
    y_train = np.vstack([imgs_mask_train[train], imgs_mask_train_aug[train]])

    # Add fp if they were withheld
    if withhold_fp:
        x_train = np.vstack([x_train, imgs_train_fp, imgs_train_aug_fp])
        y_train = np.vstack([y_train, imgs_mask_train_fp, imgs_mask_train_aug_fp])

    # Create test sets
    x_test = imgs_train[test]
    y_test = imgs_mask_train[test]

    return (x_train.astype('float32'), y_train.astype('float32'),
            x_test.astype('float32'), y_test.astype('float32'))


def final_augs(x_train, y_train, x_test, y_test,
               zca_whiten=False, gauss_noise=False):
    # Scale imgs based on train mean and std
    mean = np.mean(x_train, axis=(0, 1, 2))  # mean for data centering
    std = np.std(x_train, axis=(0, 1, 2))  # std for data normalization
    x_train -= mean
    x_train /= std
    x_train = preprocess_input(x_train)

    # Preprocess, scale val
    x_test -= mean
    x_test /= std
    x_test = preprocess_input(x_test)

    # ZCA whiten
    if zca_whiten:
        print('ZCA Whitening')
        x_train = zca.zca_whiten(x_train)
        x_test = zca.zca_whiten(x_test)

    # Gauss noise
    if gauss_noise:
        print('Gaussian noise')
        x_train = np.array([
            skimage.util.random_noise(img, mode='gaussian', var=0.01, clip=False)
            for img in x_train
        ])
        x_test = np.array([
            skimage.util.random_noise(img, mode='gaussian', var=0.01, clip=False)
            for img in x_test
        ])

    return (x_train.astype('float32'), y_train.astype('float32'),
            x_test.astype('float32'), y_test.astype('float32'))



def train(learn_rate, loss_func, band_selection, epochs=50,
          zca_whiten=False, gauss_noise=False, withhold_fp=False):
    """Master function for training"""
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    num_bands = len(band_selection)

    # Prep K-fold cross-val
    num_imgs = np.load('./data/prepped/imgs_mask_train.npy').shape[0]/2
    if withhold_fp:
        num_imgs -= 51
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed_value)

    i=0
    cvscores = {
        'iou':[],
        'f1':[],
        'precision':[],
        'recall':[]
    }
    for train, test in kfold.split(np.arange(num_imgs), np.arange(num_imgs)):
        print('-'*30)
        print('Starting Kfold {}'.format(i))
        print('-'*30)
        num_bands = len(band_selection)

        # perform split and final augmentations/scaling
        x_train, y_train, x_test, y_test = prep_and_split_kfold(
            band_selection, train, test, withhold_fp=True)
        x_train, y_train, x_test, y_test = final_augs(
            x_train, y_train, x_test, y_test,
            zca_whiten=zca_whiten, gauss_noise=gauss_noise)
        print(x_train.dtype, y_train.dtype, x_test.dtype, y_test.dtype)

        base_model = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, input_shape=(None, None, num_bands))
        output = Cropping2D(cropping=(CROP_SIZE, CROP_SIZE))(base_model.layers[-1].output)
        model = Model(base_model.inputs, output, name=base_model.name)
        print(model.summary())

        optimizer = Adam(learning_rate=learn_rate, decay=5E-4)

        model.compile(optimizer, loss=loss, metrics=[
            sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,
            sm.metrics.f1_score])

        model_checkpoint = ModelCheckpoint('weights_{}.h5'.format(i),
                                           save_best_only=True,
                                           monitor='iou_score',
                                           mode='max',
                                           save_weights_only=True)
        print('Train = {}'.format(x_train.shape))
        print('Val = {}'.format(x_test.shape))
        model.fit(
                x=x_train,
                y=y_train,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                validation_data=(x_test, y_test),
                verbose=2,
                callbacks=[model_checkpoint],
                shuffle=True
        )

        scores = model.evaluate(x_test, y_test, verbose=0)
        cvscores['iou'].append(scores[1]*100)
        cvscores['f1'].append(scores[2]*100)
        cvscores['precision'].append(scores[3]*100)
        cvscores['recall'].append(scores[4]*100)

        tf.keras.backend.clear_session()
        i+=1

        break


if __name__=='__main__':
    train(1E-4, [0, 1, 2, 3, 4, 5, 12, 13, 14, 15],
          epochs=60, withhold_fp=True,
          zca_whiten=True, gauss_noise=False)

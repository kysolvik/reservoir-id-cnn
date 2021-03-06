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

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 583

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
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

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 10.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

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
VAL = True
# Includes separate validation set (separate from test)


def scale_image_tobyte(ar):
    """Scale larger data type array to byte"""
    min_val = np.min(ar)
    max_val = np.max(ar)
    byte_ar = (np.round(255.0 * (ar - min_val) / (max_val - min_val))
               .astype(np.uint8))
    byte_ar[ar == 0] = 0

    return(byte_ar)


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2*((prec*rec)/(prec+rec+K.epsilon()))


def get_unet(img_rows, img_cols, nbands, loss_func, learn_rate, crop=0):
    """U-Net Structure

    @author: jocicmarko
    @url: https://github.com/jocicmarko/ultrasound-nerve-segmentation
    """

    inputs = Input((img_rows, img_cols, nbands))
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2),
                                       padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2),
                                       padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2),
                                       padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2),
                                       padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    crop10 = Cropping2D(cropping=(crop, crop))(conv10)

    model = Model(inputs=[inputs], outputs=[crop10])

#     lr_schedule = ExponentialDecay(
#             learn_rate,
#             decay_steps=10,
#             decay_rate=0.8,
#             staircase=True)

    model.compile(optimizer=Adam(lr=learn_rate, decay=1E-4),
                  loss=loss_func,
                  metrics=[lf.jaccard_coef, lf.dice_coef,
                           precision, recall, f1])
    # Save structure
    model_json = model.to_json()
    with open("unet_10band.txt", 'w') as outfile:
        outfile.write(model_json)

    print(model.summary())

    return model


# def resize_imgs(imgs, nbands):
#     """Resize numpy array of images"""
#     imgs_p = np.ndarray((imgs.shape[0], RESIZE_ROWS, RESIZE_COLS, nbands))
#
#     for i in range(imgs.shape[0]):
#         imgs_p[i] = transform.resize(imgs[i],
#                                      (RESIZE_ROWS, RESIZE_COLS, nbands),
#                                      preserve_range = True)
#
#     return imgs_p


def preprocess(imgs, masks, band_selection, mask_crop=0):
    """Preprocess imgs and masks, returning preprocessed copies"""
    num_bands = len(band_selection)
    # Select target bands
    imgs = imgs[:, :, :, band_selection]

#     imgs = resize_imgs(imgs, num_bands)
#     masks = resize_imgs(masks, 1)
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


def train(learn_rate, loss_func, band_selection, val):
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

    # Scale imgs based on train mean and std
    mean = np.mean(imgs_train, axis=(0,1,2))  # mean for data centering
    std = np.std(imgs_train, axis=(0,1,2))  # std for data normalization
    np.save('mean_std.npy', np.vstack((mean, std)))
    imgs_train -= mean
    imgs_train /= std

    # Prep val
    if val:
        val_path = './data/prepped/imgs_val.npy'
        val_mask_path = './data/prepped/imgs_mask_val.npy'
    else:
        # If no val set, test data is used as val/early stopping set
        val_path = './data/prepped/imgs_test.npy'
        val_mask_path = './data/prepped/imgs_mask_test.npy'

    # Load val data
    imgs_val = np.load(val_path)
    imgs_mask_val = np.load(val_mask_path)
    imgs_val, imgs_mask_val = preprocess(imgs_val, imgs_mask_val, band_selection,
                                         mask_crop=0)
    imgs_val -= mean
    imgs_val /= std
    val_data = (imgs_val, imgs_mask_val)

    print(imgs_val.shape)
    print(imgs_train.shape)
    print(imgs_mask_val.shape)
    print(imgs_mask_train.shape)
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    num_bands = len(band_selection)
    model = get_unet(TIF_ROWS, TIF_COLS, num_bands, loss_func, learn_rate,
                     crop=CROP_SIZE)

    # Setup callbacks
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss',
                                       mode='min', save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_images=False)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25,
                                   verbose=0, mode='min')


    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, imgs_mask_train, batch_size=BATCH_SIZE, epochs=500,
              verbose=2, shuffle=True,
             validation_data=val_data,
              callbacks=[model_checkpoint, tensorboard, early_stopping])

    # Record results as dictionary
    out_dict = {}

    print('-'*30)
    print('Loading saved weights for val, testing...')
    print('-'*30)
    model.load_weights('weights.h5')
    val_eval = model.evaluate(imgs_val, imgs_mask_val, batch_size=BATCH_SIZE, verbose=0)
    print('Final Val Scores: {}'.format(val_eval))
    # Validation results
    out_dict['val_f1'] = val_eval[-1]
    out_dict['val_recall'] = val_eval[-2]
    out_dict['val_prec'] = val_eval[-3]

    if not val:
        # Val and test are together, so we'll run tests on val set
        imgs_test = imgs_val
        imgs_mask_test = imgs_mask_val
    else:
        print('-'*30)
        print('Loading and preprocessing test data...')
        print('-'*30)
        imgs_test = np.load('./data/prepped/imgs_test.npy')
        imgs_mask_test = np.load('./data/prepped/imgs_mask_test.npy')
        imgs_test, imgs_mask_test = preprocess(imgs_test, imgs_mask_test,
                                               band_selection, mask_crop=0)
        imgs_test -= mean
        imgs_test /= std

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    pred_test_masks = model.predict(imgs_test, batch_size=BATCH_SIZE, verbose=0)

    # Save predicted masks
    predict_dir = './data/predict/'
    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    np.save('{}pred_test_masks.npy'.format(predict_dir), pred_test_masks)
    test_img_names = open('./data/prepped/test_names.csv').read().splitlines()


    # For calculating total error metrics
    total_res_pixels = 0
    total_true_positives = 0
    total_false_positives = 0
    for i in range(pred_test_masks.shape[0]):
        pred_mask = pred_test_masks[i]
        true_mask = imgs_mask_test[i]

        # Get ndwi as byte
        ndwi_img = imgs_test[i,:,:,num_bands-2]
        ndwi_img = transform.resize(ndwi_img,
                                    (OG_ROWS, OG_COLS),
                                    preserve_range = True)
        ndwi_img = scale_image_tobyte(ndwi_img)
        ndwi_img = ndwi_img.astype('uint8')

        # Resize masks
#         pred_mask = transform.resize(pred_mask,
#                                     (OG_ROWS, OG_COLS),
#                                     preserve_range = True)
        pred_mask = (pred_mask[:, :, 0] * 255).astype(np.uint8)
        pred_mask_binary = 255*(pred_mask > (PRED_THRESHOLD*255))
#         true_mask = transform.resize(true_mask,
#                                     (OG_ROWS, OG_COLS),
#                                     preserve_range = True)
        true_mask = (true_mask[:, :, 0] * 255).astype(np.uint8)

        # Save predicted masks
        pred_mask_filename = test_img_names[i].replace('og.tif', 'predmask.png')
        io.imsave('{}{}'.format(predict_dir, pred_mask_filename), pred_mask)

        # Save NDWI, predicted mask, and actual masks side by side
        compare_filename = test_img_names[i].replace('og.tif', 'results.png')
        compare_im = 255 * np.ones((OG_ROWS, OG_COLS * 3 + 20), dtype=np.uint8)
        compare_im[0:OG_ROWS, 0:OG_COLS] = ndwi_img
        compare_im[0:OG_ROWS, (OG_COLS + 10):(OG_COLS * 2 + 10)] = true_mask
        compare_im[0:OG_ROWS, (OG_COLS * 2 + 20):] = pred_mask
        io.imsave('{}{}'.format(predict_dir, compare_filename), compare_im)

        # Calculate basic error
        total_res_pixels += np.sum(true_mask == 255)
        total_true_positives += np.sum((true_mask == 255) * (pred_mask_binary == 255))
        total_false_positives += np.sum((true_mask == 0) * (pred_mask_binary == 255))

    print('Total Res Pixels: {}'.format(total_res_pixels))
    print('Total True Pos: {} ({})'.format(
        total_true_positives, total_true_positives/total_res_pixels))
    print('Total False Pos: {} ({})'.format(
        total_false_positives, total_false_positives/total_res_pixels))

#         # Format test masks for eval
#         imgs_mask_test = resize_imgs(imgs_mask_test, 1)
#         imgs_mask_test = imgs_mask_test.astype('float32')
#         imgs_mask_test /= 255.  # scale masks to [0, 1]
#         imgs_mask_test[imgs_mask_test >= 0.5] = 1
#         imgs_mask_test[imgs_mask_test < 0.5] = 0

    # Test results
    test_eval = model.evaluate(imgs_test, imgs_mask_test,
                                batch_size=BATCH_SIZE, verbose=0)
    out_dict['test_f1'] = test_eval[-1]
    out_dict['test_recall'] = test_eval[-2]
    out_dict['test_prec'] = test_eval[-3]
    print('Final Test Scores: {}'.format(test_eval))

    return out_dict

if __name__=='__main__':
    train(7.5E-5, lf.dice_coef_wgt_loss, [0, 1, 2, 3, 4,5, 12, 13, 14, 15],
          val=VAL)

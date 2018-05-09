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
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import backend as K
from skimage import io


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

OG_ROWS = 500
OG_COLS = 500
# Original image dimensions
RESIZE_ROWS = 512
RESIZE_COLS = 512
# Resized dimensions for training/testing.
NUM_BANDS = 6
# Number of bands in image.
SMOOTH = 1.
# Smoothing factor for jaccard_coef
PRED_THRESHOLD = 0.5
# Prediction threshold. > PRED_THRESHOLD will be classified as res.


def jaccard_coef(y_true, y_pred, smooth=SMOOTH):
    """Keras jaccard coefficient

    @author: Vladimir Iglovikov
    """

    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_distance_loss(y_true, y_pred, smooth=100):
    """Keras jaccard loss function

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """

    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth



def scale_image_tobyte(ar):
    """Scale larger data type array to byte"""

    min_val = np.min(ar)
    max_val = np.max(ar)
    byte_ar = np.round(255.0 * (ar - min_val) / (max_val - min_val)) \
        .astype(np.uint8)
    byte_ar[ar == 0] = 0

    return(byte_ar)


def stretch_n(bands, lower_percent=0, higher_percent=100):
    out = np.zeros_like(bands)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)


def dice_coef(y_true, y_pred, smooth=SMOOTH):
    """Keras implementation of Dice coefficient"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_wgt(y_true, y_pred, smooth=SMOOTH):
    """Modified Dice, with Positive class given double weight"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = K.cast(K.greater(y_true_f, 0.5), dtype='float32')
    weights = weights + K.ones_like(weights)
    intersection = K.sum(y_true_f * y_pred_f * weights)

    return (2. * intersection + smooth) / (K.sum(y_true_f * weights) + K.sum(y_pred_f * weights) + smooth)


def dice_coef_loss(y_true, y_pred):
    """Loss function is simply dice coefficient * -1"""
    return -dice_coef(y_true, y_pred)


def get_unet(img_rows, img_cols, nbands):
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

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(lr=5e-5),
                  loss=dice_coef_loss,
                  metrics=[jaccard_coef, dice_coef,
                           'accuracy'])

    return model


def resize_imgs(imgs, nbands):
    """Resize numpy array of images"""
    imgs_p = np.ndarray((imgs.shape[0], RESIZE_ROWS, RESIZE_COLS, nbands))

    for i in range(imgs.shape[0]):
        imgs_p[i] = transform.resize(imgs[i],
                                     (RESIZE_ROWS, RESIZE_COLS, nbands),
                                     preserve_range = True)

    return imgs_p


def train():
    """Master function for training"""
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    # Preprocess. Should write a func for this
    imgs_train = np.load('./data/prepped/imgs_train.npy')
    imgs_mask_train = np.load('./data/prepped/imgs_mask_train.npy')
    imgs_val = np.load('./data/prepped/imgs_val.npy')
    imgs_mask_val = np.load('./data/prepped/imgs_mask_val.npy')

    imgs_train = resize_imgs(imgs_train, NUM_BANDS)
    imgs_mask_train = resize_imgs(imgs_mask_train, 1)
    imgs_val = resize_imgs(imgs_val, NUM_BANDS)
    imgs_mask_val = resize_imgs(imgs_mask_val, 1)

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_val = imgs_val.astype('float32')
    imgs_mask_val = imgs_mask_val.astype('float32')

    mean = np.mean(imgs_train, axis=(0,1,2))  # mean for data centering
    std = np.std(imgs_train, axis=(0,1,2))  # std for data normalization

    # Save the mean and std values
    np.save('mean_std.npy', np.vstack((mean, std)))

    imgs_train -= mean
    imgs_train /= std
    imgs_val -= mean
    imgs_val /= std


    imgs_mask_train /= 255.  # scale masks to [0, 1]
    imgs_mask_train[imgs_mask_train >= 0.5] = 1
    imgs_mask_train[imgs_mask_train < 0.5] = 0
    imgs_mask_val /= 255.  # scale masks to [0, 1]
    imgs_mask_val[imgs_mask_val >= 0.5] = 1
    imgs_mask_val[imgs_mask_val < 0.5] = 0

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(RESIZE_ROWS, RESIZE_COLS, NUM_BANDS)

    # Setup callbacks
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss',
                                       save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                              write_images=True)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=8,
                                   verbose=0, mode='auto')


    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, imgs_mask_train, batch_size=8, epochs=500,
              verbose=1, shuffle=True,
              validation_data=(imgs_val, imgs_mask_val),
              callbacks=[model_checkpoint, tensorboard, early_stopping])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test = np.load('./data/prepped/imgs_test.npy')
    imgs_mask_test = np.load('./data/prepped/imgs_mask_test.npy')

    imgs_test = resize_imgs(imgs_test, NUM_BANDS)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Saving complete model...')
    print('-'*30)
    model.save('full_model.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    pred_test_masks = model.predict(imgs_test, batch_size=8, verbose=1)

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
        ndwi_img = imgs_test[i,:,:,4]
        ndwi_img = transform.resize(ndwi_img,
                                    (OG_ROWS, OG_COLS),
                                    preserve_range = True)
        ndwi_img = scale_image_tobyte(ndwi_img)
        ndwi_img = ndwi_img.astype('uint8')

        print(np.min(pred_mask), np.max(pred_mask))
        pred_mask = transform.resize(pred_mask,
                                     (OG_ROWS, OG_COLS),
                                     preserve_range = True)
        pred_mask = (pred_mask[:, :, 0] * 255.).astype(np.uint8)

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
        pred_mask = 255*(pred_mask > (PRED_THRESHOLD*255)) # move this to earlier
        total_res_pixels += np.sum(true_mask == 255)
        total_true_positives += np.sum((true_mask == 255) * (pred_mask == 255))
        total_false_positives += np.sum((true_mask == 0) * (pred_mask == 255))

        i += 1

    print('Total Res Pixels: {}'.format(total_res_pixels))
    print('Total True Pos: {} ({})'
          .format(total_true_positives,
                  total_true_positives/total_res_pixels))
    print('Total False Pos: {} ({})'
          .format(total_false_positives,
                  total_false_positives/total_res_pixels))

    return

if __name__=='__main__':
    train()

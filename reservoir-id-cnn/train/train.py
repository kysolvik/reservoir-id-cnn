#!/usr/bin/env python3
"""

"""

import os
from skimage import transform
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from skimage import io
from keras.layers.advanced_activations import LeakyReLU

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

RESIZE_ROWS = 256
RESIZE_COLS = 256
# Resized dimensions for training/testing.
NUM_BANDS = 4
# Number of bands in image.
SMOOTH = 1e-1


def jaccard_coef(y_true, y_pred, smooth=SMOOTH):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred, smooth=SMOOTH):
    # __author__ = Vladimir Iglovikov
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return K.mean(jac)


def jaccard_distance_loss(y_true, y_pred, smooth=SMOOTH):
    """
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


def dice_coef(y_true, y_pred, smooth=SMOOTH):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred, smooth=SMOOTH):
    return -dice_coef(y_true, y_pred)


def get_unet(img_rows, img_cols, nbands):

    inputs = Input((img_rows, img_cols, 4))
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

    model.compile(optimizer=Adam(lr=0.00001), loss=dice_coef_loss,
                  metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    return model


def img_resize(imgs, nbands):
    imgs_p = np.ndarray((imgs.shape[0], RESIZE_ROWS, RESIZE_COLS, nbands))

    for i in range(imgs.shape[0]):
        imgs_p[i] = transform.resize(imgs[i],
                                     (RESIZE_ROWS, RESIZE_COLS, nbands),
                                     preserve_range = True)

    # imgs_p = imgs_p[..., np.newaxis]
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train = np.load('./data/prepped/imgs_train.npy')
    imgs_mask_train = np.load('./data/prepped/imgs_mask_train.npy')

#     null_masks = np.ones(imgs_mask_train.shape[0], dtype=bool)
#     for i in range(imgs_mask_train.shape[0]):
#         max_mask = np.max(imgs_mask_train[i,:,:])
#         if max_mask < 255:
#             null_masks[i] = 0
#
#     imgs_train = imgs_train[null_masks]
#     imgs_mask_train = imgs_mask_train[null_masks]

    imgs_train = img_resize(imgs_train, 4)
    imgs_mask_train = img_resize(imgs_mask_train, 1)

    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = get_unet(RESIZE_ROWS, RESIZE_COLS, NUM_BANDS)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss',
                                       save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=20,
              verbose=1, shuffle=True, validation_split=0.2,
              callbacks=[model_checkpoint])

    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)
    imgs_test = np.load('./data/prepped/imgs_test.npy')
    imgs_test = img_resize(imgs_test, 4)

    imgs_test = imgs_test.astype('float32')
    imgs_test -= mean
    imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    pred_test_masks = model.predict(imgs_test, verbose=1)

    # Save predicted masks
    predict_dir = './data/predict/'
    if not os.path.isdir(predict_dir):
        os.makedirs(predict_dir)

    np.save('{}pred_test_masks.npy'.format(predict_dir), pred_test_masks)
    test_img_names = open('./data/prepped/test_names.csv').read().splitlines()
    for i in range(pred_test_masks.shape[0]):
        pred_mask = pred_test_masks[i]
        print(np.min(pred_mask), np.max(pred_mask))
        pred_mask = pred_mask > 0.51
        pred_mask = (pred_mask[:, :, 0] * 255.).astype(np.uint8)
        pred_mask_filename = test_img_names[i].replace('og.tif', 'predmask.png')
        io.imsave('{}{}'.format(predict_dir, pred_mask_filename), pred_mask)
        i += 1

    return

if __name__=='__main__':
    train_and_predict()

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
from tensorflow.keras.layers import Cropping2D, GaussianNoise, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
from skimage import io, transform
import segmentation_models as sm
sm.set_framework('tf.keras')
# sm.set_framework('keras')

# Seed value
# Apparently you may use different seed values at each stage
seed_value= 583

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

BACKBONE = 'resnet50'
loss = sm.losses.DiceLoss()
preprocess_input = sm.get_preprocessing(BACKBONE)

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
VAL = True
# Includes separate validation set (separate from test)
NOISE = False
# Add gaussian noise to inputs
AUGMENT = False
# Data augmentation
REGULARIZE = False
# Add regularization

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr

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


def train(learn_rate, band_selection, val, epochs=200, noise=False,
          aug=False, regularize=False):
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
    mean = np.mean(imgs_train, axis=(0,1,2), dtype='float64', keepdims=True)
    std = np.std(imgs_train, axis=(0,1,2), dtype='float64', keepdims=True)
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

    print(imgs_train.shape)
    imgs_train = preprocess_input(imgs_train)
    imgs_val = preprocess_input(imgs_val)
    print(imgs_train.shape)



    print('-'*30)
    print('Data ready...')
    print('-'*30)
    print(imgs_train.shape)
    print(imgs_mask_train.shape)
    print(imgs_val.shape)
    print(imgs_mask_val.shape)
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)

    num_bands = len(band_selection)

    if noise:
        base_model = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, input_shape=(None, None, num_bands))
        inp = Input(shape=(None,None,num_bands))
        glayer = GaussianNoise(0.025, input_shape = (None, None, num_bands))(inp)
        gauss_out = base_model(glayer)
        output = Cropping2D(cropping=(CROP_SIZE, CROP_SIZE))(gauss_out)
        model = Model(inp, output, name=base_model.name)
    else:
        base_model = sm.Unet(backbone_name=BACKBONE, encoder_weights=None, input_shape=(None, None, num_bands))
        output = Cropping2D(cropping=(CROP_SIZE, CROP_SIZE))(base_model.layers[-1].output)
        model = Model(base_model.inputs, output, name=base_model.name)


    optimizer = Adam(learning_rate=learn_rate) #, decay=1E-3)

    l2 = tf.keras.regularizers.l2(1e-4)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            model.add_loss(lambda layer=layer: l2(layer.kernel))


    model.compile(optimizer, loss=loss, metrics=[
        sm.metrics.iou_score, sm.metrics.precision, sm.metrics.recall,
        sm.metrics.f1_score])

    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_iou_score',
                                       mode='max', save_best_only=VAL)
    early_stopping = EarlyStopping(monitor='val_iou_score', min_delta=0, patience=25,
                                   verbose=0, mode='max')

    if aug:
        datagen = ImageDataGenerator(
            zca_whitening=False,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest',
            shear_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True)
        datagen.fit(imgs_train)
        model.fit(datagen.flow(imgs_train, imgs_mask_train,
                               batch_size=BATCH_SIZE),
                  steps_per_epoch = len(imgs_train)/BATCH_SIZE,
                  epochs=epochs,
                  validation_data=(imgs_val, imgs_mask_val),
                  verbose=2,
                  callbacks=[model_checkpoint, early_stopping],
                  shuffle=True
        )


    else:
        model.fit(
                x=imgs_train,
                y=imgs_mask_train,
                batch_size=BATCH_SIZE,
                epochs=epochs,
                validation_data=(imgs_val, imgs_mask_val),
                verbose=2,
                callbacks=[model_checkpoint, early_stopping],
                shuffle=True
        )

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
    train(2E-4, [0, 1, 2, 3, 4, 5, 12, 13, 14, 15],
#     train(1E-3, [0, 1, 2, 3, 4, 5, 12, 13, 14, 15],
           val=VAL, epochs=200, noise=NOISE, aug=AUGMENT,
          regularize=REGULARIZE)

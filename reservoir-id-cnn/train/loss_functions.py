#!/usr/bin/env python3

from keras import backend as K

SMOOTH = 1.
# Smoothing factor for dice and jaccard coefficients

def jaccard_coef(y_true, y_pred, smooth=SMOOTH):
    """Keras jaccard coefficient

    @author: Vladimir Iglovikov
    """

    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_distance_loss(y_true, y_pred, smooth=SMOOTH):
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


def dice_coef(y_true, y_pred, smooth=SMOOTH):
    """Keras implementation of Dice coefficient"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice_coef

def dice_coef_wgt(y_true, y_pred, smooth=SMOOTH):
    """Modified Dice, with Positive class given double weight"""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    weights = K.cast(K.greater(y_true_f, 0.5), dtype='float32')
    weights = weights + K.ones_like(weights)
    intersection = K.sum(y_true_f * y_pred_f * weights)
    dice_coef = (2. * intersection + smooth) / (K.sum(y_true_f * weights) + K.sum(y_pred_f * weights) + smooth)
    return dice_coef


def dice_coef_loss(y_true, y_pred):
    """Loss function is simply dice coefficient * -1"""
    return -dice_coef(y_true, y_pred)


def dice_coef_wgt_loss(y_true, y_pred):
    """Loss function is simply dice coefficient * -1"""
    return -dice_coef_wgt(y_true, y_pred)

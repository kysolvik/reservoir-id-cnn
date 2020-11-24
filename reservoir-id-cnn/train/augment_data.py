#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for augmenting training data"""

import numpy as np
import math
import random as random
from skimage import transform
import skimage

def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    angle = math.radians(angle)
    if w <= 0 or h <= 0:
        return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5*side_short
        wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a*cos_a - sin_a*sin_a
        wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

    x_offset = abs(int(np.ceil((w-wr)/2)))
    y_offset = abs(int(np.ceil((h-hr)/2)))
    x_coords = [x_offset, w-x_offset]
    y_coords = [y_offset, h-y_offset]

    return x_coords, y_coords



def random_aug(img, mask, resize_range=(0.6, 0.95), noise=False):
    """Perform a random assortment of augmentationso to an img-mask pair."""
    # Save datatypes and range to convert back to later
    img_dtype = img.dtype
    mask_dtype = mask.dtype
    img_bandranges = np.ptp(img, axis=(0, 1))
    img_bandmins = np.amin(img, axis=(0,1))
    img = 2.*(img - img_bandmins)/img_bandranges-1

    # Flip
    flip_v = random.randint(0, 1)
    flip_h = random.randint(0, 1)
    if flip_v:
        img = np.flip(img, 0)
        mask = np.flip(mask, 0)
    if flip_h:
        img = np.flip(img, 1)
        mask = np.flip(mask, 1)
#
#     # Crop and resize
    img_og_size = img.shape[0]
    mask_og_size = mask.shape[0]
    pad_size = int((img_og_size-mask_og_size)/2)
    crop_ratio = random.uniform(resize_range[0], resize_range[1])
    mask_crop_size = int(np.round(mask_og_size * crop_ratio))
    mask_crop_row = random.randint(0, mask_og_size - mask_crop_size)
    mask_crop_col = random.randint(0, mask_og_size - mask_crop_size)
    mask = mask[mask_crop_row:(mask_crop_row+mask_crop_size),
            mask_crop_col:(mask_crop_col+mask_crop_size)]
    img_crop_size = int(np.round(img_og_size * crop_ratio))
    img_crop_row = int(np.round(mask_crop_row+pad_size*(1-crop_ratio)))
    img_crop_col = int(np.round(mask_crop_col+pad_size*(1-crop_ratio)))
    img = img[img_crop_row:(img_crop_row+img_crop_size), img_crop_col:(img_crop_col+img_crop_size)]

    # Rotate
    rotate_degrees = random.randint(0,359)
    img = transform.rotate(img, rotate_degrees, resize=False, preserve_range=True)
    img_x_crop, img_y_crop = rotatedRectWithMaxArea(img.shape[0], img.shape[1], rotate_degrees)
    img = img[img_x_crop[0]:img_x_crop[1], img_y_crop[0]:img_y_crop[1]]
    mask = transform.rotate(mask, rotate_degrees, resize=False, preserve_range=True)
    mask_x_crop, mask_y_crop = rotatedRectWithMaxArea(mask.shape[0], mask.shape[1], rotate_degrees)
    mask = mask[mask_x_crop[0]:mask_x_crop[1], mask_y_crop[0]:mask_y_crop[1]]

    # Transform back to original shape
    img = transform.resize(img, (img_og_size, img_og_size),
            preserve_range=True)
    mask = transform.resize(mask, (mask_og_size, mask_og_size),
            preserve_range=True)

    # Gaussian noise
    if noise:
        img = skimage.util.random_noise(img, mode='gaussian', var=0.01,
                                        clip=True)

    # Convert back to original range datatypes
    img = img_bandranges*(img + 1)/2 + img_bandmins
    img = img.astype(img_dtype)
    mask = mask.astype(mask_dtype)

    # Fix masks as binary
    mask[mask>127] = 255
    mask[mask<=127] = 0

    return img, mask


#/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functions for augmenting training data"""

import numpy as np
import random as random
from skimage import transform


def random_aug(img, mask, resize_range=(0.7, 1.0)):
    """Perform a random assortment of augmentationso to an img-mask pair."""
    # Save datatypes to convert back to later
    img_dtype = img.dtype
    mask_dtype = mask.dtype

    # Flip
    flip_v = random.randint(0, 1)
    flip_h = random.randint(0, 1)
    if flip_v:
        img = np.flip(img, 0)
        mask = np.flip(mask, 0)
    if flip_h:
        img = np.flip(img, 1)
        mask = np.flip(mask, 1)

    # Crop and resize
    og_size = img.shape[0]
    crop_ratio = random.uniform(resize_range[0], resize_range[1])
    crop_size = int(np.floor(og_size * crop_ratio))
    crop_row = random.randint(0, og_size - crop_size)
    crop_col = random.randint(0, og_size - crop_size)
    img = img[crop_row:crop_row+crop_size, crop_col:crop_col+crop_size, :]
    mask = mask[crop_row:crop_row+crop_size, crop_col:crop_col+crop_size]
    img = transform.resize(img, (og_size, og_size), preserve_range=True)
    mask = transform.resize(mask, (og_size, og_size), preserve_range=True)

    # Rotate
    rotate_degrees = random.choice([0, 90, 180, 270])
    img = transform.rotate(img, rotate_degrees)
    mask = transform.rotate(mask, rotate_degrees)

    # Convert back to original datatypes
    img = img.astype(img_dtype)
    mask = mask.astype(mask_dtype)

    return img, mask


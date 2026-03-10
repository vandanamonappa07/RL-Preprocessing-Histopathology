"""
File: preprocessing.py
Author: Vandan V.S
Description:
Contains image preprocessing operations used as actions in the RL framework.
"""

import cv2
import numpy as np
from skimage import exposure


def color_normalization(img, reference=None):
    """
    Perform min-max color normalization.
    """
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def histogram_matching(img, reference):
    """
    Match histogram of input image with reference image.
    """
    if reference is None:
        raise ValueError("Reference image required for histogram matching")

    reference_resized = cv2.resize(reference, (img.shape[1], img.shape[0]))

    return exposure.match_histograms(img, reference_resized, channel_axis=-1)


def contrast_enhancement(img):
    """
    Enhance image contrast.
    """
    return cv2.convertScaleAbs(img, alpha=1.5, beta=0)


def gamma_correction(img, gamma=1.2):
    """
    Perform gamma correction.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")

    return cv2.LUT(img, table)


def no_operation(img):
    """
    Return original image without modification.
    """
    return img


# Action space for RL
ACTIONS = [
    color_normalization,
    histogram_matching,
    contrast_enhancement,
    gamma_correction,
    no_operation
]
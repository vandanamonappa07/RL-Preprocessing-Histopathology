"""
File: preprocessing.py
Author: Vandana B.S
Description:
Contains image preprocessing operations used as actions in the RL framework.
"""

import cv2
import numpy as np


def histogram_equalization(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5, 5), 0)


def sharpen(img):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)


def edge_enhancement(img):
    edges = cv2.Canny(img, 100, 200)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def histogram_matching(img, reference):
    return cv2.addWeighted(img, 0.5, reference, 0.5, 0)


ACTIONS = [
    histogram_equalization,
    gaussian_blur,
    sharpen,
    edge_enhancement,
    histogram_matching
]

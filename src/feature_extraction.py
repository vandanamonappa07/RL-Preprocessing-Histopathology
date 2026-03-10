"""
File: feature_extraction.py
Author: Vandana B.S
Description:
Extracts statistical and texture features from histopathological images.
"""

import cv2
import numpy as np


def extract_features(img):
    """
    Extract image quality features.

    Features:
    1. Average nucleus area (proxy)
    2. Texture density
    3. Entropy
    4. Contrast
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()

    entropy = -np.sum((gray / 255.0) * np.log2(gray / 255.0 + 1e-9))

    texture_density = cv2.Laplacian(gray, cv2.CV_64F).var()

    avg_nucleus_area = np.mean(gray > np.mean(gray))

    return np.array([
        avg_nucleus_area,
        texture_density,
        entropy,
        contrast
    ])
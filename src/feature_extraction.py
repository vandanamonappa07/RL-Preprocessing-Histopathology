"""
File: feature_extraction.py
Author: Vandana B.S
Description:
Extracts statistical and texture features from histopathological images.
"""

import cv2
import numpy as np
from skimage.measure import shannon_entropy


def extract_features(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Average intensity
    mean_intensity = np.mean(gray)

    # Standard deviation (contrast)
    contrast = np.std(gray)

    # Entropy
    entropy = shannon_entropy(gray)

    # Edge density
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size

    return np.array([mean_intensity, contrast, entropy, edge_density, contrast])

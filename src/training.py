"""
File: training.py
Author: Vandan V.S
Description:
Training pipeline for RL-based preprocessing selection.
"""

import os
import cv2
from skimage.metrics import structural_similarity as ssim

from preprocessing import ACTIONS
from feature_extraction import extract_features
from rl_agent import PreprocessingRLAgent


def compute_reward(img_before, img_after, reference=None, alpha=0.5, beta=0.5):
    """
    Compute reward using contrast, entropy improvement, and SSIM similarity.
    """

    feat_before = extract_features(img_before)
    feat_after = extract_features(img_after)

    delta_contrast = feat_after[-1] - feat_before[-1]
    delta_entropy = feat_after[2] - feat_before[2]

    similarity = 0

    if reference is not None:
        reference_resized = cv2.resize(reference, (img_after.shape[1], img_after.shape[0]))

        gray1 = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(reference_resized, cv2.COLOR_BGR2GRAY)

        similarity = ssim(gray1, gray2)

    reward = alpha * (delta_contrast + delta_entropy) + beta * similarity

    return reward


def select_reference(img, reference_images):
    """
    Select the most appropriate reference image.
    """

    features_img = extract_features(img)
    avg_area_img = features_img[0]

    best_ref = reference_images[0]
    min_diff = float('inf')

    for ref in reference_images:

        features_ref = extract_features(ref)
        avg_area_ref = features_ref[0]

        diff = abs(avg_area_img - avg_area_ref)

        if diff < min_diff:
            min_diff = diff
            best_ref = cv2.resize(ref, (img.shape[1], img.shape[0]))

    return best_ref


def train_agents(dataset_path, reference_images):

    groups = [
        "Group_A_Small_Scale",
        "Group_B_Medium_Scale",
        "Group_C_Large_Scale"
    ]

    trained_agents = {}

    for group in groups:

        agent = PreprocessingRLAgent()

        group_path = os.path.join(dataset_path, group)

        image_files = [
            os.path.join(group_path, f)
            for f in os.listdir(group_path)
            if f.lower().endswith(".jpg")
        ]

        for epoch in range(5):

            for img_path in image_files:

                img = cv2.imread(img_path)

                features = extract_features(img)

                best_reference = select_reference(img, reference_images)

                action_idx = agent.select_action(features)

                action_fn = ACTIONS[action_idx]

                if action_fn.__name__ == "histogram_matching":
                    img_trans = action_fn(img, best_reference)
                else:
                    img_trans = action_fn(img)

                reward = compute_reward(img, img_trans, best_reference)

                agent.update_q(features, action_idx, reward)

        trained_agents[group] = agent

        print(f"Training completed for {group}")

    return trained_agents
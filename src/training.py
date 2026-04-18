"""
File: training.py
Author: Vandana B.S
Description:
Training pipeline for RL-based preprocessing selection.
"""

import os
import cv2
import pickle
from skimage.metrics import structural_similarity as ssim

from preprocessing import ACTIONS
from feature_extraction import extract_features
from rl_agent import PreprocessingRLAgent


# -----------------------------
# REWARD FUNCTION
# -----------------------------
def compute_reward(img_before, img_after, reference=None, alpha=0.5, beta=0.5):

    feat_before = extract_features(img_before)
    feat_after = extract_features(img_after)

    delta_contrast = feat_after[1] - feat_before[1]
    delta_entropy = feat_after[2] - feat_before[2]

    similarity = 0.0

    if reference is not None:
        ref_resized = cv2.resize(
            reference,
            (img_after.shape[1], img_after.shape[0])
        )

        gray1 = cv2.cvtColor(img_after, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)

        similarity = ssim(
            gray1,
            gray2,
            data_range=gray1.max() - gray1.min() + 1e-5
        )

    reward = alpha * (delta_contrast + delta_entropy) + beta * similarity
    return float(reward)


# -----------------------------
# REFERENCE SELECTION
# -----------------------------
def select_reference(img, reference_images):

    if len(reference_images) == 0:
        return None

    features_img = extract_features(img)
    avg_feat_img = features_img[0]

    best_ref = reference_images[0]
    min_diff = float('inf')

    for ref in reference_images:
        features_ref = extract_features(ref)
        avg_feat_ref = features_ref[0]

        diff = abs(avg_feat_img - avg_feat_ref)

        if diff < min_diff:
            min_diff = diff
            best_ref = cv2.resize(
                ref,
                (img.shape[1], img.shape[0])
            )

    return best_ref


# -----------------------------
# TRAINING LOOP
# -----------------------------
def train_agents(dataset_path, reference_images):

    os.makedirs("output", exist_ok=True)

    agent = PreprocessingRLAgent()

    image_files = [
        os.path.join(dataset_path, f)
        for f in os.listdir(dataset_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if len(image_files) == 0:
        print("No images found in dataset folder!")
        return

    for epoch in range(5):

        for img_path in image_files:

            img = cv2.imread(img_path)
            if img is None:
                continue

            features = extract_features(img)
            best_reference = select_reference(img, reference_images)

            action_idx = agent.select_action(features)
            action_fn = ACTIONS[action_idx]

            # unified action call (safe)
            try:
                img_trans = action_fn(img, best_reference)
            except TypeError:
                img_trans = action_fn(img)

            reward = compute_reward(img, img_trans, best_reference)

            agent.update_q(features, action_idx, reward)

            print(
                f"Epoch {epoch} | "
                f"Image: {os.path.basename(img_path)} | "
                f"Reward: {reward:.4f}"
            )

            out_path = os.path.join("output", os.path.basename(img_path))
            cv2.imwrite(out_path, img_trans)

    with open("output/agent.pkl", "wb") as f:
        pickle.dump(agent, f)

    print("Training completed successfully")


# -----------------------------
# MAIN ENTRY POINT
# -----------------------------
if __name__ == "__main__":

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    dataset_path = os.path.join(BASE_DIR, "..", "dataset")
    reference_path = os.path.join(BASE_DIR, "..", "reference")

    reference_images = []

    if os.path.exists(reference_path):
        for f in os.listdir(reference_path):
            if f.lower().endswith((".jpg", ".png", ".jpeg")):
                img = cv2.imread(os.path.join(reference_path, f))
                if img is not None:
                    reference_images.append(img)
    else:
        print("Reference folder not found:", reference_path)

    train_agents(dataset_path, reference_images)

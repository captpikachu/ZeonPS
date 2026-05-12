"""
Solution 2: Grid-Cell CNN-Style Tube Detector
=============================================

UPDATED VERSION:
- Saves COMPLETE bbox information in CSV (variable bbox_w, bbox_h predicted by model)
- Saves partial checkpoints after EVERY image
- Can resume safely if IDE crashes
- Angle predicted in [0, 360) using sin/cos regression + atan2 (never 0-1)
- bbox_w and bbox_h regressed from GT (variable, not fixed)
- bbox_rotation is the OBB rotation from GT (separate from angle_deg)
- Evaluation matches on center_x/center_y, compares angle_deg (per README)

Outputs:
    predictions.csv
    predictions_partial.csv
    metrics.json
    checkpoints/
"""

import cv2
import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import pickle

from typing import List, Dict, Tuple, Optional
from pathlib import Path

from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from skimage.feature import hog, local_binary_pattern
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────
# Shared imports
# ─────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))
from evaluate import compute_metrics, load_ground_truth, print_report

# ─────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────

GRID_PARAMS = {
    "cell_size": 40,
    "stride": 20,
    "patch_size": 60,
    "pos_radius": 18,
    "neg_min_dist": 42,
    "nms_radius": 40,
    "score_threshold": 0.7,
    "distance_threshold": 20,
}

# Fallback bbox dimensions (used only if model has no positive training samples)
FALLBACK_BBOX_W = 42.0
FALLBACK_BBOX_H = 43.0

# ─────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extract_patch(img_bgr: np.ndarray, cx: int, cy: int, size: int) -> np.ndarray:

    h, w = img_bgr.shape[:2]

    half = size // 2
    pad = half + 2

    padded = cv2.copyMakeBorder(
        img_bgr,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_REFLECT
    )

    px = cx + pad
    py = cy + pad

    patch = padded[
        py - half: py + half,
        px - half: px + half
    ]

    return cv2.resize(patch, (size, size))


def compute_features(patch: np.ndarray) -> np.ndarray:

    size = patch.shape[0]

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # HOG
    hog_feats = hog(
        gray,
        orientations=8,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
        feature_vector=True,
    )

    # LBP
    lbp = local_binary_pattern(gray, P=16, R=2, method="uniform")

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=18,
        range=(0, 18),
        density=True
    )

    # HSV histograms
    h_hist, _ = np.histogram(
        hsv[:, :, 0].ravel(),
        bins=12,
        range=(0, 180),
        density=True
    )

    s_hist, _ = np.histogram(
        hsv[:, :, 1].ravel(),
        bins=8,
        range=(0, 256),
        density=True
    )

    v_hist, _ = np.histogram(
        hsv[:, :, 2].ravel(),
        bins=8,
        range=(0, 256),
        density=True
    )

    # Statistics — base stats + radial/strip brightness stats (from detect.py)
    stats = np.array([
        np.mean(gray) / 255.0,
        np.std(gray) / 255.0,
        np.mean(hsv[:, :, 1]) / 255.0,
        np.mean(hsv[:, :, 2]) / 255.0,
        np.std(hsv[:, :, 2]) / 255.0,
        # Radial gradient: centre region vs surround brightness
        float(np.mean(gray[size // 4: 3 * size // 4, size // 4: 3 * size // 4])) / 255,
        float(np.mean(gray[:4, :])) / 255,     # top strip
        float(np.mean(gray[-4:, :])) / 255,    # bottom strip
        float(np.mean(gray[:, :4])) / 255,     # left strip
        float(np.mean(gray[:, -4:])) / 255,    # right strip
    ])

    return np.concatenate([
        hog_feats,
        lbp_hist,
        h_hist,
        s_hist,
        v_hist,
        stats
    ])


# ─────────────────────────────────────────────
# TRAINING SAMPLE GENERATION
# ─────────────────────────────────────────────

def generate_training_samples(
    img_bgr,
    gt_tubes,
    patch_size,
    pos_radius,
    neg_min_dist,
):
    """
    Generate positive/negative training samples.

    Targets per positive sample:
        (dx, dy, angle_deg, bbox_w, bbox_h, bbox_rotation)

    - dx, dy           : offset from cell centre to tube lid centre (px)
    - angle_deg        : joint-to-tab direction in [0, 360) — the primary metric
    - bbox_w, bbox_h   : OBB dimensions from annotation (variable per tube)
    - bbox_rotation    : OBB clockwise rotation from annotation
    """

    h, w = img_bgr.shape[:2]
    stride = GRID_PARAMS["stride"]

    gt_centers = np.array([
        [g["center_x"], g["center_y"]]
        for g in gt_tubes
    ])

    gt_angles        = np.array([g["angle_deg"]       for g in gt_tubes])
    gt_bbox_ws       = np.array([g["bbox_w"]          for g in gt_tubes])
    gt_bbox_hs       = np.array([g["bbox_h"]          for g in gt_tubes])
    gt_bbox_rots     = np.array([g["bbox_rotation"]   for g in gt_tubes])

    patches = []
    labels  = []
    targets = []  # (dx, dy, angle_deg, bbox_w, bbox_h, bbox_rotation)

    for cy_cell in range(0, h, stride):
        for cx_cell in range(0, w, stride):

            cx = cx_cell + stride // 2
            cy = cy_cell + stride // 2

            dists = np.sqrt(
                (gt_centers[:, 0] - cx) ** 2 +
                (gt_centers[:, 1] - cy) ** 2
            )

            min_idx  = np.argmin(dists)
            min_dist = dists[min_idx]

            patch = extract_patch(img_bgr, cx, cy, patch_size)

            if min_dist <= pos_radius:
                label = 1
                dx    = float(gt_centers[min_idx, 0] - cx)
                dy    = float(gt_centers[min_idx, 1] - cy)
                targets.append((
                    dx,
                    dy,
                    float(gt_angles[min_idx]),
                    float(gt_bbox_ws[min_idx]),
                    float(gt_bbox_hs[min_idx]),
                    float(gt_bbox_rots[min_idx]),
                ))

            elif min_dist >= neg_min_dist:
                label = 0
                targets.append((0.0, 0.0, 0.0, FALLBACK_BBOX_W, FALLBACK_BBOX_H, 0.0))

            else:
                # Ambiguous zone — skip
                continue

            patches.append(patch)
            labels.append(label)

    return patches, labels, targets


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

class TubeDetectorModel:
    """
    Random-Forest based tube detector.

    Predicts per positive cell:
        - objectness score
        - (dx, dy)           center offset
        - (sin, cos) of angle_deg  →  angle_deg in [0, 360) via atan2
        - bbox_w, bbox_h     variable OBB size
        - bbox_rotation      OBB rotation (clockwise, degrees)
    """

    def __init__(self):

        self.objectness_clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )

        self.dx_reg  = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42)
        self.dy_reg  = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42)

        # Angle encoded as (sin, cos) to avoid 0/360 discontinuity.
        # atan2(sin, cos) then gives the angle back in full [-180,180] → remapped [0,360).
        self.angle_sin_reg = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42)
        self.angle_cos_reg = RandomForestRegressor(n_estimators=80, max_depth=12, n_jobs=-1, random_state=42)

        # Variable bbox size regressors (trained on GT values, not fixed constants)
        self.bbox_w_reg   = RandomForestRegressor(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42)
        self.bbox_h_reg   = RandomForestRegressor(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42)

        # bbox_rotation: the OBB clockwise rotation (separate from angle_deg)
        # Also encoded via sin/cos to handle the 0/360 wrap.
        self.brot_sin_reg = RandomForestRegressor(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42)
        self.brot_cos_reg = RandomForestRegressor(n_estimators=60, max_depth=10, n_jobs=-1, random_state=42)

        self.scaler = StandardScaler()
        self.is_trained = False

    def fit(self, X, y_cls, y_dx, y_dy, y_angle, y_bbox_w, y_bbox_h, y_bbox_rot):

        print(f"Training on {len(X)} samples  ({int(y_cls.sum())} positive)")

        Xs = self.scaler.fit_transform(X)
        self.objectness_clf.fit(Xs, y_cls)

        pos_mask = (y_cls == 1)
        if pos_mask.sum() == 0:
            print("  WARNING: no positive samples — regressors not trained.")
            self.is_trained = True
            return

        Xs_pos = Xs[pos_mask]

        # Offset regressors
        self.dx_reg.fit(Xs_pos, y_dx[pos_mask])
        self.dy_reg.fit(Xs_pos, y_dy[pos_mask])

        # Angle regressors  (sin/cos encoding → atan2 decoding → [0,360))
        angle_rad = np.deg2rad(y_angle[pos_mask])
        self.angle_sin_reg.fit(Xs_pos, np.sin(angle_rad))
        self.angle_cos_reg.fit(Xs_pos, np.cos(angle_rad))

        # BBox size regressors (GT values, so output is in real pixels)
        self.bbox_w_reg.fit(Xs_pos, y_bbox_w[pos_mask])
        self.bbox_h_reg.fit(Xs_pos, y_bbox_h[pos_mask])

        # BBox rotation regressors (sin/cos encoding)
        brot_rad = np.deg2rad(y_bbox_rot[pos_mask])
        self.brot_sin_reg.fit(Xs_pos, np.sin(brot_rad))
        self.brot_cos_reg.fit(Xs_pos, np.cos(brot_rad))

        self.is_trained = True

    # ── Inference helpers ──────────────────────────────────────────────────

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.objectness_clf.predict_proba(Xs)[:, 1]

    def predict_offsets(self, X: np.ndarray):
        Xs = self.scaler.transform(X)
        return self.dx_reg.predict(Xs), self.dy_reg.predict(Xs)

    def predict_angle(self, X: np.ndarray) -> np.ndarray:
        """
        Returns angle_deg in [0, 360).
        Uses sin/cos regressors + atan2 — output is NEVER bounded to [0,1].
        """
        Xs = self.scaler.transform(X)
        sin_pred = self.angle_sin_reg.predict(Xs)
        cos_pred = self.angle_cos_reg.predict(Xs)
        angle_rad = np.arctan2(sin_pred, cos_pred)           # [-pi, pi]
        return (np.rad2deg(angle_rad) + 360) % 360           # [0, 360)

    def predict_bbox(self, X: np.ndarray):
        """
        Returns (bbox_w, bbox_h, bbox_rotation_deg).
        bbox sizes are real pixel values (variable per detection).
        bbox_rotation is in [0, 360).
        """
        Xs = self.scaler.transform(X)

        bbox_w = self.bbox_w_reg.predict(Xs)
        bbox_h = self.bbox_h_reg.predict(Xs)

        sin_brot = self.brot_sin_reg.predict(Xs)
        cos_brot = self.brot_cos_reg.predict(Xs)
        brot_rad = np.arctan2(sin_brot, cos_brot)
        bbox_rot = (np.rad2deg(brot_rad) + 360) % 360

        # Clamp sizes to realistic range (27–60 px per annotation stats)
        bbox_w = np.clip(bbox_w, 20.0, 70.0)
        bbox_h = np.clip(bbox_h, 20.0, 70.0)

        return bbox_w, bbox_h, bbox_rot


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def visualize_detections(
    img_bgr: np.ndarray,
    detections: List[Dict],
    ground_truth: Optional[List[Dict]] = None,
    title: str = "",
) -> np.ndarray:
    """
    Draw predicted detections and ground truth on the image.

    Predictions (Cyan):
        - Rotated bounding box  (bbox_w × bbox_h, rotated by bbox_rotation)
        - Center circle
        - Angle arrow  (angle_deg: joint-to-tab direction)
        - angle_deg label + confidence

    Ground truth (Orange):
        - Rotated bounding box from annotation
        - Center circle
        - Angle arrow
    """
    vis = img_bgr.copy()

    # ── Ground truth ────────────────────────────────────────────────────────
    if ground_truth:
        for gt in ground_truth:
            cx, cy = int(gt["center_x"]), int(gt["center_y"])

            # Rotated OBB
            bw = float(gt.get("bbox_w", 42))
            bh = float(gt.get("bbox_h", 42))
            br = float(gt.get("bbox_rotation", 0))
            box = cv2.boxPoints(((cx, cy), (bw, bh), -br))  # cv2 uses clockwise
            box = np.intp(box)
            cv2.drawContours(vis, [box], 0, (255, 140, 0), 2)

            # Center
            cv2.circle(vis, (cx, cy), 5, (255, 140, 0), -1)

            # Angle arrow (angle_deg: counter-clockwise from +X, Y-down → flip sin)
            angle_rad = np.deg2rad(gt["angle_deg"])
            ex = int(cx + 25 * np.cos(angle_rad))
            ey = int(cy - 25 * np.sin(angle_rad))
            cv2.arrowedLine(vis, (cx, cy), (ex, ey), (255, 140, 0), 2, tipLength=0.3)

    # ── Predictions ─────────────────────────────────────────────────────────
    for det in detections:
        cx, cy = int(det["center_x"]), int(det["center_y"])

        # Rotated OBB (predicted variable size)
        bw = float(det["bbox_w"])
        bh = float(det["bbox_h"])
        br = float(det["bbox_rotation"])
        box = cv2.boxPoints(((cx, cy), (bw, bh), -br))
        box = np.intp(box)
        cv2.drawContours(vis, [box], 0, (0, 220, 255), 2)

        # Center
        cv2.circle(vis, (cx, cy), 5, (0, 220, 255), -1)

        # Angle arrow (angle_deg)
        angle_rad = np.deg2rad(det["angle_deg"])
        ex = int(cx + 25 * np.cos(angle_rad))
        ey = int(cy - 25 * np.sin(angle_rad))
        cv2.arrowedLine(vis, (cx, cy), (ex, ey), (0, 220, 255), 2, tipLength=0.3)

        # Label: angle + confidence
        cv2.putText(
            vis,
            f"{det['angle_deg']:.0f}\u00b0 [{det['confidence']:.2f}]",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1,
        )

    # ── Legend ───────────────────────────────────────────────────────────────
    legend = "Cyan=Pred  Orange=GT"
    if title:
        legend = title + "  |  " + legend
    cv2.putText(vis, legend, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.putText(vis, legend, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,   0,   0), 1)

    return vis


# ─────────────────────────────────────────────
# NMS
# ─────────────────────────────────────────────

def nms_detections(detections: List[Dict], radius: float) -> List[Dict]:

    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)

    kept = []

    for det in detections:
        cx = det["center_x"]
        cy = det["center_y"]

        valid = all(
            np.sqrt((cx - k["center_x"]) ** 2 + (cy - k["center_y"]) ** 2) >= radius
            for k in kept
        )

        if valid:
            kept.append(det)

    return kept


# ─────────────────────────────────────────────
# DETECTION
# ─────────────────────────────────────────────

def detect_tubes(img_bgr: np.ndarray, model: TubeDetectorModel) -> List[Dict]:

    h, w = img_bgr.shape[:2]

    stride     = GRID_PARAMS["stride"]
    patch_size = GRID_PARAMS["patch_size"]
    threshold  = GRID_PARAMS["score_threshold"]

    positions = []
    patches   = []

    for cy_cell in range(0, h, stride):
        for cx_cell in range(0, w, stride):
            cx = cx_cell + stride // 2
            cy = cy_cell + stride // 2
            patch = extract_patch(img_bgr, cx, cy, patch_size)
            positions.append((cx, cy))
            patches.append(patch)

    features = Parallel(n_jobs=-1)(
        delayed(compute_features)(p) for p in patches
    )

    X = np.asarray(features, dtype=np.float32)

    scores   = model.predict_scores(X)
    pos_mask = scores >= threshold

    if not pos_mask.any():
        return []

    pos_X         = X[pos_mask]
    pos_scores    = scores[pos_mask]
    pos_positions = [positions[i] for i in np.where(pos_mask)[0]]

    dx_pred, dy_pred           = model.predict_offsets(pos_X)
    angle_pred                 = model.predict_angle(pos_X)       # [0, 360)
    bbox_w_pred, bbox_h_pred, bbox_rot_pred = model.predict_bbox(pos_X)

    detections = []

    for i, (cx_cell, cy_cell) in enumerate(pos_positions):

        # Refined center (from cell centre + predicted offset)
        cx = float(np.clip(cx_cell + dx_pred[i], 0, w - 1))
        cy = float(np.clip(cy_cell + dy_pred[i], 0, h - 1))

        # angle_deg: joint-to-tab direction in [0, 360)  ← primary metric
        angle = float(angle_pred[i])

        # Variable bbox dimensions (predicted by regressors, not fixed constants)
        bw   = float(bbox_w_pred[i])
        bh   = float(bbox_h_pred[i])
        brot = float(bbox_rot_pred[i])

        # bbox top-left corner (axis-aligned interpretation for csv)
        bbox_x = cx - bw / 2
        bbox_y = cy - bh / 2

        detections.append({
            "center_x":     cx,
            "center_y":     cy,
            "bbox_x":       bbox_x,
            "bbox_y":       bbox_y,
            "bbox_w":       bw,
            "bbox_h":       bh,
            "bbox_rotation": brot,   # OBB rotation (clockwise, degrees)  ≠ angle_deg
            "angle_deg":    angle,   # joint-to-tab direction (counter-clockwise from +X)
            "confidence":   float(pos_scores[i]),
        })

    return nms_detections(detections, GRID_PARAMS["nms_radius"])


# ─────────────────────────────────────────────
# SAVE CHECKPOINT
# ─────────────────────────────────────────────

def save_partial_predictions(output_dir: str, image_name: str, detections: List[Dict]):

    csv_path = os.path.join(output_dir, "predictions_partial.csv")

    rows = []
    for d in detections:
        rows.append({
            "image":         image_name,
            "center_x":      round(d["center_x"],      1),
            "center_y":      round(d["center_y"],      1),
            "bbox_x":        round(d["bbox_x"],        1),
            "bbox_y":        round(d["bbox_y"],        1),
            "bbox_w":        round(d["bbox_w"],        1),
            "bbox_h":        round(d["bbox_h"],        1),
            "bbox_rotation": round(d["bbox_rotation"], 1),
            "angle_deg":     round(d["angle_deg"],     1),
            "confidence":    round(d["confidence"],    4),
        })

    df = pd.DataFrame(rows)

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run(
    images_dir: str,
    annotations_csv: str,
    output_dir: str = "outputs",
    visualize: bool = True,
):

    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # Load ground truth — includes center_x/y, angle_deg, bbox_x/y/w/h, bbox_rotation
    gt_all = load_ground_truth(annotations_csv)

    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    print(f"Processing {len(image_files)} images")

    # ── PRE-EXTRACT FEATURES ────────────────────────────────────────────────
    print("\n[1/3] Extracting features...")
    image_data = {}

    for fname in image_files:

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt = gt_all.get(fname, [])
        if not gt:
            continue

        patches, labels, targets = generate_training_samples(
            img,
            gt,
            GRID_PARAMS["patch_size"],
            GRID_PARAMS["pos_radius"],
            GRID_PARAMS["neg_min_dist"],
        )

        features = Parallel(n_jobs=4)(delayed(compute_features)(p) for p in patches)
        features = np.asarray(features, dtype=np.float32)

        image_data[fname] = (img, features, np.array(labels), targets)
        n_pos = sum(1 for t in targets if labels[targets.index(t)] == 1) if targets else sum(labels)
        print(f"  {fname}: {len(patches)} samples  ({sum(labels)} pos)")

    # ── LEAVE-ONE-OUT TRAINING & EVALUATION ────────────────────────────────
    print("\n[2/3] Leave-One-Out training & evaluation...")
    pred_all = {}

    for test_fname in image_files:

        if test_fname not in image_data:
            continue

        print(f"\n  Testing: {test_fname}")

        # Collect training data from all OTHER images
        X_train   = []
        y_cls     = []
        y_dx      = []
        y_dy      = []
        y_angle   = []
        y_bbox_w  = []
        y_bbox_h  = []
        y_bbox_rot= []

        for fname, (img, feats, labs, tgts) in image_data.items():
            if fname == test_fname:
                continue

            X_train.append(feats)
            y_cls.extend(labs)

            for t in tgts:
                # t = (dx, dy, angle_deg, bbox_w, bbox_h, bbox_rotation)
                y_dx.append(t[0])
                y_dy.append(t[1])
                y_angle.append(t[2])
                y_bbox_w.append(t[3])
                y_bbox_h.append(t[4])
                y_bbox_rot.append(t[5])

        X_train    = np.vstack(X_train)
        y_cls      = np.array(y_cls)
        y_dx       = np.array(y_dx)
        y_dy       = np.array(y_dy)
        y_angle    = np.array(y_angle)
        y_bbox_w   = np.array(y_bbox_w)
        y_bbox_h   = np.array(y_bbox_h)
        y_bbox_rot = np.array(y_bbox_rot)

        model = TubeDetectorModel()
        model.fit(X_train, y_cls, y_dx, y_dy, y_angle, y_bbox_w, y_bbox_h, y_bbox_rot)

        test_img = image_data[test_fname][0]
        dets     = detect_tubes(test_img, model)

        pred_all[test_fname] = dets

        # ── Running metrics checkpoint ──────────────────────────────────────
        partial_metrics = compute_metrics(
            pred_all,
            gt_all,
            GRID_PARAMS["distance_threshold"]
        )

        with open(os.path.join(output_dir, "metrics_checkpoint.json"), "w") as f:
            json.dump(partial_metrics, f, indent=2)

        gt = gt_all.get(test_fname, [])
        print(
            f"    Detected {len(dets)} / GT {len(gt)}  |  "
            f"Running F1: {partial_metrics['f1']:.4f}  "
            f"P: {partial_metrics['precision']:.4f}  "
            f"R: {partial_metrics['recall']:.4f}"
        )
        if dets:
            angles = [d["angle_deg"] for d in dets]
            print(f"    Angle range: [{min(angles):.1f}°, {max(angles):.1f}°]  "
                  f"BBox W range: [{min(d['bbox_w'] for d in dets):.1f}, "
                  f"{max(d['bbox_w'] for d in dets):.1f}]")

        # ── Partial CSV checkpoint ──────────────────────────────────────────
        save_partial_predictions(output_dir, test_fname, dets)

        # ── Visualisation ────────────────────────────────────────────────────
        if visualize:
            vis_img = visualize_detections(
                test_img,
                dets,
                ground_truth=gt_all.get(test_fname, []),
                title="LOO-CV",
            )
            vis_path = os.path.join(vis_dir, f"result_{test_fname}")
            cv2.imwrite(vis_path, vis_img)
            print(f"    Saved vis: {vis_path}")

        # ── Pickle checkpoint ───────────────────────────────────────────────
        checkpoint_path = os.path.join(output_dir, f"checkpoint_{test_fname}.pkl")
        with open(checkpoint_path, "wb") as f:
            pickle.dump(dets, f)

    # ── FINAL METRICS ───────────────────────────────────────────────────────
    print("\n[3/3] Final evaluation...")
    #
    # Matching is done on center_x / center_y (per README: we evaluate tube
    # lid centre position and rotation angle).
    # The angle metric compares angle_deg (joint-to-tab, [0,360)) with circular
    # wrap-around handled by angular_error() in evaluate.py.
    #
    metrics = compute_metrics(
        pred_all,
        gt_all,
        GRID_PARAMS["distance_threshold"]
    )

    print_report(metrics, "Grid Cell Detector (variable bbox + correct angles)")

    # ── SAVE FINAL METRICS ──────────────────────────────────────────────────
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ── SAVE FINAL CSV ──────────────────────────────────────────────────────
    rows = []

    for img_name, dets in pred_all.items():
        for d in dets:
            rows.append({
                "image":         img_name,
                "center_x":      round(d["center_x"],      1),
                "center_y":      round(d["center_y"],      1),
                "bbox_x":        round(d["bbox_x"],        1),
                "bbox_y":        round(d["bbox_y"],        1),
                "bbox_w":        round(d["bbox_w"],        1),
                "bbox_h":        round(d["bbox_h"],        1),
                "bbox_rotation": round(d["bbox_rotation"], 1),
                "angle_deg":     round(d["angle_deg"],     1),
                "confidence":    round(d["confidence"],    4),
            })

    final_csv = os.path.join(output_dir, "predictions.csv")
    pd.DataFrame(rows).to_csv(final_csv, index=False)
    print(f"\nSaved final CSV : {final_csv}")

    return metrics


# ─────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Grid-Cell Tube Detector (fixed)")

    parser.add_argument("--images",      default=r"F:\Python\ZeonPS\data\annotated_images")
    parser.add_argument("--annotations", default=r"F:\Python\ZeonPS\data\annotations.csv")
    parser.add_argument("--output",      default="outputs")
    parser.add_argument("--no-vis",      action="store_true", help="Skip saving visualizations")

    args = parser.parse_args()

    run(args.images, args.annotations, args.output, visualize=not args.no_vis)
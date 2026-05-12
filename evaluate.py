"""
Improved evaluation utilities for tube detection.
Fixes greedy matching issues and makes metrics stable.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment


# ─────────────────────────────────────────────
# Angle error (circular)
# ─────────────────────────────────────────────
def angular_error(pred_angle: float, gt_angle: float) -> float:
    diff = abs(pred_angle - gt_angle) % 360
    return min(diff, 360 - diff)


# ─────────────────────────────────────────────
# Build cost matrix (center distance)
# ─────────────────────────────────────────────
def build_cost_matrix(preds, gts):
    P = len(preds)
    G = len(gts)

    cost = np.zeros((P, G), dtype=np.float32)

    for i, p in enumerate(preds):
        for j, g in enumerate(gts):
            dx = p["center_x"] - g["center_x"]
            dy = p["center_y"] - g["center_y"]
            cost[i, j] = np.sqrt(dx * dx + dy * dy)

    return cost


# ─────────────────────────────────────────────
# Optimal matching (Hungarian algorithm)
# ─────────────────────────────────────────────
def match_detections(
    predictions: List[Dict],
    ground_truth: List[Dict],
    distance_threshold: float = 40.0,   # IMPORTANT: relaxed threshold
) -> Tuple[List[Tuple], List[int], List[int]]:

    if len(predictions) == 0:
        return [], [], list(range(len(ground_truth)))

    if len(ground_truth) == 0:
        return [], list(range(len(predictions))), []

    cost = build_cost_matrix(predictions, ground_truth)

    # Hungarian optimal assignment
    pred_idx, gt_idx = linear_sum_assignment(cost)

    matches = []
    matched_preds = set()
    matched_gts = set()

    for p, g in zip(pred_idx, gt_idx):
        if cost[p, g] <= distance_threshold:
            matches.append((p, g))
            matched_preds.add(p)
            matched_gts.add(g)

    unmatched_preds = [i for i in range(len(predictions)) if i not in matched_preds]
    unmatched_gts = [i for i in range(len(ground_truth)) if i not in matched_gts]

    return matches, unmatched_preds, unmatched_gts


# ─────────────────────────────────────────────
# Full metrics computation
# ─────────────────────────────────────────────
def compute_metrics(
    all_predictions: Dict[str, List[Dict]],
    all_ground_truth: Dict[str, List[Dict]],
    distance_threshold: float = 40.0,
) -> Dict:

    total_tp = 0
    total_fp = 0
    total_fn = 0

    angle_errors = []
    position_errors = []

    per_image = {}

    all_images = set(all_predictions.keys()) | set(all_ground_truth.keys())

    for img in sorted(all_images):

        preds = all_predictions.get(img, [])
        gts = all_ground_truth.get(img, [])

        matches, unmatched_preds, unmatched_gts = match_detections(
            preds, gts, distance_threshold
        )

        tp = len(matches)
        fp = len(unmatched_preds)
        fn = len(unmatched_gts)

        img_angle_errors = []
        img_pos_errors = []

        for pi, gi in matches:

            dx = preds[pi]["center_x"] - gts[gi]["center_x"]
            dy = preds[pi]["center_y"] - gts[gi]["center_y"]
            pos_err = np.sqrt(dx * dx + dy * dy)

            ang_err = angular_error(
                preds[pi]["angle_deg"],
                gts[gi]["angle_deg"]
            )

            position_errors.append(pos_err)
            angle_errors.append(ang_err)

            img_pos_errors.append(pos_err)
            img_angle_errors.append(ang_err)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_image[img] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "num_preds": len(preds),
            "num_gt": len(gts),
            "mean_angle_error": float(np.mean(img_angle_errors)) if img_angle_errors else None,
            "mean_position_error": float(np.mean(img_pos_errors)) if img_pos_errors else None,
        }

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)

    f1 = (2 * precision * recall) / (precision + recall + 1e-9)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),

        # These are VERY important for your report
        "mean_angle_error_deg": float(np.mean(angle_errors)) if angle_errors else None,
        "mean_position_error_px": float(np.mean(position_errors)) if position_errors else None,

        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,

        "distance_threshold_px": distance_threshold,
        "per_image": per_image,
    }


# ─────────────────────────────────────────────
# Ground truth loader
# ─────────────────────────────────────────────
def load_ground_truth(annotations_csv: str) -> Dict[str, List[Dict]]:
    df = pd.read_csv(annotations_csv)

    gt = {}

    for _, row in df.iterrows():
        img = row["image"]

        if img not in gt:
            gt[img] = []

        gt[img].append({
            "center_x": float(row["center_x"]),
            "center_y": float(row["center_y"]),
            "angle_deg": float(row["angle_deg"]),

            # kept for compatibility (not used in metric)
            "bbox_x": float(row["bbox_x"]),
            "bbox_y": float(row["bbox_y"]),
            "bbox_w": float(row["bbox_w"]),
            "bbox_h": float(row["bbox_h"]),
            "bbox_rotation": float(row["bbox_rotation"]),
        })

    return gt


# ─────────────────────────────────────────────
# Report printer
# ─────────────────────────────────────────────
def print_report(metrics: Dict, name: str):

    print("\n" + "=" * 60)
    print(f"  EVALUATION: {name}")
    print("=" * 60)

    print(f"  TP / FP / FN   : {metrics['total_tp']} / {metrics['total_fp']} / {metrics['total_fn']}")
    print(f"  Precision      : {metrics['precision']}")
    print(f"  Recall         : {metrics['recall']}")
    print(f"  F1 Score       : {metrics['f1']}")

    if metrics["mean_position_error_px"] is not None:
        print(f"  Pos Error (px) : {metrics['mean_position_error_px']:.2f}")

    if metrics["mean_angle_error_deg"] is not None:
        print(f"  Angle Error    : {metrics['mean_angle_error_deg']:.2f}°")

    print("=" * 60 + "\n")
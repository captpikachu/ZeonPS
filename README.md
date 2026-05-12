# Microcentrifuge Tube Detection

This repository contains a machine learning pipeline for detecting the center position and rotation angle of microcentrifuge tube lids in overhead RGB images. 

## Technical Approach

To detect the test tubes and their orientations, because the type of object is consistent across different locations, we extract local image patches and use **Histogram of Oriented Gradients (HOG)** to learn the features of the test tubes and knowing how it looks. 

The angles of the test tubes vary, which is reflected in particular angle bars having higher frequencies in our HOG and because we only consider overhead images, the gradient of the tubes remains almost similar across the dataset.

Key technical details:
 The image is divided into a grid, and local patches are extracted. We compute HOG (for shape/orientation), Local Binary Patterns (LBP for texture), and HSV color histograms for each patch.
*   **Model:** We use Scikit-Learn's `RandomForestClassifier` for objectness (is it a tube?) and several `RandomForestRegressor` models for bounding box offsets (`dx`, `dy`) and dimensions (`bbox_w`, `bbox_h`).
*   **Angle Regression:** The orientation angle (`angle_deg`) is predicted using a continuous sin/cos regression approach combined with `arctan2`. This gracefully handles the $0^\circ/360^\circ$ circular wrap-around discontinuity.
*   **Validation:** Due to the small size of the dataset (70 images), the model is evaluated using **Leave-One-Out Cross-Validation (LOO-CV)** to maximize the training data available for each prediction. This doesn't affect other predictions as each prediction is independent and no overfit happens which is clearly seen in results.

## Results and Analysis

*   **Running Time:** ~2 minutes total for the entire dataset (using parallel feature extraction via `joblib`).

### Angle Error Analysis
While position detection is highly accurate, the angle error metric may largely be impacted by:
1. Let us assume 5 deg acceptable error 
2.  because the tubes are visually symmetric, say about 4% of the tubes are detected perfectly but assigned an angle exactly $180^\circ$ more or less for this dataset, these $180^\circ$ anomalies artificially inflate the mean angle error by about $9^\circ$.
3. tubes positioned very close to the edge of the image provide less structural information in their patches this, combined with distracting background brightness, interferes with the HOG orientation calculation explaining additional $28^\circ$ 


## Next Steps Proposal

Removing these specific failure modes, the acceptable baseline angle error for c
*   `numpy`, `pandas`, `scikit-learn`, `scikit-image`, `scipy`, `joblib`

### Running the Detector
The main execution script is `detect.py`. It handles feature extraction, LOO training/evaluation, Non-Maximum Suppression (NMS), metric calculation, and checkpointing.

```bash
# Basic run command
@@ -54,4 +53,4 @@ python newDetect.py --images data/annotated_images/ --annotations data/annotatio
*   `predictions.csv`: Final detection outputs with predicted bounding boxes and angles.
*   `metrics.json`: Detailed F1, Precision, Recall, and Error metrics.
*   `visualizations/`: Overlay images showing Ground Truth (Orange) vs Predictions (Cyan).
*   `predictions_partial.csv` & checkpoints: Enables safe recovery if execution is interrupted. (I deleted this file after complete execution as it was no longer required to continue but kept checkpoints only).

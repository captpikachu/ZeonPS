# Microcentrifuge Tube Detection

This repository contains a machine learning pipeline for detecting the center position and rotation angle of microcentrifuge tube lids in overhead RGB images. 

## Technical Approach

To detect the test tubes and their orientations, we implemented a **Grid-Cell Patch-Based Detector** (conceptually similar to a CNN sliding window approach but powered by robust traditional ML algorithms). Because the type of object is consistent across different locations, we extract local image patches and use **Histogram of Oriented Gradients (HOG)** to learn the structural features of the test tubes. 

The angles of the test tubes vary, which is reflected in particular angle bins having higher frequencies in our HOG descriptors. Furthermore, because we only consider overhead images, the gradient profile of the tubes remains relatively similar across the dataset.

Key technical details:
*   **Grid Search & Feature Engineering:** The image is divided into a grid, and local patches are extracted. We compute HOG (for shape/orientation), Local Binary Patterns (LBP for texture), and HSV color histograms for each patch.
*   **Model:** We use Scikit-Learn's `RandomForestClassifier` for objectness (is it a tube?) and several `RandomForestRegressor` models for bounding box offsets (`dx`, `dy`) and dimensions (`bbox_w`, `bbox_h`).
*   **Angle Regression:** The orientation angle (`angle_deg`) is predicted using a continuous sin/cos regression approach combined with `arctan2`. This gracefully handles the $0^\circ/360^\circ$ circular wrap-around discontinuity.
*   **Validation:** Due to the small size of the dataset (70 images), the model is evaluated using **Leave-One-Out Cross-Validation (LOO-CV)** to maximize the training data available for each prediction.

## Results and Analysis

*   **Precision:** ~92% (Detected objects are almost always correct test tubes).
*   **Recall:** ~87% (The vast majority of tubes present in the images are successfully detected).
*   **Running Time:** ~2 minutes total for the entire dataset (using parallel feature extraction via `joblib`).

### Angle Error Analysis
While position detection is highly accurate, the angle error metric is impacted by two primary edge cases:
1.  **180-Degree Ambiguity:** Because the tubes are visually symmetric, about 4% of the tubes are detected perfectly but assigned an angle exactly $180^\circ$ in the wrong direction. For this dataset, these $180^\circ$ anomalies artificially inflate the mean angle error by about $9^\circ$.
2.  **Edge Interference:** Tubes positioned very close to the edge of the image provide less structural information in their bounding boxes. This, combined with distracting background brightness, interferes with the HOG orientation calculation, contributing an estimated $28^\circ$ of error in those specific edge cases. 

Removing these specific failure modes, the acceptable baseline angle error for correctly oriented, fully visible tubes is approximately $5^\circ$.

## Next Steps Proposal

1.  **Multi-Scale Patching:** Implement parallel computation for different patch sizes to better handle tubes that appear slightly smaller/larger or are truncated at the image boundaries.
2.  **Expanded Dataset:** The current LOO evaluation is a result of data scarcity. If a dataset of a few hundred images becomes available, the model can learn the HOG features in just a few minutes and only needs to be trained *once*. The pipeline would then be capable of immediate, near real-time prediction for any given image.
3.  **Data Augmentation:** Rotating training patches during the generation phase to teach the Random Forest to recognize tubes at all $360^\circ$ angles equally, which will help mitigate the $180^\circ$ ambiguity problem.

---

## Usage

### Requirements
*   Python 3.8+
*   `opencv-python`
*   `numpy`, `pandas`, `scikit-learn`, `scikit-image`, `scipy`, `joblib`

### Running the Detector
The main execution script is `newDetect.py`. It handles feature extraction, LOO training/evaluation, Non-Maximum Suppression (NMS), metric calculation, and checkpointing.

```bash
# Basic run command
python newDetect.py --images data/annotated_images/ --annotations data/annotations.csv --output outputs/
```

**Outputs will be saved in the `--output` directory:**
*   `predictions.csv`: Final detection outputs with predicted bounding boxes and angles.
*   `metrics.json`: Detailed F1, Precision, Recall, and Error metrics.
*   `visualizations/`: Overlay images showing Ground Truth (Orange) vs Predictions (Cyan).
*   `predictions_partial.csv` & checkpoints: Enables safe recovery if execution is interrupted.

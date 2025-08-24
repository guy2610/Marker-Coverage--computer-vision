# Algorithm (High-Level)

This document describes the high-level algorithm used to detect and validate a 3×3 marker grid from an input image.

---

## 1. Preprocessing
- Convert the input image from BGR → HSV color space.
- Apply Gaussian blur to reduce noise.
- For each of the six marker colors (red, green, blue, yellow, cyan, magenta):
  - Apply cv::inRange with tuned HSV thresholds.
  - Apply morphological operations (open/close) to clean the mask.
- Detect contours in each mask, and compute for each patch:
  - Bounding box, center, area.
  - Filter by min/max relative area.

---

## 2. Rotation Normalization (PCA)
- Extract all patch centers.
- Run PCA to align the data with the dominant axes.
- Rotate patch coordinates into a new coordinate system (x′, y′).
- This step normalizes tilted grids (up to ~45° inclination).

---

## 3. Grid Construction (3×3)
- Cluster patches along y′ axis into 3 rows using KMeans.
- Cluster patches along x′ axis into 3 columns using KMeans.
- Compute target row/column centers.
- Greedy assignment:
  - For each of the 9 target cells, assign the closest patch (with tie-breaking bonus if KMeans label matches).
- Require exactly 9 patches to form a valid grid.

---

## 4. Grid Validation
- Sort patches in each row by x′ and in each column by y′.
- Compute normalized distances:
  - dx = spacing between columns
  - dy = spacing between rows
  - Normalize each by local mean.
- Compute coefficient of variation:
  - cvx = stdev(dx) / mean(dx)
  - cvy = stdev(dy) / mean(dy)
- Thresholds:
  - Accept if cvx ≤ 0.55 and cvy ≤ 0.65.
  - Allow soft fallback if slightly worse but coverage is high.

---

## 5. Convex Hull & Coverage
- Collect all 9 patch bounding-box corners.
- Compute convex hull over these points.
- Compute hull area and the bounding box area covering all 9 patches.
- Coverage ratio = hull_area / bbox_area.
- Require coverage ≥ 0.70.
- Fallback: accept if coverage ≥ 0.80 even when spacing fails.

---

## 6. Decision Logic
- If all validations succeed → marker_found.
- Otherwise → marker_not_found with a failure reason (FR_1…FR_6).

---

## 7. Output
- Program outputs one line per image:
  - "marker_found <image path>"
  - or "marker_not_found <image path> FR_x" where FR_x indicates the failure reason.
- Debug mode (--debug) shows intermediate masks, detections, and timing.

---

This algorithm ensures robustness to:
- Color variations (via HSV thresholds).
- Noise (via morphology).
- Rotation (via PCA).
- Missing/extra patches (via clustering + greedy assignment).
- Perspective distortions (via convex hull coverage).

# Algorithm (High-Level)

This document describes the high-level algorithm used to detect and validate a 3×3 marker grid from an input image.

---

## 1. Preprocessing
- Read the image as-is (no resizing).
- Convert from BGR → HSV.
- Apply a light Gaussian blur to reduce noise.
- For each expected color (red, green, blue, yellow, cyan, magenta):
  - Threshold with tuned HSV ranges (`cv::inRange`).
  - Clean the mask using morphology (open/close).
- Find contours per mask and build patch candidates with:
  - Bounding box, center (from moments), and area.
  - Area filtering using relative min/max to reject noise/outliers.

---

## 2. Rotation Normalization (PCA)
- Collect all patch centers and run PCA to find dominant axes.
- Rotate centers into a stable coordinate system (x′, y′).
- This normalizes tilted grids (robust up to ~45°).

---

## 3. Grid Construction (3×3)
- Cluster along **y′** into 3 rows (k-means).
- Cluster along **x′** into 3 columns (k-means).
- Compute target row/column centers in (x′, y′).
- Greedy assignment of patches to the 9 cell intersections:
  - Pick closest candidates; gentle bonus when k-means labels agree.
- Require exactly 9 assigned cells for a valid grid.

---

## 4. Grid Validation (Spacing Consistency)
- For each row (sorted by x′) compute two horizontal gaps; normalize by the local mean.
- For each column (sorted by y′) compute two vertical gaps; normalize by the local mean.
- Coefficients of variation:
  - `cvx = stdev(dx_norm) / mean(dx_norm)`
  - `cvy = stdev(dy_norm) / mean(dy_norm)`
- Thresholds:
  - **Pass** if `cvx ≤ 0.55` and `cvy ≤ 0.65`.
  - **Soft fallback** if slightly worse but coverage is strong (see §5/§6).

---

## 5. Convex Hull & Coverage
- Collect the 4 corners of each of the 9 patch bounding boxes.
- Compute the convex hull over these 36 points and its area `hull_area`.
- Compute the **image area** `image_area = width × height`.
- **Coverage** = `hull_area / image_area`  (**final metric** used in the decision).

---

## 6. Decision Logic
- Let `coverage_thresh = 0.45`, `coverage_soft = 0.50`, `coverage_fallback = 0.55`.
- Compute spacing_ok from §4.
- Fallbacks:
  - If `!spacing_ok` and `coverage ≥ coverage_fallback` → accept.
  - If `!spacing_ok` and `cvx ≤ 0.60 && cvy ≤ 0.70 && coverage ≥ coverage_soft` → accept.
- **Final pass** if `(spacing_ok) AND (coverage ≥ coverage_thresh)`.
- Otherwise: fail with a specific failure reason (FR_1…FR_6).

---

## 7. Output & Debug
- Per image:
  - `marker_found <path>`
  - or `marker_not_found <path> FR_x`
- In `--debug` mode print spacing stats (cvx/cvy), coverage details, and timing.

---

### Robustness Notes
- Color variation handled via HSV thresholds + morphology.
- Rotation handled via PCA alignment.
- Extra/missing patches handled via clustering + greedy assignment.
- Perspective skew covered by convex-hull coverage over all patch corners.

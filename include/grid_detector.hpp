#pragma once
#include "types.hpp"
#include <opencv2/opencv.hpp>

struct GridParams {
    float cvx_thresh = 0.55f;
    float cvy_thresh = 0.65f;
    float coverage_thresh  = 0.45f;
    float coverage_fallback= 0.55f;
    float coverage_soft    = 0.50f;
    bool  debug = false;
};

// Runs PCA, row/col KMeans, greedy assignment, spacing checks.
// Fills GridDetection and returns FailureReason (or NONE).
FailureReason detect_grid_and_spacing(
    const std::vector<Patch>& patches,
    GridDetection& out,
    const GridParams& params);

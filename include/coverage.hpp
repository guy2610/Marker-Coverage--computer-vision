#pragma once
#include "types.hpp"
#include <opencv2/opencv.hpp>

struct CoverageResult {
    double hull_area = 0.0;
    double bbox_area = 0.0;
    double image_area = 0.0;
    double ratio_bbox = 0.0; // hull / bbox (Internal verification )
    double ratio = 0.0; // hull / image (the output)
    std::vector<cv::Point2f> hull;
};

// Builds convex hull from grid rect corners and computes coverage ratios.
// img_size inorder to compute hull/image.
CoverageResult compute_coverage_from_grid(const Patch grid[3][3], const cv::Size& img_size);

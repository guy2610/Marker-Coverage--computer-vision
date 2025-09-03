#pragma once
#include "types.hpp"
#include <vector>
#include <opencv2/opencv.hpp>

struct SegmentationParams {
    double min_area_ratio = 0.0006; // relative to image area
    double max_area_ratio = 0.2;
    bool   debug = false;
};

std::vector<Patch> segment_color_patches(const cv::Mat& bgr, const SegmentationParams& params);

#pragma once
#include <opencv2/opencv.hpp>
#include <string>

struct Patch {
    std::string color;
    cv::Rect    box;
    cv::Point2f center;
    double      area;
    int         id;
};

enum class FailureReason {
    NONE = 0,
    FEW_PATCHES = 1,
    ASSIGN_GRID = 2,
    SPACING = 3,
    SMALL_HULL = 4,
    BAD_BBOX = 5,
    LOW_COVERAGE = 6
};

inline const char* fr_to_cstr(FailureReason fr) {
    switch (fr) {
    case FailureReason::NONE:         return "OK";
    case FailureReason::FEW_PATCHES:  return "FEW_PATCHES";
    case FailureReason::ASSIGN_GRID:  return "GRID_ASSIGNMENT_FAILED";
    case FailureReason::SPACING:      return "SPACING_VALIDATION_FAILED";
    case FailureReason::SMALL_HULL:   return "HULL_TOO_SMALL";
    case FailureReason::BAD_BBOX:     return "INVALID_BBOX";
    case FailureReason::LOW_COVERAGE: return "LOW_COVERAGE";
    default:                          return "UNKNOWN";
    }
}

// A light wrapper to return what main needs
struct GridDetection {
    bool ok = false;                // 3×3 assigned?
    Patch grid[3][3];               // final grid
    std::vector<cv::Point2f> rot;   // rotated centers (x',y')
    float cvx = 1e9f, cvy = 1e9f;   // spacing CVs
    bool spacing_ok = false;        // spacing gate
};
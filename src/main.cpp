// main.cpp
#include "types.hpp"
#include "color_segmentation.hpp"
#include "grid_detector.hpp"
#include "coverage.hpp"

#include <opencv2/opencv.hpp>
#include <filesystem>
#include <chrono>
#include <iostream>
#include <cmath>

using clk = std::chrono::high_resolution_clock;

static void emit_marker_result(const std::string& name, bool ok, FailureReason fr, bool debug_mode) {
    if (debug_mode) {
        if (ok)  std::cout << "marker_found " << name << "\n";
        else     std::cout << "marker_not_found " << name << " " << fr_to_cstr(fr) << "\n";
    }
    else {
        // In non-debug we print only the required "<image> <percent>%"
        // (on failure we will print "<image> 0%")
        (void)fr; (void)ok; // no-op
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "the program needs pictures names as arguments\n";
        return 1;
    }

    bool debug_mode = false;
    std::vector<std::string> images;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--debug") { debug_mode = true; continue; }
        if (std::filesystem::exists(a)) images.push_back(a);
        else std::cerr << a << " is not a valid picture path\n";
    }
    if (images.empty()) return 1;

    int pass_count = 0, fail_count = 0;
    bool any_fail = false;

    SegmentationParams segp; segp.debug = debug_mode;
    GridParams gp; gp.debug = debug_mode;
    // thresholds as in your tuned logic
    gp.coverage_thresh = 0.45f; // must-have
    gp.coverage_fallback = 0.55f; // accept even if spacing failed
    gp.coverage_soft = 0.50f; // soft acceptance if cv within near-range

    for (const auto& path : images) {
        auto t0 = clk::now();

        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty()) {
            emit_marker_result(path, false, FailureReason::FEW_PATCHES, debug_mode);
            if (!debug_mode) std::cout << path << " 0%\n";
            ++fail_count; any_fail = true;
            continue;
        }

        // 1) Color segmentation -> candidate patches
        auto patches = segment_color_patches(img, segp);
        if (patches.size() < 3) {
            emit_marker_result(path, false, FailureReason::FEW_PATCHES, debug_mode);
            if (!debug_mode) std::cout << path << " 0%\n";
            ++fail_count; any_fail = true;
            continue;
        }

        // 2) Grid detection + spacing validation (PCA-rotated coords, CV thresholds)
        GridDetection gd;
        FailureReason fr = detect_grid_and_spacing(patches, gd, gp);
        if (fr == FailureReason::ASSIGN_GRID || fr == FailureReason::SPACING) {
            emit_marker_result(path, false, fr, debug_mode);
            if (!debug_mode) std::cout << path << " 0%\n";
            ++fail_count; any_fail = true;

            if (debug_mode) {
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
                std::cout << path << " took " << ms << " ms\n";
            }
            continue;
        }

        // 3) Coverage (convex hull of 9 rect corners) vs IMAGE area
        // NOTE: ensure your coverage.hpp/.cpp signature is:
        // CoverageResult compute_coverage_from_grid(const Patch grid[3][3], cv::Size imageSize);
        auto cov = compute_coverage_from_grid(gd.grid, img.size());
        if (cov.hull_area <= 0.0 || cov.image_area <= 0.0) {
            emit_marker_result(path, false, FailureReason::BAD_BBOX, debug_mode);
            if (!debug_mode) std::cout << path << " 0%\n";
            ++fail_count; any_fail = true;
            continue;
        }

        // 4) Thresholds + fallbacks
        bool spacing_ok = gd.spacing_ok;
        if (!spacing_ok && cov.ratio >= gp.coverage_fallback) spacing_ok = true;
        if (!spacing_ok && gd.cvx <= 0.60f && gd.cvy <= 0.70f && cov.ratio >= gp.coverage_soft) spacing_ok = true;

        bool ok = (spacing_ok && cov.ratio >= gp.coverage_thresh);
        emit_marker_result(path, ok, ok ? FailureReason::NONE : FailureReason::LOW_COVERAGE, debug_mode);

        if (ok) {
            ++pass_count;
            int pct = (int)std::lround(cov.ratio * 100.0);
            if (!debug_mode) std::cout << path << " " << pct << "%\n";
            if (debug_mode) {
                std::cout << "[coverage] hull=" << cov.hull_area
                    << " image=" << cov.image_area
                    << " ratio=" << cov.ratio << " (" << pct << "%)\n";
            }
        }
        else {
            ++fail_count; any_fail = true;
            if (!debug_mode) std::cout << path << " 0%\n";
            if (debug_mode) {
                std::cout << "[final] cvx=" << gd.cvx << " cvy=" << gd.cvy
                    << " coverage_ratio=" << cov.ratio << "\n";
            }
        }

        if (debug_mode) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() - t0).count();
            if (ms > 200) std::cerr << "[warn] " << path << " took " << ms << " ms (>200ms)\n";
            else          std::cout << path << " took " << ms << " ms\n";
        }
    }

    std::cout << "\nSummary: passed=" << pass_count
        << " failed=" << fail_count
        << " out of " << (pass_count + fail_count) << std::endl;

    return any_fail ? 1 : 0;
}

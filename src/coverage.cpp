#include "coverage.hpp"
#include <cfloat>
using namespace cv;

CoverageResult compute_coverage_from_grid(const Patch grid[3][3], const cv::Size& img_size) {
    CoverageResult r;
    std::vector<Point2f> corners; corners.reserve(9 * 4);
    float minx = +FLT_MAX, miny = +FLT_MAX, maxx = -FLT_MAX, maxy = -FLT_MAX;

    for (int i = 0;i < 3;++i) for (int j = 0;j < 3;++j) {
        const Rect& b = grid[i][j].box;
        corners.emplace_back((float)b.x, (float)b.y);
        corners.emplace_back((float)(b.x + b.width), (float)b.y);
        corners.emplace_back((float)(b.x + b.width), (float)(b.y + b.height));
        corners.emplace_back((float)b.x, (float)(b.y + b.height));

        minx = std::min(minx, (float)b.x);
        miny = std::min(miny, (float)b.y);
        maxx = std::max(maxx, (float)(b.x + b.width));
        maxy = std::max(maxy, (float)(b.y + b.height));
    }

    convexHull(corners, r.hull);
    if (r.hull.size() < 3) return r;

    r.hull_area = std::abs(contourArea(r.hull));

    double bbox_w = std::max(0.f, maxx - minx);
    double bbox_h = std::max(0.f, maxy - miny);
    r.bbox_area = (double)bbox_w * (double)bbox_h;

    r.image_area = (double)img_size.width * (double)img_size.height;

    if (r.bbox_area > 0.0) r.ratio_bbox = r.hull_area / r.bbox_area;
    if (r.image_area > 0.0) r.ratio = r.hull_area / r.image_area;

    return r;
}

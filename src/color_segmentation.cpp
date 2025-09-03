#include "color_segmentation.hpp"
using namespace cv;
using std::vector; using std::string;

static Mat inRangeHSV(const Mat& hsv, Scalar low, Scalar high) {
    Mat mask; inRange(hsv, low, high, mask); return mask;
}
static void cleanMask(Mat& mask) {
    Mat k = getStructuringElement(MORPH_ELLIPSE, {3,3});
    morphologyEx(mask, mask, MORPH_OPEN, k);
    morphologyEx(mask, mask, MORPH_CLOSE, k);
}

std::vector<Patch> segment_color_patches(const cv::Mat& bgr, const SegmentationParams& params) {
    CV_Assert(!bgr.empty());
    Mat hsv; cvtColor(bgr, hsv, COLOR_BGR2HSV);
    GaussianBlur(hsv, hsv, Size(3,3), 0);

    Mat red1 = inRangeHSV(hsv, { 0,   80, 60 }, { 10, 255, 255 });
    Mat red2 = inRangeHSV(hsv, {170,  80, 60 }, {180, 255, 255 });
    Mat red; bitwise_or(red1, red2, red);
    Mat green   = inRangeHSV(hsv, { 35, 60, 60 }, { 85, 255, 255 });
    Mat blue    = inRangeHSV(hsv, { 90, 60, 60 }, {130, 255, 255 });
    Mat yellow  = inRangeHSV(hsv, { 20, 60, 60 }, { 35, 255, 255 });
    Mat cyan    = inRangeHSV(hsv, { 80, 60, 60 }, { 95, 255, 255 });
    Mat magenta = inRangeHSV(hsv, {140, 60, 60 }, {170, 255, 255 });

    for (Mat* m : { &red,&green,&blue,&yellow,&cyan,&magenta }) cleanMask(*m);

    vector<Patch> patches;
    const double img_area = (double)bgr.cols * (double)bgr.rows;
    const double min_area = params.min_area_ratio * img_area;
    const double max_area = params.max_area_ratio * img_area;

    vector<std::pair<string,Mat>> masks = {
        {"red", red}, {"green",green}, {"blue",blue},
        {"yellow",yellow},{"cyan",cyan},{"magenta",magenta}
    };

    int next_id = 0;
    for (auto& lm : masks) {
        const string& label = lm.first;
        Mat mask = lm.second.clone();
        vector<vector<Point>> contours; vector<Vec4i> hier;
        findContours(mask, contours, hier, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (const auto& cnt : contours) {
            double a = contourArea(cnt); if (a < min_area || a > max_area) continue;
            Rect box = boundingRect(cnt);
            Moments m = moments(cnt); if (m.m00 <= 0) continue;
            Point2f center((float)(m.m10/m.m00), (float)(m.m01/m.m00));
            patches.push_back(Patch{label, box, center, a, next_id++});
        }
    }
    return patches;
}

// Sodyo assigment.cpp : main
#define NOMINMAX
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <initializer_list>
#include <cmath>
#include <numeric>    // std::accumulate



struct Patch {
    std::string color;
    cv::Rect box;
    cv::Point2f center;
    double area;
    int id;                 // << new: index to map into rot[x',y']
};
enum class FailureReason {
    NONE = 0,
    FEW_PATCHES = 1,     // פחות מדי פאצ'ים
    ASSIGN_GRID = 2,     // נכשל בבניית 3על3
    SPACING = 3,         // סטיות מרחקים (cvx/cvy) גדולות מדי/סידור שגוי
    SMALL_HULL = 4,      // מעט מדי נקודות בקמור
    BAD_BBOX = 5,        // bbox_area לא תקין
    LOW_COVERAGE = 6     // יחס כיסוי מתחת לסף
};

static const char* fr_to_cstr(FailureReason fr) {
    switch (fr) {
    case FailureReason::NONE:         return "FR_0";
    case FailureReason::FEW_PATCHES:  return "FR_1";
    case FailureReason::ASSIGN_GRID:  return "FR_2";
    case FailureReason::SPACING:      return "FR_3";
    case FailureReason::SMALL_HULL:   return "FR_4";
    case FailureReason::BAD_BBOX:     return "FR_5";
    case FailureReason::LOW_COVERAGE: return "FR_6";
    default:                          return "FR_?";
    }
}

static void emit_marker_result(const std::string& name, bool ok, FailureReason fr, bool debug_mode, int pct=-1) {
    if (ok) {
        std::cout << name << " " << pct << "%" << "\n";
    }
    else {
        std::cout << "marker_not_found " << name << " " << fr_to_cstr(fr) << "\n";
    }
}

using namespace cv;
using namespace std;
using clk = std::chrono::high_resolution_clock;
namespace {
    Mat inRangeHSV(const Mat& hsv, Scalar low, Scalar high) {
        Mat mask;
        inRange(hsv, low, high, mask);
        return mask;
    }
    void cleanMask(Mat& mask) {
        Mat k = getStructuringElement(MORPH_ELLIPSE, { 3,3 });
        morphologyEx(mask, mask, MORPH_OPEN, k);
        morphologyEx(mask, mask, MORPH_CLOSE, k);
    }
    void showOrSaveMask(const string& title, const Mat& mask, bool debug_mode) {
        if (debug_mode) {
            imshow(title, mask);
        }
    }
    bool comparePatches(const Patch& a, const Patch& b) {
        if (fabs(a.center.y - b.center.y) > 1.0)
            return a.center.y < b.center.y;
        return a.center.x < b.center.x;
    }
    double median_of(std::vector<double> v) {
        if (v.empty()) return 0.0;
        std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
        double med = v[v.size() / 2];
        if (v.size() % 2 == 0) {
            auto max_lower = *std::max_element(v.begin(), v.begin() + v.size() / 2);
            med = 0.5 * (med + max_lower);
        }
        return med;
    }
}

// ------- helpers -------
static float stdev(const std::vector<float>& v) {
    if (v.size() < 2) return 0.f;
    float m = std::accumulate(v.begin(), v.end(), 0.f) / (float)v.size();
    float s2 = 0.f;
    for (auto x : v) s2 += (x - m) * (x - m);
    return std::sqrt(s2 / (float)(v.size() - 1));
}
template<typename T>
static T median_vec(std::vector<T> v) {
    if (v.empty()) return T();
    std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
    return v[v.size() / 2];
}

// PCA rotate to x', y'
static void pca_rotate(const std::vector<cv::Point2f>& pts,
    std::vector<cv::Point2f>& rotated,
    cv::Point2f& mean,
    cv::Mat& eigvecs) {
    cv::Mat data((int)pts.size(), 2, CV_32F);
    for (int i = 0;i < (int)pts.size();++i) {
        data.at<float>(i, 0) = pts[i].x;
        data.at<float>(i, 1) = pts[i].y;
    }
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    eigvecs = pca.eigenvectors.clone(); // 2x2
    mean = cv::Point2f(pca.mean.at<float>(0, 0), pca.mean.at<float>(0, 1));
    rotated.resize(pts.size());
    for (int i = 0;i < (int)pts.size();++i) {
        cv::Vec2f v(pts[i].x - mean.x, pts[i].y - mean.y);
        cv::Mat vm = (cv::Mat_<float>(1, 2) << v[0], v[1]);
        cv::Mat r = vm * eigvecs.t();
        rotated[i] = cv::Point2f(r.at<float>(0, 0), r.at<float>(0, 1)); // (x’, y’)
    }
}

// pick 3 columns in a row using KMeans on x' (returns medoids)
static std::vector<Patch> pick_triplet_kmeans_xprime(const std::vector<Patch>& row,
    const std::vector<cv::Point2f>& rot) {
    if ((int)row.size() <= 3) {
        auto out = row;
        std::sort(out.begin(), out.end(),
            [&](const Patch& a, const Patch& b) { return rot[a.id].x < rot[b.id].x; });
        return out;
    }
    cv::Mat samples((int)row.size(), 1, CV_32F);
    for (int i = 0;i < (int)row.size();++i)
        samples.at<float>(i, 0) = rot[row[i].id].x;

    int K = 3;
    cv::Mat labels, centers;
    int attempts = 5;
    cv::kmeans(samples, K, labels,
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-3),
        attempts, cv::KMEANS_PP_CENTERS, centers);

    std::vector<std::vector<int>> groups(K);
    for (int i = 0;i < (int)row.size(); ++i) {
        int k = labels.at<int>(i, 0);
        groups[k].push_back(i);
    }
    // fallback if any empty cluster
    for (int k = 0;k < K;++k) if (groups[k].empty()) {
        auto out = row;
        std::sort(out.begin(), out.end(),
            [&](const Patch& a, const Patch& b) { return rot[a.id].x < rot[b.id].x; });
        out.resize(3);
        return out;
    }

    std::vector<Patch> triplet; triplet.reserve(3);
    for (int k = 0;k < K;++k) {
        float cx = centers.at<float>(k, 0);
        int bestIdx = -1; float bestD = 1e9f;
        for (int i_in_row : groups[k]) {
            float x = rot[row[i_in_row].id].x;
            float d = std::abs(x - cx);
            if (d < bestD) { bestD = d; bestIdx = i_in_row; }
        }
        triplet.push_back(row[bestIdx]);
    }
    std::sort(triplet.begin(), triplet.end(),
        [&](const Patch& a, const Patch& b) { return rot[a.id].x < rot[b.id].x; });
    return triplet;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "the program needs pictures names as arguments" << endl;
        return 1;
    }
    bool debug_mode = false;
    vector<string> valid_images_names;
    vector<string> valid_extensions = { ".png",".jpg", ".jpeg" };
    bool any_fail = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            debug_mode = true;
            continue;
        }
        std::filesystem::path name = arg;
        string ext = name.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        bool ext_ok = std::count(valid_extensions.begin(), valid_extensions.end(), ext) > 0;
        if (std::filesystem::exists(name) && ext_ok) {
            valid_images_names.push_back(name.string());
        }
        else {
            cout << name << " is not valid picture address" << endl;
            any_fail = true;
        }
    }

    if (valid_images_names.empty()) return 1;
    int pass_count = 0, fail_count = 0;

    for (size_t i = 0; i < valid_images_names.size(); i++)
    {
        auto t0 = clk::now();
        Mat image = imread(valid_images_names[i], IMREAD_COLOR);
        if (image.empty()) {
            cout << "failed_to_load " << valid_images_names[i] << endl;
            cout << valid_images_names[i] << " 0%" << std::endl;
            any_fail = true;
            continue;
        }
        const int MAX_W = 640, MAX_H = 480;
        if (image.cols > MAX_W || image.rows > MAX_H) {
            double sx = (double)MAX_W / image.cols;
            double sy = (double)MAX_H / image.rows;
            double s = std::min(sx, sy);
            cv::resize(image, image, cv::Size(), s, s, cv::INTER_AREA);
        }
        Mat hsv_image;
        cvtColor(image, hsv_image, COLOR_BGR2HSV);
        GaussianBlur(hsv_image, hsv_image, Size(3, 3), 0);

        if (debug_mode) { imshow("Debug - Original ", image); waitKey(1); }

        // --- color masks ---
        cv::Mat red1 = inRangeHSV(hsv_image, { 0,   80, 60 }, { 10, 255, 255 });
        cv::Mat red2 = inRangeHSV(hsv_image, { 170, 80, 60 }, { 180,255, 255 });
        cv::Mat red; cv::bitwise_or(red1, red2, red);
        cv::Mat green = inRangeHSV(hsv_image, { 35,  60, 60 }, { 85,  255, 255 });
        cv::Mat blue = inRangeHSV(hsv_image, { 90,  60, 60 }, { 130, 255, 255 });
        cv::Mat yellow = inRangeHSV(hsv_image, { 20,  60, 60 }, { 35,  255, 255 });
        cv::Mat cyan = inRangeHSV(hsv_image, { 80,  60, 60 }, { 95,  255, 255 });
        cv::Mat magenta = inRangeHSV(hsv_image, { 140, 60, 60 }, { 170, 255, 255 });

        for (Mat* m : initializer_list<Mat*>{ &red,&green,&blue,&yellow,&cyan,&magenta }) cleanMask(*m);

        showOrSaveMask("mask_red", red, debug_mode);
        showOrSaveMask("mask_green", green, debug_mode);
        showOrSaveMask("mask_blue", blue, debug_mode);
        showOrSaveMask("mask_yellow", yellow, debug_mode);
        showOrSaveMask("mask_cyan", cyan, debug_mode);
        showOrSaveMask("mask_magenta", magenta, debug_mode);

        vector<Patch> patches;
        vector<pair<string, Mat>> labeled_masks = {
            {"red", red}, {"green", green}, {"blue", blue},{"yellow", yellow}, {"cyan", cyan}, {"magenta", magenta}
        };
        const double img_area = static_cast<double>(image.cols) * image.rows;
        const double min_area = 0.0006 * img_area;   
        const double max_area = 0.2 * img_area;

        int next_id = 0; // << id assignment per image

        for (auto& lm : labeled_masks) {
            const string& label = lm.first;
            Mat mask_for_cnt = lm.second.clone();
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(mask_for_cnt, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            for (const auto& cnt : contours) {
                double a = contourArea(cnt);
                if (a < min_area || a > max_area) continue;

                Rect box = boundingRect(cnt);
                Moments m = moments(cnt);
                if (m.m00 <= 0) continue;
                Point2f center(static_cast<float>(m.m10 / m.m00), static_cast<float>(m.m01 / m.m00));
                patches.push_back(Patch{ label, box, center, a, next_id++ });
            }
        }

        if (debug_mode) {
            cv::Mat vis = image.clone();
            for (const auto& p : patches) {
                cv::rectangle(vis, p.box, cv::Scalar(0, 255, 0), 2);
                cv::circle(vis, p.center, 3, cv::Scalar(0, 0, 255), cv::FILLED);
            }
            cv::imshow("Detections", vis);
            cv::waitKey(1);
        }

        // sanity
        if (patches.size() < 3) {
            if (debug_mode) {
                std::cout << valid_images_names[i] << " marker_not_found: patches.size() < 3\n";
                cv::imshow("HSV (debug)", hsv_image);
                auto t1 = clk::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::cout << valid_images_names[i] << " took " << ms << " ms\n";
            }
            emit_marker_result(valid_images_names[i], false, FailureReason::FEW_PATCHES, debug_mode);
            fail_count++; any_fail = true;
            continue;
        }


        //PCA rotate to (x', y') for stability
        std::vector<cv::Point2f> centers_only; centers_only.reserve(patches.size());
        for (const auto& p : patches) centers_only.push_back(p.center);
        std::vector<cv::Point2f> rot; cv::Point2f meanP; cv::Mat eigvecs;
        pca_rotate(centers_only, rot, meanP, eigvecs);

        //KMeans rows on y'
        cv::Mat samples((int)patches.size(), 1, CV_32F);
        for (int idx = 0; idx < (int)patches.size(); ++idx)
            samples.at<float>(idx, 0) = rot[idx].y; // y'

        int K = 3;
        cv::Mat labels, centersY;
        int attempts = 5;
        cv::kmeans(samples, K, labels,
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-3),
            attempts, cv::KMEANS_PP_CENTERS, centersY);

        struct ClusterInfo { float cy; int k; };
        std::vector<ClusterInfo> order;
        order.reserve(K);
        for (int k = 0; k < K; ++k) order.push_back({ centersY.at<float>(k, 0), k });
        std::sort(order.begin(), order.end(), [](const ClusterInfo& a, const ClusterInfo& b) { return a.cy < b.cy; });

        int label2row[3];
        for (int r = 0; r < K; ++r) label2row[order[r].k] = r;

        std::vector<std::vector<Patch>> rows(3);
        for (int idx = 0; idx < (int)patches.size(); ++idx) {
            int k = labels.at<int>(idx, 0);
            int r = label2row[k];
            rows[r].push_back(patches[idx]);
        }

        if (debug_mode) {
            std::cout << "[kmeans] rows counts (pre-pick): ";
            for (int r = 0; r < 3; ++r) std::cout << rows[r].size() << " ";
            std::cout << "\n";
        }

        // KMeans columns on x' (על כל הפאצ'ים)
        cv::Mat samplesX((int)patches.size(), 1, CV_32F);
        for (int idx = 0; idx < (int)patches.size(); ++idx)
            samplesX.at<float>(idx, 0) = rot[idx].x;  // x'

        cv::Mat labelsX, centersX;
        cv::kmeans(samplesX, 3, labelsX,
            cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 100, 1e-3),
            5, cv::KMEANS_PP_CENTERS, centersX);

        // order col- left to right
        struct Cx { float x; int k; };
        std::vector<Cx> orderX; orderX.reserve(3);
        for (int k = 0; k < 3; ++k) orderX.push_back({ centersX.at<float>(k,0), k });
        std::sort(orderX.begin(), orderX.end(), [](const Cx& a, const Cx& b) { return a.x < b.x; });
        int label2col[3];
        for (int c = 0; c < 3; ++c) label2col[orderX[c].k] = c;

        // בונה רשימות לפי שורה/עמודה
        std::vector<std::vector<int>> byRow(3), byCol(3);
        for (int idx = 0; idx < (int)patches.size(); ++idx) {
            int r = label2row[labels.at<int>(idx, 0)];   // לפי y'
            int c = label2col[labelsX.at<int>(idx, 0)];  // לפי x'
            byRow[r].push_back(idx);
            byCol[c].push_back(idx);
        }

        // מרכזים במרחב x',y'
        float rowCenterY[3], colCenterX[3];
        //if byRow[r] is empty we will usr the center of the cluster (order by centersY)
        for (int r = 0; r < 3; ++r) {
            if (byRow[r].empty()) {
                // centersY is 'center' vector. order[r].k is the index of the cluster
                rowCenterY[r] = centersY.at<float>(order[r].k, 0);
            }
            else {
                float s = 0.f;
                for (int idx : byRow[r]) s += rot[idx].y;
                rowCenterY[r] = s / (float)byRow[r].size();
            }
        }
        //same for rows (centersX by orderX)
        for (int c = 0; c < 3; ++c) {
            if (byCol[c].empty()) {
                colCenterX[c] = centersX.at<float>(orderX[c].k, 0);
            }
            else {
                float s = 0.f;
                for (int idx : byCol[c]) s += rot[idx].x;
                colCenterX[c] = s / (float)byCol[c].size();
            }
        }

        // (cx[c], cy[r]) is the nearset point in Row x Col intersection
        // Build 3x3 grid by greedy assignment to the 9 target intersections 
        struct Cand { int r, c, idx; float d; };
        std::vector<Cand> cands; cands.reserve(patches.size() * 9);

        // (x',y') targets of the 9 cells: center of each row/column in the rotation space
        for (int r = 0;r < 3;++r) {
            for (int c = 0;c < 3;++c) {
                for (int idx = 0; idx < (int)patches.size(); ++idx) {
                    int rr = label2row[labels.at<int>(idx, 0)];
                    int cc = label2col[labelsX.at<int>(idx, 0)];
                    float bonus = (rr == r) + (cc == c); 
                    float dx = rot[idx].x - colCenterX[c];
                    float dy = rot[idx].y - rowCenterY[r];
                    float d = std::abs(dx) + std::abs(dy);      // L1 יציב
                    d /= (1.0f + 0.25f * bonus);                // priority of the cell that matches the labels
                    cands.push_back({ r,c,idx,d });
                }
            }
        }

        //  sort by distance-ascending
        std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.d < b.d; });

        //patch greedy, take the closest candidates
        Patch grid[3][3];
        bool cell_used[3][3] = { {false,false,false},{false,false,false},{false,false,false} };
        std::vector<char> patch_used(patches.size(), 0);
        int assigned = 0;

        for (const auto& c : cands) {
            if (cell_used[c.r][c.c]) continue;
            if (patch_used[c.idx]) continue;
            grid[c.r][c.c] = patches[c.idx];
            cell_used[c.r][c.c] = true;
            patch_used[c.idx] = 1;
            if (++assigned == 9) break;
        }

        if (assigned != 9) {
            if (debug_mode) {
                std::cout << "[grid] greedy assignment failed (" << assigned << "/9)\n";
                cv::imshow("HSV (debug)", hsv_image);
                auto t1 = clk::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::cout << valid_images_names[i] << " took " << ms << " ms\n";
            }
            emit_marker_result(valid_images_names[i], false, FailureReason::ASSIGN_GRID, debug_mode);
            fail_count++; any_fail = true;
            continue;
        }

        if (debug_mode) std::cout << valid_images_names[i] << " grid_detected" << std::endl;
      
        // validation (normalized spacing) on rotated coords 
        bool grid_failed = false;
        float cvx = 1e9f, cvy = 1e9f;

        auto sort_row_by_xp = [&](int r) {
            std::array<Patch, 3> row = { grid[r][0], grid[r][1], grid[r][2] };
            std::sort(row.begin(), row.end(), [&](const Patch& a, const Patch& b) {
                return rot[a.id].x < rot[b.id].x;
                });
            return row;
            };

        auto sort_col_by_yp = [&](int c) {
            std::array<Patch, 3> col = { grid[0][c], grid[1][c], grid[2][c] };
            std::sort(col.begin(), col.end(), [&](const Patch& a, const Patch& b) {
                return rot[a.id].y < rot[b.id].y;
                });
            return col;
            };

        auto mean_vec = [](const std::vector<float>& v) {
            if (v.empty()) return 0.0f;
            float s = 0.f; for (float x : v) s += x; return s / (float)v.size();
            };

        std::vector<float> dx_norm, dy_norm;

        // rows: use x' in ascending order
        for (int r = 0; r < 3; ++r) {
            auto row = sort_row_by_xp(r);
            float x1 = rot[row[0].id].x, x2 = rot[row[1].id].x, x3 = rot[row[2].id].x;
            float d1 = x2 - x1, d2 = x3 - x2;
            if (d1 <= 0 || d2 <= 0) { grid_failed = true; break; }
            float m = 0.5f * (d1 + d2);
            if (m <= 0) { grid_failed = true; break; }
            dx_norm.push_back(d1 / m);
            dx_norm.push_back(d2 / m);
        }
        if (!grid_failed) {
            // cols: use y' in ascending order (top -> bottom)
            for (int c = 0; c < 3; ++c) {
                auto col = sort_col_by_yp(c);
                float y1 = rot[col[0].id].y, y2 = rot[col[1].id].y, y3 = rot[col[2].id].y;
                float d1 = y2 - y1, d2 = y3 - y2;
                if (d1 <= 0 || d2 <= 0) { grid_failed = true; break; }
                float m = 0.5f * (d1 + d2);
                if (m <= 0) { grid_failed = true; break; }
                dy_norm.push_back(d1 / m);
                dy_norm.push_back(d2 / m);
            }
        }

        // guards
        if (grid_failed || dx_norm.size() != 6 || dy_norm.size() != 6) {
            if (debug_mode) {
                std::cout << "[grid] spacing: invalid ordering or empty gaps\n";
                cv::imshow("HSV (debug)", hsv_image);
                auto t1 = clk::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::cout << valid_images_names[i] << " took " << ms << " ms\n";
            }
            emit_marker_result(valid_images_names[i], false, FailureReason::SPACING, debug_mode);
            fail_count++; any_fail = true;
            continue;
        }
        float meanDxN = mean_vec(dx_norm), meanDyN = mean_vec(dy_norm);
        if (meanDxN <= 0 || meanDyN <= 0) {
            if (debug_mode) {
                std::cout << "[grid] spacing: zero mean\n";
                cv::imshow("HSV (debug)", hsv_image);
                auto t1 = clk::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::cout << valid_images_names[i] << " took " << ms << " ms\n";
            }
            emit_marker_result(valid_images_names[i], false, FailureReason::SPACING, debug_mode);
            fail_count++; any_fail = true;
            continue;
        }

        auto stdev = [](const std::vector<float>& v) {
            if (v.size() < 2) return 0.f;
            float m = 0.f; for (auto x : v) m += x; m /= (float)v.size();
            float s2 = 0.f; for (auto x : v) s2 += (x - m) * (x - m);
            return std::sqrt(s2 / (float)(v.size() - 1));
            };

        float sdDxN = stdev(dx_norm), sdDyN = stdev(dy_norm);
        cvx = sdDxN / meanDxN;
        cvy = sdDyN / meanDyN;

        // thresholds
        bool spacing_ok = (cvx <= 0.55f && cvy <= 0.65f);
        if (debug_mode && !spacing_ok) {
            std::cout << "[grid] spacing CV initial failed: cvx=" << cvx << " cvy=" << cvy << "\n";
        }


        // debug draw final grid 
        if (debug_mode) {
            cv::Mat vis = image.clone();
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 3; ++c) {
                    const auto& p = grid[r][c];
                    cv::rectangle(vis, p.box, cv::Scalar(0, 255, 0), 2);
                    cv::Point center_i(cvRound(p.center.x), cvRound(p.center.y));
                    cv::circle(vis, center_i, 3, cv::Scalar(0, 0, 255), cv::FILLED);
                    cv::putText(vis, "(" + std::to_string(r) + "," + std::to_string(c) + ")",
                        center_i + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1);
                }
            }
            cv::imshow("Detections", vis);
            cv::waitKey(1);
        }
        // Convex Hull (9 patches)
        std::vector<cv::Point2f> corners;
        corners.reserve(9 * 4);
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const cv::Rect& b = grid[r][c].box;
                corners.emplace_back((float)b.x, (float)b.y);
                corners.emplace_back((float)(b.x + b.width), (float)b.y);
                corners.emplace_back((float)(b.x + b.width), (float)(b.y + b.height));
                corners.emplace_back((float)b.x, (float)(b.y + b.height));
            }
        }
        std::vector<cv::Point2f> hull;
        cv::convexHull(corners, hull);

        // hull sanity
        if (hull.size() < 3) {
            if (debug_mode) {
                std::cout << "[hull] too small: " << hull.size() << "\n";
                cv::imshow("HSV (debug)", hsv_image);
                auto t1 = clk::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::cout << valid_images_names[i] << " took " << ms << " ms\n";
            }
            emit_marker_result(valid_images_names[i], false, FailureReason::SMALL_HULL, debug_mode);
            fail_count++; any_fail = true;
            continue;
        }

        if (debug_mode) {
            cv::Mat hullVis = image.clone();
            for (size_t j = 0; j < hull.size(); ++j) {
                cv::line(hullVis, hull[j], hull[(j + 1) % hull.size()], cv::Scalar(0, 255, 255), 2);
            }
            cv::imshow("Hull (debug)", hullVis);
            cv::waitKey(1);
        }

        // Coverage ratio 
        double hull_area = std::abs(cv::contourArea(hull));

        // bounding box of the patches by their Rect
        float minx = std::numeric_limits<float>::max();
        float miny = std::numeric_limits<float>::max();
        float maxx = -std::numeric_limits<float>::max();
        float maxy = -std::numeric_limits<float>::max();

        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 3; ++c) {
                const cv::Rect& b = grid[r][c].box;
                minx = std::min(minx, (float)b.x);
                miny = std::min(miny, (float)b.y);
                maxx = std::max(maxx, (float)(b.x + b.width));
                maxy = std::max(maxy, (float)(b.y + b.height));
            }
        }

        double bbox_w = std::max(0.0f, maxx - minx);
        double bbox_h = std::max(0.0f, maxy - miny);
        double bbox_area = bbox_w * bbox_h;

        if (bbox_area <= 0.0) {
            if (debug_mode) {
                std::cout << "[coverage] invalid bbox area\n";
                cv::imshow("HSV (debug)", hsv_image);
                auto t1 = clk::now();
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
                std::cout << valid_images_names[i] << " took " << ms << " ms\n";
            }
            emit_marker_result(valid_images_names[i], false, FailureReason::BAD_BBOX, debug_mode);
            fail_count++; any_fail = true;
            continue;
        }

        double coverage_ratio = hull_area / bbox_area;
        int pct = (int)std::lround(coverage_ratio * 100.0);
        if (debug_mode) {
            std::cout << "[coverage] hull=" << hull_area
                << " bbox=" << bbox_area
                << " ratio=" << coverage_ratio
                << " (" << pct << "%)" << std::endl;
        }
        constexpr double COVERAGE_THRESH = 0.70; // must
        constexpr double COVERAGE_FALLBACK = 0.80; //if spacing fail by coverage is strong , we accept
        constexpr double COVERAGE_SOFT = 0.75; // soft

        // fallback 1: pacing fail by coverage is strong
        if (!spacing_ok && coverage_ratio >= COVERAGE_FALLBACK) {
            if (debug_mode) std::cout << "[grid] spacing fallback accepted (coverage=" << coverage_ratio << ")\n";
            spacing_ok = true;
        }
        // fallback 2
        if (!spacing_ok && cvx <= 0.60f && cvy <= 0.70f && coverage_ratio >= COVERAGE_SOFT) {
            if (debug_mode) std::cout << "[grid] spacing soft-fallback accepted (cvx=" << cvx << ", cvy=" << cvy
                << ", coverage=" << coverage_ratio << ")\n";
            spacing_ok = true;
        }
        // Final output
        bool ok = (spacing_ok && coverage_ratio >= COVERAGE_THRESH);
        if (ok) {
            emit_marker_result(valid_images_names[i], true, FailureReason::NONE, debug_mode,pct);
            pass_count++;
        }
        else {
            // if spacing_ok fail – FR_3, otherwise FR_6
            FailureReason fr = spacing_ok ? FailureReason::LOW_COVERAGE : FailureReason::SPACING;
            if (debug_mode) {
                std::cout << "[final] cvx=" << cvx << " cvy=" << cvy
                    << " coverage_ratio=" << coverage_ratio << "\n";
            }
            emit_marker_result(valid_images_names[i], false, fr, debug_mode);
            fail_count++; any_fail = true;
        }

        // runtime only in debug
        if (debug_mode) {
            auto t1 = clk::now();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
            std::cout << valid_images_names[i] << " took " << ms << " ms\n";
        }
    }
    

    if (debug_mode) {
        std::cout << "\nSummary: passed=" << pass_count
            << " failed=" << fail_count
            << " out of " << (pass_count + fail_count) << std::endl;
        cout << "Press any key to close debug windows..." << endl;
        waitKey(0);
    }

    return any_fail ? 1 : 0;
}

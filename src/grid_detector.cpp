#include "grid_detector.hpp"
using namespace cv;
using std::vector;

static void pca_rotate(const std::vector<cv::Point2f>& pts,
                       std::vector<cv::Point2f>& rotated,
                       cv::Point2f& mean,
                       cv::Mat& eigvecs) {
    Mat data((int)pts.size(), 2, CV_32F);
    for (int i=0;i<(int)pts.size();++i){ data.at<float>(i,0)=pts[i].x; data.at<float>(i,1)=pts[i].y; }
    PCA pca(data, Mat(), PCA::DATA_AS_ROW);
    eigvecs = pca.eigenvectors.clone();
    mean = Point2f(pca.mean.at<float>(0,0), pca.mean.at<float>(0,1));
    rotated.resize(pts.size());
    for (int i=0;i<(int)pts.size();++i) {
        Vec2f v(pts[i].x-mean.x, pts[i].y-mean.y);
        Mat vm = (Mat_<float>(1,2) << v[0], v[1]);
        Mat r = vm * eigvecs.t();
        rotated[i] = Point2f(r.at<float>(0,0), r.at<float>(0,1));
    }
}

static float stdev(const std::vector<float>& v){
    if (v.size()<2) return 0.f;
    float m=0.f; for(float x:v) m+=x; m/= (float)v.size();
    float s2=0.f; for(float x:v) s2+=(x-m)*(x-m);
    return std::sqrt(s2/(float)(v.size()-1));
}

FailureReason detect_grid_and_spacing(
    const std::vector<Patch>& patches,
    GridDetection& out,
    const GridParams& params)
{
    if (patches.size() < 3) return FailureReason::FEW_PATCHES;

    // PCA rotate
    std::vector<Point2f> centers; centers.reserve(patches.size());
    for (auto& p: patches) centers.push_back(p.center);
    Point2f meanP; Mat eigvecs;
    pca_rotate(centers, out.rot, meanP, eigvecs);

    // KMeans rows (y')
    Mat sampY((int)patches.size(),1,CV_32F);
    for (int i=0;i<(int)patches.size();++i) sampY.at<float>(i,0)=out.rot[i].y;
    Mat labelsY, centersY;
    kmeans(sampY, 3, labelsY, TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER,100,1e-3), 5, KMEANS_PP_CENTERS, centersY);

    struct ClusterInfo{ float cy; int k; };
    std::vector<ClusterInfo> orderY; orderY.reserve(3);
    for (int k=0;k<3;++k) orderY.push_back({ centersY.at<float>(k,0), k });
    std::sort(orderY.begin(), orderY.end(), [](auto&a, auto&b){ return a.cy<b.cy; });
    int label2row[3]; for (int r=0;r<3;++r) label2row[orderY[r].k]=r;

    // KMeans cols (x')
    Mat sampX((int)patches.size(),1,CV_32F);
    for (int i=0;i<(int)patches.size();++i) sampX.at<float>(i,0)=out.rot[i].x;
    Mat labelsX, centersX;
    kmeans(sampX, 3, labelsX, TermCriteria(TermCriteria::EPS+TermCriteria::MAX_ITER,100,1e-3), 5, KMEANS_PP_CENTERS, centersX);

    struct Cx{ float x; int k; };
    std::vector<Cx> orderX; orderX.reserve(3);
    for (int k=0;k<3;++k) orderX.push_back({ centersX.at<float>(k,0), k });
    std::sort(orderX.begin(), orderX.end(), [](auto&a, auto&b){ return a.x<b.x; });
    int label2col[3]; for (int c=0;c<3;++c) label2col[orderX[c].k]=c;

    // Row/Col centers in x',y'
    float rowCenterY[3], colCenterX[3];
    std::vector<std::vector<int>> byRow(3), byCol(3);
    for (int idx=0; idx<(int)patches.size(); ++idx) {
        int r = label2row[labelsY.at<int>(idx,0)];
        int c = label2col[labelsX.at<int>(idx,0)];
        byRow[r].push_back(idx);
        byCol[c].push_back(idx);
    }
    for (int r=0;r<3;++r){
        if (byRow[r].empty()) rowCenterY[r] = centersY.at<float>(orderY[r].k,0);
        else {
            float s=0.f; for(int idx:byRow[r]) s += out.rot[idx].y;
            rowCenterY[r]= s/(float)byRow[r].size();
        }
    }
    for (int c=0;c<3;++c){
        if (byCol[c].empty()) colCenterX[c] = centersX.at<float>(orderX[c].k,0);
        else {
            float s=0.f; for(int idx:byCol[c]) s += out.rot[idx].x;
            colCenterX[c]= s/(float)byCol[c].size();
        }
    }

    // Greedy assignment to 9 intersections
    struct Cand { int r,c,idx; float d; };
    std::vector<Cand> cands; cands.reserve(patches.size()*9);
    for (int r=0;r<3;++r) for(int c=0;c<3;++c){
        for (int idx=0; idx<(int)patches.size(); ++idx){
            int rr = label2row[labelsY.at<int>(idx,0)];
            int cc = label2col[labelsX.at<int>(idx,0)];
            float bonus = (rr==r) + (cc==c);
            float dx = out.rot[idx].x - colCenterX[c];
            float dy = out.rot[idx].y - rowCenterY[r];
            float d = std::abs(dx) + std::abs(dy);
            d /= (1.0f + 0.25f*bonus);
            cands.push_back({r,c,idx,d});
        }
    }
    std::sort(cands.begin(), cands.end(), [](auto&a, auto&b){ return a.d<b.d; });

    bool cell_used[3][3] = {{0}};
    std::vector<char> patch_used(patches.size(), 0);
    int assigned = 0;
    for (const auto& c : cands) {
        if (cell_used[c.r][c.c]) continue;
        if (patch_used[c.idx]) continue;
        out.grid[c.r][c.c] = patches[c.idx];
        cell_used[c.r][c.c] = true;
        patch_used[c.idx] = 1;
        if (++assigned==9) break;
    }
    if (assigned != 9) return FailureReason::ASSIGN_GRID;

    // Spacing check on rotated coords
    auto sort_row_by_xp = [&](int r){
        std::array<Patch,3> row = { out.grid[r][0], out.grid[r][1], out.grid[r][2] };
        std::sort(row.begin(), row.end(), [&](const Patch&a,const Patch&b){ return out.rot[a.id].x < out.rot[b.id].x; });
        return row;
    };
    auto sort_col_by_yp = [&](int c){
        std::array<Patch,3> col = { out.grid[0][c], out.grid[1][c], out.grid[2][c] };
        std::sort(col.begin(), col.end(), [&](const Patch&a,const Patch&b){ return out.rot[a.id].y < out.rot[b.id].y; });
        return col;
    };
    auto mean_vec = [](const std::vector<float>& v){ float s=0.f; for(float x:v) s+=x; return v.empty()?0.f:s/(float)v.size(); };

    bool grid_failed=false;
    std::vector<float> dx_norm, dy_norm;
    for (int r=0;r<3;++r){
        auto row = sort_row_by_xp(r);
        float x1=out.rot[row[0].id].x, x2=out.rot[row[1].id].x, x3=out.rot[row[2].id].x;
        float d1=x2-x1, d2=x3-x2; if (d1<=0 || d2<=0) { grid_failed=true; break; }
        float m=0.5f*(d1+d2); if (m<=0) { grid_failed=true; break; }
        dx_norm.push_back(d1/m); dx_norm.push_back(d2/m);
    }
    if (!grid_failed){
        for (int c=0;c<3;++c){
            auto col = sort_col_by_yp(c);
            float y1=out.rot[col[0].id].y, y2=out.rot[col[1].id].y, y3=out.rot[col[2].id].y;
            float d1=y2-y1, d2=y3-y2; if (d1<=0 || d2<=0) { grid_failed=true; break; }
            float m=0.5f*(d1+d2); if (m<=0) { grid_failed=true; break; }
            dy_norm.push_back(d1/m); dy_norm.push_back(d2/m);
        }
    }
    if (grid_failed || dx_norm.size()!=6 || dy_norm.size()!=6) return FailureReason::SPACING;

    float meanDxN=mean_vec(dx_norm), meanDyN=mean_vec(dy_norm);
    if (meanDxN<=0 || meanDyN<=0) return FailureReason::SPACING;

    float sdDxN=stdev(dx_norm), sdDyN=stdev(dy_norm);
    out.cvx = sdDxN/meanDxN;
    out.cvy = sdDyN/meanDyN;

    out.spacing_ok = (out.cvx <= params.cvx_thresh && out.cvy <= params.cvy_thresh);
    return FailureReason::NONE;
}

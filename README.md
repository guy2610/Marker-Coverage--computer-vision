# Marker Coverage Estimator

## Overview
This project implements a **Marker Coverage Estimator** using OpenCV.  
It detects a **3×3 grid of colored patches** (red, green, blue, yellow, cyan, magenta), validates geometry, and computes a **coverage ratio**.  
The algorithm is robust to **rotation (≤45°)**, scale, and perspective distortions.

Final output:
- `marker_found <filename>` if the marker passes validation.  
- `marker_not_found <filename> <FR_X>` with a failure reason otherwise. 
---

## Algorithm (High-Level)
1. Convert image to HSV and threshold by the six expected colors.  
2. Extract contours, filter by relative area, keep bounding boxes.  
3. Rotate patch centers with **PCA** to normalize tilt/rotation.  
4. Cluster with **k-means** into 3 rows × 3 columns.  
5. Greedy assignment to fill all 9 grid cells.  
6. Validate spacing by checking uniformity of normalized gaps:  
   - Accept if `cvx ≤ 0.55` and `cvy ≤ 0.65`.  
   - Allow fallback if slightly higher but coverage is strong.  
7. Compute convex hull of the 9 patches and measure coverage ratio (`hull_area / image_area`).  
8. Accept marker if ratio ≥ 70% and spacing passes thresholds, or via fallback rules.

(Full description in [ALGORITHM.md](ALGORITHM.md))  
---

## Project Structure

├── CMakeLists.txt
├── README.md
├── ALGORITHM.md
├── src/
│ ├── main.cpp
│ ├── color_segmentation.cpp
│ ├── grid_detector.cpp
│ ├── coverage.cpp
├── include/
│ ├── types.hpp
│ ├── color_segmentation.hpp
│ ├── grid_detector.hpp
│ ├── coverage.hpp
├── data/ # Example input images
└── build/ # Build output (ignored in git)
 
---

## Build Instructions

### Prerequisites
- CMake ≥ 3.15  
- OpenCV ≥ 4.5 (via [vcpkg](https://github.com/microsoft/vcpkg) or system package)  
- Compiler: MSVC (Windows) / g++ or clang++ (Linux/MacOS)  

### Windows
powershell
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

### Linux/MacOS
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

###Usage
./MarkerCoverageEstimator [--debug] ./data/hi1.png ./data/hi2.png ...

###Functional Requirements Coverage
FR-1: Input validation & segmentation
FR-2: Grid construction (PCA + clustering + assignment)
FR-3: Robustness to rotation/scale/tilt (≤45°)
FR-4: Runtime <200ms @ 640×480 (typical: 40–70ms)
FR-5: Coverage ratio via convex hull / bounding box
FR-6: Decision: marker found / not found
FR-7: Documentation: README + algorithm description

#####Example######
The first run is with debug mode, and the second is without

C:\....\Release>SodyoAssignment.exe "C:\....\data\hi1.png" "C:\....\data\hi2.png" "C:\....\data\hi3.png" "C:\....\data\hi4.png" "C:\....\data\hi5.png" "C:\....\data\hi6.png" "C:\....\data\hi7.png" "C:\....\data\hi8.png" "C:\....\data\hi9.png" "C:\....\data\hi10.png" "C:\....\data\hi11.png" "C:\....\data\hi12.png" "C:\....\data\hi13.png" "C:\....\data\hi14.png" "C:\....\data\hi15.png" "C:\....\data\hi16.png" "C:\....\data\hi17.png" "C:\....\data\hi18.png" "C:\....\data\hi19.png" "C:\....\data\hi20.png"
C:\....\data\hi1.png 53%
C:\....\data\hi2.png 64%
C:\....\data\hi3.png 66%
C:\....\data\hi4.png 69%
C:\....\data\hi5.png 64%
C:\....\data\hi6.png 0%
C:\....\data\hi7.png 77%
C:\....\data\hi8.png 0%
C:\....\data\hi9.png 77%
C:\....\data\hi10.png 75%
C:\....\data\hi11.png 55%
C:\....\data\hi12.png 66%
C:\....\data\hi13.png 57%
C:\....\data\hi14.png 53%
C:\....\data\hi15.png 71%
C:\....\data\hi16.png 57%
C:\....\data\hi17.png 64%
C:\....\data\hi18.png 77%
C:\....\data\hi19.png 70%
C:\....\data\hi20.png 77%

Summary: passed=18 failed=2 out of 20

C:\....\Release>SodyoAssignment.exe "--debug" "C:\....\data\hi1.png" "C:\....\data\hi2.png" "C:\....\data\hi3.png" "C:\....\data\hi4.png" "C:\....\data\hi5.png" "C:\....\data\hi6.png" "C:\....\data\hi7.png" "C:\....\data\hi8.png" "C:\....\data\hi9.png" "C:\....\data\hi10.png" "C:\....\data\hi11.png" "C:\....\data\hi12.png" "C:\....\data\hi13.png" "C:\....\data\hi14.png" "C:\....\data\hi15.png" "C:\....\data\hi16.png" "C:\....\data\hi17.png" "C:\....\data\hi18.png" "C:\....\data\hi19.png" "C:\....\data\hi20.png"
marker_found C:\....\data\hi1.png
[coverage] hull=1660 image=3120 ratio=0.532051 (53%)
C:\....\data\hi1.png took 8 ms
marker_found C:\....\data\hi2.png
[coverage] hull=8578 image=13500 ratio=0.635407 (64%)
C:\....\data\hi2.png took 3 ms
marker_found C:\....\data\hi3.png
[coverage] hull=3414.5 image=5168 ratio=0.6607 (66%)
C:\....\data\hi3.png took 2 ms
marker_found C:\....\data\hi4.png
[coverage] hull=5171.5 image=7475 ratio=0.691839 (69%)
C:\....\data\hi4.png took 2 ms
marker_found C:\....\data\hi5.png
[coverage] hull=5173.5 image=8085 ratio=0.639889 (64%)
C:\....\data\hi5.png took 2 ms
marker_not_found C:\....\data\hi6.png LOW_COVERAGE
[final] cvx=0.407866 cvy=0.771897 coverage_ratio=0.407244
C:\....\data\hi6.png took 2 ms
marker_found C:\....\data\hi7.png
[coverage] hull=4497.5 image=5850 ratio=0.768803 (77%)
C:\....\data\hi7.png took 2 ms
marker_not_found C:\....\data\hi8.png LOW_COVERAGE
[final] cvx=0.117667 cvy=0.208022 coverage_ratio=0.385823
C:\....\data\hi8.png took 1 ms
marker_found C:\....\data\hi9.png
[coverage] hull=16017 image=20850 ratio=0.768201 (77%)
C:\....\data\hi9.png took 2 ms
marker_found C:\....\data\hi10.png
[coverage] hull=2326.5 image=3102 ratio=0.75 (75%)
C:\....\data\hi10.png took 2 ms
marker_found C:\....\data\hi11.png
[coverage] hull=6687.5 image=12096 ratio=0.552869 (55%)
C:\....\data\hi11.png took 2 ms
marker_found C:\....\data\hi12.png
[coverage] hull=3414.5 image=5168 ratio=0.6607 (66%)
C:\....\data\hi12.png took 1 ms
marker_found C:\....\data\hi13.png
[coverage] hull=2028.5 image=3577 ratio=0.567095 (57%)
C:\....\data\hi13.png took 1 ms
marker_found C:\....\data\hi14.png
[coverage] hull=5220 image=9785 ratio=0.53347 (53%)
C:\....\data\hi14.png took 2 ms
marker_found C:\....\data\hi15.png
[coverage] hull=7106 image=9951 ratio=0.714099 (71%)
C:\....\data\hi15.png took 1 ms
marker_found C:\....\data\hi16.png
[coverage] hull=1957.5 image=3430 ratio=0.5707 (57%)
C:\....\data\hi16.png took 1 ms
marker_found C:\....\data\hi17.png
[coverage] hull=8055 image=12502 ratio=0.644297 (64%)
C:\....\data\hi17.png took 1 ms
marker_found C:\....\data\hi18.png
[coverage] hull=8108.5 image=10580 ratio=0.766399 (77%)
C:\....\data\hi18.png took 1 ms
marker_found C:\....\data\hi19.png
[coverage] hull=2368 image=3382 ratio=0.700177 (70%)
C:\....\data\hi19.png took 1 ms
marker_found C:\....\data\hi20.png
[coverage] hull=8496 image=11024 ratio=0.770682 (77%)
C:\....\data\hi20.png took 1 ms

Summary: passed=18 failed=2 out of 20

C:\....\Release>
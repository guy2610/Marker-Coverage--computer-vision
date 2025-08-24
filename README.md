# Marker Coverage Estimator

## Overview
This project implements a **Marker Coverage Estimator** based on computer vision using OpenCV.  
The task is to detect a **3×3 grid of colored patches** (red, green, blue, yellow, cyan, magenta) inside an image, validate its geometry, and compute its **coverage ratio**.  
The algorithm is designed to be robust against **rotation (up to 45°), scale, and perspective distortions**, while rejecting invalid detections.  
Final output:  
- `marker_found <filename>` if the marker passes all validation steps.  
- `marker_not_found <filename> <FR_X>` with a failure reason otherwise.

---

## Algorithm (High-Level, ~150 words)
The algorithm starts by resizing the input image to a maximum of 640×480 for performance.  
It converts the image to HSV color space and applies thresholding to segment the six expected colors.  
Contours are extracted from each color mask, filtered by area, and their bounding boxes are used to define patch candidates.  

To build the 3×3 grid, patch centers are rotated using **PCA** to align with the dominant orientation, then clustered into rows and columns using **k-means**.  
A greedy assignment matches the best candidates to the 9 grid cells.  
Validation checks ensure that patch spacing is uniform (low variance in normalized gaps).  

Finally, the convex hull of the 9 patches is computed, and its area compared to the bounding box area to produce a **coverage ratio**.  
If the ratio exceeds the threshold (≥70%), and spacing constraints are satisfied, the marker is accepted.  

---

## Functional Requirements Coverage
- **FR-1**: Input validation and color segmentation.  
- **FR-2**: Grid construction (PCA + clustering + greedy assignment).  
- **FR-3**: Robustness to scale/rotation/tilt (tested up to ~45°).  
- **FR-4**: Runtime <200ms on 640×480 images (typical 40–70ms).  
- **FR-5**: Coverage ratio computed using convex hull vs bounding box.  
- **FR-6**: Final decision: marker found/not found.  
- **FR-7**: Unit tests (basic assertions provided separately).  
- **FR-8**: Documentation (this README + algorithm description).  

---

## Build Instructions

### Prerequisites
- CMake ≥ 3.15  
- OpenCV ≥ 4.5 (installed via [vcpkg](https://github.com/microsoft/vcpkg) or system package)  
- Compiler: MSVC (Windows) / g++ or clang++ (Linux, MacOS)

### Windows (MSVC + vcpkg)
```bash
cd "C:\Users\<User>\Documents\Sodyo assigment\Sodyo assigment"
cmake -S . -B build -G "Visual Studio 17 2022" -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release


#####results######
The first run is with debug mode, and the second is without

C:\Users\guyev\Documents\Sodyo assigment\Sodyo assigment\build\Release>SodyoAssignment.exe "--debug" "C:\Users\guyev\Documents\Sodyo assigment\data\hi1.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi2.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi3.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi4.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi5.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi6.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi7.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi8.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi9.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi10.png"
[kmeans] rows counts (pre-pick): 4 5 3
C:\Users\guyev\Documents\Sodyo assigment\data\hi1.png grid_detected
[coverage] hull=1660 bbox=2310 ratio=0.718615 (72%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi1.png 72%
C:\Users\guyev\Documents\Sodyo assigment\data\hi1.png took 507 ms
[kmeans] rows counts (pre-pick): 5 3 2
C:\Users\guyev\Documents\Sodyo assigment\data\hi2.png grid_detected
[coverage] hull=8578 bbox=9660 ratio=0.887992 (89%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi2.png 89%
C:\Users\guyev\Documents\Sodyo assigment\data\hi2.png took 46 ms
[kmeans] rows counts (pre-pick): 8 4 6
C:\Users\guyev\Documents\Sodyo assigment\data\hi3.png grid_detected
[coverage] hull=3414.5 bbox=4118 ratio=0.829165 (83%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi3.png 83%
C:\Users\guyev\Documents\Sodyo assigment\data\hi3.png took 46 ms
[kmeans] rows counts (pre-pick): 3 7 7
C:\Users\guyev\Documents\Sodyo assigment\data\hi4.png grid_detected
[coverage] hull=5171.5 bbox=5865 ratio=0.881756 (88%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi4.png 88%
C:\Users\guyev\Documents\Sodyo assigment\data\hi4.png took 48 ms
[kmeans] rows counts (pre-pick): 4 4 6
C:\Users\guyev\Documents\Sodyo assigment\data\hi5.png grid_detected
[coverage] hull=5173.5 bbox=6248 ratio=0.828025 (83%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi5.png 83%
C:\Users\guyev\Documents\Sodyo assigment\data\hi5.png took 45 ms
[kmeans] rows counts (pre-pick): 7 4 1
C:\Users\guyev\Documents\Sodyo assigment\data\hi6.png grid_detected
[grid] spacing CV initial failed: cvx=0.407866 cvy=0.771897
[coverage] hull=1433.5 bbox=1856 ratio=0.77236 (77%)
[final] cvx=0.407866 cvy=0.771897 coverage_ratio=0.77236
marker_not_found C:\Users\guyev\Documents\Sodyo assigment\data\hi6.png FR_3
C:\Users\guyev\Documents\Sodyo assigment\data\hi6.png took 48 ms
[kmeans] rows counts (pre-pick): 7 8 2
C:\Users\guyev\Documents\Sodyo assigment\data\hi7.png grid_detected
[coverage] hull=4497.5 bbox=4950 ratio=0.908586 (91%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi7.png 91%
C:\Users\guyev\Documents\Sodyo assigment\data\hi7.png took 45 ms
[kmeans] rows counts (pre-pick): 3 3 3
C:\Users\guyev\Documents\Sodyo assigment\data\hi8.png grid_detected
[coverage] hull=1265.5 bbox=1575 ratio=0.803492 (80%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi8.png 80%
C:\Users\guyev\Documents\Sodyo assigment\data\hi8.png took 48 ms
[kmeans] rows counts (pre-pick): 3 6 8
C:\Users\guyev\Documents\Sodyo assigment\data\hi9.png grid_detected
[coverage] hull=16017 bbox=18348 ratio=0.872956 (87%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi9.png 87%
C:\Users\guyev\Documents\Sodyo assigment\data\hi9.png took 46 ms
[kmeans] rows counts (pre-pick): 6 4 1
C:\Users\guyev\Documents\Sodyo assigment\data\hi10.png grid_detected
[coverage] hull=2326.5 bbox=2640 ratio=0.88125 (88%)
C:\Users\guyev\Documents\Sodyo assigment\data\hi10.png 88%
C:\Users\guyev\Documents\Sodyo assigment\data\hi10.png took 48 ms

Summary: passed=9 failed=1 out of 10
Press any key to close debug windows...

C:\Users\guyev\Documents\Sodyo assigment\Sodyo assigment\build\Release>SodyoAssignment.exe "C:\Users\guyev\Documents\Sodyo assigment\data\hi1.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi2.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi3.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi4.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi5.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi6.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi7.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi8.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi9.png" "C:\Users\guyev\Documents\Sodyo assigment\data\hi10.png"
C:\Users\guyev\Documents\Sodyo assigment\data\hi1.png 72%
C:\Users\guyev\Documents\Sodyo assigment\data\hi2.png 89%
C:\Users\guyev\Documents\Sodyo assigment\data\hi3.png 83%
C:\Users\guyev\Documents\Sodyo assigment\data\hi4.png 88%
C:\Users\guyev\Documents\Sodyo assigment\data\hi5.png 83%
marker_not_found C:\Users\guyev\Documents\Sodyo assigment\data\hi6.png FR_3
C:\Users\guyev\Documents\Sodyo assigment\data\hi7.png 91%
C:\Users\guyev\Documents\Sodyo assigment\data\hi8.png 80%
C:\Users\guyev\Documents\Sodyo assigment\data\hi9.png 87%
C:\Users\guyev\Documents\Sodyo assigment\data\hi10.png 88%

C:\Users\guyev\Documents\Sodyo assigment\Sodyo assigment\build\Release>
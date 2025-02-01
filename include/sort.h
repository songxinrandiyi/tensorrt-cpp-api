#pragma once

#include "kalman_tracker.h"
#include "hungarian.h"
#include <vector>
#include <opencv2/opencv.hpp>

class Sort {
public:
    Sort();
    ~Sort();

    // Update the trackers with the current frame detections
    std::vector<std::vector<float>> update(const std::vector<cv::Rect_<float>> &rects);

private:
    std::vector<KalmanTracker> trackers;  // List of active trackers
    int frameCount;  // Frame counter

    // Helper to calculate the IoU (Intersection over Union)
    double calculateIoU(const cv::Rect_<float> &box1, const cv::Rect_<float> &box2);
};

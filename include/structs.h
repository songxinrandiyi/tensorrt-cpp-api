#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};

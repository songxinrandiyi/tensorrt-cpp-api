#pragma once

#include <opencv2/opencv.hpp>

class KalmanTracker
{
public:
    KalmanTracker(cv::Rect_<float> initBox);
    ~KalmanTracker();
private:
    cv::KalmanFilter kf;
    cv::Mat mat;
};


#pragma once

#include <opencv2/opencv.hpp>

class KalmanTracker {
public:
    KalmanTracker(cv::Rect_<float> initBox);
    ~KalmanTracker();

    cv::Rect_<float> predict();
    void update(cv::Rect_<float> stateMat);
    void setLabel(int value) { label = value; } 
    int getLabel() const { return label; }
    int getID() const { return trackID; }
    int getAge() const { return age; }
    int getTotalVisibleCount() const { return totalVisibleCount; }
    int getConsecutiveInvisibleCount() const { return consecutiveInvisibleCount; }
    void increaseConsecutiveInvisibleCount() { consecutiveInvisibleCount++; }

private:
    static int kfCount;
    int trackID;
    int label;

    cv::KalmanFilter kf;
    cv::Mat state;
    cv::Mat meas;
    int age;
    int totalVisibleCount;
    int consecutiveInvisibleCount;

    void initKF(cv::Rect_<float> initBox);
};

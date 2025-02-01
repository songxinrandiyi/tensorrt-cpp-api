#include "kalman_tracker.h"

int KalmanTracker::kfCount = 0;

KalmanTracker::KalmanTracker(cv::Rect_<float> initBox) {
    if (kfCount > 1000) kfCount = 0;
    trackID = kfCount++;
    age = 0;
    totalVisibleCount = 0;
    consecutiveInvisibleCount = 0;
    initKF(initBox);
}

KalmanTracker::~KalmanTracker() {}

void KalmanTracker::initKF(cv::Rect_<float> initBox) {
    kf = cv::KalmanFilter(7, 4, 0);

    kf.transitionMatrix = (cv::Mat_<float>(7, 7) <<
        1, 0, 0, 0, 1, 0, 0,  
        0, 1, 0, 0, 0, 1, 0,  
        0, 0, 1, 0, 0, 0, 1,  
        0, 0, 0, 1, 0, 0, 0,  
        0, 0, 0, 0, 1, 0, 0,  
        0, 0, 0, 0, 0, 1, 0,  
        0, 0, 0, 0, 0, 0, 1);

    state = cv::Mat::zeros(7, 1, CV_32F);
    meas = cv::Mat::zeros(4, 1, CV_32F);

    kf.statePre.at<float>(0) = initBox.x;
    kf.statePre.at<float>(1) = initBox.y;
    kf.statePre.at<float>(2) = initBox.width;
    kf.statePre.at<float>(3) = initBox.height;
    kf.statePre.at<float>(4) = 0;  // dx
    kf.statePre.at<float>(5) = 0;  // dy
    kf.statePre.at<float>(6) = 0;  // dwidth, dheight are assumed constant

    kf.statePost = kf.statePre.clone();  

    setIdentity(kf.measurementMatrix);
    setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-3)); 
    setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-2)); 
    setIdentity(kf.errorCovPost, cv::Scalar::all(1));
}

cv::Rect2f KalmanTracker::predict() {
    cv::Mat pred = kf.predict();
    return cv::Rect_<float>(
        pred.at<float>(0), pred.at<float>(1),
        pred.at<float>(2), pred.at<float>(3));
}

void KalmanTracker::update(cv::Rect_<float> stateMat) {
    meas.at<float>(0) = stateMat.x;
    meas.at<float>(1) = stateMat.y;
    meas.at<float>(2) = stateMat.width;
    meas.at<float>(3) = stateMat.height;

    if (!kf.transitionMatrix.empty()) {
        kf.correct(meas);
    }

    age++;
    totalVisibleCount++;
    consecutiveInvisibleCount = 0;  // Reset on successful detection
}

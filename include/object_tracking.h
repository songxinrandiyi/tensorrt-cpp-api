#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "kalman_tracker.h"
#include "config.h"
#include "structs.h"

// Function to calculate IoU (Intersection over Union)
float calculateIoU(const cv::Rect2f& boxA, const cv::Rect2f& boxB);

// Function to associate detections with trackers using IoU
void associateDetectionsWithTrackers(const std::vector<Object>& detections, std::vector<KalmanTracker>& trackers, float iouThreshold);

void drawObjects(cv::Mat &image, const cv::Rect2f &box, const int id, const int label, unsigned int scale = 2);


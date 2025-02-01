#include "sort.h"
#include "hungarian.h"
#include <iostream> // For debugging output

Sort::Sort() : frameCount(0) {}

Sort::~Sort() {}

const int MIN_TRACKER_LIFETIME = 10; // Minimum frames a tracker must live

double Sort::calculateIoU(const cv::Rect_<float> &box1, const cv::Rect_<float> &box2) {
    float x1 = std::max(box1.x, box2.x);
    float y1 = std::max(box1.y, box2.y);
    float x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    float y2 = std::min(box1.y + box1.height, box2.y + box2.height);

    float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    float unionArea = box1.area() + box2.area() - intersection;

    float iou = (unionArea > 0.0) ? (intersection / unionArea) : 0.0;
    std::cout << "IoU between box1: " << box1 << " and box2: " << box2 << " = " << iou << std::endl;
    return iou;
}

std::vector<std::vector<float>> Sort::update(const std::vector<cv::Rect_<float>> &rects) {
    frameCount++;
    std::vector<std::vector<float>> results;

    // Step 1: Predict the next position of each tracker
    std::vector<cv::Rect_<float>> predictedRects;
    for (auto &tracker : trackers) {
        predictedRects.push_back(tracker.predict());
    }

    // Step 2: Compute cost matrix using IoU
    std::vector<std::vector<double>> costMatrix(rects.size(), std::vector<double>(predictedRects.size(), 0));
    for (size_t i = 0; i < rects.size(); ++i) {
        for (size_t j = 0; j < predictedRects.size(); ++j) {
            costMatrix[i][j] = 1.0 - calculateIoU(rects[i], predictedRects[j]); // Cost = 1 - IoU
            std::cout << "Cost[" << i << "][" << j << "] = " << costMatrix[i][j] << std::endl;
        }
    }

    // Step 3: Solve the assignment problem using Hungarian algorithm
    std::vector<int> assignment;
    HungarianAlgorithm hungarian;
    hungarian.solve(costMatrix, assignment);

    std::cout << "Assignments: ";
    for (auto a : assignment) {
        std::cout << a << " ";
    }
    std::cout << std::endl;

    // Step 4: Update trackers with assigned detections or create new ones
    std::vector<bool> trackerAssigned(trackers.size(), false);
    for (size_t i = 0; i < assignment.size(); ++i) {
        if (assignment[i] >= 0 && costMatrix[i][assignment[i]] < 1.0) { // Valid assignment
            trackers[assignment[i]].update(rects[i]);
            trackerAssigned[assignment[i]] = true;
        } else {
            std::cout << "Creating new tracker for detection: " << rects[i] << std::endl;
            trackers.push_back(KalmanTracker(rects[i]));
        }
    }

    // Remove trackers that were not assigned any detections, with lifetime check
    trackers.erase(std::remove_if(trackers.begin(), trackers.end(),
                                  [&trackerAssigned, idx = 0](KalmanTracker &tracker) mutable {
                                      if (!trackerAssigned[idx] && tracker.getAge() >= MIN_TRACKER_LIFETIME) {
                                          std::cout << "Removing tracker ID " << tracker.getID() << " due to inactivity." << std::endl;
                                          return true;
                                      }
                                      idx++;
                                      return false;
                                  }),
                   trackers.end());

    // Step 5: Collect results (bounding boxes and IDs)
    for (auto &tracker : trackers) {
        cv::Rect_<float> predRect = tracker.predict();
        results.push_back({predRect.x, predRect.y, predRect.x + predRect.width, predRect.y + predRect.height, 
                           static_cast<float>(tracker.getID())});
    }

    return results;
}


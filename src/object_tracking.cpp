#include "object_tracking.h"

// Function to calculate IoU (Intersection over Union)
float calculateIoU(const cv::Rect2f& boxA, const cv::Rect2f& boxB) {
    float xA = std::max(boxA.x, boxB.x);
    float yA = std::max(boxA.y, boxB.y);
    float xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    float yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    float interArea = std::max(0.f, xB - xA) * std::max(0.f, yB - yA);
    float boxAArea = boxA.width * boxA.height;
    float boxBArea = boxB.width * boxB.height;

    float iou = interArea / (boxAArea + boxBArea - interArea);
    return iou;
}

// Function to associate detections with trackers using IoU
void associateDetectionsWithTrackers(const std::vector<Object>& detections, std::vector<KalmanTracker>& trackers, float iouThreshold) {
    std::vector<bool> detectionUsed(detections.size(), false);
    
    // Loop through each tracker
    for (auto it = trackers.begin(); it != trackers.end();) {
        cv::Rect2f predictedBox = it->predict();

        float bestIoU = 0;
        int bestMatchIndex = -1;

        // Find the best match for the predicted tracker box
        for (size_t i = 0; i < detections.size(); ++i) {
            if (!detectionUsed[i]) {
                float iou = calculateIoU(predictedBox, detections[i].rect);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestMatchIndex = i;
                }
            }
        }

        // If a match is found, update the tracker
        if (bestIoU > iouThreshold) {
            it->update(detections[bestMatchIndex].rect);
            it->setLabel(detections[bestMatchIndex].label);
            detectionUsed[bestMatchIndex] = true;  
            ++it;  // Move to the next tracker
        } else {
            // If no match, increase the invisible count
            it->increaseConsecutiveInvisibleCount();

            // If invisible for too long, remove the tracker
            if (it->getConsecutiveInvisibleCount() > 3) { // Threshold: 5 frames
                it = trackers.erase(it);  // Remove and move to next
            } else {
                ++it;  // Move to next tracker
            }
        }
    }

    // Create new trackers for unmatched detections
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detectionUsed[i]) {
            trackers.push_back(KalmanTracker(detections[i].rect));  // Add a new tracker for unmatched detection
        }
    }
}

void drawObjects(cv::Mat &image, const cv::Rect2f &box, const int id, const int label, unsigned int scale) {
    // Choose the color
    int colorIndex = label % Config::COLOR_LIST.size(); // We have only defined 80 unique colors
    cv::Scalar color = cv::Scalar(Config::COLOR_LIST[colorIndex][0], Config::COLOR_LIST[colorIndex][1], Config::COLOR_LIST[colorIndex][2]);
    float meanColor = cv::mean(color)[0];
    cv::Scalar txtColor;
    if (meanColor > 0.5) {
        txtColor = cv::Scalar(0, 0, 0);
    } else {
        txtColor = cv::Scalar(255, 255, 255);
    }

    const auto &rect = box;

    // Draw rectangles and text
    char text[256];
    sprintf(text, "ID: %d  %s", id, Config::classNames[label].c_str());

    int baseLine = 0;
    cv::Size labelSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, scale, &baseLine);

    cv::Scalar txt_bk_color = color * 0.7 * 255;

    int x = box.x;
    int y = box.y + 1;

    cv::rectangle(image, rect, color * 255, scale + 1);

    cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(labelSize.width, labelSize.height + baseLine)), txt_bk_color, -1);

    cv::putText(image, text, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.35 * scale, txtColor, scale);
}

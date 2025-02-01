#include <gtest/gtest.h>
#include <iostream>
#include "kalman_tracker.h"

// Test if KalmanTracker initializes correctly
TEST(KalmanTrackerTest, Initialization) {
    cv::Rect2f initBox(100, 100, 50, 50);
    KalmanTracker tracker(initBox);

    EXPECT_EQ(tracker.getAge(), 0);
    EXPECT_GE(tracker.getID(), 0);  // ID should be non-negative
}

// Test Kalman filter prediction for multiple frames
TEST(KalmanTrackerTest, PredictStepMultipleFrames) {
    cv::Rect2f initBox(100, 100, 50, 50);
    KalmanTracker tracker(initBox);

    std::cout << "\nKalmanTracker Prediction Over Multiple Frames:\n";
    for (int i = 0; i < 10; ++i) {
        cv::Rect2f pred = tracker.predict();
        std::cout << "Frame " << i + 1 << " - Predicted: ("
                  << pred.x << ", " << pred.y << ", " 
                  << pred.width << ", " << pred.height << ")\n";

        // Basic expectation: Prediction shouldn't deviate too much from the initial position
        EXPECT_NEAR(pred.x, 100, 10.0);
        EXPECT_NEAR(pred.y, 100, 10.0);
    }
}

// Test Kalman filter update step
TEST(KalmanTrackerTest, UpdateStepMultipleFrames) {
    cv::Rect2f initBox(100, 100, 50, 50);
    KalmanTracker tracker(initBox);

    std::cout << "\nKalmanTracker Update Over Multiple Frames:\n";
    for (int i = 0; i < 30; ++i) {
        // Simulated detection with slight movement
        cv::Rect2f detection(100 + i * 2, 100 + i, 50, 50);
        tracker.update(detection);
        cv::Rect2f pred = tracker.predict();

        std::cout << "Frame " << i + 1 << " - Updated with: ("
                  << detection.x << ", " << detection.y << "), "
                  << "Predicted: (" << pred.x << ", " << pred.y << ")\n";

        EXPECT_NEAR(pred.x, detection.x, 10.0);
        EXPECT_NEAR(pred.y, detection.y, 10.0);
    }
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

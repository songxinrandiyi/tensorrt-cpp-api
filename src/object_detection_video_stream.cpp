#include "cmd_line_util.h"
#include "yolov8.h"
#include "object_tracking.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
    YoloV8Config config;
    std::string onnxModelPath;
    std::string trtModelPath;
    std::string inputVideo;
    std::string outputFileName;

    // Parse command line arguments
    if (!parseArguments(argc, argv, config, onnxModelPath, trtModelPath, inputVideo, outputFileName)) {
        return -1;
    }

    // Initialize YoloV8 object
    std::unique_ptr<YoloV8> yoloV8 = std::make_unique<YoloV8>(onnxModelPath, trtModelPath, config);
    cv::VideoCapture cap;

    try {
        cap.open(std::stoi(inputVideo));  // Try to open video from numeric index
    } catch (const std::exception &e) {
        cap.open(inputVideo);  // Otherwise, open the video file
    }

    if (!cap.isOpened()) {
        throw std::runtime_error("Unable to open video capture with input '" + inputVideo + "'");
    }

    // Check if outputFileName is non-empty for recording video
    cv::VideoWriter writer;
    if (!outputFileName.empty()) {
        // Set up video output if outputFileName is provided
        std::string outputVideo = "../outputs/" + outputFileName;  // Use the provided output file name

        // Get the first frame to initialize VideoWriter
        cv::Mat firstFrame;
        cap >> firstFrame;
        if (firstFrame.empty()) {
            throw std::runtime_error("Unable to read first frame.");
        }

        int codec = cv::VideoWriter::fourcc('H', '2', '6', '4');
        double fps = cap.get(cv::CAP_PROP_FPS);
        cv::Size frameSize(firstFrame.cols, firstFrame.rows);

        writer.open(outputVideo, codec, fps, frameSize, true);
        if (!writer.isOpened()) {
            throw std::runtime_error("Could not open the output video file for writing.");
        }
    }

    std::vector<KalmanTracker> trackers;

    while (true) {
        cv::Mat img;
        cap >> img;

        if (img.empty() || cv::waitKey(1) >= 0) {
            std::cout << "Exiting...\n";
            break;
        }

        const auto detections = yoloV8->detectObjects(img);

        associateDetectionsWithTrackers(detections, trackers, 0.3f);

        for (auto &tracker : trackers) {
            cv::Rect2f predictedBox = tracker.predict();
            drawObjects(img, predictedBox, tracker.getID(), tracker.getLabel(), 1);
        }

        // Save frame to video if outputFileName is not empty
        if (!outputFileName.empty()) {
            writer.write(img);  // Save the frame to the video file
        }

        cv::imshow("Object Detection and Tracking", img);
    }

    // Ensure video is saved safely if recording was enabled
    if (!outputFileName.empty()) {
        writer.release();
    }
    cap.release();
    cv::destroyAllWindows();
    std::cout << "Video processing completed." << std::endl;

    return 0;
}

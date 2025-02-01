<p align="center">
  <a href="https://github.com/songxinrandiyi/tensorrt-cpp-api">
    <img width="70%" src="assets/Roadtraffic video for object recognition(result).gif" alt="Result">
  </a>
</p>

# Object Tracking with Kalman Filter and YOLO Inference

## Overview
This project demonstrates object tracking using the Kalman Filter, integrated with YOLO for object detection. It leverages NVIDIA's TensorRT for fast inference of pre-trained deep learning models, enabling real-time object detection and tracking.

In this implementation:
1. YOLO performs object detection on frames of a video.
2. The Kalman Filter is applied for object tracking across frames.

This project is based on the original [TensorRT C++ API](https://github.com/cyrusbehr/tensorrt-cpp-api) for efficient TensorRT usage.

## Prerequisites
In addition to the prerequisites listed in the [TensorRT C++ API](https://github.com/cyrusbehr/tensorrt-cpp-api), the following are required:
   
1. **GTest**: Used for unit testing the Kalman Filter implementation. Install [Google Test](https://github.com/google/googletest) for running tests.

## Building the Project
- `mkdir build`
- `cd build`
- `cmake ..`
- `make -j$(nproc)`

## Running the Executables
To run the executables, use the following commands:

1. Using camera as input:
`./detect_object_video --model /path/to/your/onnx/model.onnx --input 0`

2. Using video as input:
`./detect_object_video --model /path/to/your/onnx/model.onnx --input /path/to/videofile`

3. To record the result:
`./detect_object_video --model /path/to/your/onnx/model.onnx --input /path/to/videofile --output validmp4filename` The video will be saved in the `outputs` folder.

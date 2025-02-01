cmake_minimum_required(VERSION 3.18)
project(tensorrt_cpp_api)

# Use ccache to speed up rebuilds
include(cmake/ccache.cmake)

# Set C++ version and optimization level
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG -Wno-deprecated-declarations")

# For finding FindTensorRT.cmake
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# TODO: Specify the path to TensorRT root dir
if (NOT TensorRT_DIR)
    set(TensorRT_DIR "/usr/src/tensorrt")
endif()
# Use the correct version of CUDA
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)

# We require CUDA, OpenCV, and TensorRT
find_package(TensorRT REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(fmt REQUIRED)
find_package(GTest REQUIRED)

add_library(tensorrt_cpp_api SHARED
        src/engine.cpp)

target_include_directories(tensorrt_cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS} include)
target_link_libraries(tensorrt_cpp_api PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES} fmt::fmt)

# Build the YoloV8 library
add_library(YoloV8_TRT SHARED src/yolov8.cpp)
target_link_libraries(YoloV8_TRT PUBLIC tensorrt_cpp_api)

add_library(Tracking SHARED
    src/kalman_tracker.cpp
    src/object_tracking.cpp)

target_include_directories(Tracking PUBLIC ${OpenCV_INCLUDE_DIRS} include)
target_link_libraries(Tracking PUBLIC ${OpenCV_LIBS})

add_executable(detect_object_video src/object_detection_video_stream.cpp)
target_link_libraries(detect_object_video YoloV8_TRT Tracking)

# Enable testing
enable_testing()

add_executable(test_kalman_tracker tests/test_kalman_tracker.cpp)
target_link_libraries(test_kalman_tracker PUBLIC GTest::GTest GTest::Main Tracking)

add_test(NAME KalmanTrackerTests COMMAND test_kalman_tracker)


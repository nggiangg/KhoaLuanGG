cmake_minimum_required(VERSION 3.0.2)
project(jetracer_camera_reader)

# Add C++14 support
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ROS packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  cv_bridge
  image_transport
  sensor_msgs
)

# Find OpenCV
find_package(OpenCV 4 REQUIRED)

# Find CUDA
find_package(CUDA REQUIRED)

# Find TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/x86_64-linux-gnu /usr/include
  PATH_SUFFIXES tensorrt
)

find_library(TENSORRT_LIBRARY_INFER nvinfer
  HINTS /usr/lib/x86_64-linux-gnu /usr/lib
)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser
  HINTS /usr/lib/x86_64-linux-gnu /usr/lib
)

# Debug output
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDA_VERSION}")
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT libraries: ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_ONNXPARSER}")

# Configure catkin package
catkin_package(
  CATKIN_DEPENDS roscpp cv_bridge image_transport sensor_msgs
  DEPENDS OpenCV
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
)

# Add TensorRT detector
add_library(onnx_detector src/ONNXDetector.cpp)
target_link_libraries(onnx_detector
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_ONNXPARSER}
)

# Add executable
add_executable(camera_reader src/camera_reader.cpp)
add_dependencies(camera_reader ${catkin_EXPORTED_TARGETS})

# Add SSD node
add_executable(ssd_node src/ssd_node.cpp src/ssd_detector.cpp)
target_link_libraries(ssd_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  nvinfer
  nvinfer_plugin
  nvonnxparser
)
# Link libraries
target_link_libraries(camera_reader
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${CUDA_LIBRARIES}
  ${TENSORRT_LIBRARY_INFER}
  ${TENSORRT_LIBRARY_ONNXPARSER}
  onnx_detector
)


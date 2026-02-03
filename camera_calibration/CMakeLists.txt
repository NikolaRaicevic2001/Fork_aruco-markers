cmake_minimum_required(VERSION 3.16.3)
project(camera_calibration)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${OpenCV_LIBRARY_DIRS})

# Find librealsense2
find_path(REALSENSE2_INCLUDE_DIR NAMES librealsense2/rs.hpp
  PATHS /usr/include /usr/local/include
)
find_library(REALSENSE2_LIBRARY NAMES realsense2 librealsense2
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib
)
if(NOT REALSENSE2_INCLUDE_DIR OR NOT REALSENSE2_LIBRARY)
  message(FATAL_ERROR "Could not find librealsense2 headers or library")
endif()

# Define the executable
add_executable(camera_calibration src/main.cpp)
target_include_directories(camera_calibration PRIVATE ${REALSENSE2_INCLUDE_DIR})
target_link_libraries(camera_calibration PRIVATE ${OpenCV_LIBRARIES} ${REALSENSE2_LIBRARY})
target_compile_options(camera_calibration PRIVATE -O3)
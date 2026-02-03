cmake_minimum_required(VERSION 3.16.3)
project(pose_estimation)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/../common/include)

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

# Degfine the executable
add_executable(pose_estimation src/main.cpp)
target_link_directories(pose_estimation PRIVATE ${REALSENSE2_INCLUDE_DIR})
target_link_libraries(pose_estimation PRIVATE ${OpenCV_LIBRARIES} ${REALSENSE2_LIBRARY})
target_compile_options(pose_estimation PRIVATE -O3)
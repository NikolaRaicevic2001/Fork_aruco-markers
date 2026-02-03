cmake_minimum_required(VERSION 3.16.3)
project(detect_markers)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

include_directories(${OPENCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/../common/include)

link_directories(${OpenCV_LIBRARY_DIRS})


add_executable(detect_markers src/detect_markers.cpp)
target_link_libraries(detect_markers ${OpenCV_LIBRARIES})
target_compile_options(detect_markers PRIVATE -O3 -std=c++11)

# Find librealsense2
find_path(REALSENSE2_INCLUDE_DIR NAMES librealsense2/rs.hpp
  PATHS /usr/include /usr/local/include
)
find_library(REALSENSE2_LIBRARY NAMES realsense2
  PATHS /usr/lib/x86_64-linux-gnu /usr/local/lib
)
if(NOT REALSENSE2_INCLUDE_DIR OR NOT REALSENSE2_LIBRARY)
  message(FATAL_ERROR "Could not find librealsense2 headers or library")
endif()

add_executable(detect_markers_realsense src/detect_markers_realsense.cpp)
target_include_directories(detect_markers_realsense PRIVATE ${REALSENSE2_INCLUDE_DIR})
target_link_libraries(detect_markers_realsense PRIVATE ${OpenCV_LIBRARIES} ${REALSENSE2_LIBRARY})
target_compile_options(detect_markers_realsense PRIVATE -O3)




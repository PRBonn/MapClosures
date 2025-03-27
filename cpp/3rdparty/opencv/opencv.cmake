# MIT License
#
# Copyright (c) 2024 Tiziano Guadagnino, Meher Malladi, Saurabh Gupta
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

set(BUILD_opencv_core ON CACHE BOOL "Build OpenCV core module")
set(BUILD_opencv_features2d ON CACHE BOOL "Build OpenCV features2d module")
set(BUILD_opencv_imgproc ON CACHE BOOL "Build OpenCV imgproc module")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
set(BUILD_WITH_STATIC_CRT OFF CACHE BOOL "Build with statically linked CRT")
set(BUILD_JAVA OFF CACHE BOOL "Build Java bindings")
set(BUILD_PERF_TESTS OFF CACHE BOOL "Build performance tests")
set(BUILD_PROTOBUF OFF CACHE BOOL "Build protobuf")
set(BUILD_TESTS OFF CACHE BOOL "Build tests")
set(BUILD_opencv_apps OFF CACHE BOOL "Build OpenCV apps")
set(BUILD_opencv_calib3d OFF CACHE BOOL "Build OpenCV calib3d module")
set(BUILD_opencv_dnn OFF CACHE BOOL "Build OpenCV DNN module")
set(BUILD_opencv_flann OFF CACHE BOOL "Build OpenCV flann module")
set(BUILD_opencv_gapi OFF CACHE BOOL "Build OpenCV G-API")
set(BUILD_opencv_highgui OFF CACHE BOOL "Build OpenCV HighGUI")
set(BUILD_opencv_imgcodecs OFF CACHE BOOL "Build OpenCV imgcodecs")
set(BUILD_opencv_java_bindings_generator OFF CACHE BOOL "Build OpenCV Java bindings generator")
set(BUILD_opencv_js_bindings_generator OFF CACHE BOOL "Build OpenCV JavaScript bindings generator")
set(BUILD_opencv_ml OFF CACHE BOOL "Build OpenCV machine learning module")
set(BUILD_opencv_objc_bindings_generator OFF CACHE BOOL
                                                   "Build OpenCV Objective-C bindings generator")
set(BUILD_opencv_objdetect OFF CACHE BOOL "Build OpenCV object detection module")
set(BUILD_opencv_photo OFF CACHE BOOL "Build OpenCV photo module")
set(BUILD_opencv_python3 OFF CACHE BOOL "Build OpenCV Python 3 bindings")
set(BUILD_opencv_python_bindings_generator OFF CACHE BOOL "Build OpenCV Python bindings generator")
set(BUILD_opencv_python_tests OFF CACHE BOOL "Build OpenCV Python tests")
set(BUILD_opencv_stitching OFF CACHE BOOL "Build OpenCV stitching module")
set(BUILD_opencv_video OFF CACHE BOOL "Build OpenCV video module")
set(BUILD_opencv_videoio OFF CACHE BOOL "Build OpenCV video IO module")
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type")

message(STATUS "Fetching OpenCV from Github")
include(FetchContent)
FetchContent_Declare(opencv URL https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz)
FetchContent_MakeAvailable(opencv)
# OpenCV_INCLUDE_DIRS is set by OpenCVConfig.cmake and is unavailable after simply building
set(OpenCV_INCLUDE_DIRS "${opencv_SOURCE_DIR}/include")

add_library(OpenCV INTERFACE)
target_link_libraries(OpenCV INTERFACE opencv_core opencv_features2d opencv_imgproc)
# Taken from an issue in the OpenCV project (https://github.com/opencv/opencv/issues/20548#issuecomment-1325751099)
# Include files from OpenCV modules are not forwarded to the corresponding targets, so we need to add them manually
target_include_directories(
  OpenCV
  INTERFACE ${OPENCV_CONFIG_FILE_INCLUDE_DIR} ${OPENCV_MODULE_opencv_core_LOCATION}/include
            ${OPENCV_MODULE_opencv_features2d_LOCATION}/include
            ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include ${OpenCV_INCLUDE_DIRS})

# try to hide this madness to the end-user
set(OpenCV_LIBS OpenCV)

# MIT License
#
# Copyright (c) 2024 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
# Ignacio Vizzo, Cyrill Stachniss.
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
if(CMAKE_VERSION VERSION_GREATER 3.24)
  cmake_policy(SET CMP0135 OLD)
endif()

if(${USE_SYSTEM_EIGEN3})
  find_package(Eigen3 QUIET NO_MODULE)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/eigen/eigen.cmake)
endif()

if(${USE_SYSTEM_TBB})
  find_package(TBB QUIET NO_MODULE)
endif()
if(NOT TARGET TBB::tbb)
  include(${CMAKE_CURRENT_LIST_DIR}/tbb/tbb.cmake)
endif()

if(${USE_SYSTEM_OPENCV})
  find_package(OpenCV QUIET NO_MODULE)
endif()
if(NOT TARGET opencv_features2d)
  include(${CMAKE_CURRENT_LIST_DIR}/opencv/opencv.cmake)
endif()
# Taken from an issue in the OpenCV project (https://github.com/opencv/opencv/issues/20548#issuecomment-1325751099)
add_library(OpenCV4 INTERFACE)
target_link_libraries(OpenCV4 INTERFACE opencv_core opencv_features2d opencv_imgproc)
target_include_directories(
  OpenCV4
  INTERFACE ${OPENCV_CONFIG_FILE_INCLUDE_DIR} ${OPENCV_MODULE_opencv_core_LOCATION}/include
            ${OPENCV_MODULE_opencv_features2d_LOCATION}/include
            ${OPENCV_MODULE_opencv_imgproc_LOCATION}/include ${OpenCV_INCLUDE_DIRS})

include(${CMAKE_CURRENT_LIST_DIR}/hbst/hbst.cmake)

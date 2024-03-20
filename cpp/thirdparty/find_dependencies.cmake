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

if(${USE_SYSTEM_EIGEN3})
  find_package(Eigen3 QUIET NO_MODULE)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/eigen/eigen.cmake)
endif()

if(${USE_SYSTEM_OPENCV})
  find_package(OpenCV QUIET)
else()
  include(${CMAKE_CURRENT_LIST_DIR}/opencv/opencv.cmake)
endif()
# Taken from a issue in the OpenCv project (https://github.com/opencv/opencv/issues/20548#issuecomment-1325751099)
# for some reason OpenCV does not want to convert to a FetchContent Friendly format so we need to use this trick
add_library(OpenCV4 INTERFACE)
target_link_libraries(OpenCV4 INTERFACE ${OpenCV_LIBS})
target_include_directories(OpenCV4 INTERFACE
          ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
          ${OPENCV_MODULE_opencv_core_LOCATION}/include
          ${OPENCV_MODULE_opencv_highgui_LOCATION}/include
          ${OpenCV_INCLUDE_DIRS}
          )

include(${CMAKE_CURRENT_LIST_DIR}/hbst/hbst.cmake)

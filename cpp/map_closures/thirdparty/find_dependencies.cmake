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

# CMake arguments for configuring ExternalProjects.
set(ExternalProject_CMAKE_ARGS
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON)

if(${USE_SYSTEM_EIGEN3})
  find_package(EIGEN3 QUIET NO_MODULE)
endif()
if(NOT ${USE_SYSTEM_EIGEN3} OR NOT TARGET Eigen3::Eigen)
  set(USE_SYSTEM_EIGEN3 OFF PARENT_SCOPE)
  include(${CMAKE_CURRENT_LIST_DIR}/eigen/eigen.cmake)
endif()

set(SRRG_HBST_HAS_EIGEN true)
add_definitions(-DSRRG_HBST_HAS_EIGEN)

find_package(OpenCV REQUIRED)
if(${OpenCV_FOUND})
  set(SRRG_HBST_HAS_OPENCV true)
  add_definitions(-DSRRG_HBST_HAS_OPENCV)
  add_definitions(-DSRRG_MERGE_DESCRIPTORS)
  include(${CMAKE_CURRENT_LIST_DIR}/hbst/hbst.cmake)
endif()

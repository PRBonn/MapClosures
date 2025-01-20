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

# Inspired by KISS-ICP https://github.com/PRBonn/kiss-icp/blob/d51329990462ad85e6802810e69feda4d30017c3/cpp/kiss_icp/3rdparty/find_dependencies.cmake#L28C1-L38C14
# AN IMPORTANT DIFFERENCE: In this case we need to use a macro instead of a function. you can find the differences here: https://cmake.org/cmake/help/latest/command/macro.html. In essence, a function
# will transfer the control from the calling scope to the function and the back to the calling scope once the function
# as done whatever it has to do. This transfer will cause how black magic forwarded include dirs for the OpenCV targets to dont work anymore, as this will be confined to the local scope of the cmake function.
# The easier solution for us was to just change function to macro, as the later will just copy paste the content adjusting the parameters (exactly like a C macro). This is horrible of course, but it is caused by the build system of OpenCV.
macro(find_external_dependency PACKAGE_NAME TARGET_NAME INCLUDED_CMAKE_PATH)
  string(TOUPPER ${PACKAGE_NAME} PACKAGE_NAME_UP)
  set(USE_FROM_SYSTEM_OPTION "USE_SYSTEM_${PACKAGE_NAME_UP}")
  if(${${USE_FROM_SYSTEM_OPTION}})
    find_package(${PACKAGE_NAME} QUIET NO_MODULE)
  endif()
  if(NOT TARGET ${TARGET_NAME})
    include(${INCLUDED_CMAKE_PATH})
  endif()
endmacro()

find_external_dependency("Eigen3" "Eigen3::Eigen" "${CMAKE_CURRENT_LIST_DIR}/eigen/eigen.cmake")
find_external_dependency("OpenCV" "opencv_features2d"
                         "${CMAKE_CURRENT_LIST_DIR}/opencv/opencv.cmake")
find_external_dependency("Sophus" "Sophus::Sophus" "${CMAKE_CURRENT_LIST_DIR}/sophus/sophus.cmake")

include(${CMAKE_CURRENT_LIST_DIR}/hbst/hbst.cmake)

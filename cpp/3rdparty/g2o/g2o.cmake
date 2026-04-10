# MIT License

# Copyright (c) 2025 Tiziano Guadagnino, Benedikt Mersch, Saurabh Gupta, Cyrill
# Stachniss.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
include(FetchContent)

set(G2O_BUILD_APPS OFF CACHE BOOL "Build g2o apps")
set(G2O_BUILD_EXAMPLES OFF CACHE BOOL "Build g2o examples")
set(G2O_USE_LOGGING OFF CACHE BOOL "Try to use spdlog for logging")
set(G2O_BUILD_WITH_MARCH_NATIVE OFF CACHE BOOL "Build with \"-march native\"")
set(G2O_USE_OPENMP OFF CACHE BOOL "Build g2o with OpenMP support (EXPERIMENTAL)")
set(G2O_USE_OPENGL OFF CACHE BOOL "Build g2o with OpenGL support for visualization")
set(BUILD_SHARED_LIBS OFF
    CACHE BOOL "Build Shared Libraries (preferred and required for the g2o plugin system)")
set(G2O_BUILD_SLAM3D_ADDON_TYPES OFF CACHE BOOL "no SLAM 3D addons")
set(G2O_BUILD_SIM3_TYPES OFF CACHE BOOL "no sim3")
set(G2O_BUILD_SBA_TYPES OFF CACHE BOOL "no sparse bundle adjustment")
set(G2O_BUILD_ICP_TYPES OFF CACHE BOOL "no icp")
set(G2O_BUILD_DATA_TYPES OFF CACHE BOOL "Build SLAM2D data types")
set(G2O_BUILD_SLAM2D_TYPES OFF CACHE BOOL "no SLAM 2D types")
set(G2O_BUILD_SCLAM2D_TYPES OFF CACHE BOOL "Build SCLAM2D types")
set(G2O_BUILD_SLAM2D_ADDON_TYPES OFF CACHE BOOL "Build SLAM2D addon types")

set(G2O_USE_CHOLMOD ON CACHE BOOL "Build g2o with CHOLMOD support")
set(G2O_BUILD_SLAM3D_TYPES ON CACHE BOOL "need just slam 3d types")

FetchContent_Declare(
  g2o SYSTEM EXCLUDE_FROM_ALL
  URL https://github.com/RainerKuemmerle/g2o/archive/refs/tags/20241228_git.tar.gz)
FetchContent_MakeAvailable(g2o)
add_library(g2o::core ALIAS core)
add_library(g2o::types_slam3d ALIAS types_slam3d)
add_library(g2o::solver_cholmod ALIAS solver_cholmod)

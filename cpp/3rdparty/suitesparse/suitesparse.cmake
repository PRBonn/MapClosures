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

set(SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;amd;camd;ccolamd;colamd;cholmod"
    CACHE STRING "Only install required packages")

set(SUITESPARSE_USE_PYTHON OFF CACHE BOOL "build Python interfaces for SuiteSparse packages (SPEX)")
set(SUITESPARSE_USE_OPENMP OFF CACHE BOOL "Use OpenMP in libraries by default if available")
set(SUITESPARSE_USE_CUDA OFF CACHE BOOL "Build SuiteSparse with CUDA support")
set(CHOLMOD_SUPERNODAL OFF CACHE BOOL "Build SuiteSparse SuperNodal library")
set(BUILD_TESTING OFF CACHE BOOL "SuiteSparse Build Testing")
set(SUITESPARSE_USE_FORTRAN OFF CACHE BOOL "use Fortran")
set(SUITESPARSE_DEMOS OFF CACHE BOOL "SuiteSparse Demos")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "shared off")
# Keep this to avoid BLAS config error
set(SuiteSparse_BLAS_integer "int64_t" CACHE STRING "BLAS Integer type")

set(SUITESPARSE_USE_STRICT ON CACHE BOOL "treat all _USE__ settings as strict")
set(BUILD_STATIC_LIBS ON CACHE BOOL "static on")

FetchContent_Declare(
  suitesparse SYSTEM EXCLUDE_FROM_ALL
  URL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v7.10.1.tar.gz
  PATCH_COMMAND patch -p1 < ${CMAKE_CURRENT_LIST_DIR}/suitesparse.patch UPDATE_DISCONNECTED 1)
FetchContent_MakeAvailable(suitesparse)
if(TARGET SuiteSparse::CHOLMOD)
  set(SuiteSparse_CHOLMOD_FOUND ON CACHE BOOL "SuiteSparse::CHOLMOD Exists")
  set(SuiteSparse_FOUND ON CACHE BOOL "SuiteSparse exists if target SuiteSparse::CHOLMOD exists")
  set(SuiteSparse_NO_CMAKE ON CACHE BOOL "Do not try to find SuiteSparse")
endif()

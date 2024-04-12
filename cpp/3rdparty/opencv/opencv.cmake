set(BUILD_opencv_core ON CACHE BOOL "Build OpenCV core module")
set(BUILD_opencv_features2d ON CACHE BOOL "Build OpenCV features2d module")
set(BUILD_opencv_imgproc ON CACHE BOOL "Build OpenCV imgproc module")

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
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

include(FetchContent)
FetchContent_Declare(opencv URL https://github.com/opencv/opencv/archive/refs/tags/4.9.0.tar.gz
)# downloading the zip is faster than cloning a repo
# GIT_REPOSITORY https://github.com/opencv/opencv.git GIT_TAG 4.x GIT_SHALLOW
# TRUE GIT_PROGRESS TRUE)
FetchContent_MakeAvailable(opencv)
# OpenCV_INCLUDE_DIRS is set by OpenCVConfig.cmake and is unavailable after simply building
set(OpenCV_INCLUDE_DIRS "${opencv_SOURCE_DIR}/include")

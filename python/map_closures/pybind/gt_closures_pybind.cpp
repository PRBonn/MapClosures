// MIT License
//
// Copyright (c) 2024 Saurabh Gupta, Tiziano Guadagnino, Benedikt Mersch,
// Ignacio Vizzo, Cyrill Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <Eigen/Core>
#include <vector>

#include "gt_closures/GTClosures.hpp"
#include "stl_vector_eigen.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);
PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector2i>);

namespace py = pybind11;
using namespace py::literals;

namespace gt_closures {

PYBIND11_MODULE(gt_closures_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);
    auto vector2ivector = pybind_eigen_vector_of_vector<Eigen::Vector2i>(
        m, "_Vector2iVector", "std::vector<Eigen::Vector2i>",
        py::py_array_to_vectors_int<Eigen::Vector2i>);

    py::class_<GTClosures, std::shared_ptr<GTClosures>> gt_closures(m, "_GTClosures", "");
    gt_closures
        .def(py::init<int, double, double, double, double>(), "dataset_size"_a,
             "sampling_distance"_a, "overlap_threshold"_a, "voxel_size"_a, "max_range"_a)
        .def("_AddPointCloud", &GTClosures::AddPointCloud, "idx"_a, "pointcloud"_a, "pose"_a)
        .def("_GetSegments", &GTClosures::GetSegments)
        .def("_ComputeClosuresForQuerySegment", &GTClosures::ComputeClosuresForQuerySegment,
             "query_segment_idx"_a);
}
}  // namespace gt_closures

// MIT License
//
// Copyright (c) 2026 Saurabh Gupta
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
#include <memory>
#include <vector>

#include "map_closures/VoxelMap.hpp"
#include "stl_vector_eigen.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace py = pybind11;
using namespace py::literals;

namespace map_closures {

PYBIND11_MODULE(voxel_map_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_Vector3dVector", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    py::class_<VoxelMap> internal_map(m, "_VoxelMap", "Don't use this");
    internal_map.def(py::init<double, double>(), "voxel_size"_a, "max_distance"_a)
        .def("_clear", &VoxelMap::Clear)
        .def("_num_voxels", &VoxelMap::NumVoxels)
        .def("_add_points", &VoxelMap::AddPoints, "points"_a)
        .def("_integrate_frame", &VoxelMap::IntegrateFrame, "points"_a, "pose"_a)
        .def("_point_cloud", &VoxelMap::Pointcloud)
        .def("_per_voxel_mean_and_normal", &VoxelMap::PerVoxelMeanAndNormal)
        .def("_remove_far_away_points", &VoxelMap::RemovePointsFarFromLocation, "origin"_a);
}
}  // namespace map_closures

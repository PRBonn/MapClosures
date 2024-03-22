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
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <tuple>
#include <vector>

#include "pipeline/MapClosures.hpp"
#include "stl_vector_eigen.h"

PYBIND11_MAKE_OPAQUE(std::vector<Eigen::Vector3d>);

namespace py = pybind11;
using namespace py::literals;

namespace map_closures {
Config GetConfigFromYAML(const py::dict &yaml_cfg) {
    Config cpp_config;
    cpp_config.density_threshold = yaml_cfg["density_threshold"].cast<float>();
    cpp_config.density_map_resolution = yaml_cfg["density_map_resolution"].cast<float>();
    cpp_config.hamming_distance_threshold = yaml_cfg["hamming_distance_threshold"].cast<int>();
    return cpp_config;
}

PYBIND11_MODULE(map_closures_pybind, m) {
    auto vector3dvector = pybind_eigen_vector_of_vector<Eigen::Vector3d>(
        m, "_VectorEigen3d", "std::vector<Eigen::Vector3d>",
        py::py_array_to_vectors_double<Eigen::Vector3d>);

    py::class_<MapClosures, std::shared_ptr<MapClosures>> map_closures(m, "_MapClosures", "");
    map_closures
        .def(py::init([](const py::dict &cfg) {
                 auto config = GetConfigFromYAML(cfg);
                 return std::make_shared<MapClosures>(config);
             }),
             "config"_a)
        .def(
            "_MatchAndAddLocalMap",
            [](MapClosures &self, const int map_idx, const std::vector<Eigen::Vector3d> &local_map,
               const int top_k) {
                const auto &[ref_map_indices, density_map_cv] =
                    self.MatchAndAddLocalMap(map_idx, local_map, top_k);
                Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> density_map_eigen;
                cv::cv2eigen(density_map_cv, density_map_eigen);
                return std::make_tuple(ref_map_indices, density_map_eigen);
            },
            "map_idx"_a, "local_map"_a, "top_k"_a)
        .def(
            "_DetectLoopClosureAndAddToDatabase",
            [](MapClosures &self, const int map_idx,
               const std::vector<Eigen::Vector3d> &local_map) {
                const auto &[ref_idx, query_idx, T, num_inliers] =
                    self.DetectLoopClosureAndAddToDatabase(map_idx, local_map);
                return std::make_tuple(ref_idx, query_idx, T, num_inliers);
            },
            "map_idx", "local_map")
        .def("_CheckForClosure", &MapClosures::CheckForClosure, "ref_idx"_a, "query_idx"_a);
}
}  // namespace map_closures
